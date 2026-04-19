// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
)

// DefaultReplayExtractModel is the cheap extraction model used when the
// caller does not override it via [ReplayOpts.ExtractModel]. Haiku is
// strong enough to produce structured JSON at replay volume without
// dominating the run's cost budget.
const DefaultReplayExtractModel = "claude-haiku-4-5"

// defaultReplayConcurrency bounds the in-flight extraction LLM calls
// when the caller leaves [ReplayOpts.Concurrency] at zero. Tuned for
// cheap extract models whose rate limits sit comfortably above 16 RPS.
const defaultReplayConcurrency = 32

// maxReplayConcurrency guards against pathological settings that would
// trip upstream rate limits regardless of how fast the workers are.
const maxReplayConcurrency = 512

// ReplayOpts configures the replay ingest pipeline.
type ReplayOpts struct {
	// ProjectPath is passed through to the extraction pipeline so the
	// project slug matches the retrieval path. Defaults to "/eval/lme".
	ProjectPath string

	// ExtractModel overrides the extraction model. Empty selects
	// [DefaultReplayExtractModel]. Useful when an operator wants to pin
	// extraction on a cheaper or stronger SKU than the actor / judge.
	ExtractModel string

	// Concurrency caps the number of in-flight extraction LLM calls.
	// Defaults to 16 when zero; clamped to [1, 128]. The write step
	// (ApplyExtractions) stays serial so the store commits once.
	Concurrency int

	// Contextualiser, when non-nil, generates a short situating prefix
	// per extracted fact so BM25 and the dense encoder see the
	// enclosing session context around every chunk.
	Contextualiser *memory.Contextualiser
}

// ReplayResult records what the replay ingest produced.
type ReplayResult struct {
	SessionsProcessed int
	FactsExtracted    int
	FactsWritten      int
	Warnings          []string
}

// sessionData is the intermediate carrier between deduplication and
// extraction. Holds a rendered session transcript with the session id
// and date attached so the extract worker can inject the right
// frontmatter metadata onto every fact it produces.
type sessionData struct {
	id   string
	date string
	text string
}

// IngestReplay processes an LME dataset's haystack sessions through the
// memory extraction pipeline and persists the resulting facts back to
// the brain store. This is the canonical scoring path: the downstream
// runner then retrieves over the extracted memories rather than raw
// session text, mirroring production retrieval shape.
//
// Deduplication: when the same session id appears across multiple
// questions the session is only extracted once.
func IngestReplay(
	ctx context.Context,
	store brain.Store,
	ds *Dataset,
	provider llm.Provider,
	opts ReplayOpts,
) (*ReplayResult, error) {
	if provider == nil {
		return nil, fmt.Errorf("lme replay ingest: provider is required")
	}
	if store == nil {
		return nil, fmt.Errorf("lme replay ingest: store is required")
	}
	if ds == nil {
		return nil, fmt.Errorf("lme replay ingest: dataset is required")
	}

	projectPath := opts.ProjectPath
	if projectPath == "" {
		projectPath = "/eval/lme"
	}
	extractModel := opts.ExtractModel
	if extractModel == "" {
		extractModel = DefaultReplayExtractModel
	}

	mem := memory.New(store)
	result := &ReplayResult{}

	sessions := deduplicateSessions(ds.Questions)
	if len(sessions) == 0 {
		return result, nil
	}

	concurrency := opts.Concurrency
	if concurrency <= 0 {
		concurrency = defaultReplayConcurrency
	}
	if concurrency > maxReplayConcurrency {
		concurrency = maxReplayConcurrency
	}
	if concurrency > len(sessions) {
		concurrency = len(sessions)
	}

	total := len(sessions)
	startTime := time.Now()
	fmt.Fprintf(os.Stderr, "[replay] extracting from %d unique sessions, %d workers, model=%s\n",
		total, concurrency, extractModel)
	var done atomic.Int64
	var contextualisedFacts atomic.Int64

	// Stage 1: fan extraction + contextualisation across workers. Each
	// worker extracts a session's facts, post-processes the content
	// (date prefixes, session ids, auto-tags), then kicks off the
	// per-fact contextualiser call before returning. The write step
	// stays single-writer so the store commits once per replay.
	type sessionResult struct {
		sess             sessionData
		extract          []memory.ExtractedMemory
		modifiedOverride string
		sessionDateISO   string
		err              error
	}

	jobs := make(chan sessionData)
	results := make(chan sessionResult, concurrency)

	var wg sync.WaitGroup
	wg.Add(concurrency)
	for range concurrency {
		go func() {
			defer wg.Done()
			for sess := range jobs {
				messages := sessionToMessages(sess)
				if len(messages) < 2 {
					results <- sessionResult{
						sess: sess,
						err:  fmt.Errorf("too few messages (%d)", len(messages)),
					}
					continue
				}
				callStart := time.Now()
				extracted, err := memory.ExtractFromMessages(ctx, provider, extractModel, mem, projectPath, messages)
				n := done.Add(1)
				// Throttle logs at higher concurrency settings so stderr
				// does not flood.
				if n%50 == 0 || n == int64(total) {
					elapsed := time.Since(startTime).Seconds()
					rate := float64(n) / elapsed
					var eta time.Duration
					if rate > 0 {
						eta = time.Duration(float64(int64(total)-n)/rate) * time.Second
					}
					status := "ok"
					if err != nil {
						status = fmt.Sprintf("err=%v", err)
					}
					fmt.Fprintf(os.Stderr,
						"[replay] %d/%d session=%s %s (%dms) rate=%.1f/s eta=%s\n",
						n, total, sess.id, status,
						time.Since(callStart).Milliseconds(),
						rate, eta.Truncate(time.Second))
				}
				if err != nil {
					results <- sessionResult{sess: sess, err: err}
					continue
				}
				// Post-process the session's facts (date injection,
				// session ids, auto-tags) so the contextualise call
				// sees the same body the collect loop will persist.
				modifiedOverride, sessionDateISO := postProcessSessionFacts(sess, extracted)
				if opts.Contextualiser.Enabled() && len(extracted) > 0 {
					summary := sessionContextSummary(sess)
					for i := range extracted {
						prefix := opts.Contextualiser.BuildPrefix(ctx, sess.id, summary, extracted[i].Content)
						if prefix != "" {
							extracted[i].ContextPrefix = prefix
							contextualisedFacts.Add(1)
						}
					}
				}
				results <- sessionResult{
					sess:             sess,
					extract:          extracted,
					modifiedOverride: modifiedOverride,
					sessionDateISO:   sessionDateISO,
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Respect ctx so a cancelled run stops promptly rather than draining
	// the whole queue.
	go func() {
		defer close(jobs)
		for _, sess := range sessions {
			select {
			case <-ctx.Done():
				return
			case jobs <- sess:
			}
		}
	}()

	// Stage 2: collect results and apply to the store. Single-writer
	// so we keep one commit per ingest rather than one per session.
	var pending []memory.ExtractedMemory

	for r := range results {
		if r.err != nil {
			result.Warnings = append(result.Warnings,
				fmt.Sprintf("session %s: extraction failed: %v", r.sess.id, r.err))
			continue
		}
		result.SessionsProcessed++
		if len(r.extract) == 0 {
			slog.Debug("lme replay: no facts extracted", "session", r.sess.id)
			continue
		}

		result.FactsExtracted += len(r.extract)
		pending = append(pending, r.extract...)
	}

	slug := memory.ProjectSlug(projectPath)

	if ctx.Err() != nil {
		return result, ctx.Err()
	}

	if len(pending) > 0 {
		fmt.Fprintf(os.Stderr, "[replay] applying %d extracted facts to store\n", len(pending))
		applyStart := time.Now()
		if err := mem.ApplyExtractions(ctx, slug, pending); err != nil {
			result.Warnings = append(result.Warnings,
				fmt.Sprintf("apply extractions failed: %v", err))
			return result, nil
		}
		result.FactsWritten = len(pending)
		fmt.Fprintf(os.Stderr, "[replay] apply done in %s (%d facts written); ingest total %s\n",
			time.Since(applyStart).Truncate(time.Second), result.FactsWritten,
			time.Since(startTime).Truncate(time.Second))
	}

	ingestWall := time.Since(startTime)
	ctxFresh := contextualisedFacts.Load()
	cacheHits := int64(0)
	if opts.Contextualiser.Enabled() {
		if diff := int64(result.FactsExtracted) - ctxFresh; diff > 0 {
			cacheHits = diff
		}
	}
	fmt.Fprintf(os.Stderr,
		"[replay] extracted=%d sessions, contextualised=%d facts (cache_hits=%d), ingest_wall=%ss\n",
		result.SessionsProcessed, ctxFresh, cacheHits,
		fmtDurationSecs(ingestWall))

	return result, nil
}

// deduplicateSessions flattens every question's haystack sessions into
// a unique list keyed by session id. When the same id appears more
// than once, only the first transcript survives; the LME dataset never
// ships conflicting content under one id so this is safe.
func deduplicateSessions(questions []Question) []sessionData {
	seen := make(map[string]bool)
	var out []sessionData
	for _, q := range questions {
		for i, sess := range q.HaystackSessions {
			sid := ""
			if i < len(q.SessionIDs) {
				sid = q.SessionIDs[i]
			}
			if sid == "" {
				sid = fmt.Sprintf("%s-s%d", q.ID, i)
			}
			if seen[sid] {
				continue
			}
			seen[sid] = true

			date := ""
			if i < len(q.HaystackDates) {
				date = q.HaystackDates[i]
			}

			var b strings.Builder
			for _, m := range sess {
				fmt.Fprintf(&b, "[%s]: %s\n\n", m.Role, m.Content)
			}
			out = append(out, sessionData{
				id:   sid,
				date: date,
				text: b.String(),
			})
		}
	}
	return out
}

// postProcessSessionFacts applies temporal metadata, session ids, and
// auto-tag merging that every replayed fact needs before it reaches the
// store.
func postProcessSessionFacts(sess sessionData, extract []memory.ExtractedMemory) (modifiedOverride, sessionDateISO string) {
	if len(extract) == 0 {
		return "", ""
	}

	// Inject temporal metadata into extracted fact content so the FTS
	// index can match temporal queries, and override the frontmatter
	// modified field so recency-aware retrieval treats the fact as
	// having originated on the session date rather than the wall-clock
	// replay time.
	if sess.date != "" {
		modifiedOverride = parseSessionDateRFC3339(sess.date)
		dateTokens := buildDateTokens(modifiedOverride)
		sessionDateISO = shortISODate(modifiedOverride)
		for i := range extract {
			if !strings.HasPrefix(strings.TrimSpace(extract[i].Content), "[Date:") {
				extract[i].Content = fmt.Sprintf("%s[Observed on %s]\n\n%s",
					dateTokens, sess.date, extract[i].Content)
			}
			if modifiedOverride != "" {
				extract[i].ModifiedOverride = modifiedOverride
				if strings.TrimSpace(extract[i].ObservedOn) == "" {
					extract[i].ObservedOn = modifiedOverride
				}
			}
		}
	}

	// Tag every fact with its originating session so multi-session
	// queries can aggregate or filter by origin without re-parsing the
	// body. Also seed frontmatter tags with auto-extracted entities so
	// BM25 matches against the tags column rather than body-only.
	for i := range extract {
		extract[i].SessionID = sess.id
		if sessionDateISO != "" {
			extract[i].SessionDate = sessionDateISO
		}
		auto := autoFactTags(extract[i].Content)
		if len(auto) > 0 {
			seen := make(map[string]bool, len(extract[i].Tags)+len(auto))
			merged := make([]string, 0, len(extract[i].Tags)+len(auto))
			for _, t := range extract[i].Tags {
				t = strings.TrimSpace(t)
				if t == "" || seen[t] {
					continue
				}
				seen[t] = true
				merged = append(merged, t)
			}
			for _, t := range auto {
				if seen[t] {
					continue
				}
				seen[t] = true
				merged = append(merged, t)
			}
			extract[i].Tags = merged
		}
	}

	return modifiedOverride, sessionDateISO
}

// sessionToMessages converts an LME session transcript into
// [memory.Message] values suitable for the extraction pipeline. A
// system message with the session date is prepended when a date is
// available so the extractor can anchor temporal claims.
func sessionToMessages(sess sessionData) []memory.Message {
	var messages []memory.Message

	if sess.date != "" {
		messages = append(messages, memory.Message{
			Role:    memory.RoleSystem,
			Content: fmt.Sprintf("This conversation took place on %s.", sess.date),
		})
	}

	lines := strings.Split(sess.text, "\n")
	var currentRole memory.Role
	var currentContent strings.Builder

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			if currentContent.Len() > 0 {
				currentContent.WriteByte('\n')
			}
			continue
		}

		// Detect role markers like "[user]: " or "[assistant]: ".
		if strings.HasPrefix(line, "[") {
			if idx := strings.Index(line, "]: "); idx > 0 {
				role := memory.Role(line[1:idx])
				if role == memory.RoleUser || role == memory.RoleAssistant {
					if currentContent.Len() > 0 && currentRole != "" {
						messages = append(messages, memory.Message{
							Role:    currentRole,
							Content: strings.TrimSpace(currentContent.String()),
						})
						currentContent.Reset()
					}
					currentRole = role
					rest := line[idx+3:]
					currentContent.WriteString(rest)
					continue
				}
			}
		}

		if currentContent.Len() > 0 {
			currentContent.WriteByte('\n')
		}
		currentContent.WriteString(line)
	}

	if currentContent.Len() > 0 && currentRole != "" {
		messages = append(messages, memory.Message{
			Role:    currentRole,
			Content: strings.TrimSpace(currentContent.String()),
		})
	}

	// If parsing produced no structured messages (e.g. sessions without
	// role markers), fall back to a single user message with the raw
	// text so the extractor still has something to chew on.
	if len(messages) == 0 || (len(messages) == 1 && messages[0].Role == memory.RoleSystem) {
		messages = append(messages, memory.Message{
			Role:    memory.RoleUser,
			Content: sess.text,
		})
	}

	return messages
}

// autoFactTags derives a small set of tokens from an extracted fact's
// content that are worth indexing at the tag column's higher BM25
// weight: ISO dates, weekday names, currency amounts, quantity+unit
// pairs, and capitalised multi-letter words (proper nouns).
func autoFactTags(content string) []string {
	if content == "" {
		return nil
	}
	body := content
	if len(body) > 4096 {
		body = body[:4096]
	}

	seen := make(map[string]bool)
	add := func(tok string) {
		tok = strings.TrimSpace(tok)
		if tok == "" || seen[tok] {
			return
		}
		seen[tok] = true
	}

	for _, m := range autoTagDateRe.FindAllString(body, -1) {
		add(m)
	}
	// ISO date + weekday + month let temporal queries match even when
	// the body only carries one form.
	for _, m := range autoTagDateRe.FindAllString(body, -1) {
		if t, err := time.Parse("2006-01-02", strings.ReplaceAll(m, "/", "-")); err == nil {
			add(t.Weekday().String())
			add(t.Month().String())
		}
	}
	for _, m := range autoTagWeekdayRe.FindAllString(body, -1) {
		low := strings.ToLower(m)
		if low == "" {
			continue
		}
		add(strings.ToUpper(low[:1]) + low[1:])
	}
	// Prefer fuller matches ("45 minutes", "$185") over bare numbers.
	for _, m := range autoTagMoneyRe.FindAllString(body, -1) {
		add(strings.TrimSpace(m))
	}
	for _, m := range autoTagUnitQuantityRe.FindAllStringSubmatch(body, -1) {
		if len(m) >= 3 {
			qty := strings.TrimSpace(m[1] + " " + m[2])
			add(qty)
		}
	}
	for _, m := range autoTagQuantityRe.FindAllString(body, -1) {
		add(m)
	}
	for _, m := range autoTagProperNounRe.FindAllString(body, -1) {
		if len(m) < 3 {
			continue
		}
		lower := strings.ToLower(m)
		if autoTagStopNoun[lower] {
			continue
		}
		add(m)
	}

	if len(seen) == 0 {
		return nil
	}
	out := make([]string, 0, len(seen))
	for t := range seen {
		out = append(out, t)
	}
	return out
}

var (
	autoTagDateRe         = regexp.MustCompile(`\b\d{4}[-/]\d{2}[-/]\d{2}\b`)
	autoTagWeekdayRe      = regexp.MustCompile(`(?i)\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b`)
	autoTagQuantityRe     = regexp.MustCompile(`\b\d{1,6}(?:\.\d+)?\b`)
	autoTagProperNounRe   = regexp.MustCompile(`\b[A-Z][a-zA-Z]+\b`)
	autoTagMoneyRe        = regexp.MustCompile(`[\$£€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?`)
	autoTagUnitQuantityRe = regexp.MustCompile(`(?i)\b(\d{1,6}(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?|km|kilometres?|miles?|metres?|meters?|kg|kilograms?|pounds?|lbs?|grams?|percent|%)\b`)
)

// autoTagStopNoun lists capitalised words that are common sentence
// starts rather than entities. Lowercase lookup keys.
var autoTagStopNoun = map[string]bool{
	"the": true, "this": true, "that": true, "these": true, "those": true,
	"when": true, "where": true, "what": true, "who": true, "why": true, "how": true,
	"observed": true, "date": true, "mon": true, "tue": true, "wed": true, "thu": true,
	"fri": true, "sat": true, "sun": true, "user": true, "assistant": true,
}

// buildDateTokens returns a short prefix string carrying ISO date,
// weekday, year, and month so FTS BM25 matches temporal queries
// against the body itself.
func buildDateTokens(rfc3339 string) string {
	if rfc3339 == "" {
		return ""
	}
	t, err := time.Parse(time.RFC3339, rfc3339)
	if err != nil {
		return ""
	}
	weekday := t.Weekday().String()
	month := t.Month().String()
	iso := t.Format("2006-01-02")
	year := t.Format("2006")
	return fmt.Sprintf("[Date: %s %s %s %s]\n\n", iso, weekday, month, year)
}

func shortISODate(rfc3339 string) string {
	if rfc3339 == "" {
		return ""
	}
	t, err := time.Parse(time.RFC3339, rfc3339)
	if err != nil {
		return ""
	}
	return t.Format("2006-01-02")
}

func sessionContextSummary(sess sessionData) string {
	var parts []string
	if sess.date != "" {
		parts = append(parts, fmt.Sprintf("session_date=%s", sess.date))
	}
	for _, line := range strings.Split(sess.text, "\n") {
		if strings.HasPrefix(line, "[user]: ") {
			first := strings.TrimPrefix(line, "[user]: ")
			first = strings.TrimSpace(first)
			if len(first) > 240 {
				first = first[:240] + "..."
			}
			parts = append(parts, "first_user_turn="+first)
			break
		}
	}
	return strings.Join(parts, "; ")
}

// parseSessionDateRFC3339 converts the LME session date string into
// RFC3339 form. Returns the empty string when nothing parses.
func parseSessionDateRFC3339(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	for _, layout := range []string{
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 15:04",
		"2006/01/02",
		"2006-01-02 15:04",
		"2006-01-02",
		time.RFC3339,
	} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC().Format(time.RFC3339)
		}
	}
	return ""
}

func fmtDurationSecs(d time.Duration) string {
	return fmt.Sprintf("%.1f", d.Seconds())
}
