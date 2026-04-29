// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// readerTodayAnchor renders a "YYYY-MM-DD (Weekday)" string for the
// reader's temporal grounding line, falling back to the raw input when
// nothing parses.
func readerTodayAnchor(questionDate string) string {
	s := strings.TrimSpace(questionDate)
	if s == "" {
		return "unknown"
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
			return fmt.Sprintf("%s (%s)", t.Format("2006-01-02"), t.Weekday().String())
		}
	}
	return s
}

// readerMaxTokens matches the official run_generation.py CoT setting.
const readerMaxTokens = 800
const readerTemperature = 0.0
const readerTransientRetryMax = 3

// ReaderMaxTokens and ReaderTemperature are exported so the HTTP /ask
// handler can run the same generation parameters as the in-process LME
// reader when a caller opts in to the augmented prompt.
const (
	ReaderMaxTokens   = readerMaxTokens
	ReaderTemperature = readerTemperature
)

// readerUserTemplate is the LME CoT prompt augmented with recency and
// enumeration guidance so multi-session list/count and knowledge-update
// questions score correctly.
const readerUserTemplate = `I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.

Resolving conflicting information:
- Each fact is tagged with a date. When the same topic appears with different values on different dates, prefer the value from the most recent session date.
- Treat explicit supersession phrases as hard overrides regardless of how often the old value appears: "now", "currently", "most recently", "actually", "correction", "I updated", "I changed", "no longer".
- Do not vote by frequency. One later correction outweighs any number of earlier mentions.
- Never use a fact dated after the current date.
- When the question names a specific item, event, place, or descriptor, prefer the fact that matches that target most directly. Do not substitute a broader category match or a different example from the same topic.
- A direct statement of the full usual value outranks a newer note about only one segment, leg, or example from that routine unless the newer note explicitly says the full value changed.
- For habit and routine questions ("usually", "normally", "every week", "on Saturdays", "on weekdays"), prefer explicit habitual statements over isolated single-day examples.
- Do not let an example note about a narrower segment override the whole routine. For example, a "30-minute morning commute" note does not replace a direct statement of a "45-minute daily commute to work".
- When one fact names the event and another fact gives the associated submission, booking, or join date for that same event or venue, combine them if the connection is explicit in the retrieved facts.

Enumeration and counting:
- When the question asks to list, count, enumerate, or total ("how many", "list", "which", "what are all", "total", "in total"), return every matching item you find across the retrieved facts, one per line, each tagged with its session date. Then state the count or total explicitly at the end.
- Do not summarise into a single sentence when the question demands a list.
- Add numeric values across sessions when the question asks for a total (hours, days, money, items). Show the arithmetic.
- When both atomic event facts and retrospective roll-up summaries are present, prefer the atomic event facts and avoid double counting the roll-up.
- Treat first-person past-tense purchases, gifts, sales, earnings, completions, or submissions as confirmed historical events even when they appear inside a planning or advice conversation. Exclude only clearly hypothetical or planned amounts.
- For totals over historical activities, count each distinct dated event or session as its own contribution unless a later fact explicitly says it corrects, replaces, or cancels the earlier event. The same item, title, place, or person appearing on different dates is not by itself a correction.
- If a spending or earnings question does not explicitly restrict the timeframe ("today", "this time", "most recent", "current"), include all confirmed historical amounts for the same subject across sessions.
- For totals over named items, sum only the facts that match those named items directly. Do not add alternative purchases, adjacent examples, or broader category summaries unless the note clearly says they refer to the same item.
- When a total names multiple specific items, people, or occasions, every named part must be supported directly. If any named part is missing or lacks an amount, do not return a partial total. State that the information provided is not enough.
- When the question names a singular item plus another category, choose the single best-matching fact for that singular item. Do not combine multiple different handbags, flights, meals, or other same-category purchases unless the question explicitly asks for all of them.
- When multiple notes appear to describe the same purchase, gift, booking, or transaction, count it once. Prefer the most direct transactional fact over recap notes, budget summaries, tracker entries, or assistant bookkeeping.
- For "spent", "cost", and "total amount" questions, prefer direct transactional facts over plans, budgets, broad summaries, or calculations that only restate the same purchase.

Preference-sensitive questions:
- When the user asks for ideas, advice, inspiration, or recommendations, anchor the answer in explicit prior preferences, recent projects, recurring habits, and stated dislikes from the retrieved facts.
- Avoid generic suggestions when the history already contains concrete tastes or recent examples. Reuse those specifics directly in the answer.
- Infer durable preferences from concrete desired features or liked attributes even when the earlier example was tied to a different city, venue, or product.
- When concrete amenities or features are present, prefer them over generic travel style or budget signals.
- Ignore unrelated hostel, budget, or solo-travel examples when the retrieved facts already contain a clearer accommodation-feature preference and the question does not ask about price.
- When the question asks for a specific or exact previously recommended item, answer with the narrowest directly supported set from the retrieved facts. Do not widen the answer with adjacent frameworks, resource catalogues, or loosely related examples.

Unanswerable questions:
- If the retrieved facts do not directly answer the question, state that clearly in the first sentence.
- Keep the extraction step brief and limited to the missing subject. Do not narrate your search process.
- Do not pad the answer with near-miss facts about a different city, person, product, or date unless they directly explain why the requested fact is unavailable.
- End with a direct abstention that the information provided is not enough to answer the question.

Temporal reasoning:
- Today is %s (this is the current date). Resolve relative references ("recently", "last week", "a few days ago", "this month") against this anchor.
- For date-arithmetic questions ("how many days between X and Y", "how many days ago did X happen"), first extract each event's ISO date from the fact tags, then compute the difference in days. If a retrieved fact has a days_before_question label, use it as the exclusive day difference.

History Chats:

%s

Current Date: %s
Question: %s
Answer (step by step):`

// BuildReaderPrompt renders the augmented LME CoT user prompt that the
// in-process reader sends to the LLM. Exposed so the HTTP /ask handler
// can opt in to the same prompt shape (recency, enumeration, temporal
// guidance, today anchor, step-by-step CoT) used by the eval harness.
//
// questionDate accepts the LME "YYYY/MM/DD (Mon) HH:MM" form and a few
// siblings; an empty string falls through to "unknown".
func BuildReaderPrompt(question, questionDate, retrievedContent string) string {
	date := questionDate
	if date == "" {
		date = "unknown"
	}
	return fmt.Sprintf(
		readerUserTemplate,
		readerTodayAnchor(questionDate),
		retrievedContent,
		date,
		question,
	)
}

// ProcessSessionContextForQuestion is the exported entry point to the
// session-block aware preprocessor. The HTTP /ask handler calls this
// when retrieved chunks carry session_id frontmatter so the augmented
// reader sees the same chronologically-sorted, assistant-filtered shape
// as the in-process pipeline.
func ProcessSessionContextForQuestion(raw, question string) string {
	return processSessionContextForQuestion(raw, question)
}

// ReaderConfig controls the reading/answering step.
type ReaderConfig struct {
	Provider llm.Provider
	Model    string

	// ContentBudget caps the character length of retrieved content
	// passed to the reader. Zero uses defaultJudgeContentBudget.
	ContentBudget int

	// CacheDir stores reader answers keyed by the fully rendered prompt.
	// When multiple tri-SDK runs feed the same retrieved evidence through
	// the same reader model, the first completion is reused so identical
	// evidence is scored identically across SDKs.
	CacheDir string
}

func resolveReaderContentBudget(cfg ReaderConfig) int {
	if cfg.ContentBudget > 0 {
		return cfg.ContentBudget
	}
	return judgeContentBudgetFor(cfg.Provider)
}

// ReadAnswer generates a concise answer from retrieved session content by
// passing it through an LLM reader. Uses the official LME CoT prompt
// format. Returns the usage reported by the provider so callers can
// account for the spend.
//
// Reader failures do not silently degrade into raw retrieval text. The
// caller receives the read error so benchmark runs can record a question
// error instead of switching answer semantics mid-run.
func ReadAnswer(ctx context.Context, cfg ReaderConfig, question, questionDate, retrievedContent string) (string, Usage, error) {
	if cfg.Provider == nil {
		return retrievedContent, Usage{}, nil
	}

	if retrievedContent == "" {
		return "", Usage{}, nil
	}

	content := truncateReaderContent(retrievedContent, resolveReaderContentBudget(cfg), question)

	prompt := BuildReaderPrompt(question, questionDate, content)

	if cacheDir := strings.TrimSpace(cfg.CacheDir); cacheDir != "" {
		answer, usage, err := readCachedReaderAnswer(ctx, cacheDir, cfg.Model, prompt, func() (string, Usage, error) {
			return completeReaderPrompt(ctx, cfg, prompt)
		})
		if err != nil {
			return "", Usage{}, err
		}
		return answer, usage, nil
	}

	return completeReaderPrompt(ctx, cfg, prompt)
}

func completeReaderPrompt(ctx context.Context, cfg ReaderConfig, prompt string) (string, Usage, error) {
	req := llm.CompleteRequest{
		Model: cfg.Model,
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: prompt},
		},
		MaxTokens:   readerMaxTokens,
		Temperature: readerTemperature,
	}
	resp, err := completeReaderWithTransientRetry(ctx, cfg.Provider, req)
	if err != nil {
		return "", Usage{}, err
	}

	answer := strings.TrimSpace(resp.Text)
	usage := usageFromResponse(resp)
	if answer == "" {
		return "", usage, nil
	}
	return answer, usage, nil
}

func completeReaderWithTransientRetry(
	ctx context.Context,
	provider llm.Provider,
	req llm.CompleteRequest,
) (llm.CompleteResponse, error) {
	var lastErr error
	for attempt := 0; attempt < readerTransientRetryMax; attempt++ {
		resp, err := provider.Complete(ctx, req)
		if err == nil {
			return resp, nil
		}
		if !isTransientErr(err) {
			return llm.CompleteResponse{}, err
		}
		lastErr = err
		if attempt == readerTransientRetryMax-1 {
			break
		}
		IncTransientRetry()
		base := time.Duration(1<<uint(attempt)) * time.Second
		jitter := time.Duration(float64(base) * (rand.Float64()*0.4 - 0.2))
		delay := base + jitter
		if delay > 30*time.Second {
			delay = 30 * time.Second
		}
		select {
		case <-ctx.Done():
			return llm.CompleteResponse{}, ctx.Err()
		case <-time.After(delay):
		}
	}
	return llm.CompleteResponse{}, lastErr
}

func truncateReaderContent(content string, budget int, question string) string {
	if budget <= 0 {
		return ""
	}
	if len(content) <= budget {
		return content
	}
	if strings.TrimSpace(question) == "" {
		return truncateSmartly(content, budget)
	}

	sections := splitSessions(content)
	if len(sections) <= 1 {
		return relevantSnippetForQuestion(content, question, budget)
	}

	parts := make([]string, 0, len(sections))
	remaining := budget
	for _, section := range sections {
		sep := 0
		if len(parts) > 0 {
			sep = len("\n\n---\n\n")
		}
		if remaining <= sep {
			break
		}
		alloc := remaining - sep
		if len(section) <= alloc {
			parts = append(parts, section)
			remaining -= sep + len(section)
			continue
		}
		snippet := relevantSnippetForQuestion(section, question, alloc)
		if snippet == "" {
			break
		}
		parts = append(parts, snippet)
		break
	}
	if len(parts) == 0 {
		return relevantSnippetForQuestion(content, question, budget)
	}
	return strings.Join(parts, "\n\n---\n\n")
}

func relevantSnippetForQuestion(content, question string, budget int) string {
	if budget <= 0 {
		return ""
	}
	if len(content) <= budget {
		return content
	}
	tokens := questionTokens(question)
	if len(tokens) == 0 {
		return headTailTruncate(content, budget)
	}

	lines := strings.Split(content, "\n")
	prefixEnd := 0
	for prefixEnd < len(lines) {
		line := lines[prefixEnd]
		if strings.HasPrefix(line, "[user]:") || strings.HasPrefix(line, "[assistant]:") {
			break
		}
		prefixEnd++
	}

	selected := make([]bool, len(lines))
	for i := 0; i < prefixEnd; i++ {
		selected[i] = true
	}

	matched := false
	for i := prefixEnd; i < len(lines); i++ {
		if scoreChunkRelevance(lines[i], tokens) == 0 {
			continue
		}
		matched = true
		from := max(0, i-2)
		to := min(len(lines), i+3)
		for j := from; j < to; j++ {
			selected[j] = true
		}
	}
	if !matched {
		return headTailTruncate(content, budget)
	}

	var b strings.Builder
	omitted := false
	for i, line := range lines {
		if selected[i] {
			if omitted {
				b.WriteString("[...omitted irrelevant lines...]\n")
				omitted = false
			}
			b.WriteString(line)
			b.WriteByte('\n')
			continue
		}
		if b.Len() > 0 {
			omitted = true
		}
	}

	return headTailTruncate(strings.TrimSpace(b.String()), budget)
}
