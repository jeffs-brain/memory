// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/query"
)

// sessionBlock represents a parsed session from retrieved content, with
// its date extracted for chronological sorting and assistant filtering.
type sessionBlock struct {
	sessionID string
	date      string
	sortedAt  time.Time
	hasTime   bool
	relevance int
	content   string
	userOnly  string
}

var (
	sessionDateRe     = regexp.MustCompile(`(?m)^session_date:\s*(.+?)\s*$`)
	sessionIDRe       = regexp.MustCompile(`(?m)^session_id:\s*(\S+)\s*$`)
	daysAgoQuestionRe = regexp.MustCompile(`(?i)\bhow\s+many\s+days?\s+ago\b`)
	datePrefixRe      = regexp.MustCompile(`^\d{4}[-/]\d{2}[-/]\d{2}$`)
)

// processSessionContext takes raw retrieved content (multiple sessions
// joined by double newlines) and applies the improvements observed in
// top LME systems: parse into individual session blocks, sort
// chronologically by session date, filter assistant messages, inject
// clear date headers. Returns the processed content ready for the
// reader.
func processSessionContext(raw string) string {
	return processSessionContextForQuestion(raw, "")
}

// processSessionContextForQuestion is the question-aware variant that
// keeps assistant chunks scored by token overlap against the question.
func processSessionContextForQuestion(raw, question string) string {
	blocks := parseSessionBlocksForQuestion(raw, question)
	if len(blocks) == 0 {
		return raw
	}

	sort.SliceStable(blocks, func(i, j int) bool {
		if blocks[i].relevance != blocks[j].relevance {
			return blocks[i].relevance > blocks[j].relevance
		}
		if blocks[i].hasTime && blocks[j].hasTime && !blocks[i].sortedAt.Equal(blocks[j].sortedAt) {
			return blocks[i].sortedAt.After(blocks[j].sortedAt)
		}
		if blocks[i].date != blocks[j].date {
			return blocks[i].date > blocks[j].date
		}
		return blocks[i].sessionID < blocks[j].sessionID
	})

	var b strings.Builder
	for i, block := range blocks {
		if i > 0 {
			b.WriteString("\n\n---\n\n")
		}

		if block.date != "" {
			fmt.Fprintf(&b, "=== Session Date: %s ===\n", block.date)
		}

		b.WriteString(block.userOnly)
	}

	return b.String()
}

// parseSessionBlocks splits retrieved content into individual session
// blocks.
func parseSessionBlocks(content string) []sessionBlock {
	return parseSessionBlocksForQuestion(content, "")
}

// parseSessionBlocksForQuestion forwards the question into
// filterAssistantTurns so relevant assistant turns survive the cap.
func parseSessionBlocksForQuestion(content, question string) []sessionBlock {
	parts := splitOnSessionBoundary(content)
	if len(parts) == 0 {
		return nil
	}
	tokens := questionTokens(question)

	blocks := make([]sessionBlock, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		block := sessionBlock{content: part}

		if m := sessionDateRe.FindStringSubmatch(part); len(m) > 1 {
			block.date = strings.TrimSpace(m[1])
			if parsed, ok := parseSessionTime(block.date); ok {
				block.sortedAt = parsed
				block.hasTime = true
			}
		}
		if m := sessionIDRe.FindStringSubmatch(part); len(m) > 1 {
			block.sessionID = m[1]
		}

		block.userOnly = filterAssistantTurnsForQuestion(part, question)
		block.relevance = scoreChunkRelevance(block.userOnly, tokens)
		blocks = append(blocks, block)
	}

	return blocks
}

// splitOnSessionBoundary splits content on the YAML frontmatter delimiter
// pattern that bulk ingest uses between sessions.
func splitOnSessionBoundary(content string) []string {
	parts := strings.Split(content, "\n\n---\nsession_id:")
	if len(parts) > 1 {
		result := make([]string, len(parts))
		result[0] = parts[0]
		for i := 1; i < len(parts); i++ {
			result[i] = "---\nsession_id:" + parts[i]
		}
		return result
	}

	parts = strings.Split(content, "\n\n---\n")
	if len(parts) > 1 {
		result := make([]string, len(parts))
		result[0] = parts[0]
		for i := 1; i < len(parts); i++ {
			result[i] = "---\n" + parts[i]
		}
		return result
	}

	return []string{content}
}

// filterAssistantTurns is the question-agnostic entry point.
func filterAssistantTurns(content string) string {
	return filterAssistantTurnsForQuestion(content, "")
}

// filterAssistantTurnsForQuestion keeps all user turns and the top 5
// assistant turns. Chunks are ranked by token-overlap relevance against
// the question when one is supplied.
func filterAssistantTurnsForQuestion(content, question string) string {
	lines := strings.Split(content, "\n")

	var userLines []string
	var assistantChunks []string
	var currentAssistant strings.Builder
	inAssistant := false

	for _, line := range lines {
		if strings.HasPrefix(line, "[user]:") {
			if inAssistant && currentAssistant.Len() > 0 {
				assistantChunks = append(assistantChunks, currentAssistant.String())
				currentAssistant.Reset()
			}
			inAssistant = false
			userLines = append(userLines, line)
		} else if strings.HasPrefix(line, "[assistant]:") {
			if inAssistant && currentAssistant.Len() > 0 {
				assistantChunks = append(assistantChunks, currentAssistant.String())
				currentAssistant.Reset()
			}
			inAssistant = true
			currentAssistant.WriteString(line)
			currentAssistant.WriteByte('\n')
		} else if inAssistant {
			currentAssistant.WriteString(line)
			currentAssistant.WriteByte('\n')
		} else {
			userLines = append(userLines, line)
		}
	}
	if inAssistant && currentAssistant.Len() > 0 {
		assistantChunks = append(assistantChunks, currentAssistant.String())
	}

	const maxAssistantChunks = 5
	if len(assistantChunks) > maxAssistantChunks {
		if question != "" {
			tokens := questionTokens(question)
			sort.SliceStable(assistantChunks, func(i, j int) bool {
				si, sj := scoreChunkRelevance(assistantChunks[i], tokens), scoreChunkRelevance(assistantChunks[j], tokens)
				if si != sj {
					return si > sj
				}
				return len(assistantChunks[i]) < len(assistantChunks[j])
			})
		} else {
			sort.Slice(assistantChunks, func(i, j int) bool {
				return len(assistantChunks[i]) < len(assistantChunks[j])
			})
		}
		assistantChunks = assistantChunks[:maxAssistantChunks]
	}

	var result strings.Builder
	result.WriteString(strings.Join(userLines, "\n"))
	if len(assistantChunks) > 0 {
		result.WriteString("\n\n[Assistant context (suggestions only; not confirmed user facts unless the user later affirmed them)]:\n")
		for _, chunk := range assistantChunks {
			result.WriteString(strings.TrimSpace(chunk))
			result.WriteByte('\n')
		}
	}

	return result.String()
}

// questionTokens splits a question into lowercased tokens of at least
// three characters, skipping a small stop-word set.
func questionTokens(question string) []string {
	if question == "" {
		return nil
	}
	stops := map[string]bool{
		"the": true, "and": true, "for": true, "with": true, "what": true,
		"who": true, "when": true, "where": true, "why": true, "how": true,
		"did": true, "does": true, "was": true, "were": true, "are": true,
		"you": true, "your": true, "about": true, "this": true, "that": true,
		"have": true, "has": true, "had": true, "from": true, "into": true,
		"than": true, "then": true, "them": true, "they": true, "their": true,
	}
	raw := strings.Fields(strings.ToLower(question))
	seen := make(map[string]bool, len(raw))
	out := make([]string, 0, len(raw))
	for _, tok := range raw {
		tok = strings.Trim(tok, `.,;:!?"'()[]{}<>`)
		if len(tok) < 3 || stops[tok] || seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	return out
}

// scoreChunkRelevance counts how many question tokens appear in the
// chunk, lowercased.
func scoreChunkRelevance(chunk string, tokens []string) int {
	if len(tokens) == 0 {
		return 0
	}
	lower := strings.ToLower(chunk)
	var score int
	for _, t := range tokens {
		if strings.Contains(lower, t) {
			score++
		}
	}
	return score
}

func parseSessionTime(value string) (time.Time, bool) {
	s := strings.TrimSpace(value)
	if s == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 15:04",
		"2006/01/02",
		"2006-01-02 15:04",
		"2006-01-02",
		time.RFC3339,
	} {
		if parsed, err := time.Parse(layout, s); err == nil {
			return parsed.UTC(), true
		}
	}
	return time.Time{}, false
}

// RetrievedPassage is the retrieve-only rendering shape used by the
// actor-backed LongMemEval path.
type RetrievedPassage struct {
	Path            string
	Score           float64
	Body            string
	Date            string
	SessionID       string
	SourceRole      string
	EventDate       string
	EvidenceKind    string
	EvidenceGroup   string
	StateKey        string
	ClaimStatus     string
	ValidFrom       string
	ValidTo         string
	ArtefactType    string
	ArtefactOrdinal string
	ArtefactSection string
}

// RenderRetrievedPassages renders retrieve-only evidence with explicit
// boundaries so the reader can distinguish neighbouring hits.
func RenderRetrievedPassages(passages []RetrievedPassage, question, questionDate string) string {
	ordered := clusterPassagesBySession(passages)
	if len(ordered) == 0 {
		return ""
	}

	var parts []string
	if hint := resolvedTemporalHintLine(question, questionDate); hint != "" {
		parts = append(parts, hint, "")
	}
	parts = append(parts, fmt.Sprintf("Retrieved facts (%d):", len(ordered)), "")
	for i, passage := range ordered {
		labels := []string{fmt.Sprintf("[%s]", passageDate(passage))}
		if sessionID := passageSessionID(passage); sessionID != "" {
			labels = append(labels, fmt.Sprintf("[session=%s]", sessionID))
		}
		if sourceRole := passageSourceRole(passage); sourceRole != "" {
			labels = append(labels, fmt.Sprintf("[source_role=%s]", sourceRole))
		}
		if eventDate := passageEventDate(passage); eventDate != "" {
			labels = append(labels, fmt.Sprintf("[event_date=%s]", eventDate))
		}
		if temporalDelta := passageTemporalDeltaLabel(passage, question, questionDate); temporalDelta != "" {
			labels = append(labels, fmt.Sprintf("[%s]", temporalDelta))
		}
		if evidenceKind := passageEvidenceKind(passage); evidenceKind != "" {
			labels = append(labels, fmt.Sprintf("[evidence=%s]", evidenceKind))
		}
		if evidenceGroup := passageEvidenceGroup(passage); evidenceGroup != "" {
			labels = append(labels, fmt.Sprintf("[group=%s]", evidenceGroup))
		}
		if stateKey := passageStateKey(passage); stateKey != "" {
			labels = append(labels, fmt.Sprintf("[state=%s]", stateKey))
		}
		if claimStatus := passageClaimStatus(passage); claimStatus != "" {
			labels = append(labels, fmt.Sprintf("[claim=%s]", claimStatus))
		}
		if validWindow := passageValidWindow(passage); validWindow != "" {
			labels = append(labels, fmt.Sprintf("[valid=%s]", validWindow))
		}
		if artefactLabel := passageArtefactLabel(passage); artefactLabel != "" {
			labels = append(labels, fmt.Sprintf("[artefact=%s]", artefactLabel))
		}
		if source := sourceTagFromPath(passage.Path); source != "" {
			labels = append(labels, fmt.Sprintf("[%s]", source))
		}
		parts = append(parts, fmt.Sprintf("%2d. %s", i+1, strings.Join(labels, " ")))
		parts = append(parts, passageDisplayBody(passage), "")
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func clusterPassagesBySession(passages []RetrievedPassage) []RetrievedPassage {
	if len(passages) <= 1 {
		out := make([]RetrievedPassage, len(passages))
		copy(out, passages)
		return out
	}
	order := make([]string, 0, len(passages))
	groups := make(map[string][]RetrievedPassage, len(passages))
	for i, passage := range passages {
		key := passageSessionID(passage)
		if key == "" {
			key = fmt.Sprintf("__solo_%d__", i)
		}
		if _, ok := groups[key]; !ok {
			order = append(order, key)
		}
		groups[key] = append(groups[key], passage)
	}
	out := make([]RetrievedPassage, 0, len(passages))
	for _, key := range order {
		out = append(out, groups[key]...)
	}
	return out
}

func passageDate(passage RetrievedPassage) string {
	if trimmed := strings.TrimSpace(passage.Date); trimmed != "" {
		return trimmed
	}
	for _, key := range []string{"session_date", "observed_on", "modified"} {
		if value := firstFrontmatterValue(passage.Body, key); value != "" {
			return value
		}
	}
	return "unknown"
}

func passageSessionID(passage RetrievedPassage) string {
	if trimmed := strings.TrimSpace(passage.SessionID); trimmed != "" {
		return trimmed
	}
	return firstFrontmatterValue(passage.Body, "session_id")
}

func passageSourceRole(passage RetrievedPassage) string {
	role := strings.ToLower(strings.TrimSpace(passage.SourceRole))
	if role == "" {
		role = strings.ToLower(strings.TrimSpace(firstFrontmatterValue(passage.Body, "source_role")))
	}
	switch role {
	case "user", "assistant", "mixed":
		return role
	default:
		return ""
	}
}

func passageEventDate(passage RetrievedPassage) string {
	if trimmed := strings.TrimSpace(passage.EventDate); trimmed != "" {
		return trimmed
	}
	return firstFrontmatterValue(passage.Body, "event_date")
}

func passageTemporalDeltaLabel(passage RetrievedPassage, question, questionDate string) string {
	if !daysAgoQuestionRe.MatchString(question) {
		return ""
	}
	anchor, ok := parsePassageDate(questionDate)
	if !ok {
		return ""
	}
	dateValue := passageEventDate(passage)
	if dateValue == "" {
		dateValue = passageDate(passage)
	}
	eventDate, ok := parsePassageDate(dateValue)
	if !ok {
		return ""
	}
	days := int(anchor.Sub(eventDate).Hours() / 24)
	if days < 0 {
		return ""
	}
	return fmt.Sprintf("days_before_question=%d", days)
}

func parsePassageDate(raw string) (time.Time, bool) {
	candidate := normalisePassageDate(raw)
	if candidate == "" {
		return time.Time{}, false
	}
	for _, layout := range []string{
		time.RFC3339,
		"2006/01/02 (Mon) 15:04",
		"2006/01/02 (Mon)",
		"2006/01/02 15:04",
		"2006/01/02",
		"2006-01-02",
	} {
		parsed, err := time.Parse(layout, candidate)
		if err == nil {
			year, month, day := parsed.Date()
			return time.Date(year, month, day, 0, 0, 0, 0, time.UTC), true
		}
	}
	return time.Time{}, false
}

func normalisePassageDate(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" || trimmed == "unknown" {
		return ""
	}
	if len(trimmed) >= len("2006-01-02") {
		prefix := trimmed[:10]
		if datePrefixRe.MatchString(prefix) {
			return prefix
		}
	}
	return trimmed
}

func passageEvidenceKind(passage RetrievedPassage) string {
	kind := strings.ToLower(strings.TrimSpace(passage.EvidenceKind))
	switch kind {
	case "atomic", "rollup", "recap", "plan", "unknown":
		return kind
	default:
		return ""
	}
}

func passageEvidenceGroup(passage RetrievedPassage) string {
	group := strings.TrimSpace(passage.EvidenceGroup)
	if group == "" {
		return ""
	}
	return strings.Map(func(r rune) rune {
		if r == '\n' || r == '\r' || r == '\t' {
			return -1
		}
		return r
	}, group)
}

func passageStateKey(passage RetrievedPassage) string {
	return safePassageLabel(passage.StateKey)
}

func passageClaimStatus(passage RetrievedPassage) string {
	status := strings.ToLower(strings.TrimSpace(passage.ClaimStatus))
	switch status {
	case "asserted", "currently_true", "superseded", "unknown":
		return status
	default:
		return ""
	}
}

func passageValidWindow(passage RetrievedPassage) string {
	from := safePassageLabel(passage.ValidFrom)
	to := safePassageLabel(passage.ValidTo)
	if from == "" && to == "" {
		return ""
	}
	if from == "" {
		from = "unknown"
	}
	if to == "" {
		to = "present"
	}
	return from + ".." + to
}

func passageArtefactLabel(passage RetrievedPassage) string {
	kind := safePassageLabel(passage.ArtefactType)
	if kind == "" {
		return ""
	}
	if ordinal := safePassageLabel(passage.ArtefactOrdinal); ordinal != "" {
		kind += "#" + ordinal
	}
	if section := safePassageLabel(passage.ArtefactSection); section != "" {
		kind += ":" + section
	}
	return kind
}

func safePassageLabel(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return strings.Map(func(r rune) rune {
		if r == '\n' || r == '\r' || r == '\t' || r == ' ' {
			return '_'
		}
		return r
	}, value)
}

func passageDisplayBody(passage RetrievedPassage) string {
	_, body := memory.ParseFrontmatter(passage.Body)
	return strings.TrimSpace(body)
}

func firstFrontmatterValue(content, key string) string {
	lines := strings.Split(content, "\n")
	inFrontmatter := false
	prefix := key + ":"
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			if inFrontmatter {
				break
			}
			inFrontmatter = true
			continue
		}
		if !inFrontmatter || !strings.HasPrefix(trimmed, prefix) {
			continue
		}
		return strings.TrimSpace(strings.TrimPrefix(trimmed, prefix))
	}
	return ""
}

func sourceTagFromPath(path string) string {
	base := path
	if idx := strings.LastIndexByte(base, '/'); idx >= 0 && idx < len(base)-1 {
		base = base[idx+1:]
	}
	return strings.TrimSuffix(base, ".md")
}

func resolvedTemporalHintLine(question, questionDate string) string {
	expansion := query.ExpandTemporal(question, questionDate)
	if !expansion.Resolved || len(expansion.DateHints) == 0 {
		return ""
	}
	return fmt.Sprintf("[Resolved temporal references: %s]", strings.Join(expansion.DateHints, ", "))
}
