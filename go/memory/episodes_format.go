// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// normaliseOutcome coerces a string into a valid EpisodeOutcome.
func normaliseOutcome(value string) EpisodeOutcome {
	switch EpisodeOutcome(strings.ToLower(strings.TrimSpace(value))) {
	case EpisodeOutcomeSuccess:
		return EpisodeOutcomeSuccess
	case EpisodeOutcomePartial:
		return EpisodeOutcomePartial
	case EpisodeOutcomeFailure:
		return EpisodeOutcomeFailure
	default:
		return EpisodeOutcomeUnknown
	}
}

// normaliseScope coerces a string into a valid EpisodeScope.
func normaliseEpisodeScope(value string) EpisodeScope {
	switch EpisodeScope(strings.ToLower(strings.TrimSpace(value))) {
	case EpisodeScopeGlobal:
		return EpisodeScopeGlobal
	case EpisodeScopeProject:
		return EpisodeScopeProject
	case EpisodeScopeAgent:
		return EpisodeScopeAgent
	default:
		return EpisodeScopeProject
	}
}

// buildEpisodeFileContent renders an EpisodeRecord as a markdown file
// with YAML frontmatter and a structured body containing a JSON
// payload block.
func buildEpisodeFileContent(ep EpisodeRecord) string {
	var b strings.Builder

	b.WriteString("---\n")
	b.WriteString(fmt.Sprintf("name: \"%s\"\n", ep.Name))
	b.WriteString(fmt.Sprintf("description: \"%s\"\n", escapeFrontmatterStr(ep.Summary)))
	b.WriteString("type: episode\n")
	b.WriteString(fmt.Sprintf("scope: %s\n", ep.Scope))
	b.WriteString(fmt.Sprintf("created: %s\n", ep.Created))
	b.WriteString(fmt.Sprintf("modified: %s\n", ep.Modified))
	b.WriteString("source: episode\n")
	b.WriteString(fmt.Sprintf("session_id: %s\n", ep.SessionID))
	b.WriteString(fmt.Sprintf("actor_id: %s\n", ep.ActorID))
	b.WriteString(fmt.Sprintf("outcome: %s\n", ep.Outcome))
	if ep.StartedAt != "" {
		b.WriteString(fmt.Sprintf("started_at: %s\n", ep.StartedAt))
	}
	if ep.EndedAt != "" {
		b.WriteString(fmt.Sprintf("ended_at: %s\n", ep.EndedAt))
	}
	if len(ep.Tags) > 0 {
		b.WriteString("tags:\n")
		for _, tag := range ep.Tags {
			b.WriteString(fmt.Sprintf("  - %s\n", tag))
		}
	}
	b.WriteString("---\n\n")

	b.WriteString(fmt.Sprintf("# %s\n\n", ep.Name))
	b.WriteString("## Summary\n\n")
	if ep.Summary != "" {
		b.WriteString(ep.Summary + "\n\n")
	} else {
		b.WriteString("_no summary_\n\n")
	}

	b.WriteString("## Outcome\n\n")
	b.WriteString(string(ep.Outcome) + "\n\n")

	if ep.RetryFeedback != "" {
		b.WriteString("## Retry feedback\n\n")
		b.WriteString(ep.RetryFeedback + "\n\n")
	}

	if len(ep.OpenQuestions) > 0 {
		b.WriteString("## Open questions\n\n")
		for _, q := range ep.OpenQuestions {
			b.WriteString(fmt.Sprintf("- %s\n", q))
		}
		b.WriteString("\n")
	}

	if len(ep.Heuristics) > 0 {
		b.WriteString("## Heuristics\n\n")
		for _, h := range ep.Heuristics {
			marker := "[pattern]"
			if h.AntiPattern {
				marker = "[anti-pattern]"
			}
			ctx := h.Context
			if ctx == "" {
				ctx = "the same type of work"
			}
			b.WriteString(fmt.Sprintf("- %s %s _(context: %s; confidence: %s; category: %s; scope: %s)_\n",
				marker, h.Rule, ctx, h.Confidence, h.Category, h.Scope))
		}
		b.WriteString("\n")
	}

	b.WriteString("## Signals\n\n")
	b.WriteString(fmt.Sprintf("- write_signal: %s\n", boolStr(ep.Signals.WriteSignal)))
	b.WriteString(fmt.Sprintf("- edit_signal: %s\n", boolStr(ep.Signals.EditSignal)))
	b.WriteString(fmt.Sprintf("- tool_signal: %s\n", boolStr(ep.Signals.ToolSignal)))
	b.WriteString(fmt.Sprintf("- message_count: %d\n", ep.Signals.MessageCount))
	b.WriteString(fmt.Sprintf("- substantive_message_count: %d\n", ep.Signals.SubstantiveMessageCount))
	b.WriteString(fmt.Sprintf("- tool_call_count: %d\n", ep.Signals.ToolCallCount))
	b.WriteString("\n")

	b.WriteString("## Episode data\n\n")
	b.WriteString("```json\n")
	payload, _ := json.MarshalIndent(ep, "", "  ")
	b.Write(payload)
	b.WriteString("\n```\n")

	return b.String()
}

// parseEpisodeFileContent parses a markdown episode file into an
// EpisodeRecord.
func parseEpisodeFileContent(path brain.Path, raw string) (EpisodeRecord, error) {
	fm, body := ParseFrontmatter(raw)
	payload := parseEpisodePayloadJSON(body)

	sessionID := firstNonEmpty(
		fmExtra(fm, "session_id"),
		payloadStr(payload, "session_id"),
		sessionIDFromPath(path),
	)
	if sessionID == "" {
		return EpisodeRecord{}, fmt.Errorf("episode has no session_id")
	}

	actorID := firstNonEmpty(fmExtra(fm, "actor_id"), payloadStr(payload, "actor_id"))
	scope := normaliseEpisodeScope(firstNonEmpty(fmExtra(fm, "scope"), fm.scopeVal(), payloadStr(payload, "scope")))
	summary := collapseWhitespaceStr(firstNonEmpty(fm.Description, payloadStr(payload, "summary")))
	outcome := normaliseOutcome(firstNonEmpty(fmExtra(fm, "outcome"), payloadStr(payload, "outcome")))
	name := firstNonEmpty(fm.Name, "Episode "+sessionID)

	var record EpisodeRecord
	record.Path = path
	record.SessionID = sessionID
	record.ActorID = actorID
	record.Scope = scope
	record.Name = name
	record.Summary = summary
	record.Outcome = outcome
	record.ShouldRecordEpisode = true
	record.Created = fm.Created
	record.Modified = fm.Modified

	if payload != nil {
		record.RetryFeedback = collapseWhitespaceStr(payloadStr(payload, "retry_feedback"))
		record.OpenQuestions = payloadStrSlice(payload, "open_questions")
		record.Heuristics = payloadHeuristics(payload)
		record.Tags = payloadStrSlice(payload, "tags")
		record.StartedAt = payloadStr(payload, "started_at")
		record.EndedAt = payloadStr(payload, "ended_at")
		record.Signals = payloadSignals(payload)
	} else {
		record.OpenQuestions = []string{}
		record.Heuristics = []EpisodeHeuristic{}
		record.Tags = dedupeTagSlice(fm.Tags)
	}

	if record.Tags == nil {
		record.Tags = []string{}
	}
	if record.OpenQuestions == nil {
		record.OpenQuestions = []string{}
	}
	if record.Heuristics == nil {
		record.Heuristics = []EpisodeHeuristic{}
	}

	return record, nil
}

// parseEpisodePayloadJSON extracts the JSON payload from the
// "## Episode data" code block in the body.
func parseEpisodePayloadJSON(body string) map[string]any {
	idx := strings.Index(body, "## Episode data")
	if idx < 0 {
		return nil
	}
	remainder := body[idx:]
	startMarker := "```json\n"
	startIdx := strings.Index(remainder, startMarker)
	if startIdx < 0 {
		startMarker = "```json"
		startIdx = strings.Index(remainder, startMarker)
		if startIdx < 0 {
			return nil
		}
	}
	jsonStart := startIdx + len(startMarker)
	endIdx := strings.Index(remainder[jsonStart:], "```")
	if endIdx < 0 {
		return nil
	}
	jsonStr := strings.TrimSpace(remainder[jsonStart : jsonStart+endIdx])
	if jsonStr == "" {
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &payload); err != nil {
		return nil
	}
	return payload
}

// --- Frontmatter access helpers ---

// fmExtra reads a key from the Frontmatter that is not part of the
// standard set. Since the Go Frontmatter struct does not store
// arbitrary extra keys, we return empty. The actual data comes from
// the JSON payload block.
func fmExtra(_ Frontmatter, _ string) string {
	return ""
}

// scopeVal returns the Frontmatter scope as a string.
func (fm Frontmatter) scopeVal() string {
	return ""
}

// --- Payload access helpers ---

func payloadStr(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	v, ok := m[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(s)
}

func payloadStrSlice(m map[string]any, key string) []string {
	if m == nil {
		return []string{}
	}
	v, ok := m[key]
	if !ok {
		return []string{}
	}
	arr, ok := v.([]any)
	if !ok {
		return []string{}
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		s, ok := item.(string)
		if !ok {
			continue
		}
		s = strings.TrimSpace(s)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func payloadHeuristics(m map[string]any) []EpisodeHeuristic {
	if m == nil {
		return []EpisodeHeuristic{}
	}
	v, ok := m["heuristics"]
	if !ok {
		return []EpisodeHeuristic{}
	}
	arr, ok := v.([]any)
	if !ok {
		return []EpisodeHeuristic{}
	}
	out := make([]EpisodeHeuristic, 0, len(arr))
	for _, item := range arr {
		obj, ok := item.(map[string]any)
		if !ok {
			continue
		}
		h := EpisodeHeuristic{
			Rule:        payloadStr(obj, "rule"),
			Context:     payloadStr(obj, "context"),
			Confidence:  payloadStr(obj, "confidence"),
			Category:    payloadStr(obj, "category"),
			Scope:       payloadStr(obj, "scope"),
			AntiPattern: payloadBool(obj, "anti_pattern"),
		}
		out = append(out, h)
	}
	return out
}

func payloadSignals(m map[string]any) EpisodeSignals {
	if m == nil {
		return EpisodeSignals{}
	}
	v, ok := m["signals"]
	if !ok {
		return EpisodeSignals{}
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return EpisodeSignals{}
	}
	return EpisodeSignals{
		MessageCount:            payloadInt(obj, "message_count"),
		SubstantiveMessageCount: payloadInt(obj, "substantive_message_count"),
		UserMessageCount:        payloadInt(obj, "user_message_count"),
		AssistantMessageCount:   payloadInt(obj, "assistant_message_count"),
		ToolMessageCount:        payloadInt(obj, "tool_message_count"),
		ToolCallCount:           payloadInt(obj, "tool_call_count"),
		WriteSignal:             payloadBool(obj, "write_signal"),
		EditSignal:              payloadBool(obj, "edit_signal"),
		ToolSignal:              payloadBool(obj, "tool_signal"),
	}
}

func payloadInt(m map[string]any, key string) int {
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	default:
		return 0
	}
}

func payloadBool(m map[string]any, key string) bool {
	v, ok := m[key]
	if !ok {
		return false
	}
	switch b := v.(type) {
	case bool:
		return b
	case string:
		return strings.ToLower(strings.TrimSpace(b)) == "true"
	default:
		return false
	}
}

// --- String helpers ---

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		trimmed := strings.TrimSpace(v)
		if trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func collapseWhitespaceStr(s string) string {
	parts := strings.Fields(s)
	return strings.Join(parts, " ")
}

func boolStr(v bool) string {
	if v {
		return "true"
	}
	return "false"
}

func escapeFrontmatterStr(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}

func sessionIDFromPath(p brain.Path) string {
	s := string(p)
	idx := strings.LastIndex(s, "/")
	if idx >= 0 {
		s = s[idx+1:]
	}
	return strings.TrimSuffix(s, ".md")
}

func dedupeTagSlice(tags []string) []string {
	seen := make(map[string]bool, len(tags))
	out := make([]string, 0, len(tags))
	for _, tag := range tags {
		normalised := normaliseTagStr(tag)
		if normalised == "" || seen[normalised] {
			continue
		}
		seen[normalised] = true
		out = append(out, normalised)
	}
	return out
}

func normaliseTagStr(tag string) string {
	tag = strings.ToLower(strings.TrimSpace(tag))
	var b strings.Builder
	for _, r := range tag {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '.', r == '_', r == ':', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	result := b.String()
	for strings.Contains(result, "--") {
		result = strings.ReplaceAll(result, "--", "-")
	}
	result = strings.Trim(result, "-")
	return result
}
