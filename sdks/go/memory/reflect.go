// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/jeffs-brain/memory/go/llm"
)

// Reflection thresholds.
const (
	reflectMaxTokens   = 4096
	reflectTemperature = 0.3
	reflectMinMessages = 8
	reflectMaxRecent   = 60
)

// Reflector manages background session reflection.
type Reflector struct {
	mem *Memory

	mu         sync.Mutex
	lastCursor int
	inProgress bool
}

// NewReflector creates a new Reflector bound to the supplied Memory.
func NewReflector(mem *Memory) *Reflector {
	return &Reflector{mem: mem}
}

// ReflectionResult holds the parsed output from the reflection agent.
type ReflectionResult struct {
	Outcome             string      `json:"outcome"`
	Summary             string      `json:"summary"`
	RetryFeedback       string      `json:"retry_feedback"`
	Heuristics          []Heuristic `json:"heuristics"`
	ShouldRecordEpisode bool        `json:"should_record_episode"`
}

// Heuristic represents a single generalisable pattern extracted from a
// session.
type Heuristic struct {
	Rule        string `json:"rule"`
	Context     string `json:"context"`
	Confidence  string `json:"confidence"`
	Category    string `json:"category"`
	Scope       string `json:"scope"`
	AntiPattern bool   `json:"anti_pattern"`
}

// SessionAnalysis summarises a coding session for the reflection
// prompt.
type SessionAnalysis struct {
	TaskDescription   string
	ToolCallSummary   string
	ErrorsEncountered []string
	UserCorrections   []string
	IterationCount    int
	WriteToolCalls    int
	Outcome           string
}

// reflectionSystemPrompt drives generalisation, not memorisation.
// Ported verbatim from jeff.
const reflectionSystemPrompt = `You are a reflection agent. You analyse completed coding sessions to extract lasting wisdom.

Your job is NOT to summarise what happened — it is to identify GENERALISABLE PATTERNS.

Good heuristic: "When working on Go projects with generated code, check for //go:generate directives before modifying generated files."
Bad heuristic: "The file cmd/server/main.go has a bug on line 42." (Too specific.)

## Output format
Respond with ONLY valid JSON:
{
  "outcome": "success|partial|failure",
  "summary": "one paragraph",
  "retry_feedback": "what to do differently if retrying this specific task",
  "heuristics": [
    {
      "rule": "imperative, actionable pattern",
      "context": "when this applies (language, framework, problem type)",
      "confidence": "low|medium|high",
      "category": "approach|debugging|architecture|testing|communication",
      "scope": "project|global",
      "anti_pattern": false
    }
  ],
  "should_record_episode": true
}

## When to produce heuristics
- User corrected the agent → HIGH confidence (possibly anti_pattern=true)
- Multiple approaches tried before success → MEDIUM confidence
- Non-obvious error encountered → LOW confidence
- Routine session → empty array is fine

## Anti-pattern signals
Look for: "no", "don't", "stop", "instead", "that's wrong", "not like that", agent backtracking, multiple failed attempts`

// correctionPatterns are phrases that signal the user corrected the
// agent.
var correctionPatterns = []string{
	"no,", "no ", "don't", "do not", "stop", "instead",
	"that's wrong", "not like that", "not what i",
	"please revert", "undo that", "try again",
}

// MaybeReflect checks if reflection should run and, if so, analyses
// the session to extract generalisable heuristics.
func (r *Reflector) MaybeReflect(
	ctx context.Context,
	provider llm.Provider,
	model string,
	projectPath string,
	messages []Message,
) {
	r.mu.Lock()
	if r.inProgress {
		r.mu.Unlock()
		return
	}
	r.inProgress = true
	cursor := r.lastCursor
	r.mu.Unlock()

	defer func() {
		r.mu.Lock()
		r.inProgress = false
		r.mu.Unlock()
	}()

	if !shouldReflect(messages, cursor) {
		return
	}

	result := r.reflect(ctx, provider, model, projectPath, messages, cursor)
	if result == nil {
		return
	}

	if len(result.Heuristics) > 0 {
		if err := r.mem.ApplyHeuristics(ctx, ProjectSlug(projectPath), result.Heuristics); err != nil {
			slog.Warn("memory: apply heuristics failed", "err", err)
		}
	}

	r.mu.Lock()
	r.lastCursor = len(messages)
	r.mu.Unlock()
}

// ForceReflect skips the shouldReflect check and returns the result
// directly.
func (r *Reflector) ForceReflect(
	ctx context.Context,
	provider llm.Provider,
	model string,
	projectPath string,
	messages []Message,
) *ReflectionResult {
	r.mu.Lock()
	cursor := r.lastCursor
	r.mu.Unlock()

	result := r.reflect(ctx, provider, model, projectPath, messages, cursor)
	if result == nil {
		return nil
	}

	if len(result.Heuristics) > 0 {
		if err := r.mem.ApplyHeuristics(ctx, ProjectSlug(projectPath), result.Heuristics); err != nil {
			slog.Warn("memory: apply heuristics failed", "err", err)
		}
	}

	r.mu.Lock()
	r.lastCursor = len(messages)
	r.mu.Unlock()

	return result
}

// reflect performs the actual LLM call and result parsing.
func (r *Reflector) reflect(
	ctx context.Context,
	provider llm.Provider,
	model string,
	projectPath string,
	messages []Message,
	cursor int,
) *ReflectionResult {
	recent := messages[cursor:]
	if len(recent) > reflectMaxRecent {
		recent = recent[len(recent)-reflectMaxRecent:]
	}

	analysis := analyseSession(recent)
	userPrompt := buildReflectionPrompt(analysis, recent)

	resp, err := provider.Complete(ctx, llm.CompleteRequest{
		Model: model,
		Messages: []llm.Message{
			{Role: RoleSystem, Content: reflectionSystemPrompt},
			{Role: RoleUser, Content: userPrompt},
		},
		MaxTokens:   reflectMaxTokens,
		Temperature: reflectTemperature,
	})
	if err != nil {
		return nil
	}

	result := parseReflectionResult(resp.Text)
	if result.Outcome == "" {
		return nil
	}

	return &result
}

// shouldReflect determines whether the session warrants reflection.
func shouldReflect(messages []Message, cursor int) bool {
	newMessages := messages[cursor:]

	if len(newMessages) < reflectMinMessages {
		return false
	}

	writes := countWriteToolCalls(newMessages)
	corrections := findUserCorrections(newMessages)
	iterations := countAssistantToolIterations(newMessages)

	if writes == 0 && iterations < 10 && len(corrections) == 0 {
		return false
	}

	return true
}

// analyseSession builds a structured summary of the session.
func analyseSession(messages []Message) SessionAnalysis {
	return SessionAnalysis{
		TaskDescription:   extractTaskDescription(messages),
		ToolCallSummary:   summariseToolCalls(messages),
		ErrorsEncountered: findErrors(messages),
		UserCorrections:   findUserCorrections(messages),
		IterationCount:    countAssistantToolIterations(messages),
		WriteToolCalls:    countWriteToolCalls(messages),
		Outcome:           inferOutcome(messages),
	}
}

// extractTaskDescription returns the first substantive user message as
// a proxy for the task description.
func extractTaskDescription(messages []Message) string {
	for _, m := range messages {
		if m.Role != RoleUser {
			continue
		}
		content := strings.TrimSpace(m.Content)
		if len(content) < 5 {
			continue
		}
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		return content
	}
	return ""
}

// summariseToolCalls groups tool calls by name and extracts file paths
// for write/edit operations.
func summariseToolCalls(messages []Message) string {
	type toolInfo struct {
		count int
		files []string
	}
	tools := make(map[string]*toolInfo)
	var order []string

	for _, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			name := tc.Name
			info, ok := tools[name]
			if !ok {
				info = &toolInfo{}
				tools[name] = info
				order = append(order, name)
			}
			info.count++

			if name == "write" || name == "edit" {
				if path := extractFilePath(tc.Arguments); path != "" {
					info.files = append(info.files, path)
				}
			}
		}
	}

	var b strings.Builder
	for _, name := range order {
		info := tools[name]
		b.WriteString(fmt.Sprintf("%s: %d calls", name, info.count))
		if len(info.files) > 0 {
			seen := make(map[string]bool)
			var unique []string
			for _, f := range info.files {
				base := baseFilename(f)
				if !seen[base] {
					seen[base] = true
					unique = append(unique, base)
				}
			}
			b.WriteString(fmt.Sprintf(" (%s)", strings.Join(unique, ", ")))
		}
		b.WriteString("\n")
	}

	return strings.TrimSpace(b.String())
}

// extractFilePath attempts to pull a file_path from JSON tool
// arguments.
func extractFilePath(args json.RawMessage) string {
	var parsed struct {
		FilePath string `json:"file_path"`
	}
	if err := json.Unmarshal(args, &parsed); err != nil {
		return ""
	}
	return parsed.FilePath
}

// baseFilename returns the last component of a file path.
func baseFilename(path string) string {
	parts := strings.Split(path, "/")
	if len(parts) == 0 {
		return path
	}
	return parts[len(parts)-1]
}

// findUserCorrections scans user messages for correction patterns.
func findUserCorrections(messages []Message) []string {
	var corrections []string
	for _, m := range messages {
		if m.Role != RoleUser {
			continue
		}
		lower := strings.ToLower(m.Content)
		for _, pattern := range correctionPatterns {
			if strings.Contains(lower, pattern) {
				content := m.Content
				if len(content) > 200 {
					content = content[:200] + "..."
				}
				corrections = append(corrections, content)
				break
			}
		}
	}
	return corrections
}

// findErrors scans tool results for error indicators.
func findErrors(messages []Message) []string {
	var errors []string
	for _, m := range messages {
		if m.Role != RoleTool {
			continue
		}
		lower := strings.ToLower(m.Content)
		if strings.Contains(lower, "error") || strings.Contains(lower, "failed") ||
			strings.Contains(lower, "panic") || strings.Contains(lower, "cannot") ||
			strings.Contains(lower, "fatal") {
			content := m.Content
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			errors = append(errors, content)
		}

		for _, block := range m.Blocks {
			if block.ToolResult != nil && block.ToolResult.IsError {
				content := block.ToolResult.Content
				if len(content) > 300 {
					content = content[:300] + "..."
				}
				errors = append(errors, content)
			}
		}
	}
	return errors
}

// countWriteToolCalls counts write and edit tool calls in the
// messages.
func countWriteToolCalls(messages []Message) int {
	count := 0
	for _, m := range messages {
		if m.Role != RoleAssistant {
			continue
		}
		for _, tc := range m.ToolCalls {
			if tc.Name == "write" || tc.Name == "edit" {
				count++
			}
		}
	}
	return count
}

// countAssistantToolIterations counts assistant messages that contain
// tool calls.
func countAssistantToolIterations(messages []Message) int {
	count := 0
	for _, m := range messages {
		if m.Role == RoleAssistant && len(m.ToolCalls) > 0 {
			count++
		}
	}
	return count
}

// inferOutcome guesses the session outcome from the final messages.
func inferOutcome(messages []Message) string {
	if len(messages) == 0 {
		return "unknown"
	}

	end := len(messages)
	start := end - 5
	if start < 0 {
		start = 0
	}

	for i := end - 1; i >= start; i-- {
		m := messages[i]
		lower := strings.ToLower(m.Content)

		if m.Role == RoleUser {
			if strings.Contains(lower, "thanks") || strings.Contains(lower, "perfect") ||
				strings.Contains(lower, "great") || strings.Contains(lower, "looks good") {
				return "success"
			}
			if strings.Contains(lower, "that's wrong") || strings.Contains(lower, "not working") ||
				strings.Contains(lower, "broken") || strings.Contains(lower, "revert") {
				return "failure"
			}
		}
	}

	return "partial"
}

// buildReflectionPrompt constructs the user message for the reflection
// agent.
func buildReflectionPrompt(analysis SessionAnalysis, messages []Message) string {
	var b strings.Builder

	b.WriteString("## Session analysis\n\n")

	if analysis.TaskDescription != "" {
		b.WriteString(fmt.Sprintf("**Task:** %s\n\n", analysis.TaskDescription))
	}

	if analysis.ToolCallSummary != "" {
		b.WriteString(fmt.Sprintf("**Tool usage:**\n%s\n\n", analysis.ToolCallSummary))
	}

	b.WriteString(fmt.Sprintf("**Iterations:** %d tool-call rounds\n", analysis.IterationCount))
	b.WriteString(fmt.Sprintf("**Write/edit calls:** %d\n", analysis.WriteToolCalls))
	b.WriteString(fmt.Sprintf("**Inferred outcome:** %s\n\n", analysis.Outcome))

	if len(analysis.ErrorsEncountered) > 0 {
		b.WriteString("**Errors encountered:**\n")
		for _, e := range analysis.ErrorsEncountered {
			b.WriteString(fmt.Sprintf("- %s\n", e))
		}
		b.WriteString("\n")
	}

	if len(analysis.UserCorrections) > 0 {
		b.WriteString("**User corrections:**\n")
		for _, c := range analysis.UserCorrections {
			b.WriteString(fmt.Sprintf("- %s\n", c))
		}
		b.WriteString("\n")
	}

	b.WriteString("## Conversation transcript\n\n")
	for _, m := range messages {
		switch m.Role {
		case RoleUser:
			content := m.Content
			if len(content) > 1000 {
				content = content[:1000] + "\n[...truncated]"
			}
			b.WriteString(fmt.Sprintf("[user]: %s\n\n", content))

		case RoleAssistant:
			content := m.Content
			if len(content) > 1000 {
				content = content[:1000] + "\n[...truncated]"
			}
			if content != "" {
				b.WriteString(fmt.Sprintf("[assistant]: %s\n\n", content))
			}
			for _, tc := range m.ToolCalls {
				args := string(tc.Arguments)
				if len(args) > 200 {
					args = args[:200] + "..."
				}
				b.WriteString(fmt.Sprintf("[tool_call %s]: %s\n\n", tc.Name, args))
			}

		case RoleTool:
			content := m.Content
			if len(content) > 300 {
				content = content[:300] + "..."
			}
			b.WriteString(fmt.Sprintf("[tool_result %s]: %s\n\n", m.Name, content))
		}
	}

	return b.String()
}

// parseReflectionResult extracts the reflection data from the model's
// JSON response.
func parseReflectionResult(content string) ReflectionResult {
	content = strings.TrimSpace(content)

	if idx := strings.Index(content, "{"); idx >= 0 {
		if end := strings.LastIndex(content, "}"); end > idx {
			content = content[idx : end+1]
		}
	}

	var result ReflectionResult
	if err := json.Unmarshal([]byte(content), &result); err != nil {
		return ReflectionResult{}
	}

	return result
}
