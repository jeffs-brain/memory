// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestNewReflector(t *testing.T) {
	mem, _ := newTestMemory(t)
	r := NewReflector(mem)
	if r == nil {
		t.Fatal("expected non-nil reflector")
	}
	if r.inProgress {
		t.Error("should not be in progress initially")
	}
	if r.lastCursor != 0 {
		t.Errorf("expected cursor 0, got %d", r.lastCursor)
	}
}

func TestShouldReflect_TooFewMessages(t *testing.T) {
	messages := make([]Message, 5)
	for i := range messages {
		messages[i] = Message{Role: RoleUser, Content: "hello"}
	}

	if shouldReflect(messages, 0) {
		t.Error("expected false for fewer than 8 messages")
	}
}

func TestShouldReflect_NoWriteTools(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Read this file"},
		{Role: RoleAssistant, Content: "Sure", ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/test.go"}`)},
		}},
		{Role: RoleTool, Content: "file contents", Name: "read"},
		{Role: RoleAssistant, Content: "Here it is"},
		{Role: RoleUser, Content: "Thanks"},
		{Role: RoleAssistant, Content: "Welcome"},
		{Role: RoleUser, Content: "What about this?"},
		{Role: RoleAssistant, Content: "Let me check", ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/other.go"}`)},
		}},
	}

	if shouldReflect(messages, 0) {
		t.Error("expected false when no write/edit tool calls and no corrections")
	}
}

func TestShouldReflect_WithWriteTools(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the bug in handler.go"},
		{Role: RoleAssistant, Content: "Reading file", ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/handler.go"}`)},
		}},
		{Role: RoleTool, Content: "package handler...", Name: "read"},
		{Role: RoleAssistant, Content: "I see the issue", ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/tmp/handler.go"}`)},
		}},
		{Role: RoleTool, Content: "file edited", Name: "edit"},
		{Role: RoleAssistant, Content: "Fixed the nil pointer check"},
		{Role: RoleUser, Content: "Great, thanks"},
		{Role: RoleAssistant, Content: "Anything else?"},
	}

	if !shouldReflect(messages, 0) {
		t.Error("expected true for session with write tool calls and 8+ messages")
	}
}

func TestShouldReflect_WithUserCorrections(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Update the config"},
		{Role: RoleAssistant, Content: "Done"},
		{Role: RoleUser, Content: "No, don't change that file"},
		{Role: RoleAssistant, Content: "Sorry, reverting"},
		{Role: RoleUser, Content: "Instead, update the other config"},
		{Role: RoleAssistant, Content: "Got it"},
		{Role: RoleUser, Content: "Better"},
		{Role: RoleAssistant, Content: "Anything else?"},
	}

	if !shouldReflect(messages, 0) {
		t.Error("expected true when user corrections detected")
	}
}

func TestShouldReflect_ManyIterations(t *testing.T) {
	var messages []Message
	messages = append(messages, Message{Role: RoleUser, Content: "Investigate the performance issue"})
	for i := 0; i < 11; i++ {
		messages = append(messages, Message{
			Role:    RoleAssistant,
			Content: "Checking...",
			ToolCalls: []ToolCall{
				{Name: "bash", Arguments: json.RawMessage(`{"command": "go test -bench ."}`)},
			},
		})
		messages = append(messages, Message{
			Role:    RoleTool,
			Content: "PASS",
			Name:    "bash",
		})
	}

	if !shouldReflect(messages, 0) {
		t.Error("expected true for 10+ tool-call iterations")
	}
}

func TestShouldReflect_RespectsOffset(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the bug"},
		{Role: RoleAssistant, Content: "Done", ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleTool, Content: "edited", Name: "edit"},
		{Role: RoleAssistant, Content: "Fixed"},
		{Role: RoleUser, Content: "Thanks"},
		{Role: RoleAssistant, Content: "Welcome"},
		{Role: RoleUser, Content: "Bye"},
		{Role: RoleAssistant, Content: "Goodbye"},
		{Role: RoleUser, Content: "One more thing"},
		{Role: RoleAssistant, Content: "Sure"},
	}

	if shouldReflect(messages, 8) {
		t.Error("expected false when cursor leaves too few new messages")
	}
}

func TestFindUserCorrections_DetectsPatterns(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{"no comma", "No, that's not right", true},
		{"no space", "no way, try the other approach", true},
		{"don't", "Please don't modify that file", true},
		{"do not", "Do not change the tests", true},
		{"stop", "Stop, let me think about this", true},
		{"instead", "Instead, use the factory pattern", true},
		{"that's wrong", "That's wrong, the API expects JSON", true},
		{"not like that", "Not like that, use a struct", true},
		{"not what i", "That's not what I asked for", true},
		{"please revert", "Please revert the last change", true},
		{"undo that", "Undo that, it broke the build", true},
		{"try again", "Try again with a different approach", true},
		{"normal message", "Looks good, ship it", false},
		{"question", "What does this function do?", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := []Message{{Role: RoleUser, Content: tt.content}}
			corrections := findUserCorrections(messages)
			got := len(corrections) > 0
			if got != tt.want {
				t.Errorf("findUserCorrections(%q) found=%v, want=%v", tt.content, got, tt.want)
			}
		})
	}
}

func TestFindUserCorrections_NoCorrections(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Please add a new endpoint for user profiles"},
		{Role: RoleAssistant, Content: "I'll create the handler"},
		{Role: RoleUser, Content: "Yes, that looks correct"},
		{Role: RoleAssistant, Content: "Done"},
		{Role: RoleUser, Content: "Perfect, thanks"},
	}

	corrections := findUserCorrections(messages)
	if len(corrections) != 0 {
		t.Errorf("expected no corrections, got %d: %v", len(corrections), corrections)
	}
}

func TestFindUserCorrections_IgnoresAssistantMessages(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "No, I think we should try again"},
		{Role: RoleUser, Content: "Looks good"},
	}

	corrections := findUserCorrections(messages)
	if len(corrections) != 0 {
		t.Errorf("expected no corrections from assistant messages, got %d", len(corrections))
	}
}

func TestFindUserCorrections_TruncatesLongMessages(t *testing.T) {
	long := strings.Repeat("x", 300) + " no, that's wrong"
	messages := []Message{
		{Role: RoleUser, Content: long},
	}

	corrections := findUserCorrections(messages)
	if len(corrections) != 1 {
		t.Fatalf("expected 1 correction, got %d", len(corrections))
	}
	if len(corrections[0]) > 210 {
		t.Errorf("expected truncated correction, got length %d", len(corrections[0]))
	}
}

func TestSummariseToolCalls_GroupsByName(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/main.go"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/handler.go"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "bash", Arguments: json.RawMessage(`{"command": "go test"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/routes.go"}`)},
		}},
	}

	summary := summariseToolCalls(messages)
	if !strings.Contains(summary, "read: 3 calls") {
		t.Errorf("expected 'read: 3 calls' in summary, got: %s", summary)
	}
	if !strings.Contains(summary, "bash: 1 calls") {
		t.Errorf("expected 'bash: 1 calls' in summary, got: %s", summary)
	}
}

func TestSummariseToolCalls_ExtractsFilePaths(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "write", Arguments: json.RawMessage(`{"file_path": "/home/user/project/main.go"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/home/user/project/handler_test.go"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "write", Arguments: json.RawMessage(`{"file_path": "/home/user/project/main.go"}`)},
		}},
	}

	summary := summariseToolCalls(messages)
	if !strings.Contains(summary, "write: 2 calls") {
		t.Errorf("expected 'write: 2 calls' in summary, got: %s", summary)
	}
	if !strings.Contains(summary, "main.go") {
		t.Errorf("expected 'main.go' in write summary, got: %s", summary)
	}
	if !strings.Contains(summary, "handler_test.go") {
		t.Errorf("expected 'handler_test.go' in edit summary, got: %s", summary)
	}
}

func TestSummariseToolCalls_DeduplicatesFilePaths(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/a/b/main.go"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/a/b/main.go"}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/a/b/main.go"}`)},
		}},
	}

	summary := summariseToolCalls(messages)
	if !strings.Contains(summary, "edit: 3 calls") {
		t.Errorf("expected 'edit: 3 calls' in summary, got: %s", summary)
	}
	count := strings.Count(summary, "main.go")
	if count != 1 {
		t.Errorf("expected main.go to appear once, appeared %d times in: %s", count, summary)
	}
}

func TestSummariseToolCalls_NoToolCalls(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi"},
	}

	summary := summariseToolCalls(messages)
	if summary != "" {
		t.Errorf("expected empty summary for no tool calls, got: %q", summary)
	}
}

func TestExtractTaskDescription_FirstUserMessage(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the authentication bug in the login handler"},
		{Role: RoleAssistant, Content: "I'll look into it"},
		{Role: RoleUser, Content: "Also check the middleware"},
	}

	desc := extractTaskDescription(messages)
	if desc != "Fix the authentication bug in the login handler" {
		t.Errorf("expected first user message, got: %q", desc)
	}
}

func TestExtractTaskDescription_SkipsShortMessages(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "hi"},
		{Role: RoleUser, Content: "Fix the broken endpoint in routes.go"},
	}

	desc := extractTaskDescription(messages)
	if desc != "Fix the broken endpoint in routes.go" {
		t.Errorf("expected second (substantive) message, got: %q", desc)
	}
}

func TestExtractTaskDescription_TruncatesLongMessages(t *testing.T) {
	long := strings.Repeat("x", 600)
	messages := []Message{
		{Role: RoleUser, Content: long},
	}

	desc := extractTaskDescription(messages)
	if !strings.HasSuffix(desc, "...") {
		t.Error("expected truncation of long task description")
	}
	if len(desc) > 510 {
		t.Errorf("expected truncated length <= 510, got %d", len(desc))
	}
}

func TestExtractTaskDescription_NoUserMessages(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "Hello, how can I help?"},
	}

	desc := extractTaskDescription(messages)
	if desc != "" {
		t.Errorf("expected empty description, got: %q", desc)
	}
}

func TestCountWriteToolCalls_CountsCorrectly(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "write", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{}`)},
			{Name: "edit", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "bash", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleUser, Content: "user message with write", ToolCalls: []ToolCall{
			{Name: "write", Arguments: json.RawMessage(`{}`)},
		}},
	}

	count := countWriteToolCalls(messages)
	if count != 3 {
		t.Errorf("expected 3 write/edit calls, got %d", count)
	}
}

func TestCountWriteToolCalls_NoWrites(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{}`)},
			{Name: "bash", Arguments: json.RawMessage(`{}`)},
		}},
	}

	count := countWriteToolCalls(messages)
	if count != 0 {
		t.Errorf("expected 0 write/edit calls, got %d", count)
	}
}

func TestFindErrors_DetectsToolErrors(t *testing.T) {
	messages := []Message{
		{Role: RoleTool, Content: "PASS\nok  \t./... 1.2s", Name: "bash"},
		{Role: RoleTool, Content: "error: cannot find package \"foo\"", Name: "bash"},
		{Role: RoleTool, Content: "build failed: missing import", Name: "bash"},
		{Role: RoleTool, Content: "panic: runtime error: nil pointer dereference", Name: "bash"},
		{Role: RoleTool, Content: "fatal: not a git repository", Name: "bash"},
		{Role: RoleTool, Content: "file contents here, all good", Name: "read"},
	}

	errors := findErrors(messages)
	if len(errors) != 4 {
		t.Errorf("expected 4 errors, got %d: %v", len(errors), errors)
	}
}

func TestFindErrors_NoErrors(t *testing.T) {
	messages := []Message{
		{Role: RoleTool, Content: "PASS\nok  \t./... 1.2s", Name: "bash"},
		{Role: RoleTool, Content: "package main\n\nfunc main() {}", Name: "read"},
		{Role: RoleUser, Content: "This has error in the user message"},
	}

	errors := findErrors(messages)
	if len(errors) != 0 {
		t.Errorf("expected no errors from clean tool results, got %d: %v", len(errors), errors)
	}
}

func TestFindErrors_TruncatesLongErrors(t *testing.T) {
	long := "error: " + strings.Repeat("x", 400)
	messages := []Message{
		{Role: RoleTool, Content: long, Name: "bash"},
	}

	errors := findErrors(messages)
	if len(errors) != 1 {
		t.Fatalf("expected 1 error, got %d", len(errors))
	}
	if len(errors[0]) > 310 {
		t.Errorf("expected truncated error, got length %d", len(errors[0]))
	}
}

func TestFindErrors_DetectsBlockErrors(t *testing.T) {
	messages := []Message{
		{
			Role: RoleTool,
			Blocks: []ContentBlock{
				{
					Type: "tool_result",
					ToolResult: &ToolResultBlock{
						ToolCallID: "tc_1",
						Content:    "permission denied",
						IsError:    true,
					},
				},
			},
		},
	}

	errors := findErrors(messages)
	if len(errors) != 1 {
		t.Errorf("expected 1 block error, got %d", len(errors))
	}
}

func TestParseReflectionResult_Valid(t *testing.T) {
	input := `{
		"outcome": "success",
		"summary": "Fixed the authentication handler by adding proper token validation.",
		"retry_feedback": "Check token format before validation",
		"heuristics": [
			{
				"rule": "Always validate JWT token format before attempting to decode",
				"context": "Go projects using JWT authentication",
				"confidence": "high",
				"category": "debugging",
				"scope": "global",
				"anti_pattern": false
			}
		],
		"should_record_episode": true
	}`

	result := parseReflectionResult(input)
	if result.Outcome != "success" {
		t.Errorf("outcome = %q, want %q", result.Outcome, "success")
	}
	if result.Summary == "" {
		t.Error("expected non-empty summary")
	}
	if result.RetryFeedback == "" {
		t.Error("expected non-empty retry feedback")
	}
	if len(result.Heuristics) != 1 {
		t.Fatalf("expected 1 heuristic, got %d", len(result.Heuristics))
	}

	h := result.Heuristics[0]
	if h.Rule == "" {
		t.Error("expected non-empty rule")
	}
	if h.Context == "" {
		t.Error("expected non-empty context")
	}
	if h.Confidence != "high" {
		t.Errorf("confidence = %q, want %q", h.Confidence, "high")
	}
	if h.Category != "debugging" {
		t.Errorf("category = %q, want %q", h.Category, "debugging")
	}
	if h.Scope != "global" {
		t.Errorf("scope = %q, want %q", h.Scope, "global")
	}
	if h.AntiPattern {
		t.Error("expected anti_pattern=false")
	}
	if !result.ShouldRecordEpisode {
		t.Error("expected should_record_episode=true")
	}
}

func TestParseReflectionResult_InvalidJSON(t *testing.T) {
	result := parseReflectionResult("not json at all")
	if result.Outcome != "" {
		t.Errorf("expected empty outcome for invalid JSON, got %q", result.Outcome)
	}
	if len(result.Heuristics) != 0 {
		t.Errorf("expected no heuristics for invalid JSON, got %d", len(result.Heuristics))
	}
}

func TestParseReflectionResult_WrappedInMarkdown(t *testing.T) {
	input := "```json\n{\"outcome\": \"partial\", \"summary\": \"Partially fixed\", \"retry_feedback\": \"\", \"heuristics\": [], \"should_record_episode\": false}\n```"

	result := parseReflectionResult(input)
	if result.Outcome != "partial" {
		t.Errorf("outcome = %q, want %q", result.Outcome, "partial")
	}
	if result.Summary != "Partially fixed" {
		t.Errorf("summary = %q, want %q", result.Summary, "Partially fixed")
	}
}

func TestParseReflectionResult_EmptyHeuristics(t *testing.T) {
	input := `{"outcome": "success", "summary": "Routine session", "retry_feedback": "", "heuristics": [], "should_record_episode": false}`

	result := parseReflectionResult(input)
	if result.Outcome != "success" {
		t.Errorf("outcome = %q, want %q", result.Outcome, "success")
	}
	if len(result.Heuristics) != 0 {
		t.Errorf("expected 0 heuristics, got %d", len(result.Heuristics))
	}
}

func TestParseReflectionResult_MultipleHeuristics(t *testing.T) {
	input := `{
		"outcome": "failure",
		"summary": "Failed to resolve dependency conflict",
		"retry_feedback": "Check go.mod replace directives first",
		"heuristics": [
			{"rule": "Check go.mod for replace directives", "context": "Go dependency issues", "confidence": "high", "category": "debugging", "scope": "global", "anti_pattern": false},
			{"rule": "Do not modify vendor directory manually", "context": "Go modules", "confidence": "high", "category": "approach", "scope": "global", "anti_pattern": true}
		],
		"should_record_episode": true
	}`

	result := parseReflectionResult(input)
	if len(result.Heuristics) != 2 {
		t.Fatalf("expected 2 heuristics, got %d", len(result.Heuristics))
	}
	if !result.Heuristics[1].AntiPattern {
		t.Error("expected second heuristic to be an anti-pattern")
	}
}

func TestAnalyseSession_Complete(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the broken test in auth_test.go"},
		{Role: RoleAssistant, Content: "Reading the test file", ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/auth_test.go"}`)},
		}},
		{Role: RoleTool, Content: "package auth\n\nfunc TestAuth(t *testing.T) {}", Name: "read"},
		{Role: RoleAssistant, Content: "I see the issue", ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/tmp/auth_test.go"}`)},
		}},
		{Role: RoleTool, Content: "file edited", Name: "edit"},
		{Role: RoleAssistant, Content: "Running tests", ToolCalls: []ToolCall{
			{Name: "bash", Arguments: json.RawMessage(`{"command": "go test ./..."}`)},
		}},
		{Role: RoleTool, Content: "error: test failed", Name: "bash"},
		{Role: RoleUser, Content: "No, don't change the assertion, fix the handler instead"},
		{Role: RoleAssistant, Content: "Got it, fixing handler", ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/tmp/handler.go"}`)},
		}},
		{Role: RoleTool, Content: "file edited", Name: "edit"},
		{Role: RoleAssistant, Content: "Running tests again", ToolCalls: []ToolCall{
			{Name: "bash", Arguments: json.RawMessage(`{"command": "go test ./..."}`)},
		}},
		{Role: RoleTool, Content: "PASS\nok  \tauth 0.5s", Name: "bash"},
		{Role: RoleAssistant, Content: "Tests pass now"},
		{Role: RoleUser, Content: "Perfect, thanks"},
	}

	analysis := analyseSession(messages)

	if analysis.TaskDescription != "Fix the broken test in auth_test.go" {
		t.Errorf("task = %q", analysis.TaskDescription)
	}
	if analysis.WriteToolCalls != 2 {
		t.Errorf("write calls = %d, want 2", analysis.WriteToolCalls)
	}
	if len(analysis.UserCorrections) != 1 {
		t.Errorf("corrections = %d, want 1", len(analysis.UserCorrections))
	}
	if len(analysis.ErrorsEncountered) != 1 {
		t.Errorf("errors = %d, want 1", len(analysis.ErrorsEncountered))
	}
	if analysis.IterationCount != 5 {
		t.Errorf("iterations = %d, want 5", analysis.IterationCount)
	}
	if analysis.Outcome != "success" {
		t.Errorf("outcome = %q, want %q", analysis.Outcome, "success")
	}

	if !strings.Contains(analysis.ToolCallSummary, "read:") {
		t.Error("expected 'read' in tool call summary")
	}
	if !strings.Contains(analysis.ToolCallSummary, "edit:") {
		t.Error("expected 'edit' in tool call summary")
	}
	if !strings.Contains(analysis.ToolCallSummary, "bash:") {
		t.Error("expected 'bash' in tool call summary")
	}
}

func TestInferOutcome_Success(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the bug"},
		{Role: RoleAssistant, Content: "Done"},
		{Role: RoleUser, Content: "Thanks, that's perfect"},
	}

	outcome := inferOutcome(messages)
	if outcome != "success" {
		t.Errorf("outcome = %q, want %q", outcome, "success")
	}
}

func TestInferOutcome_Failure(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the bug"},
		{Role: RoleAssistant, Content: "Done"},
		{Role: RoleUser, Content: "That's wrong, it's still broken"},
	}

	outcome := inferOutcome(messages)
	if outcome != "failure" {
		t.Errorf("outcome = %q, want %q", outcome, "failure")
	}
}

func TestInferOutcome_Partial(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the bug"},
		{Role: RoleAssistant, Content: "I made some changes"},
		{Role: RoleUser, Content: "OK, I'll take a look later"},
	}

	outcome := inferOutcome(messages)
	if outcome != "partial" {
		t.Errorf("outcome = %q, want %q", outcome, "partial")
	}
}

func TestInferOutcome_Empty(t *testing.T) {
	outcome := inferOutcome(nil)
	if outcome != "unknown" {
		t.Errorf("outcome = %q, want %q", outcome, "unknown")
	}
}

func TestCountAssistantToolIterations(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "thinking"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "write", Arguments: json.RawMessage(`{}`)},
			{Name: "bash", Arguments: json.RawMessage(`{}`)},
		}},
		{Role: RoleUser, Content: "hello"},
	}

	count := countAssistantToolIterations(messages)
	if count != 2 {
		t.Errorf("expected 2 iterations, got %d", count)
	}
}

func TestBuildReflectionPrompt_IncludesAnalysis(t *testing.T) {
	analysis := SessionAnalysis{
		TaskDescription:   "Fix the auth bug",
		ToolCallSummary:   "read: 3 calls\nedit: 1 calls",
		ErrorsEncountered: []string{"error: nil pointer"},
		UserCorrections:   []string{"No, use the other approach"},
		IterationCount:    5,
		WriteToolCalls:    1,
		Outcome:           "success",
	}

	messages := []Message{
		{Role: RoleUser, Content: "Fix the auth bug"},
		{Role: RoleAssistant, Content: "On it"},
	}

	prompt := buildReflectionPrompt(analysis, messages)

	if !strings.Contains(prompt, "Fix the auth bug") {
		t.Error("expected task description in prompt")
	}
	if !strings.Contains(prompt, "read: 3 calls") {
		t.Error("expected tool summary in prompt")
	}
	if !strings.Contains(prompt, "error: nil pointer") {
		t.Error("expected errors in prompt")
	}
	if !strings.Contains(prompt, "No, use the other approach") {
		t.Error("expected user corrections in prompt")
	}
	if !strings.Contains(prompt, "5 tool-call rounds") {
		t.Error("expected iteration count in prompt")
	}
	if !strings.Contains(prompt, "[user]: Fix the auth bug") {
		t.Error("expected conversation transcript in prompt")
	}
}

func TestBuildReflectionPrompt_TruncatesLongContent(t *testing.T) {
	long := strings.Repeat("x", 2000)
	messages := []Message{
		{Role: RoleUser, Content: long},
	}

	prompt := buildReflectionPrompt(SessionAnalysis{}, messages)
	if !strings.Contains(prompt, "[...truncated]") {
		t.Error("expected truncation of long user message")
	}
}

func TestBuildReflectionPrompt_IncludesToolResults(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/main.go"}`)},
		}},
		{Role: RoleTool, Content: "package main", Name: "read"},
	}

	prompt := buildReflectionPrompt(SessionAnalysis{}, messages)
	if !strings.Contains(prompt, "[tool_call read]") {
		t.Error("expected tool call in transcript")
	}
	if !strings.Contains(prompt, "[tool_result read]") {
		t.Error("expected tool result in transcript")
	}
}

func TestExtractFilePath_Valid(t *testing.T) {
	args := json.RawMessage(`{"file_path": "/home/user/main.go"}`)
	path := extractFilePath(args)
	if path != "/home/user/main.go" {
		t.Errorf("expected /home/user/main.go, got %q", path)
	}
}

func TestExtractFilePath_InvalidJSON(t *testing.T) {
	args := json.RawMessage(`not json`)
	path := extractFilePath(args)
	if path != "" {
		t.Errorf("expected empty path for invalid JSON, got %q", path)
	}
}

func TestExtractFilePath_NoFilePath(t *testing.T) {
	args := json.RawMessage(`{"command": "go test"}`)
	path := extractFilePath(args)
	if path != "" {
		t.Errorf("expected empty path when no file_path key, got %q", path)
	}
}

func TestBaseFilename(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"/home/user/project/main.go", "main.go"},
		{"main.go", "main.go"},
		{"/a/b/c/d.txt", "d.txt"},
		{"", ""},
	}

	for _, tt := range tests {
		got := baseFilename(tt.input)
		if got != tt.want {
			t.Errorf("baseFilename(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
