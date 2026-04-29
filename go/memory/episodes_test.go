// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"encoding/json"
	"testing"
)

func TestNewEpisodeRecorder(t *testing.T) {
	r := NewEpisodeRecorder()
	if r == nil {
		t.Fatal("expected non-nil recorder")
	}
}

func TestShouldRecordEpisode_TooFewMessages(t *testing.T) {
	messages := make([]Message, 5)
	for i := range messages {
		messages[i] = Message{Role: RoleUser, Content: "hello"}
	}

	if shouldRecordEpisode(messages) {
		t.Error("expected false for fewer than 8 messages")
	}
}

func TestShouldRecordEpisode_NoWriteTools(t *testing.T) {
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

	if shouldRecordEpisode(messages) {
		t.Error("expected false when no write/edit tool calls present")
	}
}

func TestShouldRecordEpisode_WithWriteTool(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the bug"},
		{Role: RoleAssistant, Content: "Reading file", ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/test.go"}`)},
		}},
		{Role: RoleTool, Content: "file contents", Name: "read"},
		{Role: RoleAssistant, Content: "I see the issue", ToolCalls: []ToolCall{
			{Name: "edit", Arguments: json.RawMessage(`{"file_path": "/tmp/test.go"}`)},
		}},
		{Role: RoleTool, Content: "file edited", Name: "edit"},
		{Role: RoleAssistant, Content: "Fixed"},
		{Role: RoleUser, Content: "Great"},
		{Role: RoleAssistant, Content: "Anything else?"},
	}

	if !shouldRecordEpisode(messages) {
		t.Error("expected true for session with edit tool call and 8+ messages")
	}
}

func TestShouldRecordEpisode_WithWriteToolExact(t *testing.T) {
	messages := make([]Message, 0, 10)
	for i := 0; i < 7; i++ {
		messages = append(messages, Message{Role: RoleUser, Content: "msg"})
	}
	messages = append(messages, Message{
		Role: RoleAssistant, Content: "writing",
		ToolCalls: []ToolCall{
			{Name: "write", Arguments: json.RawMessage(`{}`)},
		},
	})

	if !shouldRecordEpisode(messages) {
		t.Error("expected true for session with write tool call and 8 messages")
	}
}

func TestParseEpisodeResult_Valid(t *testing.T) {
	input := `{"significant": true, "summary": "Fixed auth bug", "outcome": "success", "heuristics": ["Check token expiry"], "tags": ["auth"]}`

	result := parseEpisodeResult(input)
	if !result.Significant {
		t.Error("expected significant=true")
	}
	if result.Summary != "Fixed auth bug" {
		t.Errorf("summary = %q", result.Summary)
	}
	if result.Outcome != "success" {
		t.Errorf("outcome = %q", result.Outcome)
	}
	if len(result.Heuristics) != 1 || result.Heuristics[0] != "Check token expiry" {
		t.Errorf("heuristics = %v", result.Heuristics)
	}
	if len(result.Tags) != 1 || result.Tags[0] != "auth" {
		t.Errorf("tags = %v", result.Tags)
	}
}

func TestParseEpisodeResult_NotSignificant(t *testing.T) {
	input := `{"significant": false, "summary": "", "outcome": "", "heuristics": [], "tags": []}`

	result := parseEpisodeResult(input)
	if result.Significant {
		t.Error("expected significant=false")
	}
}

func TestParseEpisodeResult_WrappedInMarkdown(t *testing.T) {
	input := "```json\n{\"significant\": true, \"summary\": \"test\", \"outcome\": \"partial\", \"heuristics\": [], \"tags\": []}\n```"

	result := parseEpisodeResult(input)
	if !result.Significant {
		t.Error("expected significant=true from markdown-wrapped JSON")
	}
	if result.Summary != "test" {
		t.Errorf("summary = %q", result.Summary)
	}
}

func TestParseEpisodeResult_InvalidJSON(t *testing.T) {
	result := parseEpisodeResult("not json at all")
	if result.Significant {
		t.Error("expected significant=false for invalid JSON")
	}
}

func TestBuildEpisodePrompt_IncludesMessages(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the authentication bug"},
		{Role: RoleAssistant, Content: "Looking at it", ToolCalls: []ToolCall{
			{Name: "read", Arguments: json.RawMessage(`{"file_path": "/tmp/auth.go"}`)},
		}},
		{Role: RoleTool, Content: "package auth...", Name: "read"},
	}

	prompt := buildEpisodePrompt(messages)

	if !contains(prompt, "[user]: Fix the authentication bug") {
		t.Error("expected user message in prompt")
	}
	if !contains(prompt, "[assistant]: Looking at it") {
		t.Error("expected assistant message in prompt")
	}
	if !contains(prompt, "[tool_call read]") {
		t.Error("expected tool call in prompt")
	}
	if !contains(prompt, "[tool_result read]") {
		t.Error("expected tool result in prompt")
	}
}

func TestBuildEpisodePrompt_TruncatesLongContent(t *testing.T) {
	longContent := make([]byte, 2000)
	for i := range longContent {
		longContent[i] = 'x'
	}

	messages := []Message{
		{Role: RoleUser, Content: string(longContent)},
	}

	prompt := buildEpisodePrompt(messages)
	if !contains(prompt, "[...truncated]") {
		t.Error("expected truncation of long user message")
	}
}
