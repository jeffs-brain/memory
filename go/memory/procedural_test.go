// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDetectProcedurals_SkillInvocation(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Deploy the staging environment"},
		{Role: RoleAssistant, Content: "Running deploy skill", ToolCalls: []ToolCall{
			{ID: "tc_1", Name: "skill", Arguments: json.RawMessage(`{"skill": "deploy", "args": "--env staging"}`)},
		}},
		{Role: RoleTool, Content: "Deployment complete", ToolCallID: "tc_1", Name: "skill"},
	}

	records := DetectProcedurals(messages)
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}

	r := records[0]
	if r.Tier != "skill" {
		t.Errorf("tier = %q, want %q", r.Tier, "skill")
	}
	if r.Name != "deploy" {
		t.Errorf("name = %q, want %q", r.Name, "deploy")
	}
	if r.Outcome != "ok" {
		t.Errorf("outcome = %q, want %q", r.Outcome, "ok")
	}
	if r.TaskContext != "Deploy the staging environment" {
		t.Errorf("task context = %q", r.TaskContext)
	}
}

func TestDetectProcedurals_AgentInvocation(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Review the PR"},
		{Role: RoleAssistant, Content: "Delegating to review agent", ToolCalls: []ToolCall{
			{ID: "tc_2", Name: "agent", Arguments: json.RawMessage(`{"type": "reviewer", "prompt": "Review PR #42"}`)},
		}},
		{Role: RoleTool, Content: "Review complete, 3 comments posted", ToolCallID: "tc_2", Name: "agent"},
	}

	records := DetectProcedurals(messages)
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}

	r := records[0]
	if r.Tier != "agent" {
		t.Errorf("tier = %q, want %q", r.Tier, "agent")
	}
	if r.Name != "reviewer" {
		t.Errorf("name = %q, want %q", r.Name, "reviewer")
	}
	if r.Outcome != "ok" {
		t.Errorf("outcome = %q, want %q", r.Outcome, "ok")
	}
	if r.TaskContext != "Review PR #42" {
		t.Errorf("task context = %q, want %q", r.TaskContext, "Review PR #42")
	}
}

func TestDetectProcedurals_EmptyForNoToolCalls(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "What is Go?"},
		{Role: RoleAssistant, Content: "Go is a programming language."},
		{Role: RoleUser, Content: "Thanks"},
	}

	records := DetectProcedurals(messages)
	if len(records) != 0 {
		t.Errorf("expected 0 records, got %d", len(records))
	}
}

func TestDetectProcedurals_MalformedSkillArgs(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Do something"},
		{Role: RoleAssistant, Content: "Trying", ToolCalls: []ToolCall{
			{ID: "tc_3", Name: "skill", Arguments: json.RawMessage(`{invalid json}`)},
		}},
	}

	records := DetectProcedurals(messages)
	if len(records) != 0 {
		t.Errorf("expected 0 records for malformed args, got %d", len(records))
	}
}

func TestInferToolCallOutcome_ErrorInResult(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "running"},
		{Role: RoleTool, Content: "error: permission denied", ToolCallID: "tc_err", Name: "skill"},
	}

	outcome := inferToolCallOutcome(messages, 0, "tc_err")
	if outcome != "error" {
		t.Errorf("outcome = %q, want %q", outcome, "error")
	}
}

func TestInferToolCallOutcome_FailedInResult(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "running"},
		{Role: RoleTool, Content: "Task failed: timeout", ToolCallID: "tc_fail", Name: "skill"},
	}

	outcome := inferToolCallOutcome(messages, 0, "tc_fail")
	if outcome != "error" {
		t.Errorf("outcome = %q, want %q", outcome, "error")
	}
}

func TestInferToolCallOutcome_SuccessResult(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "running"},
		{Role: RoleTool, Content: "All good, deployed successfully", ToolCallID: "tc_ok", Name: "skill"},
	}

	outcome := inferToolCallOutcome(messages, 0, "tc_ok")
	if outcome != "ok" {
		t.Errorf("outcome = %q, want %q", outcome, "ok")
	}
}

func TestInferToolCallOutcome_NoToolResult(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "running"},
		{Role: RoleUser, Content: "What happened?"},
	}

	outcome := inferToolCallOutcome(messages, 0, "tc_missing")
	if outcome != "partial" {
		t.Errorf("outcome = %q, want %q", outcome, "partial")
	}
}

func TestInferProceduralContext_PrecedingUserMessage(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Fix the authentication bug"},
		{Role: RoleAssistant, Content: "On it"},
		{Role: RoleUser, Content: "Also check the tests"},
		{Role: RoleAssistant, Content: "Sure"},
	}

	ctx := inferProceduralContext(messages, 3)
	if ctx != "Also check the tests" {
		t.Errorf("context = %q, want %q", ctx, "Also check the tests")
	}
}

func TestInferProceduralContext_TruncationAt160(t *testing.T) {
	longMsg := strings.Repeat("a", 200)
	messages := []Message{
		{Role: RoleUser, Content: longMsg},
		{Role: RoleAssistant, Content: "Noted"},
	}

	ctx := inferProceduralContext(messages, 1)
	if len(ctx) != 160 {
		t.Errorf("context length = %d, want 160", len(ctx))
	}
}

func TestInferProceduralContext_NoUserMessage(t *testing.T) {
	messages := []Message{
		{Role: RoleAssistant, Content: "Starting up"},
	}

	ctx := inferProceduralContext(messages, 0)
	if ctx != "" {
		t.Errorf("context = %q, want empty", ctx)
	}
}

func TestFormatProceduralRecord_ValidMarkdown(t *testing.T) {
	r := ProceduralRecord{
		Tier:        "skill",
		Name:        "deploy",
		TaskContext: "Deploy staging",
		Outcome:     "ok",
		ToolCalls:   []string{"skill"},
		Tags:        []string{"procedural", "skill", "deploy"},
	}

	out := FormatProceduralRecord(r)

	if !strings.Contains(out, "---\nname: deploy\n") {
		t.Error("expected frontmatter name")
	}
	if !strings.Contains(out, "type: procedural\n") {
		t.Error("expected type: procedural")
	}
	if !strings.Contains(out, "tier: skill\n") {
		t.Error("expected tier: skill")
	}
	if !strings.Contains(out, "outcome: ok\n") {
		t.Error("expected outcome: ok")
	}
	if !strings.Contains(out, "tags: [procedural, skill, deploy]\n") {
		t.Error("expected tags list")
	}
	if !strings.Contains(out, "## Context\n\nDeploy staging\n") {
		t.Error("expected context section")
	}
	if !strings.Contains(out, "## Tool sequence\n\nskill\n") {
		t.Error("expected tool sequence section")
	}
}

func TestFormatProceduralRecord_EmptyContext(t *testing.T) {
	r := ProceduralRecord{
		Tier:      "agent",
		Name:      "reviewer",
		Outcome:   "ok",
		ToolCalls: []string{"agent"},
		Tags:      []string{"procedural", "agent", "reviewer"},
	}

	out := FormatProceduralRecord(r)

	if strings.Contains(out, "## Context") {
		t.Error("expected no context section when TaskContext is empty")
	}
}

func TestDetectProcedurals_SkillWithEmptyName(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Do something"},
		{Role: RoleAssistant, Content: "Trying", ToolCalls: []ToolCall{
			{ID: "tc_4", Name: "skill", Arguments: json.RawMessage(`{"skill": "", "args": "test"}`)},
		}},
	}

	records := DetectProcedurals(messages)
	if len(records) != 0 {
		t.Errorf("expected 0 records for empty skill name, got %d", len(records))
	}
}

func TestDetectProcedurals_MixedSkillsAndAgents(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Build and review"},
		{Role: RoleAssistant, Content: "Building", ToolCalls: []ToolCall{
			{ID: "tc_5", Name: "skill", Arguments: json.RawMessage(`{"skill": "build", "args": ""}`)},
		}},
		{Role: RoleTool, Content: "Build succeeded", ToolCallID: "tc_5", Name: "skill"},
		{Role: RoleAssistant, Content: "Now reviewing", ToolCalls: []ToolCall{
			{ID: "tc_6", Name: "agent", Arguments: json.RawMessage(`{"type": "reviewer", "prompt": "Check the build output"}`)},
		}},
		{Role: RoleTool, Content: "Review complete", ToolCallID: "tc_6", Name: "agent"},
	}

	records := DetectProcedurals(messages)
	if len(records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(records))
	}
	if records[0].Tier != "skill" {
		t.Errorf("first record tier = %q, want %q", records[0].Tier, "skill")
	}
	if records[1].Tier != "agent" {
		t.Errorf("second record tier = %q, want %q", records[1].Tier, "agent")
	}
}

func TestDetectProcedurals_AgentPromptTruncation(t *testing.T) {
	longPrompt := strings.Repeat("x", 200)
	messages := []Message{
		{Role: RoleUser, Content: "Do something"},
		{Role: RoleAssistant, Content: "Delegating", ToolCalls: []ToolCall{
			{ID: "tc_7", Name: "agent", Arguments: json.RawMessage(`{"type": "worker", "prompt": "` + longPrompt + `"}`)},
		}},
		{Role: RoleTool, Content: "Done", ToolCallID: "tc_7", Name: "agent"},
	}

	records := DetectProcedurals(messages)
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	if len(records[0].TaskContext) != 160 {
		t.Errorf("task context length = %d, want 160", len(records[0].TaskContext))
	}
}

func TestDetectProcedurals_AgentFallsBackToUserContext(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Check the logs"},
		{Role: RoleAssistant, Content: "Delegating", ToolCalls: []ToolCall{
			{ID: "tc_8", Name: "agent", Arguments: json.RawMessage(`{"type": "monitor", "prompt": ""}`)},
		}},
		{Role: RoleTool, Content: "Logs checked", ToolCallID: "tc_8", Name: "agent"},
	}

	records := DetectProcedurals(messages)
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	if records[0].TaskContext != "Check the logs" {
		t.Errorf("task context = %q, want %q", records[0].TaskContext, "Check the logs")
	}
}
