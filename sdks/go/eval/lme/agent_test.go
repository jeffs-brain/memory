// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/store/mem"
)

// scriptedAgentProvider replays a canned sequence of Complete
// responses. Each call consumes the next scripted response; unknown
// calls return a final assistant text so a misbehaving loop cannot
// hang the test indefinitely.
type scriptedAgentProvider struct {
	mu        sync.Mutex
	responses []llm.CompleteResponse
	calls     int
	maxCtx    int
}

func (p *scriptedAgentProvider) Complete(_ context.Context, _ llm.CompleteRequest) (llm.CompleteResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.calls >= len(p.responses) {
		return llm.CompleteResponse{Text: "(fallback)", Stop: llm.StopEndTurn}, nil
	}
	resp := p.responses[p.calls]
	p.calls++
	return resp, nil
}

func (p *scriptedAgentProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("stream not supported")
}
func (p *scriptedAgentProvider) Close() error          { return nil }
func (p *scriptedAgentProvider) MaxContextTokens() int { return p.maxCtx }

// stallingAgentProvider never returns until the context is cancelled.
// Lets the timeout path be exercised deterministically.
type stallingAgentProvider struct{}

func (p *stallingAgentProvider) Complete(ctx context.Context, _ llm.CompleteRequest) (llm.CompleteResponse, error) {
	<-ctx.Done()
	return llm.CompleteResponse{}, ctx.Err()
}
func (p *stallingAgentProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, fmt.Errorf("stream not supported")
}
func (p *stallingAgentProvider) Close() error { return nil }

func TestRunQuestionsAgentic_CapturesToolCallsAndAnswer(t *testing.T) {
	args := json.RawMessage(`{"query":"red car"}`)
	provider := &scriptedAgentProvider{
		responses: []llm.CompleteResponse{
			{
				ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "kb_search", Arguments: args}},
				TokensIn:  20,
				TokensOut: 10,
			},
			{
				Text:      "The car was red.",
				Stop:      llm.StopEndTurn,
				TokensIn:  30,
				TokensOut: 8,
			},
		},
	}

	store := mem.New()
	costs := &CostAccumulator{}
	questions := []Question{
		{ID: "q1", Category: "single-session", Question: "What colour was the car?", Answer: "red", QuestionDate: "2024-03-20"},
	}

	outcomes := RunQuestionsAgentic(
		context.Background(),
		defaultAgentFactory,
		store,
		provider,
		"scripted-model",
		questions,
		AgenticOpts{MaxIterations: 5, QuestionTimeout: 5 * time.Second},
		costs,
		1,
	)

	if len(outcomes) != 1 {
		t.Fatalf("expected 1 outcome, got %d", len(outcomes))
	}
	o := outcomes[0]
	if o.ID != "q1" {
		t.Fatalf("ID = %q, want q1", o.ID)
	}
	if o.Category != "single-session" {
		t.Fatalf("Category = %q, want single-session", o.Category)
	}
	if o.QuestionDate != "2024-03-20" {
		t.Fatalf("QuestionDate = %q, want 2024-03-20", o.QuestionDate)
	}
	if o.Error != "" {
		t.Fatalf("unexpected error: %s", o.Error)
	}
	if len(o.ToolCalls) != 1 || o.ToolCalls[0] != "kb_search" {
		t.Fatalf("ToolCalls = %v, want [kb_search]", o.ToolCalls)
	}
	if o.AgentAnswer != "The car was red." {
		t.Fatalf("AgentAnswer = %q, want %q", o.AgentAnswer, "The car was red.")
	}
}

func TestRunQuestionsAgentic_Timeout(t *testing.T) {
	provider := &stallingAgentProvider{}

	store := mem.New()
	costs := &CostAccumulator{}
	questions := []Question{
		{ID: "q1", Category: "temporal", Question: "When?", Answer: "Tuesday"},
	}

	start := time.Now()
	outcomes := RunQuestionsAgentic(
		context.Background(),
		defaultAgentFactory,
		store,
		provider,
		"scripted-model",
		questions,
		AgenticOpts{MaxIterations: 3, QuestionTimeout: 50 * time.Millisecond},
		costs,
		1,
	)
	elapsed := time.Since(start)

	if len(outcomes) != 1 {
		t.Fatalf("expected 1 outcome, got %d", len(outcomes))
	}
	if outcomes[0].Error == "" {
		t.Fatalf("expected an error on timeout, got empty")
	}
	if elapsed > 5*time.Second {
		t.Fatalf("timeout test took %s, expected < 5s", elapsed)
	}
}

func TestRunQuestionsAgentic_TextOnlyNoTools(t *testing.T) {
	provider := &scriptedAgentProvider{
		responses: []llm.CompleteResponse{
			{Text: "Jeff says: hello.", Stop: llm.StopEndTurn, TokensIn: 5, TokensOut: 3},
		},
	}

	store := mem.New()
	costs := &CostAccumulator{}
	questions := []Question{
		{ID: "q1", Category: "single-session", Question: "Say hi.", Answer: "hi"},
	}

	outcomes := RunQuestionsAgentic(
		context.Background(),
		defaultAgentFactory,
		store,
		provider,
		"scripted-model",
		questions,
		AgenticOpts{MaxIterations: 5, QuestionTimeout: 5 * time.Second},
		costs,
		1,
	)

	if len(outcomes) != 1 {
		t.Fatalf("expected 1 outcome, got %d", len(outcomes))
	}
	o := outcomes[0]
	if o.Error != "" {
		t.Fatalf("unexpected error: %s", o.Error)
	}
	if len(o.ToolCalls) != 0 {
		t.Fatalf("ToolCalls = %v, want empty", o.ToolCalls)
	}
	if o.AgentAnswer != "Jeff says: hello." {
		t.Fatalf("AgentAnswer = %q, want %q", o.AgentAnswer, "Jeff says: hello.")
	}
}

func TestRunQuestionsAgentic_MaxIterations(t *testing.T) {
	// Provider keeps returning tool calls, never a final text answer.
	// The loop should bail once MaxIterations is hit.
	provider := &scriptedAgentProvider{
		responses: []llm.CompleteResponse{
			{ToolCalls: []llm.ToolCall{{ID: "1", Name: "kb_search", Arguments: json.RawMessage(`{"query":"x"}`)}}, TokensIn: 10, TokensOut: 5},
			{ToolCalls: []llm.ToolCall{{ID: "2", Name: "kb_search", Arguments: json.RawMessage(`{"query":"y"}`)}}, TokensIn: 10, TokensOut: 5},
			{ToolCalls: []llm.ToolCall{{ID: "3", Name: "kb_search", Arguments: json.RawMessage(`{"query":"z"}`)}}, TokensIn: 10, TokensOut: 5},
		},
	}

	store := mem.New()
	costs := &CostAccumulator{}
	questions := []Question{{ID: "q1", Category: "c", Question: "?", Answer: "a"}}

	outcomes := RunQuestionsAgentic(
		context.Background(),
		defaultAgentFactory,
		store,
		provider,
		"scripted-model",
		questions,
		AgenticOpts{MaxIterations: 2, QuestionTimeout: 5 * time.Second},
		costs,
		1,
	)

	if len(outcomes) != 1 {
		t.Fatalf("expected 1 outcome, got %d", len(outcomes))
	}
	if outcomes[0].Error == "" {
		t.Error("expected max-iterations error, got empty")
	}
	if len(outcomes[0].ToolCalls) != 2 {
		t.Errorf("tool calls = %d, want 2 (iteration budget)", len(outcomes[0].ToolCalls))
	}
}

func TestRunQuestionsAgentic_CostsAccumulate(t *testing.T) {
	provider := &scriptedAgentProvider{
		responses: []llm.CompleteResponse{
			{Text: "ok", Stop: llm.StopEndTurn, TokensIn: 100, TokensOut: 50},
		},
	}
	store := mem.New()
	costs := &CostAccumulator{}
	questions := []Question{{ID: "q1", Category: "c", Question: "?", Answer: "a"}}

	_ = RunQuestionsAgentic(
		context.Background(),
		defaultAgentFactory,
		store,
		provider,
		"gpt-4o-mini",
		questions,
		AgenticOpts{},
		costs,
		1,
	)

	snap := costs.Snapshot()
	if snap.AgentUSD <= 0 {
		t.Errorf("expected AgentUSD > 0, got %v", snap.AgentUSD)
	}
}

func TestExecuteAgentTool_UnknownTool(t *testing.T) {
	store := mem.New()
	res := &AgentResources{Store: store, ProjectPath: "/eval/lme"}
	result := executeAgentTool(context.Background(), res, llm.ToolCall{
		Name:      "not_a_tool",
		Arguments: json.RawMessage(`{}`),
	})
	if result == "" {
		t.Fatal("expected non-empty result")
	}
	if !containsString(result, "unknown tool") {
		t.Errorf("result = %q, want contains 'unknown tool'", result)
	}
}

func TestExecuteAgentTool_KBSearchFallbackNoRetriever(t *testing.T) {
	// Seed raw/lme/ with one session containing the query string so the
	// naive fallback can surface it.
	store := mem.New()
	ctx := context.Background()
	path := brain.Path("raw/lme/sess-001.md")
	body := "---\nsession_id: sess-001\n---\n\n[user]: I love kangaroos.\n\n[assistant]: Ok.\n"
	if err := store.Write(ctx, path, []byte(body)); err != nil {
		t.Fatalf("write: %v", err)
	}

	res := &AgentResources{Store: store, ProjectPath: "/eval/lme"}
	result := executeAgentTool(ctx, res, llm.ToolCall{
		Name:      "kb_search",
		Arguments: json.RawMessage(`{"query":"kangaroos"}`),
	})
	if !containsString(result, "kangaroos") {
		t.Errorf("expected kb_search fallback to find raw session, got: %s", result)
	}
}

func TestExecuteAgentTool_MemoryRecallMissing(t *testing.T) {
	store := mem.New()
	res := &AgentResources{Store: store, ProjectPath: "/eval/lme"}
	result := executeAgentTool(context.Background(), res, llm.ToolCall{
		Name:      "memory_recall",
		Arguments: json.RawMessage(`{"keyword":"nothing"}`),
	})
	if result == "" {
		t.Fatal("expected non-empty result")
	}
}

func TestDefaultAgentFactory_Wires(t *testing.T) {
	store := mem.New()
	res, err := defaultAgentFactory(context.Background(), store)
	if err != nil {
		t.Fatalf("defaultAgentFactory: %v", err)
	}
	if res.Store == nil {
		t.Error("Store not set")
	}
	if res.Memory == nil {
		t.Error("Memory not set")
	}
	if res.ProjectPath != "/eval/lme" {
		t.Errorf("ProjectPath = %q, want /eval/lme", res.ProjectPath)
	}
}

// containsString is a tiny wrapper around strings.Contains so the test
// helper stays co-located with the cases.
func containsString(haystack, needle string) bool {
	return len(needle) > 0 && len(haystack) >= len(needle) &&
		indexOf(haystack, needle) >= 0
}

// indexOf is a local, non-generic substring search. Kept here instead
// of importing strings again to make test intent obvious.
func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
