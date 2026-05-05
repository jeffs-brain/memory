// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestAnthropicComplete(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("path %q", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "k" {
			t.Errorf("missing key: %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("missing version: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"system":"be brief"`) {
			t.Errorf("system missing: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"id":"msg_1","type":"message","role":"assistant",
			"content":[{"type":"text","text":"hi"}],
			"stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}
		}`)
	}))
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	defer func() { _ = p.Close() }()
	resp, err := p.Complete(context.Background(), llm.CompleteRequest{
		Model: "claude-test",
		Messages: []llm.Message{
			{Role: llm.RoleSystem, Content: "be brief"},
			{Role: llm.RoleUser, Content: "hello"},
		},
		MaxTokens: 32,
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.Text != "hi" {
		t.Fatalf("text %q", resp.Text)
	}
	if resp.Stop != llm.StopEndTurn {
		t.Fatalf("stop %q", resp.Stop)
	}
	if resp.TokensIn != 3 || resp.TokensOut != 1 {
		t.Fatalf("usage wrong: %+v", resp)
	}
}

func TestAnthropicStreaming(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("flusher unavailable")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		events := []string{
			`{"type":"message_start"}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"he"}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"llo"}}`,
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
			`{"type":"message_stop"}`,
		}
		for _, ev := range events {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", ev)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "say hi"}},
		MaxTokens: 32,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var text strings.Builder
	var stop llm.StopReason
	for chunk := range ch {
		text.WriteString(chunk.DeltaText)
		if chunk.Stop != "" {
			stop = chunk.Stop
		}
	}
	if text.String() != "hello" {
		t.Fatalf("text %q", text.String())
	}
	if stop != llm.StopEndTurn {
		t.Fatalf("stop %q", stop)
	}
}

func TestAnthropicError(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprint(w, `{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`)
	}))
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), llm.CompleteRequest{
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		MaxTokens: 1,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "bad") {
		t.Fatalf("missing message: %v", err)
	}
}

func TestAnthropicRejectsEmpty(t *testing.T) {
	t.Parallel()
	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: "http://unused"})
	if _, err := p.Complete(context.Background(), llm.CompleteRequest{}); err == nil {
		t.Fatal("expected error")
	}
}

func TestAnthropicStream_SingleToolCall(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_abc123","name":"search"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"query\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"hello world\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "search for hello"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var tools []llm.ToolCall
	var stop llm.StopReason
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
		if chunk.Stop != "" {
			stop = chunk.Stop
		}
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].ID != "toolu_abc123" {
		t.Errorf("tool ID = %q, want %q", tools[0].ID, "toolu_abc123")
	}
	if tools[0].Name != "search" {
		t.Errorf("tool name = %q, want %q", tools[0].Name, "search")
	}
	if string(tools[0].Arguments) != `{"query":"hello world"}` {
		t.Errorf("tool args = %q, want %q", string(tools[0].Arguments), `{"query":"hello world"}`)
	}
	if stop != llm.StopToolUse {
		t.Errorf("stop = %q, want %q", stop, llm.StopToolUse)
	}
}

func TestAnthropicStream_TextAndToolInterleaved(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me search"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" for you."}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"web_search"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"url\":\"https://example.com\"}"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "search"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var text strings.Builder
	var tools []llm.ToolCall
	for chunk := range ch {
		text.WriteString(chunk.DeltaText)
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	if text.String() != "Let me search for you." {
		t.Errorf("text = %q, want %q", text.String(), "Let me search for you.")
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	if tools[0].Name != "web_search" {
		t.Errorf("tool name = %q, want %q", tools[0].Name, "web_search")
	}
	if string(tools[0].Arguments) != `{"url":"https://example.com"}` {
		t.Errorf("tool args = %q", string(tools[0].Arguments))
	}
}

func TestAnthropicStream_MultipleTools(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/tmp/a.txt\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Reading files..."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_2","name":"write_file"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"/tmp/b"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":".txt\",\"content\":\"hello\"}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "do stuff"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var text strings.Builder
	var tools []llm.ToolCall
	for chunk := range ch {
		text.WriteString(chunk.DeltaText)
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	if text.String() != "Reading files..." {
		t.Errorf("text = %q, want %q", text.String(), "Reading files...")
	}
	if len(tools) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(tools))
	}
	if tools[0].ID != "toolu_1" || tools[0].Name != "read_file" {
		t.Errorf("tool[0] = %+v", tools[0])
	}
	if string(tools[0].Arguments) != `{"path":"/tmp/a.txt"}` {
		t.Errorf("tool[0].args = %q", string(tools[0].Arguments))
	}
	if tools[1].ID != "toolu_2" || tools[1].Name != "write_file" {
		t.Errorf("tool[1] = %+v", tools[1])
	}
	if string(tools[1].Arguments) != `{"path":"/tmp/b.txt","content":"hello"}` {
		t.Errorf("tool[1].args = %q", string(tools[1].Arguments))
	}
}

func TestAnthropicStream_TextOnly_NoRegression(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_4","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "say hi"}},
		MaxTokens: 32,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var text strings.Builder
	var tools []llm.ToolCall
	var stop llm.StopReason
	for chunk := range ch {
		text.WriteString(chunk.DeltaText)
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
		if chunk.Stop != "" {
			stop = chunk.Stop
		}
	}
	if text.String() != "Hello world!" {
		t.Errorf("text = %q, want %q", text.String(), "Hello world!")
	}
	if len(tools) != 0 {
		t.Errorf("expected no tool calls, got %d", len(tools))
	}
	if stop != llm.StopEndTurn {
		t.Errorf("stop = %q, want %q", stop, llm.StopEndTurn)
	}
}

func TestAnthropicStream_MalformedJSON(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_5","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_bad","name":"broken"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"key\": "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"INVALID"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "break"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var tools []llm.ToolCall
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	// Should still emit the tool call (with invalid JSON as raw bytes) rather than panic.
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].Name != "broken" {
		t.Errorf("tool name = %q, want %q", tools[0].Name, "broken")
	}
	if tools[0].ID != "toolu_bad" {
		t.Errorf("tool ID = %q, want %q", tools[0].ID, "toolu_bad")
	}
	// The raw malformed JSON is preserved as-is for downstream error handling.
	if string(tools[0].Arguments) != `{"key": INVALID` {
		t.Errorf("tool args = %q, want %q", string(tools[0].Arguments), `{"key": INVALID`)
	}
}

func TestAnthropicStream_EmptyToolArguments(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_6","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_empty","name":"no_args"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "run"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var tools []llm.ToolCall
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].Name != "no_args" {
		t.Errorf("tool name = %q", tools[0].Name)
	}
	// Empty arguments should default to empty JSON object.
	if string(tools[0].Arguments) != "{}" {
		t.Errorf("tool args = %q, want %q", string(tools[0].Arguments), "{}")
	}
}

func TestAnthropicStream_LargeToolArguments(t *testing.T) {
	t.Parallel()
	// Simulate a large argument split across many delta events.
	largeValue := strings.Repeat("x", 10000)
	fullJSON := fmt.Sprintf(`{"data":"%s"}`, largeValue)

	// Split JSON into 100-byte chunks to simulate many deltas.
	var events []string
	events = append(events, `{"type":"message_start","message":{"id":"msg_7","type":"message","role":"assistant"}}`)
	events = append(events, `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_big","name":"large_tool"}}`)
	for i := 0; i < len(fullJSON); i += 100 {
		end := i + 100
		if end > len(fullJSON) {
			end = len(fullJSON)
		}
		chunk := fullJSON[i:end]
		// Escape the chunk for embedding in JSON.
		escaped := strings.ReplaceAll(chunk, `\`, `\\`)
		escaped = strings.ReplaceAll(escaped, `"`, `\"`)
		events = append(events, fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"%s"}}`, escaped))
	}
	events = append(events, `{"type":"content_block_stop","index":0}`)
	events = append(events, `{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`)
	events = append(events, `{"type":"message_stop"}`)

	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "big request"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var tools []llm.ToolCall
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if tools[0].Name != "large_tool" {
		t.Errorf("tool name = %q", tools[0].Name)
	}
	if string(tools[0].Arguments) != fullJSON {
		t.Errorf("tool args length = %d, want %d", len(tools[0].Arguments), len(fullJSON))
	}
}

func TestAnthropicStream_ToolWithEmptyObject(t *testing.T) {
	t.Parallel()
	events := []string{
		`{"type":"message_start","message":{"id":"msg_8","type":"message","role":"assistant"}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_obj","name":"ping"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"}}`,
		`{"type":"message_stop"}`,
	}
	srv := anthropicStreamServer(t, events)
	defer srv.Close()

	p := llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "ping"}},
		MaxTokens: 1024,
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}

	var tools []llm.ToolCall
	for chunk := range ch {
		if chunk.ToolCall != nil {
			tools = append(tools, *chunk.ToolCall)
		}
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tools))
	}
	if string(tools[0].Arguments) != "{}" {
		t.Errorf("tool args = %q, want %q", string(tools[0].Arguments), "{}")
	}
}

// anthropicStreamServer creates an httptest server that serves SSE events.
func anthropicStreamServer(t *testing.T, events []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("flusher unavailable")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, ev := range events {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", ev)
			flusher.Flush()
		}
	}))
}
