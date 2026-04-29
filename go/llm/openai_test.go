// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestOpenAIComplete(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth header missing: %q", got)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), "gpt-test") {
			t.Errorf("model missing: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"choices": [{"index":0,"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
		}`)
	}))
	defer srv.Close()

	p := llm.NewOpenAI(llm.OpenAIConfig{APIKey: "test-key", BaseURL: srv.URL})
	defer func() { _ = p.Close() }()
	resp, err := p.Complete(context.Background(), llm.CompleteRequest{
		Model:    "gpt-test",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.Text != "hi there" {
		t.Fatalf("text %q", resp.Text)
	}
	if resp.Stop != llm.StopEndTurn {
		t.Fatalf("stop %q", resp.Stop)
	}
	if resp.TokensIn != 5 || resp.TokensOut != 2 {
		t.Fatalf("token counts wrong: %+v", resp)
	}
}

func TestOpenAIStreaming(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("flusher unavailable")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		chunks := []string{
			`{"choices":[{"index":0,"delta":{"content":"hel"}}]}`,
			`{"choices":[{"index":0,"delta":{"content":"lo"}}]}`,
			`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", c)
			flusher.Flush()
		}
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p := llm.NewOpenAI(llm.OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
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

func TestOpenAIToolCall(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"choices": [{"index":0,"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},"finish_reason":"tool_calls"}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1}
		}`)
	}))
	defer srv.Close()

	p := llm.NewOpenAI(llm.OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	resp, err := p.Complete(context.Background(), llm.CompleteRequest{
		Model:    "m",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "call it"}},
		Tools: []llm.ToolDef{{
			Name:        "lookup",
			Description: "test",
			Schema:      map[string]any{"type": "object"},
		}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.Stop != llm.StopToolUse {
		t.Fatalf("stop %q", resp.Stop)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool calls: %+v", resp.ToolCalls)
	}
	tc := resp.ToolCalls[0]
	if tc.Name != "lookup" || tc.ID != "call_1" {
		t.Fatalf("tool call fields: %+v", tc)
	}
	var args map[string]string
	if err := json.Unmarshal(tc.Arguments, &args); err != nil || args["q"] != "x" {
		t.Fatalf("args: %v %v", args, err)
	}
}

func TestOpenAICompleteJSONResponseFormat(t *testing.T) {
	t.Parallel()

	var captured struct {
		ResponseFormat struct {
			Type string `json:"type"`
		} `json:"response_format"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"choices": [{"index":0,"message":{"role":"assistant","content":"{\"ok\":true}"},"finish_reason":"stop"}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1}
		}`)
	}))
	defer srv.Close()

	p := llm.NewOpenAI(llm.OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), llm.CompleteRequest{
		Model:              "gpt-test",
		Messages:           []llm.Message{{Role: llm.RoleUser, Content: "Return JSON."}},
		ResponseFormatJSON: true,
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if captured.ResponseFormat.Type != "json_object" {
		t.Fatalf("response_format.type = %q, want json_object", captured.ResponseFormat.Type)
	}
}

func TestOpenAIErrorParsing(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = fmt.Fprint(w, `{"error":{"message":"bad key","type":"invalid_request_error","code":"invalid_api_key"}}`)
	}))
	defer srv.Close()

	p := llm.NewOpenAI(llm.OpenAIConfig{APIKey: "k", BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), llm.CompleteRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "bad key") {
		t.Fatalf("missing error message: %v", err)
	}
	if !strings.Contains(err.Error(), "401") {
		t.Fatalf("missing status: %v", err)
	}
}

func TestOpenAIEmbedder(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"data":[{"index":0,"embedding":[0.1,0.2,0.3]},{"index":1,"embedding":[0.4,0.5,0.6]}]}`)
	}))
	defer srv.Close()

	e := llm.NewOpenAIEmbedder(llm.OpenAIEmbedConfig{APIKey: "k", BaseURL: srv.URL, Dimensions: 3})
	defer func() { _ = e.Close() }()
	vecs, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 3 || vecs[1][2] != 0.6 {
		t.Fatalf("vectors: %+v", vecs)
	}
	if e.Dimensions() != 3 {
		t.Fatalf("dims %d", e.Dimensions())
	}
}
