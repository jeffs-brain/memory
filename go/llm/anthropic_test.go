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
