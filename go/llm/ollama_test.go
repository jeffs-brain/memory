// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestOllamaComplete(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Errorf("path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{
			"model":"llama3","message":{"role":"assistant","content":"howdy"},
			"done":true,"done_reason":"stop","prompt_eval_count":4,"eval_count":2
		}`)
	}))
	defer srv.Close()

	p := llm.NewOllama(llm.OllamaConfig{BaseURL: srv.URL})
	defer func() { _ = p.Close() }()
	resp, err := p.Complete(context.Background(), llm.CompleteRequest{
		Model:    "llama3",
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.Text != "howdy" {
		t.Fatalf("text %q", resp.Text)
	}
	if resp.Stop != llm.StopEndTurn {
		t.Fatalf("stop %q", resp.Stop)
	}
	if resp.TokensIn != 4 || resp.TokensOut != 2 {
		t.Fatalf("tokens wrong: %+v", resp)
	}
}

func TestOllamaStreaming(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("flusher unavailable")
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		lines := []string{
			`{"message":{"role":"assistant","content":"he"},"done":false}`,
			`{"message":{"role":"assistant","content":"llo"},"done":false}`,
			`{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}`,
		}
		for _, line := range lines {
			_, _ = fmt.Fprintln(w, line)
			flusher.Flush()
		}
	}))
	defer srv.Close()

	p := llm.NewOllama(llm.OllamaConfig{BaseURL: srv.URL})
	ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
		Model:    "llama3",
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

func TestOllamaEmbedder(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprint(w, `{"embeddings":[[0.1,0.2],[0.3,0.4]]}`)
	}))
	defer srv.Close()

	e := llm.NewOllamaEmbedder(llm.OllamaEmbedConfig{BaseURL: srv.URL, Model: "bge-m3", Dimensions: 2})
	defer func() { _ = e.Close() }()
	vecs, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(vecs) != 2 || vecs[1][1] != 0.4 {
		t.Fatalf("vectors: %+v", vecs)
	}
	if e.Dimensions() != 2 {
		t.Fatalf("dims %d", e.Dimensions())
	}
}

func TestOllamaErrorStatus(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = fmt.Fprint(w, "boom")
	}))
	defer srv.Close()

	p := llm.NewOllama(llm.OllamaConfig{BaseURL: srv.URL})
	_, err := p.Complete(context.Background(), llm.CompleteRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err == nil || !strings.Contains(err.Error(), "500") {
		t.Fatalf("expected 500 error, got %v", err)
	}
}
