// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func envMap(m map[string]string) llm.Getenv {
	return func(k string) string { return m[k] }
}

func TestProviderFromEnvFake(t *testing.T) {
	t.Parallel()
	p, err := llm.ProviderFromEnv(envMap(map[string]string{
		llm.EnvProvider: "fake",
	}))
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	defer func() { _ = p.Close() }()
	resp, err := p.Complete(context.Background(), llm.CompleteRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.Text == "" {
		t.Fatalf("expected fake text")
	}
}

func TestProviderFromEnvOpenAIMissingKey(t *testing.T) {
	t.Parallel()
	_, err := llm.ProviderFromEnv(envMap(map[string]string{
		llm.EnvProvider: "openai",
	}))
	if err == nil {
		t.Fatal("expected error")
	}
}

// TestProviderFromEnvAnthropicBaseURL ensures the proxy override lifts
// out of the env and reaches the Anthropic client configuration. We
// exercise the round-trip by pointing the proxy at an httptest server
// and checking it was hit.
func TestProviderFromEnvAnthropicBaseURL(t *testing.T) {
	t.Parallel()
	hit := make(chan string, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case hit <- r.URL.Path:
		default:
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
	}))
	defer srv.Close()

	p, err := llm.ProviderFromEnv(envMap(map[string]string{
		llm.EnvProvider:      "anthropic",
		llm.EnvAnthropicKey:  "k",
		llm.EnvAnthropicBase: srv.URL,
		llm.EnvModel:         "claude-test",
	}))
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	defer func() { _ = p.Close() }()
	resp, err := p.Complete(context.Background(), llm.CompleteRequest{
		Model:     "claude-test",
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
		MaxTokens: 4,
	})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}
	if resp.Text != "ok" {
		t.Fatalf("text %q", resp.Text)
	}
	select {
	case path := <-hit:
		if path != "/v1/messages" {
			t.Fatalf("path %q", path)
		}
	default:
		t.Fatal("proxy was not hit; ANTHROPIC_BASE_URL override not applied")
	}
}

func TestProviderFromEnvUnknown(t *testing.T) {
	t.Parallel()
	_, err := llm.ProviderFromEnv(envMap(map[string]string{
		llm.EnvProvider: "mystery",
	}))
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestProviderFromEnvAutoDetectFallback(t *testing.T) {
	t.Parallel()
	// Point Ollama at a host guaranteed to be unreachable so we drop
	// through to the fake backend.
	p, err := llm.ProviderFromEnv(envMap(map[string]string{
		llm.EnvOllamaHost: "http://127.0.0.1:1",
	}))
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	defer func() { _ = p.Close() }()
	if _, err := p.Complete(context.Background(), llm.CompleteRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "x"}},
	}); err != nil {
		t.Fatalf("complete: %v", err)
	}
}

func TestEmbedderFromEnvFake(t *testing.T) {
	t.Parallel()
	e, err := llm.EmbedderFromEnv(envMap(map[string]string{
		llm.EnvEmbedProvider: "fake",
	}))
	if err != nil {
		t.Fatalf("embedder: %v", err)
	}
	defer func() { _ = e.Close() }()
	vecs, err := e.Embed(context.Background(), []string{"hi"})
	if err != nil || len(vecs) != 1 {
		t.Fatalf("embed: %v %v", vecs, err)
	}
}

func TestEmbedderFromEnvAnthropicUnsupported(t *testing.T) {
	t.Parallel()
	_, err := llm.EmbedderFromEnv(envMap(map[string]string{
		llm.EnvEmbedProvider: "anthropic",
	}))
	if err == nil {
		t.Fatal("expected error")
	}
}
