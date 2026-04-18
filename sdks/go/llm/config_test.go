// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
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
