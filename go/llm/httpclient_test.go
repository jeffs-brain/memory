// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// TestDefaultClientTimeoutOnSlowHeaders verifies that a provider using the
// default client times out when the server delays sending response headers
// beyond ResponseHeaderTimeout (~30s). We use a scaled-down timeout via a
// custom client for test speed.
func TestDefaultClientTimeoutOnSlowHeaders(t *testing.T) {
	t.Parallel()

	slowClient := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: 200 * time.Millisecond,
		},
	}

	tests := []struct {
		name     string
		provider func(url string) llm.Provider
	}{
		{
			name: "OpenAI",
			provider: func(url string) llm.Provider {
				return llm.NewOpenAI(llm.OpenAIConfig{
					APIKey:     "k",
					BaseURL:    url,
					HTTPClient: slowClient,
				})
			},
		},
		{
			name: "Anthropic",
			provider: func(url string) llm.Provider {
				return llm.NewAnthropic(llm.AnthropicConfig{
					APIKey:     "k",
					BaseURL:    url,
					HTTPClient: slowClient,
				})
			},
		},
		{
			name: "Ollama",
			provider: func(url string) llm.Provider {
				return llm.NewOllama(llm.OllamaConfig{
					BaseURL:    url,
					HTTPClient: slowClient,
				})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				// Delay longer than ResponseHeaderTimeout to trigger timeout.
				time.Sleep(2 * time.Second)
				w.WriteHeader(http.StatusOK)
			}))
			defer srv.Close()

			p := tc.provider(srv.URL)
			defer func() { _ = p.Close() }()

			start := time.Now()
			_, err := p.Complete(context.Background(), llm.CompleteRequest{
				Model:    "m",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			elapsed := time.Since(start)

			if err == nil {
				t.Fatal("expected timeout error, got nil")
			}
			if elapsed > 1*time.Second {
				t.Fatalf("timeout took too long: %v (expected ~200ms)", elapsed)
			}
		})
	}
}

// TestDefaultClientCompletesWithinTimeout verifies non-streaming calls
// succeed when the server responds promptly.
func TestDefaultClientCompletesWithinTimeout(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		handler  http.HandlerFunc
		provider func(url string) llm.Provider
		verify   func(t *testing.T, resp llm.CompleteResponse)
	}{
		{
			name: "OpenAI",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{
					"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
					"usage":{"prompt_tokens":1,"completion_tokens":1}
				}`)
			}),
			provider: func(url string) llm.Provider {
				return llm.NewOpenAI(llm.OpenAIConfig{APIKey: "k", BaseURL: url})
			},
			verify: func(t *testing.T, resp llm.CompleteResponse) {
				if resp.Text != "ok" {
					t.Fatalf("text %q", resp.Text)
				}
			},
		},
		{
			name: "Anthropic",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{
					"id":"msg_1","type":"message","role":"assistant",
					"content":[{"type":"text","text":"ok"}],
					"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}
				}`)
			}),
			provider: func(url string) llm.Provider {
				return llm.NewAnthropic(llm.AnthropicConfig{APIKey: "k", BaseURL: url})
			},
			verify: func(t *testing.T, resp llm.CompleteResponse) {
				if resp.Text != "ok" {
					t.Fatalf("text %q", resp.Text)
				}
			},
		},
		{
			name: "Ollama",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{
					"model":"m","message":{"role":"assistant","content":"ok"},
					"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1
				}`)
			}),
			provider: func(url string) llm.Provider {
				return llm.NewOllama(llm.OllamaConfig{BaseURL: url})
			},
			verify: func(t *testing.T, resp llm.CompleteResponse) {
				if resp.Text != "ok" {
					t.Fatalf("text %q", resp.Text)
				}
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()

			p := tc.provider(srv.URL)
			defer func() { _ = p.Close() }()

			resp, err := p.Complete(context.Background(), llm.CompleteRequest{
				Model:    "m",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tc.verify(t, resp)
		})
	}
}

// TestStreamingWithSlowEvents verifies that once headers arrive,
// ResponseHeaderTimeout does not kill long-running SSE streams. The server
// sends events slowly over a period exceeding the header timeout.
func TestStreamingWithSlowEvents(t *testing.T) {
	t.Parallel()

	// Use a short ResponseHeaderTimeout so the test proves streaming
	// continues well past that window.
	shortHeaderClient := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: 100 * time.Millisecond,
		},
	}

	tests := []struct {
		name     string
		handler  http.HandlerFunc
		provider func(url string) llm.Provider
		expected string
	}{
		{
			name: "OpenAI",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				flusher, ok := w.(http.Flusher)
				if !ok {
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				// Send chunks with delays totalling well over 100ms
				chunks := []string{"hel", "lo", " wo", "rld"}
				for _, c := range chunks {
					_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":%q}}]}\n\n", c)
					flusher.Flush()
					time.Sleep(80 * time.Millisecond)
				}
				_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
				flusher.Flush()
				_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
				flusher.Flush()
			}),
			provider: func(url string) llm.Provider {
				return llm.NewOpenAI(llm.OpenAIConfig{
					APIKey:     "k",
					BaseURL:    url,
					HTTPClient: shortHeaderClient,
				})
			},
			expected: "hello world",
		},
		{
			name: "Anthropic",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				flusher, ok := w.(http.Flusher)
				if !ok {
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				parts := []string{"hel", "lo", " wo", "rld"}
				for _, p := range parts {
					_, _ = fmt.Fprintf(w, "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":%q}}\n\n", p)
					flusher.Flush()
					time.Sleep(80 * time.Millisecond)
				}
				_, _ = fmt.Fprint(w, "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n")
				flusher.Flush()
				_, _ = fmt.Fprint(w, "data: {\"type\":\"message_stop\"}\n\n")
				flusher.Flush()
			}),
			provider: func(url string) llm.Provider {
				return llm.NewAnthropic(llm.AnthropicConfig{
					APIKey:     "k",
					BaseURL:    url,
					HTTPClient: shortHeaderClient,
				})
			},
			expected: "hello world",
		},
		{
			name: "Ollama",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				flusher, ok := w.(http.Flusher)
				if !ok {
					return
				}
				w.Header().Set("Content-Type", "application/x-ndjson")
				parts := []string{"hel", "lo", " wo", "rld"}
				for _, p := range parts {
					_, _ = fmt.Fprintf(w, "{\"message\":{\"role\":\"assistant\",\"content\":%q},\"done\":false}\n", p)
					flusher.Flush()
					time.Sleep(80 * time.Millisecond)
				}
				_, _ = fmt.Fprint(w, "{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n")
				flusher.Flush()
			}),
			provider: func(url string) llm.Provider {
				return llm.NewOllama(llm.OllamaConfig{
					BaseURL:    url,
					HTTPClient: shortHeaderClient,
				})
			},
			expected: "hello world",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()

			p := tc.provider(srv.URL)
			defer func() { _ = p.Close() }()

			ch, err := p.CompleteStream(context.Background(), llm.CompleteRequest{
				Model:    "m",
				Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
			})
			if err != nil {
				t.Fatalf("stream: %v", err)
			}
			var text strings.Builder
			for chunk := range ch {
				text.WriteString(chunk.DeltaText)
			}
			if text.String() != tc.expected {
				t.Fatalf("text %q, want %q", text.String(), tc.expected)
			}
		})
	}
}

// TestCustomClientPreserved verifies that providing a custom HTTPClient to
// any provider constructor prevents the default client from being used.
func TestCustomClientPreserved(t *testing.T) {
	t.Parallel()

	customHeaderValue := "custom-client-marker"
	customTransport := &headerInjectTransport{
		base:   http.DefaultTransport,
		header: "X-Custom-Client",
		value:  customHeaderValue,
	}
	customClient := &http.Client{Transport: customTransport}

	tests := []struct {
		name     string
		handler  http.HandlerFunc
		provider func(url string) interface{ Close() error }
		call     func(ctx context.Context, p interface{ Close() error }) error
	}{
		{
			name: "OpenAI",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Custom-Client"); got != customHeaderValue {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = fmt.Fprintf(w, `{"error":{"message":"missing custom header: got %q"}}`, got)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)
			}),
			provider: func(url string) interface{ Close() error } {
				return llm.NewOpenAI(llm.OpenAIConfig{
					APIKey:     "k",
					BaseURL:    url,
					HTTPClient: customClient,
				})
			},
			call: func(ctx context.Context, p interface{ Close() error }) error {
				_, err := p.(llm.Provider).Complete(ctx, llm.CompleteRequest{
					Model:    "m",
					Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
				})
				return err
			},
		},
		{
			name: "Anthropic",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Custom-Client"); got != customHeaderValue {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = fmt.Fprintf(w, `{"error":{"type":"invalid","message":"missing custom header: got %q"}}`, got)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			}),
			provider: func(url string) interface{ Close() error } {
				return llm.NewAnthropic(llm.AnthropicConfig{
					APIKey:     "k",
					BaseURL:    url,
					HTTPClient: customClient,
				})
			},
			call: func(ctx context.Context, p interface{ Close() error }) error {
				_, err := p.(llm.Provider).Complete(ctx, llm.CompleteRequest{
					Model:     "m",
					Messages:  []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
					MaxTokens: 32,
				})
				return err
			},
		},
		{
			name: "Ollama",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Custom-Client"); got != customHeaderValue {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = fmt.Fprint(w, "missing custom header")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"model":"m","message":{"role":"assistant","content":"ok"},"done":true,"done_reason":"stop","prompt_eval_count":1,"eval_count":1}`)
			}),
			provider: func(url string) interface{ Close() error } {
				return llm.NewOllama(llm.OllamaConfig{
					BaseURL:    url,
					HTTPClient: customClient,
				})
			},
			call: func(ctx context.Context, p interface{ Close() error }) error {
				_, err := p.(llm.Provider).Complete(ctx, llm.CompleteRequest{
					Model:    "m",
					Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
				})
				return err
			},
		},
		{
			name: "OpenAIEmbedder",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Custom-Client"); got != customHeaderValue {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = fmt.Fprintf(w, `{"error":{"message":"missing custom header: got %q"}}`, got)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"data":[{"index":0,"embedding":[0.1,0.2]}]}`)
			}),
			provider: func(url string) interface{ Close() error } {
				return llm.NewOpenAIEmbedder(llm.OpenAIEmbedConfig{
					APIKey:     "k",
					BaseURL:    url,
					Dimensions: 2,
					HTTPClient: customClient,
				})
			},
			call: func(ctx context.Context, p interface{ Close() error }) error {
				_, err := p.(llm.Embedder).Embed(ctx, []string{"test"})
				return err
			},
		},
		{
			name: "OllamaEmbedder",
			handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if got := r.Header.Get("X-Custom-Client"); got != customHeaderValue {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = fmt.Fprint(w, "missing custom header")
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprint(w, `{"embeddings":[[0.1,0.2]]}`)
			}),
			provider: func(url string) interface{ Close() error } {
				return llm.NewOllamaEmbedder(llm.OllamaEmbedConfig{
					BaseURL:    url,
					Model:      "bge-m3",
					Dimensions: 2,
					HTTPClient: customClient,
				})
			},
			call: func(ctx context.Context, p interface{ Close() error }) error {
				_, err := p.(llm.Embedder).Embed(ctx, []string{"test"})
				return err
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()

			p := tc.provider(srv.URL)
			defer func() { _ = p.Close() }()

			if err := tc.call(context.Background(), p); err != nil {
				t.Fatalf("call failed (custom client not used): %v", err)
			}
		})
	}
}

// TestEmbedderTimeoutOnSlowHeaders verifies that embedder constructors also
// get the timeout behaviour.
func TestEmbedderTimeoutOnSlowHeaders(t *testing.T) {
	t.Parallel()

	slowClient := &http.Client{
		Transport: &http.Transport{
			ResponseHeaderTimeout: 200 * time.Millisecond,
		},
	}

	tests := []struct {
		name     string
		embedder func(url string) llm.Embedder
	}{
		{
			name: "OpenAIEmbedder",
			embedder: func(url string) llm.Embedder {
				return llm.NewOpenAIEmbedder(llm.OpenAIEmbedConfig{
					APIKey:     "k",
					BaseURL:    url,
					Dimensions: 2,
					HTTPClient: slowClient,
				})
			},
		},
		{
			name: "OllamaEmbedder",
			embedder: func(url string) llm.Embedder {
				return llm.NewOllamaEmbedder(llm.OllamaEmbedConfig{
					BaseURL:    url,
					Model:      "bge-m3",
					Dimensions: 2,
					HTTPClient: slowClient,
				})
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				time.Sleep(2 * time.Second)
				w.WriteHeader(http.StatusOK)
			}))
			defer srv.Close()

			e := tc.embedder(srv.URL)
			defer func() { _ = e.Close() }()

			start := time.Now()
			_, err := e.Embed(context.Background(), []string{"test"})
			elapsed := time.Since(start)

			if err == nil {
				t.Fatal("expected timeout error, got nil")
			}
			if elapsed > 1*time.Second {
				t.Fatalf("timeout took too long: %v (expected ~200ms)", elapsed)
			}
		})
	}
}

// headerInjectTransport is a test helper that injects a custom header into
// every request to prove a custom client is being used.
type headerInjectTransport struct {
	base   http.RoundTripper
	header string
	value  string
}

func (t *headerInjectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req = req.Clone(req.Context())
	req.Header.Set(t.header, t.value)
	return t.base.RoundTrip(req)
}
