// SPDX-License-Identifier: Apache-2.0

package llm

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

// Environment variable names read by [ProviderFromEnv] and
// [EmbedderFromEnv].
const (
	EnvProvider       = "JB_LLM_PROVIDER"
	EnvModel          = "JB_LLM_MODEL"
	EnvOpenAIAPIKey   = "OPENAI_API_KEY"
	EnvOpenAIBaseURL  = "OPENAI_BASE_URL"
	EnvAnthropicKey   = "ANTHROPIC_API_KEY"
	EnvAnthropicBase  = "ANTHROPIC_BASE_URL"
	EnvOllamaHost     = "OLLAMA_HOST"
	EnvEmbedProvider  = "JB_EMBED_PROVIDER"
	EnvEmbedModel     = "JB_EMBED_MODEL"
	EnvEmbedDimension = "JB_EMBED_DIMENSIONS"
)

// Getenv is a lookup function matching [os.Getenv]. It is the sole input
// to [ProviderFromEnv] so tests can inject a mock environment.
type Getenv func(string) string

// OSGetenv is a [Getenv] backed by the real process environment.
func OSGetenv(key string) string { return os.Getenv(key) }

// ProviderFromEnv resolves a [Provider] from environment variables.
//
// The function inspects JB_LLM_PROVIDER first. If unset, it probes a local
// Ollama instance (OLLAMA_HOST or http://localhost:11434) and returns that
// provider when reachable. As a final fallback it returns a deterministic
// fake so downstream code is never left without a provider during tests.
func ProviderFromEnv(env Getenv) (Provider, error) {
	if env == nil {
		env = OSGetenv
	}
	switch strings.ToLower(strings.TrimSpace(env(EnvProvider))) {
	case "openai":
		key := env(EnvOpenAIAPIKey)
		if key == "" {
			return nil, fmt.Errorf("llm: %s is set to openai but %s is empty", EnvProvider, EnvOpenAIAPIKey)
		}
		return NewOpenAI(OpenAIConfig{APIKey: key, BaseURL: env(EnvOpenAIBaseURL), Model: env(EnvModel)}), nil
	case "anthropic":
		key := env(EnvAnthropicKey)
		if key == "" {
			return nil, fmt.Errorf("llm: %s is set to anthropic but %s is empty", EnvProvider, EnvAnthropicKey)
		}
		return NewAnthropic(AnthropicConfig{APIKey: key, BaseURL: env(EnvAnthropicBase), Model: env(EnvModel)}), nil
	case "ollama":
		return NewOllama(OllamaConfig{BaseURL: ollamaHostFromEnv(env), Model: env(EnvModel)}), nil
	case "fake":
		return NewFake([]string{"ok"}), nil
	case "":
		// Auto-detect path below.
	default:
		return nil, fmt.Errorf("llm: unknown %s=%q", EnvProvider, env(EnvProvider))
	}
	// Auto-detect. When a cloud key is present prefer the cloud provider
	// over a locally reachable Ollama — a user who set OPENAI_API_KEY
	// almost always wants that provider, not whatever happens to be
	// listening on 127.0.0.1:11434.
	if key := strings.TrimSpace(env(EnvOpenAIAPIKey)); key != "" {
		return NewOpenAI(OpenAIConfig{APIKey: key, BaseURL: env(EnvOpenAIBaseURL), Model: env(EnvModel)}), nil
	}
	if key := strings.TrimSpace(env(EnvAnthropicKey)); key != "" {
		return NewAnthropic(AnthropicConfig{APIKey: key, BaseURL: env(EnvAnthropicBase), Model: env(EnvModel)}), nil
	}
	host := ollamaHostFromEnv(env)
	if ollamaReachable(host) {
		return NewOllama(OllamaConfig{BaseURL: host}), nil
	}
	return NewFake([]string{"ok"}), nil
}

// EmbedderFromEnv resolves an [Embedder] from environment variables. If
// JB_EMBED_PROVIDER is blank, it follows the same auto-detect path as
// [ProviderFromEnv].
func EmbedderFromEnv(env Getenv) (Embedder, error) {
	if env == nil {
		env = OSGetenv
	}
	provider := strings.ToLower(strings.TrimSpace(env(EnvEmbedProvider)))
	if provider == "" {
		provider = strings.ToLower(strings.TrimSpace(env(EnvProvider)))
	}
	switch provider {
	case "openai":
		key := env(EnvOpenAIAPIKey)
		if key == "" {
			return nil, fmt.Errorf("llm: %s is set to openai but %s is empty", EnvEmbedProvider, EnvOpenAIAPIKey)
		}
		return NewOpenAIEmbedder(OpenAIEmbedConfig{
			APIKey: key,
			Model:  env(EnvEmbedModel),
		}), nil
	case "ollama":
		return NewOllamaEmbedder(OllamaEmbedConfig{
			BaseURL: ollamaHostFromEnv(env),
			Model:   env(EnvEmbedModel),
		}), nil
	case "fake":
		return NewFakeEmbedder(ollamaDefaultEmbedDims), nil
	case "":
		// Same precedence as ProviderFromEnv: a user who set an OpenAI
		// API key wants embeddings from OpenAI, not whatever happens to
		// be listening on 127.0.0.1:11434.
		if key := strings.TrimSpace(env(EnvOpenAIAPIKey)); key != "" {
			return NewOpenAIEmbedder(OpenAIEmbedConfig{
				APIKey: key,
				Model:  env(EnvEmbedModel),
			}), nil
		}
		host := ollamaHostFromEnv(env)
		if ollamaReachable(host) {
			return NewOllamaEmbedder(OllamaEmbedConfig{BaseURL: host}), nil
		}
		return NewFakeEmbedder(ollamaDefaultEmbedDims), nil
	case "anthropic":
		return nil, fmt.Errorf("llm: anthropic does not expose an embedding endpoint")
	default:
		return nil, fmt.Errorf("llm: unknown %s=%q", EnvEmbedProvider, env(EnvEmbedProvider))
	}
}

func ollamaHostFromEnv(env Getenv) string {
	host := strings.TrimSpace(env(EnvOllamaHost))
	if host == "" {
		return ollamaDefaultBase
	}
	if !strings.HasPrefix(host, "http://") && !strings.HasPrefix(host, "https://") {
		host = "http://" + host
	}
	// OLLAMA_HOST frequently omits the port (e.g. "0.0.0.0" when the
	// daemon is configured to bind all interfaces). Fall back to the
	// default 11434 when no port is present.
	if !strings.Contains(strings.TrimPrefix(strings.TrimPrefix(host, "http://"), "https://"), ":") {
		host = host + ":11434"
	}
	return host
}

// ollamaReachable issues a lightweight HEAD against base to decide whether
// an Ollama daemon is sat on the other end.
func ollamaReachable(base string) bool {
	if base == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, base+"/", nil)
	if err != nil {
		return false
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode < 500
}
