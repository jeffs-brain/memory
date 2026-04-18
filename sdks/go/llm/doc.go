// SPDX-License-Identifier: Apache-2.0

// Package llm defines the minimal cross-cutting LLM abstraction used by
// query, memory, retrieval, knowledge and eval/lme.
//
// The package is intentionally small. It exposes two core interfaces,
// Provider for chat completion and Embedder for embedding models, plus a
// handful of concrete backends: OpenAI, Anthropic, Ollama and an in-memory
// fake for deterministic tests.
//
// Configuration is driven through environment variables via
// [ProviderFromEnv] so downstream packages never need to care which backend
// is in use.
package llm
