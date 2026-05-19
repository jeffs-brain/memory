// SPDX-License-Identifier: Apache-2.0

// Package ratelimit provides a shared, per-tenant adaptive rate limiter
// using a token bucket algorithm. It is consumed by both the ingestion
// pipeline and connector layers to honour provider-imposed rate limits
// (OpenAI, Anthropic, Ollama, etc.).
//
// The limiter reads X-RateLimit-Remaining, X-RateLimit-Limit,
// X-RateLimit-Reset, and Retry-After response headers to adaptively
// adjust throughput. When no headers are present it falls back to the
// configured static rate.
//
// Concurrency safety is guaranteed: the in-memory token bucket uses
// golang.org/x/time/rate internally, which is goroutine-safe.
package ratelimit
