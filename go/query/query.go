// SPDX-License-Identifier: Apache-2.0

// Package query rewrites raw user input into structured search queries that
// downstream retrieval can consume. The public surface is [Distiller] and
// its default implementation [DefaultDistiller], plus the [ExpandTemporal]
// helper which resolves relative date phrases against a caller-supplied
// anchor.
package query

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/jeffs-brain/memory/go/llm"
)

// Query is a structured search query produced by distillation.
type Query struct {
	Text        string   `json:"text"`
	Domain      string   `json:"domain,omitempty"`
	Entities    []string `json:"entities,omitempty"`
	RecencyBias string   `json:"recency_bias,omitempty"`
	Confidence  float64  `json:"confidence,omitempty"`
}

// Options controls the distillation behaviour.
type Options struct {
	MinTokenThreshold   int
	MinSignificantTerms int
	MaxQueries          int
	DisableCache        bool
	Scope               string
	LocalModel          string
	LocalBaseURL        string
	CloudProvider       llm.Provider
}

// DefaultOptions returns sensible defaults.
func DefaultOptions() Options {
	return Options{
		MinTokenThreshold:   20,
		MinSignificantTerms: 3,
		MaxQueries:          3,
		Scope:               "search",
	}
}

// Trace records what happened during distillation for debugging and eval.
type Trace struct {
	Skipped         bool    `json:"skipped"`
	SkipReason      string  `json:"skip_reason,omitempty"`
	CacheHit        bool    `json:"cache_hit"`
	Provider        string  `json:"provider,omitempty"`
	ModelName       string  `json:"model_name,omitempty"`
	LatencyMillis   int64   `json:"latency_ms"`
	RawTokens       int     `json:"raw_tokens"`
	MechanicalTerms int     `json:"mechanical_terms"`
	Queries         []Query `json:"queries,omitempty"`
	ErrorDetail     string  `json:"error_detail,omitempty"`
	FellBackToRaw   bool    `json:"fell_back_to_raw"`
}

// Result wraps the distilled queries and the trace.
type Result struct {
	Queries []Query
	Trace   Trace
}

// Distiller rewrites raw user input into structured search queries.
type Distiller interface {
	Distill(ctx context.Context, raw string, history []llm.Message, opts Options) (Result, error)
}

// ErrNoProvider is returned by [Distiller] implementations that require an
// LLM provider when none has been configured.
var ErrNoProvider = errors.New("query: no provider configured")

// DefaultDistiller is the production distiller backed by an LLM.
type DefaultDistiller struct {
	cache *cache
}

// NewDistiller creates a DefaultDistiller with a built-in LRU cache.
func NewDistiller() *DefaultDistiller {
	return &DefaultDistiller{
		cache: newCache(512),
	}
}

// Distill rewrites raw into structured queries. Gate logic:
//  1. Cached result exists - return distilled (fast path)
//  2. Empty after trim - skip
//  3. Short input with enough significant terms - skip, return raw
//  4. Long input OR too few significant terms - distil via LLM
func (d *DefaultDistiller) Distill(ctx context.Context, raw string, history []llm.Message, opts Options) (Result, error) {
	if opts.MaxQueries <= 0 {
		opts.MaxQueries = 3
	}
	if opts.MinTokenThreshold <= 0 {
		opts.MinTokenThreshold = 20
	}
	if opts.MinSignificantTerms <= 0 {
		opts.MinSignificantTerms = 3
	}
	if opts.Scope == "" {
		opts.Scope = "search"
	}

	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Result{Trace: Trace{Skipped: true, SkipReason: "empty input"}}, nil
	}

	// Truncate before cache key computation: inputs differing only in
	// their first portion hash to the same key by design.
	truncated := truncateInput(raw, maxInputChars)

	// Check cache.
	if !opts.DisableCache {
		key := cacheKey(truncated, opts.Scope)
		if cached, ok := d.cache.get(key); ok {
			return Result{
				Queries: cached,
				Trace:   Trace{CacheHit: true, Queries: cached},
			}, nil
		}
	}

	// Count tokens and significant terms for the gate.
	tokens := countTokens(truncated)
	significantTerms := countSignificantTerms(truncated)

	// Gate: short input with enough terms - return raw unchanged.
	if tokens < opts.MinTokenThreshold && significantTerms >= opts.MinSignificantTerms {
		q := []Query{{Text: truncated, Confidence: 1.0}}
		return Result{
			Queries: q,
			Trace: Trace{
				Skipped:         true,
				SkipReason:      "below threshold",
				RawTokens:       tokens,
				MechanicalTerms: significantTerms,
				Queries:         q,
			},
		}, nil
	}

	// Distil via LLM.
	start := time.Now()

	provider := opts.CloudProvider
	providerName := "cloud"
	if provider == nil {
		return Result{
			Queries: []Query{{Text: truncated, Confidence: 1.0}},
			Trace: Trace{
				Skipped:       true,
				SkipReason:    "no provider available",
				FellBackToRaw: true,
				RawTokens:     tokens,
			},
		}, nil
	}

	queries, err := callDistillLLM(ctx, provider, truncated, history, opts.MaxQueries)
	latency := time.Since(start).Milliseconds()

	if err != nil {
		// Fall back to raw on LLM failure.
		q := []Query{{Text: truncated, Confidence: 1.0}}
		return Result{
			Queries: q,
			Trace: Trace{
				FellBackToRaw:   true,
				Provider:        providerName,
				LatencyMillis:   latency,
				RawTokens:       tokens,
				MechanicalTerms: significantTerms,
				ErrorDetail:     err.Error(),
				Queries:         q,
			},
		}, nil
	}

	if len(queries) == 0 {
		queries = []Query{{Text: truncated, Confidence: 1.0}}
	}

	// Cache the result.
	if !opts.DisableCache {
		key := cacheKey(truncated, opts.Scope)
		d.cache.put(key, queries)
	}

	return Result{
		Queries: queries,
		Trace: Trace{
			Provider:        providerName,
			LatencyMillis:   latency,
			RawTokens:       tokens,
			MechanicalTerms: significantTerms,
			Queries:         queries,
		},
	}, nil
}
