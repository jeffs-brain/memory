// SPDX-License-Identifier: Apache-2.0

package query

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

// countingFake wraps [llm.NewFake] so tests can assert call counts and
// inject errors. The canonical SDK fake does not surface either directly.
type countingFake struct {
	inner llm.Provider
	err   error
	calls atomic.Uint64
}

func newCountingFake(response string) *countingFake {
	return &countingFake{inner: llm.NewFake([]string{response})}
}

func newErrorFake(err error) *countingFake {
	return &countingFake{inner: llm.NewFake([]string{""}), err: err}
}

func (f *countingFake) Complete(ctx context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	f.calls.Add(1)
	if f.err != nil {
		return llm.CompleteResponse{}, f.err
	}
	return f.inner.Complete(ctx, req)
}

func (f *countingFake) CompleteStream(ctx context.Context, req llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	f.calls.Add(1)
	if f.err != nil {
		return nil, f.err
	}
	return f.inner.CompleteStream(ctx, req)
}

func (f *countingFake) Close() error { return f.inner.Close() }

func (f *countingFake) Calls() int { return int(f.calls.Load()) }

// --- Gate logic tests ---

func TestDistill_EmptyInput(t *testing.T) {
	d := NewDistiller()
	result, err := d.Distill(context.Background(), "", nil, DefaultOptions())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Trace.Skipped {
		t.Error("expected Skipped to be true for empty input")
	}
	if result.Trace.SkipReason != "empty input" {
		t.Errorf("expected skip reason 'empty input', got %q", result.Trace.SkipReason)
	}
	if len(result.Queries) != 0 {
		t.Errorf("expected no queries for empty input, got %d", len(result.Queries))
	}
}

func TestDistill_WhitespaceOnly(t *testing.T) {
	d := NewDistiller()
	result, err := d.Distill(context.Background(), "   \t\n  ", nil, DefaultOptions())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Trace.Skipped {
		t.Error("expected Skipped for whitespace-only input")
	}
	if result.Trace.SkipReason != "empty input" {
		t.Errorf("expected skip reason 'empty input', got %q", result.Trace.SkipReason)
	}
}

func TestDistill_ShortWithEnoughSignificantTerms(t *testing.T) {
	// "kubernetes deployment rollback" has 3 significant terms and < 20 tokens.
	d := NewDistiller()
	opts := DefaultOptions()
	result, err := d.Distill(context.Background(), "kubernetes deployment rollback", nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Trace.Skipped {
		t.Error("expected Skipped for short input with enough significant terms")
	}
	if result.Trace.SkipReason != "below threshold" {
		t.Errorf("expected skip reason 'below threshold', got %q", result.Trace.SkipReason)
	}
	if len(result.Queries) != 1 {
		t.Fatalf("expected 1 query, got %d", len(result.Queries))
	}
	if result.Queries[0].Text != "kubernetes deployment rollback" {
		t.Errorf("expected raw text passthrough, got %q", result.Queries[0].Text)
	}
	if result.Queries[0].Confidence != 1.0 {
		t.Errorf("expected confidence 1.0, got %f", result.Queries[0].Confidence)
	}
}

func TestDistill_ShortWithTooFewTermsAndNoProvider(t *testing.T) {
	// "the is a" has 0 significant terms and < 20 tokens.
	// With no provider it should fall back to raw.
	d := NewDistiller()
	opts := DefaultOptions()
	result, err := d.Distill(context.Background(), "the is a", nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Trace.SkipReason != "no provider available" {
		t.Errorf("expected skip reason 'no provider available', got %q", result.Trace.SkipReason)
	}
	if !result.Trace.FellBackToRaw {
		t.Error("expected FellBackToRaw for nil provider")
	}
}

func TestDistill_NilProviderFallsBackToRaw(t *testing.T) {
	d := NewDistiller()
	// Build a long input that exceeds the token threshold.
	longInput := strings.Repeat("error stacktrace failure panic ", 10)
	opts := DefaultOptions()
	opts.CloudProvider = nil

	result, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Trace.FellBackToRaw {
		t.Error("expected FellBackToRaw when provider is nil")
	}
	if len(result.Queries) != 1 {
		t.Fatalf("expected 1 fallback query, got %d", len(result.Queries))
	}
}

func TestDistill_LLMSuccess(t *testing.T) {
	d := NewDistiller()
	longInput := strings.Repeat("I keep getting a weird error when I try to deploy the kubernetes cluster with argocd and it fails at the sync step ", 3)

	resp := `[{"text": "kubernetes argocd deployment sync failure", "domain": "infrastructure", "entities": ["kubernetes", "argocd"], "confidence": 0.9}]`
	provider := newCountingFake(resp)

	opts := DefaultOptions()
	opts.CloudProvider = provider
	opts.DisableCache = true

	result, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Trace.FellBackToRaw {
		t.Error("did not expect FellBackToRaw on successful LLM call")
	}
	if result.Trace.Provider != "cloud" {
		t.Errorf("expected provider 'cloud', got %q", result.Trace.Provider)
	}
	if len(result.Queries) != 1 {
		t.Fatalf("expected 1 query, got %d", len(result.Queries))
	}
	if result.Queries[0].Text != "kubernetes argocd deployment sync failure" {
		t.Errorf("unexpected query text: %q", result.Queries[0].Text)
	}
	if provider.Calls() != 1 {
		t.Errorf("expected 1 provider call, got %d", provider.Calls())
	}
}

func TestDistill_LLMError_FallsBackToRaw(t *testing.T) {
	d := NewDistiller()
	longInput := strings.Repeat("something something error trace ", 10)

	provider := newErrorFake(fmt.Errorf("connection refused"))

	opts := DefaultOptions()
	opts.CloudProvider = provider
	opts.DisableCache = true

	result, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Trace.FellBackToRaw {
		t.Error("expected FellBackToRaw on LLM error")
	}
	if result.Trace.ErrorDetail == "" {
		t.Error("expected ErrorDetail to be populated")
	}
	if len(result.Queries) != 1 {
		t.Fatalf("expected 1 fallback query, got %d", len(result.Queries))
	}
}

// --- Cache tests ---

func TestDistill_CacheHit(t *testing.T) {
	d := NewDistiller()
	longInput := strings.Repeat("kubernetes deployment error sync argocd ", 5)

	resp := `[{"text": "k8s deploy error", "confidence": 0.8}]`
	provider := newCountingFake(resp)

	opts := DefaultOptions()
	opts.CloudProvider = provider

	// First call: cache miss, calls provider.
	result1, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result1.Trace.CacheHit {
		t.Error("first call should not be a cache hit")
	}
	if provider.Calls() != 1 {
		t.Errorf("expected 1 provider call after first distill, got %d", provider.Calls())
	}

	// Second call: cache hit, no additional provider call.
	result2, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result2.Trace.CacheHit {
		t.Error("second call should be a cache hit")
	}
	if provider.Calls() != 1 {
		t.Errorf("expected still 1 provider call after cache hit, got %d", provider.Calls())
	}
	if len(result2.Queries) != 1 {
		t.Fatalf("expected 1 cached query, got %d", len(result2.Queries))
	}
	if result2.Queries[0].Text != "k8s deploy error" {
		t.Errorf("unexpected cached query text: %q", result2.Queries[0].Text)
	}
}

func TestDistill_CacheDisabled(t *testing.T) {
	d := NewDistiller()
	longInput := strings.Repeat("kubernetes deployment error sync argocd ", 5)

	resp := `[{"text": "k8s deploy error", "confidence": 0.8}]`
	provider := newCountingFake(resp)

	opts := DefaultOptions()
	opts.CloudProvider = provider
	opts.DisableCache = true

	// Two calls with cache disabled should both hit the provider.
	_, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, err = d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if provider.Calls() != 2 {
		t.Errorf("expected 2 provider calls with cache disabled, got %d", provider.Calls())
	}
}

// --- Token counting tests ---

func TestCountTokens(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"hello", 1},
		{"hello world", 2},
		{"  hello   world  ", 2},
		{"one two three four five", 5},
		{"tabs\there\ttoo", 3},
		{"newlines\nwork\ntoo", 3},
	}
	for _, tt := range tests {
		got := countTokens(tt.input)
		if got != tt.want {
			t.Errorf("countTokens(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

// --- Significant term counting tests ---

func TestCountSignificantTerms(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"", 0},
		{"the is a", 0},
		{"kubernetes deployment rollback", 3},
		{"I want to deploy the application", 3}, // want, deploy, application
		{"what is kubernetes?", 1},               // kubernetes
		{"error: panic in goroutine", 3},         // error, panic, goroutine
	}
	for _, tt := range tests {
		got := countSignificantTerms(tt.input)
		if got != tt.want {
			t.Errorf("countSignificantTerms(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

// --- Cache key normalisation tests ---

func TestCacheKeyNormalisation(t *testing.T) {
	// Same content with different whitespace should produce the same key.
	k1 := cacheKey("hello  world", "search")
	k2 := cacheKey("hello world", "search")
	if k1 != k2 {
		t.Errorf("expected same key for collapsed whitespace, got %q vs %q", k1, k2)
	}

	// Case insensitive.
	k3 := cacheKey("Hello World", "search")
	if k1 != k3 {
		t.Errorf("expected same key for different case, got %q vs %q", k1, k3)
	}

	// Different scope produces different key.
	k4 := cacheKey("hello world", "recall")
	if k1 == k4 {
		t.Error("expected different keys for different scopes")
	}

	// Zero-width characters are stripped.
	k5 := cacheKey("hello\u200bworld", "search")
	k6 := cacheKey("helloworld", "search")
	if k5 != k6 {
		t.Errorf("expected same key after stripping zero-width chars, got %q vs %q", k5, k6)
	}

	// Non-breaking spaces are stripped.
	k7 := cacheKey("hello\u00a0world", "search")
	k8 := cacheKey("hello world", "search")
	if k7 != k8 {
		t.Errorf("expected same key after stripping NBSP, got %q vs %q", k7, k8)
	}
}

// --- Input truncation tests ---

func TestTruncateInput(t *testing.T) {
	// Short input: no truncation.
	s := "short"
	got := truncateInput(s, 100)
	if got != s {
		t.Errorf("expected no truncation, got %q", got)
	}

	// Long input: preserves the tail.
	long := strings.Repeat("a", 10000)
	got = truncateInput(long, 8000)
	if len(got) != 8000 {
		t.Errorf("expected 8000 chars, got %d", len(got))
	}
	// Tail should be all 'a's.
	if got != long[2000:] {
		t.Error("expected truncation to preserve the tail")
	}

	// Exact boundary: no truncation.
	exact := strings.Repeat("b", 8000)
	got = truncateInput(exact, 8000)
	if got != exact {
		t.Error("expected no truncation at exact boundary")
	}
}

// --- parseDistillResponse tests ---

func TestParseDistillResponse_ValidJSON(t *testing.T) {
	input := `[{"text": "query one", "confidence": 0.9}, {"text": "query two", "confidence": 0.7}]`
	queries, err := parseDistillResponse(input, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(queries) != 2 {
		t.Fatalf("expected 2 queries, got %d", len(queries))
	}
	if queries[0].Text != "query one" {
		t.Errorf("unexpected first query text: %q", queries[0].Text)
	}
	if queries[1].Text != "query two" {
		t.Errorf("unexpected second query text: %q", queries[1].Text)
	}
}

func TestParseDistillResponse_WrappedInText(t *testing.T) {
	input := `Here are the queries:
[{"text": "wrapped query", "confidence": 0.8}]
Hope that helps!`
	queries, err := parseDistillResponse(input, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(queries) != 1 {
		t.Fatalf("expected 1 query, got %d", len(queries))
	}
	if queries[0].Text != "wrapped query" {
		t.Errorf("unexpected query text: %q", queries[0].Text)
	}
}

func TestParseDistillResponse_EnforcesMaxQueries(t *testing.T) {
	input := `[{"text": "q1"}, {"text": "q2"}, {"text": "q3"}, {"text": "q4"}]`
	queries, err := parseDistillResponse(input, 2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(queries) != 2 {
		t.Errorf("expected 2 queries (max), got %d", len(queries))
	}
}

func TestParseDistillResponse_DropsEmptyText(t *testing.T) {
	input := `[{"text": "valid"}, {"text": ""}, {"text": "  "}]`
	queries, err := parseDistillResponse(input, 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(queries) != 1 {
		t.Fatalf("expected 1 valid query, got %d", len(queries))
	}
	if queries[0].Text != "valid" {
		t.Errorf("unexpected query text: %q", queries[0].Text)
	}
}

func TestParseDistillResponse_NoJSONArray(t *testing.T) {
	_, err := parseDistillResponse("no json here at all", 3)
	if err == nil {
		t.Error("expected error for input with no JSON array")
	}
}

func TestParseDistillResponse_MalformedJSON(t *testing.T) {
	_, err := parseDistillResponse(`[{"text": broken}]`, 3)
	if err == nil {
		t.Error("expected error for malformed JSON")
	}
}

func TestParseDistillResponse_EmptyArray(t *testing.T) {
	queries, err := parseDistillResponse("[]", 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(queries) != 0 {
		t.Errorf("expected 0 queries for empty array, got %d", len(queries))
	}
}

// --- LRU cache eviction test ---

func TestCacheEviction(t *testing.T) {
	c := newCache(2)

	c.put("k1", []Query{{Text: "q1"}})
	c.put("k2", []Query{{Text: "q2"}})
	c.put("k3", []Query{{Text: "q3"}}) // Should evict k1.

	if _, ok := c.get("k1"); ok {
		t.Error("k1 should have been evicted")
	}
	if _, ok := c.get("k2"); !ok {
		t.Error("k2 should still be in cache")
	}
	if _, ok := c.get("k3"); !ok {
		t.Error("k3 should still be in cache")
	}
}

func TestCacheLRUOrder(t *testing.T) {
	c := newCache(2)

	c.put("k1", []Query{{Text: "q1"}})
	c.put("k2", []Query{{Text: "q2"}})

	// Access k1 to make it most recent.
	c.get("k1")

	// k3 should evict k2 (least recently used), not k1.
	c.put("k3", []Query{{Text: "q3"}})

	if _, ok := c.get("k1"); !ok {
		t.Error("k1 should still be in cache after access refresh")
	}
	if _, ok := c.get("k2"); ok {
		t.Error("k2 should have been evicted as LRU")
	}
	if _, ok := c.get("k3"); !ok {
		t.Error("k3 should still be in cache")
	}
}

// --- DefaultOptions test ---

func TestDefaultOptions(t *testing.T) {
	opts := DefaultOptions()
	if opts.MinTokenThreshold != 20 {
		t.Errorf("expected MinTokenThreshold 20, got %d", opts.MinTokenThreshold)
	}
	if opts.MinSignificantTerms != 3 {
		t.Errorf("expected MinSignificantTerms 3, got %d", opts.MinSignificantTerms)
	}
	if opts.MaxQueries != 3 {
		t.Errorf("expected MaxQueries 3, got %d", opts.MaxQueries)
	}
	if opts.Scope != "search" {
		t.Errorf("expected Scope 'search', got %q", opts.Scope)
	}
}

// --- buildDistillPrompt tests ---

func TestBuildDistillPrompt_NoHistory(t *testing.T) {
	prompt := buildDistillPrompt("test query", nil, 3)
	if !strings.Contains(prompt, "test query") {
		t.Error("prompt should contain raw input")
	}
	if strings.Contains(prompt, "conversation context") {
		t.Error("prompt should not contain history section when history is nil")
	}
}

func TestBuildDistillPrompt_WithHistory(t *testing.T) {
	history := []llm.Message{
		{Role: llm.RoleAssistant, Content: "I found some results."},
		{Role: llm.RoleUser, Content: "Can you tell me more about that?"},
		{Role: llm.RoleUser, Content: "Also what about the deployment?"},
	}
	prompt := buildDistillPrompt("expand on the error", history, 3)
	if !strings.Contains(prompt, "conversation context") {
		t.Error("prompt should contain history section")
	}
	if !strings.Contains(prompt, "Also what about the deployment?") {
		t.Error("prompt should contain recent user message")
	}
	if !strings.Contains(prompt, "Can you tell me more about that?") {
		t.Error("prompt should contain earlier user message")
	}
	// Assistant messages should not appear.
	if strings.Contains(prompt, "I found some results") {
		t.Error("prompt should not contain assistant messages")
	}
}

func TestBuildDistillPrompt_HistoryTruncation(t *testing.T) {
	longContent := strings.Repeat("x", 1000)
	history := []llm.Message{
		{Role: llm.RoleUser, Content: longContent},
	}
	prompt := buildDistillPrompt("query", history, 3)
	// The content should be truncated to 500 chars + "...".
	if strings.Contains(prompt, longContent) {
		t.Error("long history content should be truncated")
	}
	if !strings.Contains(prompt, "...") {
		t.Error("truncated content should end with ellipsis")
	}
}

// --- Query JSON serialisation test ---

func TestQuery_JSONRoundTrip(t *testing.T) {
	q := Query{
		Text:        "test query",
		Domain:      "infra",
		Entities:    []string{"kubernetes", "argocd"},
		RecencyBias: "recent",
		Confidence:  0.85,
	}
	data, err := json.Marshal(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded Query
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.Text != q.Text {
		t.Errorf("text mismatch: %q vs %q", decoded.Text, q.Text)
	}
	if decoded.Domain != q.Domain {
		t.Errorf("domain mismatch: %q vs %q", decoded.Domain, q.Domain)
	}
	if len(decoded.Entities) != 2 {
		t.Errorf("expected 2 entities, got %d", len(decoded.Entities))
	}
	if decoded.RecencyBias != q.RecencyBias {
		t.Errorf("recency_bias mismatch: %q vs %q", decoded.RecencyBias, q.RecencyBias)
	}
	if decoded.Confidence != q.Confidence {
		t.Errorf("confidence mismatch: %f vs %f", decoded.Confidence, q.Confidence)
	}
}

// --- Distiller with multi-query LLM response ---

func TestDistill_MultiQueryResponse(t *testing.T) {
	d := NewDistiller()
	longInput := strings.Repeat("I need help with kubernetes deployment and also the database migrations are failing and can you check the CI pipeline too ", 3)

	resp := `[
		{"text": "kubernetes deployment troubleshooting", "confidence": 0.9},
		{"text": "database migration failures", "domain": "backend", "confidence": 0.85},
		{"text": "CI pipeline status check", "confidence": 0.7}
	]`
	provider := newCountingFake(resp)

	opts := DefaultOptions()
	opts.CloudProvider = provider
	opts.DisableCache = true

	result, err := d.Distill(context.Background(), longInput, nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Queries) != 3 {
		t.Fatalf("expected 3 queries, got %d", len(result.Queries))
	}
	if result.Queries[1].Domain != "backend" {
		t.Errorf("expected domain 'backend' on second query, got %q", result.Queries[1].Domain)
	}
}

// --- NormaliseForCache tests ---

func TestNormaliseForCache(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"  Hello  World  ", "hello world"},
		{"UPPER", "upper"},
		{"zero\u200bwidth", "zerowidth"},
		{"non\u00a0breaking", "non breaking"},
		{"bom\ufeffchar", "bomchar"},
		{"tabs\there", "tabs here"},
		{"multi\n\nline", "multi line"},
	}
	for _, tt := range tests {
		got := normaliseForCache(tt.input)
		if got != tt.want {
			t.Errorf("normaliseForCache(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// --- Distiller respects zero-value option overrides ---

func TestDistill_ZeroOptionsDefaulted(t *testing.T) {
	d := NewDistiller()
	// Pass zero-value options; the distiller should default them.
	opts := Options{}
	result, err := d.Distill(context.Background(), "kubernetes argocd rollback deploy", nil, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// With defaults: < 20 tokens, >= 3 significant terms, should skip.
	if !result.Trace.Skipped {
		t.Error("expected Skipped with defaulted options for short significant input")
	}
}
