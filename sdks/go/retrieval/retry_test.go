// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"strings"
	"testing"
)

// retryCorpus carries a handful of docs that exercise each rung of
// the ladder. Slugs are intentionally obscure so only the trigram
// fallback can surface them on a typo query.
func retryCorpus() []fakeChunk {
	return []fakeChunk{
		{
			ID:      "r1",
			Path:    "wiki/kubernetes-cluster-setup.md",
			Title:   "Kubernetes Cluster Setup",
			Summary: "Bootstrapping a kind cluster",
			Content: "Provision a three-node kind cluster and apply the base manifests.",
		},
		{
			ID:      "r2",
			Path:    "wiki/archipelago-tooling.md",
			Title:   "Archipelago tooling",
			Summary: "Custom build tooling for the archipelago stack",
			Content: "The archipelago build set manages intra-service contracts.",
		},
		{
			ID:      "r3",
			Path:    "wiki/miscellaneous-notes.md",
			Title:   "Miscellaneous Notes",
			Summary: "Catch-all",
			Content: "Kubernetes operations runbooks are kept here for ad-hoc lookup.",
		},
	}
}

func TestRetryLadder_Rung0_InitialHit(t *testing.T) {
	t.Parallel()
	src := newFakeSource(retryCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{Query: "kubernetes", Mode: ModeBM25})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.UsedRetry {
		t.Fatalf("retry ladder should not fire when initial query hits")
	}
	if len(resp.Attempts) != 1 || resp.Attempts[0].Reason != "initial" {
		t.Fatalf("unexpected attempts: %+v", resp.Attempts)
	}
}

func TestRetryLadder_Rung1_StrongestTerm(t *testing.T) {
	t.Parallel()
	src := newFakeSource(retryCorpus())
	// Force the initial query to return zero so rung 1 can fire.
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		if strings.Contains(expr, "xyz") {
			return nil, true
		}
		return nil, false
	}
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Raw query with a rubbish token that the initial compile
	// matches; strongest term "kubernetes" surfaces real hits.
	resp, err := r.Retrieve(context.Background(), Request{Query: "xyz kubernetes", Mode: ModeBM25})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if !resp.Trace.UsedRetry {
		t.Fatalf("retry ladder should have fired")
	}
	var sawStrongest bool
	for _, a := range resp.Attempts {
		if a.Reason == "strongest_term" {
			sawStrongest = true
			if a.Rung != 1 {
				t.Fatalf("strongest_term reported rung %d, want 1", a.Rung)
			}
			if a.Chunks == 0 {
				t.Fatalf("strongest_term should have found hits")
			}
		}
	}
	if !sawStrongest {
		t.Fatalf("strongest_term attempt missing: %+v", resp.Attempts)
	}
}

func TestRetryLadder_Rung3_RefreshedSanitised(t *testing.T) {
	t.Parallel()
	// A punctuation-heavy query whose sanitised form unlocks a
	// match, while the initial compiled expression returns nothing.
	chunks := []fakeChunk{
		{ID: "a", Path: "wiki/foo.md", Title: "foo", Content: "alphabet"},
	}
	src := newFakeSource(chunks)
	// First call yields nothing; subsequent calls go through the
	// normal BM25 path.
	calls := 0
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		calls++
		if calls == 1 {
			return nil, true
		}
		// Rung 1 strongest term: "alphabet" beats other tokens.
		// Force it empty too so we fall through to rung 3.
		if calls == 2 {
			return nil, true
		}
		return nil, false
	}
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{Query: "!!! alphabet ???", Mode: ModeBM25})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if !resp.Trace.UsedRetry {
		t.Fatalf("retry ladder should have fired")
	}
	var sawSanitised bool
	for _, a := range resp.Attempts {
		if a.Reason == "refreshed_sanitised" {
			sawSanitised = true
			if a.Rung != 3 {
				t.Fatalf("refreshed_sanitised rung %d, want 3", a.Rung)
			}
		}
	}
	if !sawSanitised {
		t.Fatalf("refreshed_sanitised attempt missing: %+v", resp.Attempts)
	}
}

func TestRetryLadder_Rung4_RefreshedStrongest(t *testing.T) {
	t.Parallel()
	chunks := []fakeChunk{
		{ID: "a", Path: "wiki/kubernetes.md", Title: "Kubernetes", Content: "runbook"},
	}
	src := newFakeSource(chunks)
	// Force rungs 0, 1, 3 empty so 4 fires.
	calls := 0
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		calls++
		if calls <= 3 {
			return nil, true
		}
		return nil, false
	}
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{Query: "!? kubernetes ?!", Mode: ModeBM25})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if !resp.Trace.UsedRetry {
		t.Fatalf("retry ladder should have fired")
	}
	var sawRung4 bool
	for _, a := range resp.Attempts {
		if a.Reason == "refreshed_strongest" {
			sawRung4 = true
			if a.Rung != 4 {
				t.Fatalf("refreshed_strongest rung %d, want 4", a.Rung)
			}
		}
	}
	if !sawRung4 {
		t.Fatalf("refreshed_strongest attempt missing: %+v", resp.Attempts)
	}
}

func TestRetryLadder_Rung5_TrigramFuzzy(t *testing.T) {
	t.Parallel()
	chunks := []fakeChunk{
		{ID: "kube", Path: "wiki/kubernetes.md", Title: "Kubernetes", Content: "ops"},
		{ID: "arch", Path: "wiki/archipelago.md", Title: "Archipelago", Content: "build"},
	}
	src := newFakeSource(chunks)
	// Every BM25 rung returns empty so the trigram fallback runs.
	src.bm25Override = func(expr string) ([]BM25Hit, bool) { return nil, true }
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Single-character typo: the slug "kubernetes" still gives
	// Jaccard above the 0.3 threshold against the query trigrams.
	resp, err := r.Retrieve(context.Background(), Request{Query: "kubernets", Mode: ModeBM25, TopK: 5})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if !resp.Trace.UsedRetry {
		t.Fatalf("retry ladder should have fired")
	}
	var sawTrigram bool
	for _, a := range resp.Attempts {
		if a.Reason == "trigram_fuzzy" {
			sawTrigram = true
			if a.Rung != 5 {
				t.Fatalf("trigram_fuzzy rung %d, want 5", a.Rung)
			}
			if a.Chunks == 0 {
				t.Fatalf("trigram_fuzzy should have surfaced the kubernetes slug")
			}
		}
	}
	if !sawTrigram {
		t.Fatalf("trigram_fuzzy attempt missing: %+v", resp.Attempts)
	}
	if len(resp.Chunks) == 0 {
		t.Fatalf("expected trigram hits as final results")
	}
}

func TestRetryLadder_SkipRetryLadder_BypassesRungs(t *testing.T) {
	t.Parallel()
	src := newFakeSource(retryCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) { return nil, true }
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:           "kubernetes",
		Mode:            ModeBM25,
		SkipRetryLadder: true,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.UsedRetry {
		t.Fatalf("retry ladder should be suppressed")
	}
	if len(resp.Attempts) != 1 {
		t.Fatalf("expected single initial attempt, got %d", len(resp.Attempts))
	}
}
