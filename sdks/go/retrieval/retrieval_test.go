// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func newTestCorpus() []fakeChunk {
	return []fakeChunk{
		{
			ID:      "c1",
			Path:    "wiki/invoice-processing.md",
			Title:   "Invoice Processing",
			Summary: "How we automate supplier invoice ingestion",
			Content: "Invoice automation workflow extracts line items from PDFs.",
		},
		{
			ID:      "c2",
			Path:    "wiki/order-processing.md",
			Title:   "Order Processing Pipeline",
			Summary: "Sales order ingestion for retailers",
			Content: "Automated document processing for orders captured via email.",
		},
		{
			ID:      "c3",
			Path:    "wiki/contact-centre.md",
			Title:   "Contact Centre",
			Summary: "Inbound voice routing",
			Content: "Telephony stack routes calls via SIP.",
		},
		{
			ID:      "c4",
			Path:    "memory/global/user-preference-coffee.md",
			Title:   "Coffee preferences",
			Summary: "Alex prefers flat whites",
			Content: "Alex likes flat whites with oat milk.",
		},
		{
			ID:      "c5",
			Path:    "memory/global/user-fact-birthday.md",
			Title:   "User fact: birthday",
			Summary: "Observed on 1986-08-14",
			Content: "[observed on: 1986-08-14] Alex was born on 14 August 1986.",
		},
		{
			ID:      "c6",
			Path:    "wiki/rollup/invoice-summary.md",
			Title:   "Invoice recap",
			Summary: "Roll-up summary across invoice workflows",
			Content: "Overview and summary of invoice workflow totals.",
		},
	}
}

func TestRetrieve_BM25Mode(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  3,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatalf("expected hits")
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective mode %q, want bm25", resp.Trace.EffectiveMode)
	}
	if resp.Trace.EmbedderUsed {
		t.Fatalf("embedder should not fire in bm25 mode")
	}
	if resp.Chunks[0].Path != "wiki/invoice-processing.md" {
		t.Fatalf("top path %q, want wiki/invoice-processing.md", resp.Chunks[0].Path)
	}
}

func TestRetrieve_SemanticMode(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)
	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation workflow",
		TopK:  3,
		Mode:  ModeSemantic,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeSemantic {
		t.Fatalf("effective mode %q, want semantic", resp.Trace.EffectiveMode)
	}
	if !resp.Trace.EmbedderUsed {
		t.Fatalf("embedder should have fired")
	}
	if len(resp.Chunks) == 0 {
		t.Fatalf("expected semantic hits")
	}
}

func TestRetrieve_HybridMode(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)
	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  3,
		Mode:  ModeHybrid,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeHybrid {
		t.Fatalf("effective mode %q, want hybrid", resp.Trace.EffectiveMode)
	}
	if resp.Trace.BM25Hits == 0 {
		t.Fatalf("hybrid mode: bm25 leg should find hits")
	}
	if resp.Trace.VectorHits == 0 {
		t.Fatalf("hybrid mode: vector leg should find hits")
	}
	if resp.Trace.FusedHits == 0 {
		t.Fatalf("hybrid mode: fusion should produce hits")
	}
	// Invoice-related docs should dominate.
	top := resp.Chunks[0].Path
	if top != "wiki/invoice-processing.md" && top != "wiki/order-processing.md" {
		t.Fatalf("unexpected top result %q", top)
	}
}

func TestRetrieve_HybridRerank_AppliesReranker(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	embedder := llm.NewFakeEmbedder(src.embedDim)
	// Reranker reverses whatever head it receives so we can assert
	// it actually ran by checking the new top slot.
	rr := rerankerFn(func(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
		out := make([]RetrievedChunk, len(chunks))
		for i := range chunks {
			out[i] = chunks[len(chunks)-1-i]
			out[i].RerankScore = float64(len(chunks) - i)
		}
		return out, nil
	})
	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice automation",
		TopK:  5,
		Mode:  ModeHybridRerank,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if !resp.Trace.Reranked {
		t.Fatalf("rerank did not fire; trace: %+v", resp.Trace)
	}
	if resp.Trace.RerankSkipReason != "" {
		t.Fatalf("rerank skip reason %q should be empty", resp.Trace.RerankSkipReason)
	}
}

func TestRetrieve_ModeFallback_WhenEmbedderMissing(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice",
		Mode:  ModeHybrid,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective %q, want fallback bm25", resp.Trace.EffectiveMode)
	}
	if !resp.Trace.FellBackToBM25 {
		t.Fatalf("FellBackToBM25 should be true")
	}
}

func TestRetrieve_AutoWithoutEmbedder_FallsBackSilently(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "invoice",
		Mode:  ModeAuto,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.EffectiveMode != ModeBM25 {
		t.Fatalf("effective %q, want bm25", resp.Trace.EffectiveMode)
	}
	if resp.Trace.FellBackToBM25 {
		t.Fatalf("auto fallback should not flag FellBackToBM25")
	}
}

func TestRetrieve_UnanimityShortcut_SkipsRerank(t *testing.T) {
	t.Parallel()
	// Craft a corpus where the BM25 and vector top-3 agree. We use
	// a narrow corpus with strong overlap so both legs return the
	// same three paths.
	chunks := []fakeChunk{
		{ID: "a", Path: "a.md", Title: "Alpha", Content: "alpha bravo"},
		{ID: "b", Path: "b.md", Title: "Bravo", Content: "alpha bravo"},
		{ID: "c", Path: "c.md", Title: "Charlie", Content: "alpha bravo"},
	}
	src := newFakeSource(chunks)
	embedder := llm.NewFakeEmbedder(src.embedDim)
	rerankerCalls := 0
	rr := rerankerFn(func(ctx context.Context, q string, rs []RetrievedChunk) ([]RetrievedChunk, error) {
		rerankerCalls++
		return rs, nil
	})
	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query: "alpha bravo",
		TopK:  3,
		Mode:  ModeHybridRerank,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if resp.Trace.UnanimitySkipped {
		if rerankerCalls != 0 {
			t.Fatalf("rerank ran despite unanimity shortcut")
		}
		if resp.Trace.RerankSkipReason != "unanimity" {
			t.Fatalf("skip reason %q, want unanimity", resp.Trace.RerankSkipReason)
		}
	} else {
		// If agreements fell below the threshold, the reranker
		// should have run.
		if rerankerCalls == 0 {
			t.Fatalf("reranker never called and unanimity not flagged")
		}
	}
}

func TestRetrieve_NilSource_Errors(t *testing.T) {
	t.Parallel()
	if _, err := New(Config{}); err == nil {
		t.Fatalf("expected error for nil source")
	}
}

func TestRetrieve_BM25LegError_Surfaces(t *testing.T) {
	t.Parallel()
	src := newFakeSource(newTestCorpus())
	src.bm25Fail = errors.New("boom")
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if _, err := r.Retrieve(context.Background(), Request{Query: "anything", Mode: ModeBM25}); err == nil {
		t.Fatalf("expected bm25 error to propagate")
	}
}

func TestRetrieve_Filters_NarrowCorpus(t *testing.T) {
	t.Parallel()
	corpus := newTestCorpus()
	corpus = append(corpus, fakeChunk{
		ID:    "scoped",
		Path:  "memory/project/billing/invoice.md",
		Title: "Billing invoice workflow",
	})
	src := newFakeSource(corpus)
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	resp, err := r.Retrieve(context.Background(), Request{
		Query:   "invoice",
		Mode:    ModeBM25,
		Filters: Filters{PathPrefix: "memory/project/"},
		TopK:    5,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	for _, c := range resp.Chunks {
		if c.Path != "memory/project/billing/invoice.md" {
			t.Fatalf("filter leak: %q", c.Path)
		}
	}
}

// rerankerFn is a function-as-Reranker adapter used by tests.
type rerankerFn func(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error)

func (f rerankerFn) Rerank(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
	return f(ctx, query, chunks)
}

func (f rerankerFn) Name() string { return "test-reranker" }
