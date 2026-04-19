// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"errors"
	"strings"
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

func TestRetrieve_BM25Fanout_MergesTemporalQueryVariants(t *testing.T) {
	t.Parallel()

	src := newFakeSource(newTestCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		switch {
		case strings.Contains(expr, "2024") || strings.Contains(expr, "03") || strings.Contains(expr, "08"):
			return []BM25Hit{{
				ID:      "dated",
				Path:    "raw/lme/dated.md",
				Title:   "Dated hit",
				Summary: "Temporal match",
				Content: "This happened on 2024/03/08.",
			}}, true
		case strings.Contains(strings.ToLower(expr), "friday"):
			return []BM25Hit{{
				ID:      "plain",
				Path:    "raw/lme/plain.md",
				Title:   "Plain hit",
				Summary: "Lexical match",
				Content: "We met on Friday.",
			}}, true
		default:
			return nil, false
		}
	}

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What happened last Friday?",
		QuestionDate: "2024/03/13 (Wed) 10:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) < 2 {
		t.Fatalf("expected fanout to merge at least two hits, got %d", len(resp.Chunks))
	}
	got := []string{resp.Chunks[0].Path, resp.Chunks[1].Path}
	if !containsPath(got, "raw/lme/plain.md") || !containsPath(got, "raw/lme/dated.md") {
		t.Fatalf("fanout merged paths = %v, want plain + dated hits", got)
	}
	if len(resp.Attempts) == 0 || !strings.Contains(resp.Attempts[0].Query, "||") {
		t.Fatalf("initial attempt should record fanout queries, got %+v", resp.Attempts)
	}
}

func TestRetrieve_BM25_ReweightsMostRecentDatedHits(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "older",
			Path:    "memory/global/a-older.md",
			Title:   "Market visit",
			Content: "[Observed on 2024/03/01 (Fri) 09:00]\nEarned $220 at the Downtown Farmers Market.",
		},
		{
			ID:      "newer",
			Path:    "memory/global/z-newer.md",
			Title:   "Market visit",
			Content: "[Observed on 2024/03/08 (Fri) 09:00]\nEarned $420 at the Downtown Farmers Market.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "How much did I earn at the Downtown Farmers Market on my most recent visit?",
		TopK:  5,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "newer" {
		t.Fatalf("top chunk = %+v, want newer", resp.Chunks)
	}
}

func TestRetrieve_BM25_ReweightsClosestTemporalHintDate(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "far",
			Path:    "memory/global/a-far.md",
			Title:   "Weekly note",
			Content: "[Observed on 2024/02/02 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
		},
		{
			ID:      "near",
			Path:    "memory/global/z-near.md",
			Title:   "Weekly note",
			Content: "[Observed on 2024/03/08 (Fri) 10:00]\nMet the supplier and agreed the timeline.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What happened with the supplier last Friday?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "near" {
		t.Fatalf("top chunk = %+v, want near", resp.Chunks)
	}
}

func TestRetrieve_BM25_DropsFutureDatedHitsRelativeToQuestionDate(t *testing.T) {
	t.Parallel()

	src := newFakeSource([]fakeChunk{
		{
			ID:      "past",
			Path:    "memory/global/past.md",
			Title:   "Supplier visit",
			Content: "[Observed on 2024/03/10 (Sun) 09:00]\nMet the supplier and agreed the next steps.",
		},
		{
			ID:      "future",
			Path:    "memory/global/future.md",
			Title:   "Supplier visit",
			Content: "[Observed on 2024/03/20 (Wed) 09:00]\nMet the supplier and agreed the next steps.",
		},
		{
			ID:      "undated",
			Path:    "memory/global/undated.md",
			Title:   "Supplier visit",
			Content: "Met the supplier and agreed the next steps.",
		},
	})

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query:        "What is the most recent supplier visit?",
		QuestionDate: "2024/03/15 (Fri) 09:00",
		TopK:         5,
		Mode:         ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 || resp.Chunks[0].ChunkID != "past" {
		t.Fatalf("top chunk = %+v, want past", resp.Chunks)
	}
	for _, chunk := range resp.Chunks {
		if chunk.ChunkID == "future" {
			t.Fatalf("future-dated chunk leaked into results: %+v", resp.Chunks)
		}
	}
}

func TestRetrieve_BM25Fanout_DropsDriftedTokenProbes(t *testing.T) {
	t.Parallel()

	src := newFakeSource(newTestCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		switch expr {
		case "conversation":
			return []BM25Hit{{
				ID:      "noise-conversation",
				Path:    "memory/project/noise-conversation.md",
				Title:   "Conversation note",
				Summary: "Off-topic conversation metadata",
				Content: "Conversation metadata and follow-up notes.",
			}}, true
		case "remembered":
			return []BM25Hit{{
				ID:      "noise-remembered",
				Path:    "memory/project/noise-remembered.md",
				Title:   "Remembered note",
				Summary: "Off-topic remembered preference",
				Content: "Remembered preferences and recap notes.",
			}}, true
		}
		if strings.Contains(expr, "radiation") && strings.Contains(expr, "amplified") && strings.Contains(expr, "zombie") {
			return []BM25Hit{{
				ID:      "target",
				Path:    "raw/lme/answer_sharegpt_hChsWOp_97.md",
				Title:   "",
				Summary: "",
				Content: "We finally named the Radiation Amplified zombie Fissionator.",
			}}, true
		}
		return nil, false
	}

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "I was thinking back to our previous conversation about the Radiation Amplified zombie, and I was wondering if you remembered what we finally decided to name it?",
		TopK:  5,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) == 0 {
		t.Fatal("expected at least one fanout result")
	}
	if got := resp.Chunks[0].Path; got != "raw/lme/answer_sharegpt_hChsWOp_97.md" {
		t.Fatalf("top fanout hit = %q, want target transcript", got)
	}
}

func TestBuildBM25FanoutQueries_UsesPhraseProbesForCompoundQuestions(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"What is the total amount I spent on the designer handbag and high-end skincare products?",
		"",
	)

	want := []string{
		"handbag cost",
		"high-end skincare products",
	}
	if len(got) != len(want) {
		t.Fatalf("fanout queries = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("fanout queries = %v, want %v", got, want)
		}
	}
}

func TestBuildBM25FanoutQueries_AddsSpecificBackEndLanguageProbe(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?",
		"",
	)

	if len(got) < 1 {
		t.Fatalf("fanout queries = %v, want focused back-end language probe", got)
	}
	if got[0] != "back-end programming language" {
		t.Fatalf("fanout queries = %v, want back-end programming language in slot 1", got)
	}
}

func TestBuildBM25FanoutQueries_AddsActionDateProbeForWhenDidISubmit(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"When did I submit my research paper on sentiment analysis?",
		"",
	)

	if len(got) != 2 {
		t.Fatalf("fanout queries = %v, want submission-date probe", got)
	}
	if got[0] != "sentiment analysis submission date" {
		t.Fatalf("fanout queries = %v, want focused submission probe in slot 1", got)
	}
	if got[1] != "research paper submission date" {
		t.Fatalf("fanout queries = %v, want research-paper submission probe in slot 2", got)
	}
}

func TestBuildBM25FanoutQueries_AddsInspirationSourceProbe(t *testing.T) {
	t.Parallel()

	got := buildBM25FanoutQueries(
		"How can I find new inspiration for my paintings?",
		"",
	)

	if len(got) < 2 {
		t.Fatalf("fanout queries = %v, want inspiration-source probe", got)
	}
	if got[0] != "paintings social media tutorials" {
		t.Fatalf("fanout queries = %v, want paintings social media tutorials in slot 1", got)
	}
}

func TestRetrieve_BM25Fanout_UsesPhraseProbesForCompoundTotals(t *testing.T) {
	t.Parallel()

	src := newFakeSource(newTestCorpus())
	src.bm25Override = func(expr string) ([]BM25Hit, bool) {
		switch {
		case strings.Contains(expr, "designer") && strings.Contains(expr, "handbag") && !strings.Contains(expr, "skincare"):
			return []BM25Hit{{
				ID:      "bag",
				Path:    "raw/lme/designer-handbag.md",
				Title:   "Designer handbag",
				Summary: "High-value purchase",
				Content: "I spent 1800 on the designer handbag.",
			}}, true
		case strings.Contains(expr, "skincare") && strings.Contains(expr, "products"):
			return []BM25Hit{{
				ID:      "skincare",
				Path:    "raw/lme/skincare-products.md",
				Title:   "Skincare products",
				Summary: "Beauty purchase",
				Content: "I spent 320 on the high-end skincare products.",
			}}, true
		case strings.Contains(expr, "handbag") && strings.Contains(expr, "skincare"):
			return nil, true
		default:
			return nil, false
		}
	}

	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	resp, err := r.Retrieve(context.Background(), Request{
		Query: "What is the total amount I spent on the designer handbag and high-end skincare products?",
		TopK:  5,
		Mode:  ModeBM25,
	})
	if err != nil {
		t.Fatalf("Retrieve: %v", err)
	}
	if len(resp.Chunks) < 2 {
		t.Fatalf("expected phrase fanout to merge two hits, got %d", len(resp.Chunks))
	}
	got := []string{resp.Chunks[0].Path, resp.Chunks[1].Path}
	if !containsPath(got, "raw/lme/designer-handbag.md") || !containsPath(got, "raw/lme/skincare-products.md") {
		t.Fatalf("fanout merged paths = %v, want both compound item hits", got)
	}
	if len(resp.Attempts) == 0 {
		t.Fatalf("expected initial attempt trace, got none")
	}
	if !strings.Contains(resp.Attempts[0].Query, "designer") || !strings.Contains(resp.Attempts[0].Query, "skincare") {
		t.Fatalf("initial attempt should record phrase probes, got %+v", resp.Attempts)
	}
}

func containsPath(paths []string, want string) bool {
	for _, path := range paths {
		if path == want {
			return true
		}
	}
	return false
}

// rerankerFn is a function-as-Reranker adapter used by tests.
type rerankerFn func(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error)

func (f rerankerFn) Rerank(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
	return f(ctx, query, chunks)
}

func (f rerankerFn) Name() string { return "test-reranker" }
