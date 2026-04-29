// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
	"gopkg.in/yaml.v3"
)

// goldenFixtureDir locates spec/fixtures/retrieval relative to this
// package. The Go module root is two levels above the retrieval
// package; the spec directory lives alongside the module root inside
// the memory repo so we walk up until we find it.
func goldenFixtureDir(t *testing.T) string {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	dir := cwd
	for i := 0; i < 8; i++ {
		candidate := filepath.Join(dir, "spec", "fixtures", "retrieval")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return candidate
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	t.Skipf("spec/fixtures/retrieval not reachable from %s", cwd)
	return ""
}

type goldenQuery struct {
	ID          string   `yaml:"id"`
	Q           string   `yaml:"q"`
	AnyOf       []string `yaml:"any_of"`
	MustRetrieve []string `yaml:"must_retrieve"`
	Notes       string   `yaml:"notes"`
}

type goldenSet struct {
	Queries []goldenQuery `yaml:"queries"`
}

func loadGoldenSet(t *testing.T, name string) goldenSet {
	t.Helper()
	dir := goldenFixtureDir(t)
	path := filepath.Join(dir, name)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var set goldenSet
	if err := yaml.Unmarshal(raw, &set); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
	return set
}

// goldenCorpus synthesises a small in-mem corpus for a list of
// golden queries. Every expected `any_of` path becomes a chunk with
// hand-crafted title / summary / content derived from the path slug
// so both BM25 and semantic can plausibly surface it. A fixed slate
// of distractor chunks adds realistic noise.
func goldenCorpus(queries []goldenQuery) []fakeChunk {
	chunks := make([]fakeChunk, 0)
	seen := make(map[string]bool)
	for _, q := range queries {
		for _, p := range q.AnyOf {
			if seen[p] {
				continue
			}
			seen[p] = true
			chunks = append(chunks, fakeChunkForPath(p, q.Q))
		}
		for _, p := range q.MustRetrieve {
			if seen[p] {
				continue
			}
			seen[p] = true
			chunks = append(chunks, fakeChunkForPath(p, q.Q))
		}
	}
	// Distractors: same topics, different slugs, different words
	// so unrelated queries have something other than the golden
	// answers to grab.
	distractors := []fakeChunk{
		{ID: "d1", Path: "wiki/holiday-calendar.md", Title: "Holiday calendar", Content: "Public holidays across regions."},
		{ID: "d2", Path: "wiki/office-stationery.md", Title: "Stationery budget", Content: "Pen and paper stock ledger."},
		{ID: "d3", Path: "wiki/hr-handbook.md", Title: "HR handbook", Content: "Policies on annual leave and expenses."},
		{ID: "d4", Path: "wiki/company-wifi.md", Title: "Office wifi", Content: "Joining the guest wifi network."},
	}
	chunks = append(chunks, distractors...)
	return chunks
}

// fakeChunkForPath manufactures a chunk whose title, summary and
// content all echo the path slug and the originating query so BM25
// will find the tokens and semantic cosine similarity stays high.
func fakeChunkForPath(path, query string) fakeChunk {
	slug := slugTextFor(path)
	words := strings.Fields(slug)
	title := strings.Join(words, " ")
	summary := "Reference note about " + strings.Join(words, " ")
	content := summary + ". Related query context: " + query + "."
	return fakeChunk{
		ID:      path,
		Path:    path,
		Title:   title,
		Summary: summary,
		Content: content,
	}
}

// TestGolden_HybridBM25 exercises a subset of golden-hybrid queries
// against ModeBM25. The fixture was captured against a 5K-article
// corpus we cannot redistribute, so we synthesise a minimal corpus
// keyed to the golden expectations and assert the top-5 surfaces at
// least one `any_of` hit (the fixture pass criterion).
func TestGolden_HybridBM25(t *testing.T) {
	t.Parallel()
	set := loadGoldenSet(t, "golden-hybrid.yaml")
	// Focus on queries where BM25 can reasonably surface the
	// expected paths through slug overlap.
	subset := pickQueries(set, []string{"invoice-automation", "quote-generation-tools"})
	corpus := goldenCorpus(subset)
	src := newFakeSource(corpus)
	r, err := New(Config{Source: src})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	passed := 0
	for _, q := range subset {
		resp, err := r.Retrieve(context.Background(), Request{
			Query: q.Q,
			Mode:  ModeBM25,
			TopK:  5,
		})
		if err != nil {
			t.Fatalf("Retrieve %s: %v", q.ID, err)
		}
		if topKContainsAny(resp.Chunks, q.AnyOf) {
			passed++
			continue
		}
		t.Errorf("golden %s: top-5 paths did not include any of %v; got %v", q.ID, q.AnyOf, topPaths(resp.Chunks))
	}
	if passed != len(subset) {
		t.Fatalf("BM25 golden pass %d/%d", passed, len(subset))
	}
}

// TestGolden_HybridMode exercises the same subset against ModeHybrid
// using a fake embedder. The assertion is that hybrid recall is at
// least as good as BM25 on the chosen queries.
func TestGolden_HybridMode(t *testing.T) {
	t.Parallel()
	set := loadGoldenSet(t, "golden-hybrid.yaml")
	subset := pickQueries(set, []string{"invoice-automation", "quote-generation-tools"})
	corpus := goldenCorpus(subset)
	src := newFakeSource(corpus)
	embedder := llm.NewFakeEmbedder(src.embedDim)
	r, err := New(Config{Source: src, Embedder: embedder})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, q := range subset {
		resp, err := r.Retrieve(context.Background(), Request{
			Query: q.Q,
			Mode:  ModeHybrid,
			TopK:  5,
		})
		if err != nil {
			t.Fatalf("Retrieve %s: %v", q.ID, err)
		}
		if !topKContainsAny(resp.Chunks, q.AnyOf) {
			t.Errorf("golden %s (hybrid): top-5 missed %v; got %v", q.ID, q.AnyOf, topPaths(resp.Chunks))
		}
	}
}

// TestGolden_HybridRerank asserts the rerank pass runs without
// regressing recall against the same golden subset.
func TestGolden_HybridRerank(t *testing.T) {
	t.Parallel()
	set := loadGoldenSet(t, "golden-hybrid.yaml")
	subset := pickQueries(set, []string{"invoice-automation"})
	corpus := goldenCorpus(subset)
	src := newFakeSource(corpus)
	embedder := llm.NewFakeEmbedder(src.embedDim)
	// Simple reranker that returns the inputs unchanged so top-K
	// parity with hybrid is preserved. Real rerankers re-score the
	// head; we only need to prove the wiring is intact.
	rr := rerankerFn(func(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error) {
		return chunks, nil
	})
	r, err := New(Config{Source: src, Embedder: embedder, Reranker: rr})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	for _, q := range subset {
		resp, err := r.Retrieve(context.Background(), Request{
			Query: q.Q,
			Mode:  ModeHybridRerank,
			TopK:  5,
		})
		if err != nil {
			t.Fatalf("Retrieve %s: %v", q.ID, err)
		}
		if !topKContainsAny(resp.Chunks, q.AnyOf) {
			t.Errorf("golden %s (hybrid-rerank): top-5 missed %v; got %v", q.ID, q.AnyOf, topPaths(resp.Chunks))
		}
		if !resp.Trace.Reranked && !resp.Trace.UnanimitySkipped {
			t.Errorf("golden %s: rerank did not fire and unanimity was not flagged; trace %+v", q.ID, resp.Trace)
		}
	}
}

func pickQueries(set goldenSet, ids []string) []goldenQuery {
	out := make([]goldenQuery, 0, len(ids))
	idx := make(map[string]goldenQuery, len(set.Queries))
	for _, q := range set.Queries {
		idx[q.ID] = q
	}
	for _, id := range ids {
		if q, ok := idx[id]; ok {
			out = append(out, q)
		}
	}
	return out
}

func topKContainsAny(chunks []RetrievedChunk, wanted []string) bool {
	want := make(map[string]bool, len(wanted))
	for _, w := range wanted {
		want[w] = true
	}
	for _, c := range chunks {
		if want[c.Path] {
			return true
		}
	}
	return false
}

func topPaths(chunks []RetrievedChunk) []string {
	out := make([]string, 0, len(chunks))
	for _, c := range chunks {
		out = append(out, c.Path)
	}
	return out
}
