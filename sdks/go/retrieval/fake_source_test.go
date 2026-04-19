// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"math"
	"sort"
	"strings"

	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/search"
)

// fakeChunk is the seed shape for the in-mem fake source used by the
// retrieval unit tests. Every field is optional; unset fields default
// to the empty string so callers can keep fixtures compact.
type fakeChunk struct {
	ID      string
	Path    string
	Title   string
	Summary string
	Content string
	Tags    []string
	Scope   string
	Project string
	Session string
}

// fakeSource implements [Source] over an in-mem slice. BM25 ranking
// is a deterministic token-match count over `title + summary + path
// + content`. Vector ranking uses the deterministic fake embedder
// from the llm package so semantic tests stay reproducible without
// a real model.
type fakeSource struct {
	chunks []fakeChunk
	// embedDim must match the embedder used by the retriever under
	// test.
	embedDim int
	// bm25Fail, when non-nil, is returned by every SearchBM25 call.
	bm25Fail error
	// vectorFail, when non-nil, is returned by every SearchVector
	// call.
	vectorFail error
	// bm25Override, when non-nil, replaces the computed BM25 list
	// for the supplied expression. Used by retry ladder tests to
	// force zero hits on the initial call while allowing later
	// rungs through.
	bm25Override func(expr string) ([]BM25Hit, bool)
}

func newFakeSource(chunks []fakeChunk) *fakeSource {
	out := make([]fakeChunk, len(chunks))
	copy(out, chunks)
	return &fakeSource{chunks: out, embedDim: 16}
}

func (f *fakeSource) SearchBM25(ctx context.Context, expr string, k int, filters Filters) ([]BM25Hit, error) {
	if f.bm25Fail != nil {
		return nil, f.bm25Fail
	}
	if f.bm25Override != nil {
		if hits, ok := f.bm25Override(expr); ok {
			return hits, nil
		}
	}
	tokens := tokeniseFakeExpr(expr)
	if len(tokens) == 0 {
		return nil, nil
	}

	type scored struct {
		chunk fakeChunk
		score int
	}
	scoredHits := make([]scored, 0, len(f.chunks))
	for _, c := range f.chunks {
		if !matchesFakeFilter(c, filters) {
			continue
		}
		corpus := strings.ToLower(strings.Join([]string{c.Path, c.Title, c.Summary, c.Content}, " \n "))
		score := 0
		for _, t := range tokens {
			score += strings.Count(corpus, t)
		}
		if score == 0 {
			continue
		}
		scoredHits = append(scoredHits, scored{chunk: c, score: score})
	}

	sort.SliceStable(scoredHits, func(i, j int) bool {
		if scoredHits[i].score != scoredHits[j].score {
			return scoredHits[i].score > scoredHits[j].score
		}
		return scoredHits[i].chunk.Path < scoredHits[j].chunk.Path
	})

	if k > 0 && len(scoredHits) > k {
		scoredHits = scoredHits[:k]
	}
	hits := make([]BM25Hit, 0, len(scoredHits))
	for _, s := range scoredHits {
		hits = append(hits, BM25Hit{
			ID:      s.chunk.ID,
			Path:    s.chunk.Path,
			Title:   s.chunk.Title,
			Summary: s.chunk.Summary,
			Content: s.chunk.Content,
			Score:   float64(s.score),
		})
	}
	return hits, nil
}

func (f *fakeSource) SearchVector(ctx context.Context, embedding []float32, k int, filters Filters) ([]VectorHit, error) {
	if f.vectorFail != nil {
		return nil, f.vectorFail
	}
	if len(embedding) == 0 {
		return nil, nil
	}
	embedder := llm.NewFakeEmbedder(f.embedDim)
	type scored struct {
		chunk      fakeChunk
		similarity float64
	}
	scoredHits := make([]scored, 0, len(f.chunks))
	for _, c := range f.chunks {
		if !matchesFakeFilter(c, filters) {
			continue
		}
		// Seed the chunk vector with its content so similar text
		// produces similar vectors under the fake embedder. Empty
		// content falls back to the path to avoid zero vectors.
		seed := strings.TrimSpace(strings.Join([]string{c.Title, c.Summary, c.Content}, " "))
		if seed == "" {
			seed = c.Path
		}
		vecs, err := embedder.Embed(ctx, []string{seed})
		if err != nil {
			return nil, err
		}
		if len(vecs) == 0 {
			continue
		}
		scoredHits = append(scoredHits, scored{chunk: c, similarity: cosine(embedding, vecs[0])})
	}
	sort.SliceStable(scoredHits, func(i, j int) bool {
		if scoredHits[i].similarity != scoredHits[j].similarity {
			return scoredHits[i].similarity > scoredHits[j].similarity
		}
		return scoredHits[i].chunk.Path < scoredHits[j].chunk.Path
	})
	if k > 0 && len(scoredHits) > k {
		scoredHits = scoredHits[:k]
	}
	hits := make([]VectorHit, 0, len(scoredHits))
	for _, s := range scoredHits {
		hits = append(hits, VectorHit{
			ID:         s.chunk.ID,
			Path:       s.chunk.Path,
			Title:      s.chunk.Title,
			Summary:    s.chunk.Summary,
			Content:    s.chunk.Content,
			Similarity: s.similarity,
		})
	}
	return hits, nil
}

func (f *fakeSource) Chunks(ctx context.Context) ([]trigramChunk, error) {
	out := make([]trigramChunk, 0, len(f.chunks))
	for _, c := range f.chunks {
		out = append(out, trigramChunk{
			ID:      chunkID(c),
			Path:    c.Path,
			Title:   c.Title,
			Summary: c.Summary,
			Content: c.Content,
		})
	}
	return out, nil
}

func (f *fakeSource) Lookup(ctx context.Context, ids []string) ([]search.IndexedRow, error) {
	rows := make([]search.IndexedRow, 0, len(ids))
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		for _, chunk := range f.chunks {
			if chunk.Path != id {
				continue
			}
			rows = append(rows, search.IndexedRow{
				Path:        chunk.Path,
				Title:       chunk.Title,
				Summary:     chunk.Summary,
				Content:     chunk.Content,
				Tags:        strings.Join(chunk.Tags, " "),
				Scope:       chunk.Scope,
				ProjectSlug: chunk.Project,
				SessionDate: chunk.Session,
			})
			break
		}
	}
	return rows, nil
}

// matchesFakeFilter mirrors the contract of real Source filters. A
// zero Filters value matches every chunk.
func matchesFakeFilter(c fakeChunk, f Filters) bool {
	if f.PathPrefix != "" && !strings.HasPrefix(c.Path, f.PathPrefix) {
		return false
	}
	if f.Scope != "" && c.Scope != f.Scope {
		return false
	}
	if len(f.Tags) > 0 {
		tagSet := make(map[string]bool, len(c.Tags))
		for _, t := range c.Tags {
			tagSet[t] = true
		}
		for _, want := range f.Tags {
			if !tagSet[want] {
				return false
			}
		}
	}
	return true
}

func chunkID(c fakeChunk) string {
	if c.ID != "" {
		return c.ID
	}
	return c.Path
}

// tokeniseFakeExpr strips FTS5 operators and returns the lowercase
// wordset embedded in an FTS5 MATCH expression. Good enough for
// token-count BM25 in the fake.
func tokeniseFakeExpr(expr string) []string {
	expr = strings.ToLower(expr)
	replacer := strings.NewReplacer(
		"(", " ", ")", " ", `"`, " ", "*", " ", "^", " ", ":", " ",
	)
	cleaned := replacer.Replace(expr)
	out := make([]string, 0)
	for _, tok := range strings.Fields(cleaned) {
		switch tok {
		case "and", "or", "not":
			continue
		}
		if tok == "" {
			continue
		}
		out = append(out, tok)
	}
	return out
}

func cosine(a []float32, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, aa, bb float64
	for i := range a {
		af := float64(a[i])
		bf := float64(b[i])
		dot += af * bf
		aa += af * af
		bb += bf * bf
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / (math.Sqrt(aa) * math.Sqrt(bb))
}
