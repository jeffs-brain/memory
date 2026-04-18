// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"strings"

	"github.com/jeffs-brain/memory/go/search"
)

// BM25Hit is one candidate emitted by the BM25 leg. The rank is
// 0-indexed and carries through into the fusion bucket.
type BM25Hit struct {
	ID      string
	Path    string
	Title   string
	Summary string
	Content string
	Score   float64
}

// VectorHit is one candidate emitted by the vector leg.
type VectorHit struct {
	ID         string
	Path       string
	Title      string
	Summary    string
	Content    string
	Similarity float64
}

// Source is the retrieval layer's view of an index. Production
// callers plug in [search.Index] via [NewIndexSource]; tests use an
// in-memory fake to drive deterministic retrieval behaviour without
// standing up FTS5.
//
// Implementations must be safe to call from multiple goroutines and
// must not mutate the returned slices.
type Source interface {
	// SearchBM25 returns up to k candidates ordered by BM25 rank
	// (lower is better in FTS5; the retrieval layer normalises to
	// ascending rank before fusion). Returning nil is equivalent to
	// zero hits.
	SearchBM25(ctx context.Context, expr string, k int, filters Filters) ([]BM25Hit, error)
	// SearchVector returns up to k candidates ordered by cosine
	// similarity descending. The caller supplies a query embedding;
	// implementations that do not maintain vectors should return
	// nil, nil.
	SearchVector(ctx context.Context, embedding []float32, k int, filters Filters) ([]VectorHit, error)
	// Chunks returns a snapshot of every chunk the source can
	// surface. Used by the trigram fallback so the retry ladder can
	// build a lazy slug index without touching storage directly.
	Chunks(ctx context.Context) ([]trigramChunk, error)
}

// compileToFTS is a lightweight wrapper around [search.BuildFTS5Expr]
// applied to the output of [search.ParseQuery]. The retrieval layer
// calls this before each BM25 leg so the expression is identical to
// what the FTS5 index expects. An empty input returns an empty
// string so callers can short-circuit.
func compileToFTS(q string) string {
	trimmed := strings.TrimSpace(q)
	if trimmed == "" {
		return ""
	}
	tokens := search.ParseQuery(trimmed)
	if len(tokens) == 0 {
		return ""
	}
	return search.BuildFTS5Expr(tokens)
}
