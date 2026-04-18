// SPDX-License-Identifier: Apache-2.0

package retrieval

import "context"

// Mode selects which retrievers participate in a hybrid search. The
// public constants mirror the spec's HybridMode enumeration so a Go
// caller can use the same strings that appear in traces and eval
// reports.
type Mode string

const (
	// ModeAuto picks hybrid when an embedder is supplied, otherwise
	// falls back to BM25 only. Traces mark the fallback via
	// Trace.FellBackToBM25.
	ModeAuto Mode = "auto"
	// ModeBM25 runs BM25 only, even if an embedder is configured.
	ModeBM25 Mode = "bm25"
	// ModeSemantic runs vector search only.
	ModeSemantic Mode = "semantic"
	// ModeHybrid runs BM25 and vector search in parallel and fuses
	// the two ranked lists via Reciprocal Rank Fusion with k = 60.
	ModeHybrid Mode = "hybrid"
	// ModeHybridRerank forces the hybrid pipeline plus the cross
	// encoder rerank pass. Behaviour matches ModeHybrid when no
	// reranker is configured (the pipeline records the reason).
	ModeHybridRerank Mode = "hybrid-rerank"
)

// Filters narrows retrieval to a subset of the corpus. Empty fields
// are treated as no filter. PathPrefix is inclusive of the exact
// prefix. Tags are matched against the FTS tags column; every tag
// must be present for a hit to survive.
type Filters struct {
	PathPrefix string
	Tags       []string
	Scope      string
	Project    string
}

// HasAny reports whether any field carries a non-zero filter.
func (f Filters) HasAny() bool {
	return f.PathPrefix != "" || len(f.Tags) > 0 || f.Scope != "" || f.Project != ""
}

// Request drives a single retrieval call.
type Request struct {
	Query           string
	TopK            int
	Mode            Mode
	BrainID         string
	Filters         Filters
	CandidateK      int
	RerankTopN      int
	SkipRetryLadder bool
}

// RetrievedChunk is a single ranked hit.
type RetrievedChunk struct {
	ChunkID          string
	DocumentID       string
	Path             string
	Score            float64
	Text             string
	Title            string
	Summary          string
	Metadata         map[string]any
	BM25Rank         int
	VectorSimilarity float64
	RerankScore      float64
}

// Attempt records one rung of the retry ladder. The retrieval
// pipeline emits one entry per rung it actually ran, with the hit
// count it produced. Silently-skipped rungs are omitted so the
// attempt log reflects what was tried.
type Attempt struct {
	Rung   int
	Mode   Mode
	TopK   int
	Reason string
	Chunks int
	Query  string
}

// Trace records every decision the pipeline made. Consumers should
// surface this to eval harnesses and --explain style reports.
type Trace struct {
	RequestedMode    Mode
	EffectiveMode    Mode
	Intent           string
	UsedRetry        bool
	RRFK             int
	CandidateK       int
	RerankTopN       int
	FellBackToBM25   bool
	EmbedderUsed     bool
	Reranked         bool
	RerankProvider   string
	RerankSkipReason string
	BM25Hits         int
	VectorHits       int
	FusedHits        int
	Agreements       int
	UnanimitySkipped bool
}

// Response bundles the ranked hits with the trace and attempt log.
type Response struct {
	Chunks   []RetrievedChunk
	TookMs   int
	Trace    Trace
	Attempts []Attempt
}

// Retriever is the hybrid retrieval surface. Implementations are
// concurrency-safe: the returned Response owns no shared state.
type Retriever interface {
	Retrieve(ctx context.Context, req Request) (Response, error)
}

// Reranker is the pluggable cross-encoder surface. Implementations
// must preserve the input order of any chunk they choose not to
// rescore and must return a slice of the same length as the input.
type Reranker interface {
	Rerank(ctx context.Context, query string, chunks []RetrievedChunk) ([]RetrievedChunk, error)
}
