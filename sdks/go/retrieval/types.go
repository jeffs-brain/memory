// SPDX-License-Identifier: Apache-2.0

package retrieval

import (
	"context"
	"encoding/json"
	"strings"
)

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
// prefix. Paths is an exact allow-list. Tags are matched against the
// FTS tags column; every tag must be present for a hit to survive.
type Filters struct {
	PathPrefix string   `json:"pathPrefix,omitempty"`
	Paths      []string `json:"paths,omitempty"`
	Tags       []string `json:"tags,omitempty"`
	Scope      string   `json:"scope,omitempty"`
	Project    string   `json:"project,omitempty"`
}

// HasAny reports whether any field carries a non-zero filter.
func (f Filters) HasAny() bool {
	return strings.TrimSpace(f.PathPrefix) != "" ||
		len(normalisePathList(f.Paths)) > 0 ||
		len(f.Tags) > 0 ||
		strings.TrimSpace(f.Scope) != "" ||
		strings.TrimSpace(f.Project) != ""
}

// MatchesPath reports whether path survives the exact-path and prefix
// filters carried in f.
func (f Filters) MatchesPath(path string) bool {
	pathPrefix := strings.TrimSpace(f.PathPrefix)
	if pathPrefix != "" && !strings.HasPrefix(path, pathPrefix) {
		return false
	}
	paths := normalisePathList(f.Paths)
	if len(paths) == 0 {
		return true
	}
	for _, candidate := range paths {
		if candidate == path {
			return true
		}
	}
	return false
}

// UnmarshalJSON accepts the canonical camelCase wire keys plus the
// request aliases used by the other SDKs.
func (f *Filters) UnmarshalJSON(data []byte) error {
	if strings.TrimSpace(string(data)) == "" || strings.TrimSpace(string(data)) == "null" {
		*f = Filters{}
		return nil
	}
	type rawFilters struct {
		PathPrefix       string   `json:"pathPrefix"`
		PathPrefixSnake  string   `json:"path_prefix"`
		Paths            []string `json:"paths"`
		DocumentPaths    []string `json:"documentPaths"`
		DocumentPathsAlt []string `json:"document_paths"`
		Tags             []string `json:"tags"`
		Scope            string   `json:"scope"`
		Project          string   `json:"project"`
	}
	var raw rawFilters
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*f = Filters{
		PathPrefix: firstNonEmptyString(raw.PathPrefix, raw.PathPrefixSnake),
		Paths:      firstNonEmptyPathList(raw.Paths, raw.DocumentPaths, raw.DocumentPathsAlt),
		Tags:       raw.Tags,
		Scope:      raw.Scope,
		Project:    raw.Project,
	}
	return nil
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func firstNonEmptyPathList(candidates ...[]string) []string {
	for _, values := range candidates {
		if paths := normalisePathList(values); len(paths) > 0 {
			return paths
		}
	}
	return nil
}

func normalisePathList(paths []string) []string {
	if len(paths) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(paths))
	out := make([]string, 0, len(paths))
	for _, value := range paths {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" || seen[trimmed] {
			continue
		}
		seen[trimmed] = true
		out = append(out, trimmed)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// Request drives a single retrieval call.
type Request struct {
	Query           string
	QuestionDate    string
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
	ChunkID          string         `json:"chunkId"`
	DocumentID       string         `json:"documentId"`
	Path             string         `json:"path"`
	Score            float64        `json:"score"`
	Text             string         `json:"text"`
	Title            string         `json:"title"`
	Summary          string         `json:"summary"`
	Metadata         map[string]any `json:"metadata,omitempty"`
	BM25Rank         int            `json:"bm25Rank,omitempty"`
	VectorSimilarity float64        `json:"vectorSimilarity,omitempty"`
	RerankScore      float64        `json:"rerankScore,omitempty"`
}

// Attempt records one rung of the retry ladder. The retrieval
// pipeline emits one entry per rung it actually ran, with the hit
// count it produced. Silently-skipped rungs are omitted so the
// attempt log reflects what was tried.
type Attempt struct {
	Rung   int    `json:"rung"`
	Mode   Mode   `json:"mode"`
	TopK   int    `json:"topK"`
	Reason string `json:"reason"`
	Chunks int    `json:"chunks"`
	Query  string `json:"query"`
}

// Trace records every decision the pipeline made. Consumers should
// surface this to eval harnesses and --explain style reports.
type Trace struct {
	RequestedMode    Mode   `json:"requestedMode"`
	EffectiveMode    Mode   `json:"effectiveMode"`
	Intent           string `json:"intent"`
	UsedRetry        bool   `json:"usedRetry"`
	RRFK             int    `json:"rrfK"`
	CandidateK       int    `json:"candidateK"`
	RerankTopN       int    `json:"rerankTopN"`
	FellBackToBM25   bool   `json:"fellBackToBM25"`
	EmbedderUsed     bool   `json:"embedderUsed"`
	Reranked         bool   `json:"reranked"`
	RerankProvider   string `json:"rerankProvider,omitempty"`
	RerankSkipReason string `json:"rerankSkipReason,omitempty"`
	BM25Hits         int    `json:"bm25Hits"`
	VectorHits       int    `json:"vectorHits"`
	FusedHits        int    `json:"fusedHits"`
	Agreements       int    `json:"agreements"`
	UnanimitySkipped bool   `json:"unanimitySkipped"`
}

// Response bundles the ranked hits with the trace and attempt log.
type Response struct {
	Chunks   []RetrievedChunk `json:"chunks"`
	TookMs   int              `json:"tookMs"`
	Trace    Trace            `json:"trace"`
	Attempts []Attempt        `json:"attempts,omitempty"`
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
