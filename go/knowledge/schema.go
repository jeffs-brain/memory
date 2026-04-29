// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"io"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// Document represents a persisted ingest in the brain.
//
// A Document is the top-level record produced by [Base.Ingest] or
// [Base.IngestURL]. Compilation derives zero or more [Chunk]s from the
// stored body and writes them into the search index.
type Document struct {
	ID          string
	Path        brain.Path
	Title       string
	Source      string
	ContentType string
	Tags        []string
	Summary     string
	Body        string
	Bytes       int
	Ingested    time.Time
	Modified    time.Time
}

// Chunk is a single segment of a Document, indexed for retrieval.
//
// Ordinal reflects the ordered position of the chunk within the parent
// document (zero-based). Heading carries the nearest markdown heading
// where available; it is empty for plain-text sources.
type Chunk struct {
	DocumentID string
	Ordinal    int
	Heading    string
	Text       string
	Tokens     int
}

// IngestRequest is the input to [Base.Ingest].
//
// Exactly one of Content or Path must be supplied. When Content is set,
// ContentType is required and Path is treated as a suggested source
// label. When Path points to a local file, Content is ignored and the
// file is read from disk.
type IngestRequest struct {
	BrainID     string
	Path        string
	ContentType string
	Content     io.Reader
	Title       string
	Tags        []string
}

// IngestResponse summarises the result of an ingest call.
type IngestResponse struct {
	DocumentID string
	Path       brain.Path
	ChunkCount int
	Bytes      int
	TookMs     int64
}

// CompileOptions tunes [Base.Compile].
//
// When Paths is empty the whole raw tree is compiled. MaxBatch bounds
// the number of documents compiled per call; zero means unlimited.
// DryRun walks the documents without writing to the index.
type CompileOptions struct {
	Paths    []brain.Path
	MaxBatch int
	DryRun   bool
}

// CompileResult summarises the outcome of a compile run.
type CompileResult struct {
	Compiled int
	Chunks   int
	Skipped  int
	Errors   int
	Elapsed  time.Duration
}

// SearchRequest is the input to [Base.Search].
type SearchRequest struct {
	Query      string
	MaxResults int
	Mode       SearchMode
}

// SearchMode selects which retriever participates in [Base.Search].
//
// The zero value (SearchAuto) uses the hybrid retriever when the
// package has one bound, falling back to BM25 otherwise.
type SearchMode int

const (
	// SearchAuto defers to the bound retriever, otherwise BM25.
	SearchAuto SearchMode = iota
	// SearchBM25 forces the BM25 FTS5 path via the bound index.
	SearchBM25
	// SearchHybrid forces the hybrid retrieval path; falls back to
	// BM25 when no retriever is bound.
	SearchHybrid
)

// SearchResponse is the output of [Base.Search].
type SearchResponse struct {
	Hits     []SearchHit
	Mode     string
	Elapsed  time.Duration
	FellBack bool
}

// SearchHit is a single retrieval hit.
type SearchHit struct {
	Path     brain.Path
	Title    string
	Summary  string
	Snippet  string
	Score    float64
	Modified time.Time
	// Source names which retriever surfaced this hit: "bm25",
	// "vector", "fused", or "memory" for the in-memory fallback.
	Source string
}
