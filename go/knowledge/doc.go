// SPDX-License-Identifier: Apache-2.0

// Package knowledge provides the minimal ingest, compile, and hybrid
// search surface for Jeffs Brain.
//
// The API exposes a [Base] interface that covers:
//
//   - Ingest: accept a markdown file, plain text file, PDF, or inline
//     reader and persist a Document plus Chunk index entries.
//   - IngestURL: fetch a URL, extract plain text, and ingest it.
//   - Compile: chunk persisted documents and feed them into the bound
//     [*search.Index].
//   - Search: run a hybrid retrieval against the index, delegating to
//     the retrieval package when available and falling back to BM25.
//
// The implementation is deliberately small: it ports the minimum subset
// of jeff/apps/jeff/internal/knowledge required to drive the Go SDK
// end-to-end. Advanced surfaces (wiki compilation, tool integration,
// inbox promotion, OCR, two-phase planners) are intentionally omitted
// and live in the jeff repository.
package knowledge
