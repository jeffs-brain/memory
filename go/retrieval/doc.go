// SPDX-License-Identifier: Apache-2.0

// Package retrieval composes BM25 and vector search into a hybrid
// pipeline with Reciprocal Rank Fusion, an optional cross-encoder
// rerank pass and the five-rung `forceRefreshIndex` retry ladder from
// spec/ALGORITHMS.md.
//
// The pipeline is the Go port of the TypeScript reference in
// packages/memory/src/retrieval and the Go reference buried inside
// jeff's knowledge package. Only the retrieval-specific parts are
// included here; ingestion, compilation and linting continue to live
// in their own packages.
//
// Intent-aware reweighting is English-only. Every trigger pattern is
// hand-tuned against English surface forms (regex source copied bit
// for bit from the spec), so non-English queries silently bypass the
// reweighter and receive the base RRF scores without modification.
// Additional locales are out of scope for v1.0.
package retrieval
