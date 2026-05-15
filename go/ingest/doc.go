// SPDX-License-Identifier: Apache-2.0

// Package ingest provides configurable chunking primitives for the
// ingestion pipeline. It defines [ChunkConfig] — the canonical
// configuration type shared across Go and TypeScript SDKs — and the
// token estimation function that both SDKs implement identically.
//
// This package does NOT perform actual chunking (that remains in the
// knowledge package for now). It provides the configuration types and
// validation that the knowledge package will adopt in P1-4.
//
// The canonical parameters are documented in spec/INGESTION.md.
package ingest
