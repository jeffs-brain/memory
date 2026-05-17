// SPDX-License-Identifier: Apache-2.0

// Package extract provides pluggable content extractors for the ingest
// pipeline. Each extractor converts a specific media type into plain
// text suitable for chunking and indexing.
package extract

import (
	"context"
	"time"
)

// ExtractionResult holds the output of a successful extraction.
type ExtractionResult struct {
	Content    string
	MIME       string
	Metadata   map[string]any
	Pages      int
	Language   string
	Confidence float64
}

// ExtractorCapability describes the file types an Extractor supports.
type ExtractorCapability struct {
	Extensions     []string
	MIMETypes      []string
	MagicBytes     []MagicSignature
	RequiresBinary bool
}

// MagicSignature identifies a file format by a byte sequence at a
// fixed offset within the first bytes of the file.
type MagicSignature struct {
	Offset int
	Bytes  []byte
}

// Extractor converts raw bytes into searchable plain text. Each
// implementation is optional: Available reports whether the required
// external binaries are present.
type Extractor interface {
	Name() string
	Capability() ExtractorCapability
	Extract(ctx context.Context, input []byte, opts ExtractOptions) (ExtractionResult, error)
	Available(ctx context.Context) (bool, error)
}

// ExtractOptions carries per-invocation hints to extractors.
type ExtractOptions struct {
	Filename string
	MIME     string
	Language string
	Timeout  time.Duration
}
