// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"fmt"
	"log/slog"
)

// MaxContentChars is the default maximum character count passed to the
// extractor from an ingested document.
const MaxContentChars = 128_000

// MemoryExtractor is the subset of the memory interface needed for
// post-ingest extraction.
type MemoryExtractor interface {
	Extract(ctx context.Context, messages []ExtractMessage) ([]ExtractedMemory, error)
}

// ExtractMessage is a synthetic message passed to the extractor.
type ExtractMessage struct {
	Role    string
	Content string
}

// ExtractedMemory represents a single extracted fact.
type ExtractedMemory struct {
	Filename string
	Content  string
}

// ExtractAfterIngestOptions configures post-ingest extraction.
type ExtractAfterIngestOptions struct {
	BrainID         string
	DocumentPath    string
	DocumentContent string
	Extractor       MemoryExtractor
	ActorID         string
	SessionID       string
	MaxContentChars int
	Logger          *slog.Logger
}

// ExtractAfterIngestResult holds the outcome of post-ingest extraction.
type ExtractAfterIngestResult struct {
	FactsExtracted int
	Memories       []ExtractedMemory
}

// ExtractAfterIngest runs the memory extractor on document content after
// a successful ingest. Builds a synthetic user message from the document
// content (truncated to maxContentChars) and passes it to the extractor.
//
// Extraction failure is non-fatal: an empty result is returned and the
// error is swallowed.
func ExtractAfterIngest(ctx context.Context, opts ExtractAfterIngestOptions) (ExtractAfterIngestResult, error) {
	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	maxChars := opts.MaxContentChars
	if maxChars <= 0 {
		maxChars = MaxContentChars
	}

	content := opts.DocumentContent
	if len(content) == 0 {
		return ExtractAfterIngestResult{FactsExtracted: 0}, nil
	}

	// Truncate by rune count to avoid splitting multi-byte characters.
	runes := []rune(content)
	if len(runes) > maxChars {
		runes = runes[:maxChars]
		content = string(runes)
	}

	messages := []ExtractMessage{
		{
			Role: "user",
			Content: fmt.Sprintf(
				"The following document was ingested from %q. Extract any important facts, knowledge, or structured information from it:\n\n<ingested-document>\n%s\n</ingested-document>",
				opts.DocumentPath, content,
			),
		},
	}

	extracted, err := opts.Extractor.Extract(ctx, messages)
	if err != nil {
		logger.Warn("extract-after-ingest: extraction failed", "document", opts.DocumentPath, "error", err)
		return ExtractAfterIngestResult{FactsExtracted: 0}, nil
	}

	return ExtractAfterIngestResult{
		FactsExtracted: len(extracted),
		Memories:       extracted,
	}, nil
}
