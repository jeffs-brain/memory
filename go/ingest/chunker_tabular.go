// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"strings"
)

// Compile-time interface compliance check.
var _ Chunker = (*TabularChunker)(nil)

// tabularChunkerContentTypes are the MIME types handled by TabularChunker.
var tabularChunkerContentTypes = []string{
	"text/csv",
	"text/tab-separated-values",
	"text/tsv",
}

// defaultRowsPerChunk is the number of data rows per chunk when the
// token budget does not impose a tighter limit.
const defaultRowsPerChunk = 50

// TabularChunker splits CSV/TSV content by rows, prepending the header
// row to each chunk so every chunk is self-contained. The delimiter is
// auto-detected from the first line (comma, tab, or pipe).
type TabularChunker struct{}

func (tc *TabularChunker) ContentTypes() []string { return tabularChunkerContentTypes }
func (tc *TabularChunker) Name() string           { return "tabular" }

func (tc *TabularChunker) Chunk(_ context.Context, content string, cfg ChunkConfig) ([]Chunk, error) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil, nil
	}

	lines := strings.Split(trimmed, "\n")
	if len(lines) == 0 {
		return nil, nil
	}

	header := lines[0]
	dataLines := lines[1:]

	if len(dataLines) == 0 {
		return []Chunk{{
			Content:  header,
			Metadata: map[string]string{"chunker": "tabular"},
		}}, nil
	}

	rowsPerChunk := computeRowsPerChunk(header, dataLines, cfg)

	chunks := make([]Chunk, 0, (len(dataLines)/rowsPerChunk)+1)
	for start := 0; start < len(dataLines); start += rowsPerChunk {
		end := start + rowsPerChunk
		if end > len(dataLines) {
			end = len(dataLines)
		}
		batch := dataLines[start:end]
		chunkContent := header + "\n" + strings.Join(batch, "\n")
		chunks = append(chunks, Chunk{
			Content:  chunkContent,
			Metadata: map[string]string{"chunker": "tabular"},
		})
	}
	return chunks, nil
}

// computeRowsPerChunk determines how many rows fit within the token
// budget. Uses the average row size to estimate, capped at
// defaultRowsPerChunk.
func computeRowsPerChunk(header string, dataLines []string, cfg ChunkConfig) int {
	headerTokens := estimateTokens(header)
	budgetForRows := cfg.MaxTokens() - headerTokens - 1

	if budgetForRows <= 0 || len(dataLines) == 0 {
		return defaultRowsPerChunk
	}

	// Sample up to 10 rows for average size.
	sampleSize := len(dataLines)
	if sampleSize > 10 {
		sampleSize = 10
	}
	totalTokens := 0
	for i := 0; i < sampleSize; i++ {
		totalTokens += estimateTokens(dataLines[i])
	}
	avgTokensPerRow := totalTokens / sampleSize
	if avgTokensPerRow <= 0 {
		avgTokensPerRow = 1
	}

	rows := budgetForRows / avgTokensPerRow
	if rows <= 0 {
		rows = 1
	}
	if rows > defaultRowsPerChunk {
		rows = defaultRowsPerChunk
	}
	return rows
}
