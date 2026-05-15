// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"fmt"
	"strings"
)

// Compile-time interface compliance check.
var _ Chunker = (*PageLevelChunker)(nil)

// pageLevelContentTypes lists the MIME types handled by PageLevelChunker.
var pageLevelContentTypes = []string{
	"application/pdf",
	"text/x-pdf-text",
}

// PageLevelChunker splits content at page boundaries indicated by the
// form-feed character (\f). PDF text extractors typically insert \f
// between pages. Pages that exceed maxTokens are split further using
// the recursive separator hierarchy.
type PageLevelChunker struct{}

func (pc *PageLevelChunker) ContentTypes() []string { return pageLevelContentTypes }
func (pc *PageLevelChunker) Name() string           { return "page_level" }

func (pc *PageLevelChunker) Chunk(_ context.Context, content string, cfg ChunkConfig) ([]Chunk, error) {
	if strings.TrimSpace(content) == "" {
		return nil, nil
	}

	pages := strings.Split(content, "\f")
	chunks := make([]Chunk, 0, len(pages))

	for i, page := range pages {
		trimmed := strings.TrimSpace(page)
		if trimmed == "" {
			continue
		}

		pageNum := fmt.Sprintf("%d", i+1)

		if estimateTokens(trimmed) <= cfg.MaxTokens() {
			chunks = append(chunks, Chunk{
				Content: trimmed,
				Metadata: map[string]string{
					"chunker": "page_level",
					"page":    pageNum,
				},
			})
			continue
		}

		// Page too large: recursively split.
		subPieces := recursiveSplit(trimmed, cfg.MaxTokens(), 0)
		for _, piece := range subPieces {
			t := strings.TrimSpace(piece)
			if t == "" {
				continue
			}
			chunks = append(chunks, Chunk{
				Content: t,
				Metadata: map[string]string{
					"chunker": "page_level",
					"page":    pageNum,
				},
			})
		}
	}
	return chunks, nil
}
