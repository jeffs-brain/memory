// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"strings"
)

// Compile-time interface compliance check.
var _ Chunker = (*RecursiveChunker)(nil)

// separators is the hierarchy of split points used by RecursiveChunker.
// Tried in order: paragraph breaks, single newlines, sentence endings,
// word boundaries, and finally character-level as a last resort.
var separators = []string{"\n\n", "\n", ". ", " ", ""}

// RecursiveChunker is the default chunking strategy. It recursively
// splits content using a separator hierarchy until each piece fits
// within the configured maxTokens budget. Overlap is applied by copying
// overlapTokens worth of trailing text from the previous chunk to the
// start of the next.
type RecursiveChunker struct{}

func (rc *RecursiveChunker) ContentTypes() []string {
	return []string{"text/plain", "application/octet-stream"}
}

func (rc *RecursiveChunker) Name() string { return "recursive" }

func (rc *RecursiveChunker) Chunk(ctx context.Context, content string, cfg ChunkConfig) ([]Chunk, error) {
	if strings.TrimSpace(content) == "" {
		return nil, nil
	}
	pieces := recursiveSplit(content, cfg.MaxTokens, 0)
	chunks := applyOverlapAndBuild(pieces, cfg)
	return chunks, nil
}

// recursiveSplit breaks text into pieces that each fit within maxTokens.
// It walks the separator hierarchy from coarsest to finest until the
// pieces are small enough.
func recursiveSplit(text string, maxTokens int, sepIdx int) []string {
	if estimateTokens(text) <= maxTokens {
		return []string{text}
	}
	if sepIdx >= len(separators) {
		return hardSplit(text, maxTokens)
	}
	sep := separators[sepIdx]
	if sep == "" {
		return hardSplit(text, maxTokens)
	}
	parts := strings.Split(text, sep)
	var merged []string
	var current strings.Builder
	for _, part := range parts {
		candidate := buildCandidate(current.String(), part, sep)
		if estimateTokens(candidate) > maxTokens {
			if current.Len() > 0 {
				merged = append(merged, current.String())
				current.Reset()
			}
			if estimateTokens(part) > maxTokens {
				sub := recursiveSplit(part, maxTokens, sepIdx+1)
				merged = append(merged, sub...)
			} else {
				current.WriteString(part)
			}
		} else {
			if current.Len() > 0 {
				current.WriteString(sep)
			}
			current.WriteString(part)
		}
	}
	if current.Len() > 0 {
		merged = append(merged, current.String())
	}
	return merged
}

// buildCandidate constructs the text that would result from appending
// part to existing with the separator between them.
func buildCandidate(existing, part, sep string) string {
	if existing == "" {
		return part
	}
	return existing + sep + part
}

// hardSplit divides text into maxTokens-sized character windows as a
// last resort when no separator hierarchy can break it further.
func hardSplit(text string, maxTokens int) []string {
	step := maxTokens * 4
	if step <= 0 {
		step = 1
	}
	var out []string
	for i := 0; i < len(text); i += step {
		end := i + step
		if end > len(text) {
			end = len(text)
		}
		out = append(out, text[i:end])
	}
	return out
}

// applyOverlapAndBuild takes raw split pieces, applies the overlap
// strategy, merges undersized chunks, and returns the final Chunk slice.
func applyOverlapAndBuild(pieces []string, cfg ChunkConfig) []Chunk {
	if len(pieces) == 0 {
		return nil
	}
	chunks := make([]Chunk, 0, len(pieces))
	var prevTail string
	for i, piece := range pieces {
		trimmed := strings.TrimSpace(piece)
		if trimmed == "" {
			continue
		}
		chunkContent := trimmed
		if i > 0 && cfg.OverlapTokens > 0 && prevTail != "" {
			chunkContent = prevTail + "\n" + trimmed
		}
		// Merge undersized chunks into the previous one.
		if estimateTokens(trimmed) < cfg.MinTokens && len(chunks) > 0 {
			prev := chunks[len(chunks)-1]
			prev.Content = prev.Content + "\n" + trimmed
			chunks[len(chunks)-1] = prev
			prevTail = extractTail(prev.Content, cfg.OverlapTokens)
			continue
		}
		chunks = append(chunks, Chunk{
			Content:  chunkContent,
			Metadata: map[string]string{"chunker": "recursive"},
		})
		prevTail = extractTail(trimmed, cfg.OverlapTokens)
	}
	return chunks
}

// extractTail returns approximately overlapTokens worth of characters
// from the end of text (overlapTokens * 4 chars). Used to build the
// overlap prefix for the next chunk.
func extractTail(text string, overlapTokens int) string {
	chars := overlapTokens * 4
	if chars >= len(text) {
		return text
	}
	return text[len(text)-chars:]
}
