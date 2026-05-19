// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"strings"

	"github.com/jeffs-brain/memory/go/ingest"
)

// ChunkDocument segments a document body into chunks using the given
// config. When cfg is zero-valued, spec defaults are applied via
// [ingest.DefaultChunkConfig].
func ChunkDocument(body string, headings []headingSection, cfg ingest.ChunkConfig) []Chunk {
	if cfg.MaxTokens == 0 {
		cfg = ingest.DefaultChunkConfig()
	}

	body = strings.TrimSpace(body)
	if body == "" {
		return nil
	}

	maxChars := cfg.MaxTokens * 4
	minChars := cfg.MinTokens * 4

	sections := headings
	if len(sections) == 0 {
		sections = splitByHeadings(body)
	}

	out := make([]Chunk, 0, len(sections))
	ordinal := 0
	for _, sec := range sections {
		pieces := splitLong(sec.text, maxChars)
		for _, piece := range pieces {
			piece = strings.TrimSpace(piece)
			if piece == "" {
				continue
			}
			out = append(out, Chunk{
				Heading: sec.heading,
				Text:    piece,
				Tokens:  ingest.EstimateTokens(piece),
			})
			ordinal++
		}
	}

	out = mergeSmallChunksByTokens(out, minChars)

	if cfg.OverlapTokens > 0 {
		out = applyOverlap(out, cfg.OverlapTokens)
	}

	// Recompute ordinals after all transformations.
	for i := range out {
		out[i].Ordinal = i
	}
	return out
}

// mergeSmallChunksByTokens folds chunks shorter than minChars into the
// previous chunk so the index never sees single-line stubs. Uses
// character count (minTokens * 4) for the threshold, matching the token
// estimation formula.
func mergeSmallChunksByTokens(in []Chunk, minChars int) []Chunk {
	if len(in) == 0 {
		return in
	}
	out := make([]Chunk, 0, len(in))
	for _, c := range in {
		if len(out) > 0 && len(c.Text) < minChars {
			last := &out[len(out)-1]
			last.Text = last.Text + "\n\n" + c.Text
			last.Tokens = ingest.EstimateTokens(last.Text)
			continue
		}
		out = append(out, c)
	}
	return out
}

// applyOverlap adds trailing context from the previous chunk as a
// prefix to the next chunk. The overlap is taken from the end of the
// previous chunk's text, sized to approximately overlapTokens tokens
// (overlapTokens * 4 characters). The overlap text is extracted at a
// sentence or whitespace boundary when possible.
func applyOverlap(chunks []Chunk, overlapTokens int) []Chunk {
	if len(chunks) <= 1 || overlapTokens <= 0 {
		return chunks
	}

	overlapChars := overlapTokens * 4
	out := make([]Chunk, len(chunks))
	out[0] = chunks[0]

	for i := 1; i < len(chunks); i++ {
		prev := chunks[i-1].Text
		overlap := extractTrailingOverlap(prev, overlapChars)
		if overlap == "" {
			out[i] = chunks[i]
			continue
		}
		merged := overlap + "\n\n" + chunks[i].Text
		out[i] = Chunk{
			DocumentID: chunks[i].DocumentID,
			Heading:    chunks[i].Heading,
			Text:       merged,
			Tokens:     ingest.EstimateTokens(merged),
		}
	}
	return out
}

// extractTrailingOverlap returns the last ~maxChars characters from
// text, breaking at a sentence boundary or whitespace when possible.
func extractTrailingOverlap(text string, maxChars int) string {
	if len(text) <= maxChars {
		return text
	}

	tail := text[len(text)-maxChars:]

	// Try to break at a sentence boundary within the tail.
	bestCut := -1
	for _, sep := range []string{". ", "! ", "? "} {
		idx := strings.Index(tail, sep)
		if idx >= 0 {
			candidate := idx + len(sep)
			if bestCut < 0 || candidate < bestCut {
				bestCut = candidate
			}
		}
	}
	if bestCut > 0 && bestCut < len(tail) {
		return strings.TrimSpace(tail[bestCut:])
	}

	// Fall back to paragraph or whitespace boundary.
	if idx := strings.Index(tail, "\n\n"); idx >= 0 {
		result := strings.TrimSpace(tail[idx+2:])
		if result != "" {
			return result
		}
	}
	if idx := strings.Index(tail, "\n"); idx >= 0 {
		result := strings.TrimSpace(tail[idx+1:])
		if result != "" {
			return result
		}
	}
	if idx := strings.Index(tail, " "); idx >= 0 {
		result := strings.TrimSpace(tail[idx+1:])
		if result != "" {
			return result
		}
	}

	return strings.TrimSpace(tail)
}
