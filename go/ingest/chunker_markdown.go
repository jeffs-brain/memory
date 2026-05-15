// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"regexp"
	"strings"
)

// Compile-time interface compliance check.
var _ Chunker = (*MarkdownChunker)(nil)

// atxHeadingRe matches ATX-style markdown headings (# through ######).
var atxHeadingRe = regexp.MustCompile(`^(#{1,6})\s+(.+?)\s*#*\s*$`)

// MarkdownChunker implements heading-aware splitting. It identifies
// heading boundaries, tracks the heading hierarchy, and produces chunks
// whose metadata carries the full heading path. Sections that exceed
// maxTokens are split recursively using the separator hierarchy.
type MarkdownChunker struct{}

func (mc *MarkdownChunker) ContentTypes() []string {
	return []string{"text/markdown", "text/x-markdown"}
}

func (mc *MarkdownChunker) Name() string { return "markdown" }

func (mc *MarkdownChunker) Chunk(ctx context.Context, content string, cfg ChunkConfig) ([]Chunk, error) {
	if strings.TrimSpace(content) == "" {
		return nil, nil
	}
	sections := splitMarkdownSections(content)
	chunks := make([]Chunk, 0, len(sections))
	for _, sec := range sections {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		trimmed := strings.TrimSpace(sec.content)
		if trimmed == "" {
			continue
		}
		headingPath := strings.Join(sec.headingPath, " > ")
		tokens := estimateTokens(trimmed)
		if tokens <= cfg.MaxTokens() {
			chunks = append(chunks, Chunk{
				Content: trimmed,
				Metadata: map[string]string{
					"chunker":     "markdown",
					"headingPath": headingPath,
				},
			})
			continue
		}
		subChunks := splitSectionWithOverlap(trimmed, cfg)
		for _, sub := range subChunks {
			chunks = append(chunks, Chunk{
				Content: sub,
				Metadata: map[string]string{
					"chunker":     "markdown",
					"headingPath": headingPath,
				},
			})
		}
	}
	return chunks, nil
}

// mdSection holds a parsed section with its heading hierarchy and text.
type mdSection struct {
	headingPath []string
	content     string
}

// splitMarkdownSections parses markdown content into sections bounded by
// headings. Each section carries the full heading path showing the
// hierarchy (e.g., ["## Architecture", "### Patterns"]).
func splitMarkdownSections(content string) []mdSection {
	lines := strings.Split(content, "\n")
	var sections []mdSection
	var stack []headingEntry
	var currentContent strings.Builder
	var currentPath []string

	flush := func() {
		text := currentContent.String()
		if strings.TrimSpace(text) != "" {
			sections = append(sections, mdSection{
				headingPath: append([]string{}, currentPath...),
				content:     text,
			})
		}
		currentContent.Reset()
	}

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		match := atxHeadingRe.FindStringSubmatch(line)
		if match != nil {
			flush()
			level := len(match[1])
			title := strings.TrimSpace(match[2])
			heading := match[1] + " " + title
			// Pop stack entries at same or deeper level.
			for len(stack) > 0 && stack[len(stack)-1].level >= level {
				stack = stack[:len(stack)-1]
			}
			stack = append(stack, headingEntry{level: level, title: heading})
			currentPath = make([]string, len(stack))
			for j, h := range stack {
				currentPath[j] = h.title
			}
			currentContent.WriteString(line)
			currentContent.WriteByte('\n')
			continue
		}
		// Check for setext headings (underline-style).
		if i+1 < len(lines) && strings.TrimSpace(line) != "" {
			nextLine := lines[i+1]
			if isSetextUnderline(nextLine) {
				flush()
				level := 1
				if strings.TrimSpace(nextLine)[0] == '-' {
					level = 2
				}
				title := strings.TrimSpace(line)
				heading := strings.Repeat("#", level) + " " + title
				for len(stack) > 0 && stack[len(stack)-1].level >= level {
					stack = stack[:len(stack)-1]
				}
				stack = append(stack, headingEntry{level: level, title: heading})
				currentPath = make([]string, len(stack))
				for j, h := range stack {
					currentPath[j] = h.title
				}
				currentContent.WriteString(line)
				currentContent.WriteByte('\n')
				currentContent.WriteString(nextLine)
				currentContent.WriteByte('\n')
				i++
				continue
			}
		}
		currentContent.WriteString(line)
		currentContent.WriteByte('\n')
	}
	flush()
	return sections
}

// headingEntry tracks a heading in the hierarchy stack.
type headingEntry struct {
	level int
	title string
}

// isSetextUnderline returns true when line is a setext heading underline
// (two or more = or - characters).
func isSetextUnderline(line string) bool {
	trimmed := strings.TrimSpace(line)
	if len(trimmed) < 2 {
		return false
	}
	allEq := true
	allDash := true
	for _, r := range trimmed {
		if r != '=' {
			allEq = false
		}
		if r != '-' {
			allDash = false
		}
	}
	return allEq || allDash
}

// splitSectionWithOverlap breaks an oversized section into overlapping
// pieces using the separator hierarchy. Overlap is applied from the tail
// of the previous piece to the start of the next.
func splitSectionWithOverlap(text string, cfg ChunkConfig) []string {
	pieces := recursiveSplit(text, cfg.MaxTokens(), 0)
	if len(pieces) <= 1 {
		return pieces
	}
	result := make([]string, 0, len(pieces))
	for i, piece := range pieces {
		trimmed := strings.TrimSpace(piece)
		if trimmed == "" {
			continue
		}
		if i > 0 && cfg.OverlapTokens() > 0 {
			prevTrimmed := strings.TrimSpace(pieces[i-1])
			tail := extractTail(prevTrimmed, cfg.OverlapTokens())
			trimmed = tail + "\n" + trimmed
		}
		result = append(result, trimmed)
	}
	return result
}

