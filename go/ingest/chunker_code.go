// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"regexp"
	"strings"
)

// Compile-time interface compliance check.
var _ Chunker = (*CodeChunker)(nil)

// codeChunkerContentTypes are the MIME types handled by CodeChunker.
var codeChunkerContentTypes = []string{
	"text/x-go",
	"text/x-python",
	"text/x-typescript",
	"text/x-javascript",
	"text/x-java",
	"text/x-c",
	"text/x-c++",
	"text/x-rust",
	"application/x-typescript",
	"application/javascript",
}

// Heuristic patterns used to detect function/class boundaries across
// languages. Each pattern matches the start of a top-level declaration
// line.
var (
	goFuncRe     = regexp.MustCompile(`^func\s`)
	goTypeRe     = regexp.MustCompile(`^type\s`)
	pyDefRe      = regexp.MustCompile(`^(def|class|async\s+def)\s`)
	tsFuncRe     = regexp.MustCompile(`^(export\s+)?(function|class|const|interface|type|enum)\s`)
	cFuncRe      = regexp.MustCompile(`^[a-zA-Z_].*\)\s*\{?\s*$`)
	importLineRe = regexp.MustCompile(`^(import|from|require|use|#include|package)\s`)
)

// CodeChunker splits source code at function/class/type boundaries using
// line-level heuristics. This is a Phase-1 implementation that does NOT
// use a tree-sitter AST; full AST splitting is deferred to a later phase.
//
// The algorithm:
//  1. Collect leading import/package lines into a header block.
//  2. Walk remaining lines and split at detected declaration boundaries.
//  3. Each chunk gets the header prepended for self-contained context.
type CodeChunker struct{}

func (cc *CodeChunker) ContentTypes() []string { return codeChunkerContentTypes }
func (cc *CodeChunker) Name() string           { return "code" }

func (cc *CodeChunker) Chunk(_ context.Context, content string, cfg ChunkConfig) ([]Chunk, error) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil, nil
	}

	lines := strings.Split(content, "\n")
	header, bodyStart := extractImportHeader(lines)
	sections := splitAtDeclarations(lines[bodyStart:])

	chunks := make([]Chunk, 0, len(sections))
	for _, section := range sections {
		text := strings.TrimSpace(section)
		if text == "" {
			continue
		}
		// Prepend header to each chunk for context, unless it is the header itself.
		full := text
		if header != "" && !strings.HasPrefix(text, header) {
			full = header + "\n\n" + text
		}

		if estimateTokens(full) <= cfg.MaxTokens {
			chunks = append(chunks, Chunk{
				Content:  full,
				Metadata: map[string]string{"chunker": "code"},
			})
			continue
		}
		// Section too large: fall back to recursive splitting.
		subPieces := recursiveSplit(full, cfg.MaxTokens, 0)
		for _, piece := range subPieces {
			t := strings.TrimSpace(piece)
			if t == "" {
				continue
			}
			chunks = append(chunks, Chunk{
				Content:  t,
				Metadata: map[string]string{"chunker": "code"},
			})
		}
	}

	// Merge undersized chunks.
	return mergeUndersized(chunks, cfg), nil
}

// extractImportHeader collects leading import/package/require lines and
// returns them as a single block along with the index where the body starts.
func extractImportHeader(lines []string) (string, int) {
	var headerLines []string
	bodyStart := 0
	inHeader := true

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if inHeader {
			if trimmed == "" || importLineRe.MatchString(trimmed) || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "/*") || strings.HasPrefix(trimmed, "*") {
				headerLines = append(headerLines, line)
				bodyStart = i + 1
				continue
			}
			// Also include closing paren for grouped imports: ")"
			if trimmed == ")" {
				headerLines = append(headerLines, line)
				bodyStart = i + 1
				continue
			}
			inHeader = false
		}
	}

	return strings.TrimSpace(strings.Join(headerLines, "\n")), bodyStart
}

// splitAtDeclarations walks body lines and breaks at detected function,
// class, or type declaration boundaries.
func splitAtDeclarations(lines []string) []string {
	if len(lines) == 0 {
		return nil
	}

	var sections []string
	var current strings.Builder

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if isDeclarationStart(trimmed) && current.Len() > 0 {
			sections = append(sections, current.String())
			current.Reset()
		}
		current.WriteString(line)
		current.WriteByte('\n')
	}
	if current.Len() > 0 {
		sections = append(sections, current.String())
	}
	return sections
}

// isDeclarationStart returns true if the line looks like the start of
// a top-level function, class, or type declaration.
func isDeclarationStart(line string) bool {
	if line == "" {
		return false
	}
	return goFuncRe.MatchString(line) ||
		goTypeRe.MatchString(line) ||
		pyDefRe.MatchString(line) ||
		tsFuncRe.MatchString(line) ||
		cFuncRe.MatchString(line)
}

// mergeUndersized combines consecutive chunks that are below minTokens
// into the preceding chunk.
func mergeUndersized(chunks []Chunk, cfg ChunkConfig) []Chunk {
	if len(chunks) == 0 {
		return nil
	}
	merged := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		if estimateTokens(c.Content) < cfg.MinTokens && len(merged) > 0 {
			prev := merged[len(merged)-1]
			prev.Content = prev.Content + "\n" + c.Content
			merged[len(merged)-1] = prev
			continue
		}
		merged = append(merged, c)
	}
	return merged
}
