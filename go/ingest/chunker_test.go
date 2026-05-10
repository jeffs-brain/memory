// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func TestNewChunkConfig_valid(t *testing.T) {
	t.Parallel()
	cfg, err := NewChunkConfig(512, 64, 40)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.MaxTokens() != 512 {
		t.Fatalf("MaxTokens = %d, want 512", cfg.MaxTokens())
	}
	if cfg.OverlapTokens() != 64 {
		t.Fatalf("OverlapTokens = %d, want 64", cfg.OverlapTokens())
	}
	if cfg.MinTokens() != 40 {
		t.Fatalf("MinTokens = %d, want 40", cfg.MinTokens())
	}
}

func TestNewChunkConfig_rejectsInvalid(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		max     int
		overlap int
		min     int
	}{
		{"minTokens >= maxTokens", 100, 10, 100},
		{"minTokens > maxTokens", 100, 10, 200},
		{"overlapTokens >= maxTokens", 100, 100, 10},
		{"overlapTokens > maxTokens", 100, 200, 10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := NewChunkConfig(tc.max, tc.overlap, tc.min)
			if err == nil {
				t.Fatalf("expected error for %s", tc.name)
			}
		})
	}
}

func TestNewChunkConfig_appliesDefaults(t *testing.T) {
	t.Parallel()
	cfg, err := NewChunkConfig(0, -1, -1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.MaxTokens() != DefaultMaxTokens {
		t.Fatalf("MaxTokens = %d, want %d", cfg.MaxTokens(), DefaultMaxTokens)
	}
	if cfg.OverlapTokens() != DefaultOverlapTokens {
		t.Fatalf("OverlapTokens = %d, want %d", cfg.OverlapTokens(), DefaultOverlapTokens)
	}
	if cfg.MinTokens() != DefaultMinTokens {
		t.Fatalf("MinTokens = %d, want %d", cfg.MinTokens(), DefaultMinTokens)
	}
}

func TestRecursiveChunker_shortContent(t *testing.T) {
	t.Parallel()
	rc := &RecursiveChunker{}
	cfg := DefaultChunkConfig()
	chunks, err := rc.Chunk(context.Background(), "Hello world.", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Content != "Hello world." {
		t.Fatalf("content = %q, want %q", chunks[0].Content, "Hello world.")
	}
}

func TestRecursiveChunker_emptyContent(t *testing.T) {
	t.Parallel()
	rc := &RecursiveChunker{}
	cfg := DefaultChunkConfig()
	chunks, err := rc.Chunk(context.Background(), "   \n\n  ", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks, got %d", len(chunks))
	}
}

func TestRecursiveChunker_separatorHierarchy(t *testing.T) {
	t.Parallel()
	// Build content with paragraph breaks that exceeds 64 tokens.
	paragraphs := make([]string, 10)
	for i := range paragraphs {
		paragraphs[i] = fmt.Sprintf("Paragraph %d with some content that uses tokens.", i)
	}
	content := strings.Join(paragraphs, "\n\n")
	cfg, err := NewChunkConfig(64, 16, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rc := &RecursiveChunker{}
	chunks, err := rc.Chunk(context.Background(), content, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	// Each chunk should respect maxTokens (with some tolerance for overlap).
	for i, c := range chunks {
		tokens := estimateTokens(c.Content)
		// Allow 2x max for overlap prefix; the core piece should still fit.
		if tokens > cfg.MaxTokens()*2 {
			t.Errorf("chunk %d has %d tokens, exceeds 2x maxTokens (%d)", i, tokens, cfg.MaxTokens())
		}
	}
}

func TestRecursiveChunker_overlapApplied(t *testing.T) {
	t.Parallel()
	// Create content with clear paragraph boundaries.
	paragraphs := []string{
		strings.Repeat("alpha ", 40),
		strings.Repeat("beta ", 40),
		strings.Repeat("gamma ", 40),
	}
	content := strings.Join(paragraphs, "\n\n")
	cfg, err := NewChunkConfig(64, 16, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rc := &RecursiveChunker{}
	chunks, err := rc.Chunk(context.Background(), content, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}
	// The second chunk should contain overlap from the first.
	firstContent := chunks[0].Content
	tail := extractTail(firstContent, cfg.OverlapTokens())
	if !strings.Contains(chunks[1].Content, tail) {
		t.Errorf("chunk 1 does not contain overlap tail from chunk 0")
	}
}

func TestMarkdownChunker_headingAware(t *testing.T) {
	t.Parallel()
	content := `# Introduction

This is the intro paragraph.

## Architecture

Architecture details here.

### Patterns

Pattern details.

## Conclusion

Final thoughts.`
	cfg := DefaultChunkConfig()
	mc := &MarkdownChunker{}
	chunks, err := mc.Chunk(context.Background(), content, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 3 {
		t.Fatalf("expected at least 3 chunks, got %d", len(chunks))
	}
	// First chunk should have heading path for "Introduction".
	firstPath := chunks[0].Metadata["headingPath"]
	if !strings.Contains(firstPath, "Introduction") {
		t.Errorf("first chunk headingPath = %q, want to contain 'Introduction'", firstPath)
	}
	// Find a chunk with nested heading path.
	foundNested := false
	for _, c := range chunks {
		path := c.Metadata["headingPath"]
		if strings.Contains(path, "Architecture") && strings.Contains(path, "Patterns") {
			foundNested = true
			break
		}
	}
	if !foundNested {
		t.Error("no chunk found with nested heading path containing Architecture > Patterns")
	}
}

func TestMarkdownChunker_respectsMaxTokens(t *testing.T) {
	t.Parallel()
	// Build a large section under one heading.
	var content strings.Builder
	content.WriteString("# Big Section\n\n")
	for i := 0; i < 30; i++ {
		content.WriteString(fmt.Sprintf("Paragraph %d with enough words to accumulate tokens over the limit. ", i))
		content.WriteString("More filler text to ensure we exceed the budget.\n\n")
	}
	cfg, err := NewChunkConfig(64, 16, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	mc := &MarkdownChunker{}
	chunks, err := mc.Chunk(context.Background(), content.String(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks from oversized section, got %d", len(chunks))
	}
	for _, c := range chunks {
		path := c.Metadata["headingPath"]
		if !strings.Contains(path, "Big Section") {
			t.Errorf("chunk heading path %q does not contain 'Big Section'", path)
		}
	}
}

func TestMarkdownChunker_emptyContent(t *testing.T) {
	t.Parallel()
	mc := &MarkdownChunker{}
	cfg := DefaultChunkConfig()
	chunks, err := mc.Chunk(context.Background(), "", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks for empty input, got %d", len(chunks))
	}
}

func TestChunkerRegistry_routesByContentType(t *testing.T) {
	t.Parallel()
	reg := NewChunkerRegistry()
	cfg := DefaultChunkConfig()
	content := "# Hello\n\nWorld"

	chunks, err := reg.Chunk(context.Background(), content, "text/markdown", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least 1 chunk")
	}
	if chunks[0].Metadata["chunker"] != "markdown" {
		t.Errorf("expected markdown chunker, got %q", chunks[0].Metadata["chunker"])
	}
}

func TestChunkerRegistry_fallbackToRecursive(t *testing.T) {
	t.Parallel()
	reg := NewChunkerRegistry()
	cfg := DefaultChunkConfig()
	content := "Just some plain text."

	chunks, err := reg.Chunk(context.Background(), content, "application/unknown", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least 1 chunk")
	}
	if chunks[0].Metadata["chunker"] != "recursive" {
		t.Errorf("expected recursive chunker as fallback, got %q", chunks[0].Metadata["chunker"])
	}
}

func TestChunkerRegistry_customChunker(t *testing.T) {
	t.Parallel()
	reg := NewChunkerRegistry()
	cfg := DefaultChunkConfig()

	custom := NewChunkerFunc("custom-json", []string{"application/json"}, func(_ context.Context, content string, _ ChunkConfig) ([]Chunk, error) {
		return []Chunk{{Content: content, Metadata: map[string]string{"chunker": "custom-json"}}}, nil
	})
	reg.Register(custom)

	chunks, err := reg.Chunk(context.Background(), `{"key": "value"}`, "application/json", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Metadata["chunker"] != "custom-json" {
		t.Errorf("expected custom-json chunker, got %q", chunks[0].Metadata["chunker"])
	}
}

func TestChunkerRegistry_contentTypeNormalisation(t *testing.T) {
	t.Parallel()
	reg := NewChunkerRegistry()
	cfg := DefaultChunkConfig()
	content := "# Title\n\nBody"

	// Content type with charset suffix should still route to markdown.
	chunks, err := reg.Chunk(context.Background(), content, "text/markdown; charset=utf-8", cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) == 0 {
		t.Fatal("expected at least 1 chunk")
	}
	if chunks[0].Metadata["chunker"] != "markdown" {
		t.Errorf("expected markdown chunker with charset suffix, got %q", chunks[0].Metadata["chunker"])
	}
}

func TestChunkerRegistry_cancelledContext(t *testing.T) {
	t.Parallel()
	reg := NewChunkerRegistry()
	cfg := DefaultChunkConfig()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := reg.Chunk(ctx, "some content", "text/plain", cfg)
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestMarkdownChunker_setextHeadings(t *testing.T) {
	t.Parallel()
	content := `Title
=====

Some intro text.

Subtitle
--------

Details here.`
	cfg := DefaultChunkConfig()
	mc := &MarkdownChunker{}
	chunks, err := mc.Chunk(context.Background(), content, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d", len(chunks))
	}
	// First should reference the setext h1 "Title".
	firstPath := chunks[0].Metadata["headingPath"]
	if !strings.Contains(firstPath, "Title") {
		t.Errorf("first chunk headingPath = %q, want to contain 'Title'", firstPath)
	}
}

func TestRecursiveChunker_contentTypes(t *testing.T) {
	t.Parallel()
	rc := &RecursiveChunker{}
	types := rc.ContentTypes()
	if len(types) == 0 {
		t.Fatal("expected at least one content type")
	}
	found := false
	for _, ct := range types {
		if ct == "text/plain" {
			found = true
		}
	}
	if !found {
		t.Error("expected text/plain in content types")
	}
}

func TestMarkdownChunker_contentTypes(t *testing.T) {
	t.Parallel()
	mc := &MarkdownChunker{}
	types := mc.ContentTypes()
	found := false
	for _, ct := range types {
		if ct == "text/markdown" {
			found = true
		}
	}
	if !found {
		t.Error("expected text/markdown in content types")
	}
}
