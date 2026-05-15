// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/ingest"
)

// fixturesDir returns the absolute path to spec/fixtures/ingestion
// relative to this test file.
func fixturesDir(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	return filepath.Join(filepath.Dir(file), "..", "..", "spec", "fixtures", "ingestion")
}

func TestChunkDocument_default_config(t *testing.T) {
	body := `# Alpha

This is the first section with enough content to form a standalone chunk above the minimum token threshold of thirty tokens.

## Beta

The second section also has sufficient content to remain its own chunk after the merge pass runs through the document body.

## Gamma

The third section contains enough words to meet the minimum token threshold and survive as an independent chunk in the output.`

	sections := splitByHeadings(body)
	chunks := ChunkDocument(body, sections, ingest.ChunkConfig{})
	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(chunks))
	}
	if chunks[0].Heading != "Alpha" {
		t.Fatalf("chunk[0].Heading = %q, want Alpha", chunks[0].Heading)
	}
	if chunks[1].Heading != "Beta" {
		t.Fatalf("chunk[1].Heading = %q, want Beta", chunks[1].Heading)
	}
	if chunks[2].Heading != "Gamma" {
		t.Fatalf("chunk[2].Heading = %q, want Gamma", chunks[2].Heading)
	}
}

func TestChunkDocument_custom_max_tokens(t *testing.T) {
	// Build a body with two sections, each around 300 chars (~75 tokens).
	// With MaxTokens=50 both should be split, producing more chunks than
	// the default config.
	body := `# One

` + strings.Repeat("word ", 60) + `

# Two

` + strings.Repeat("word ", 60)

	sections := splitByHeadings(body)

	defaultChunks := ChunkDocument(body, sections, ingest.ChunkConfig{})

	cfg := ingest.ChunkConfig{
		MaxTokens:     50,
		OverlapTokens: 0,
		MinTokens:     10,
		Strategy:      ingest.StrategyMarkdown,
		Separators:    ingest.DefaultSeparators,
	}
	customChunks := ChunkDocument(body, sections, cfg)

	if len(customChunks) <= len(defaultChunks) {
		t.Fatalf("custom (MaxTokens=50) produced %d chunks, want more than default %d",
			len(customChunks), len(defaultChunks))
	}
}

func TestChunkDocument_overlap(t *testing.T) {
	body := `# Part A

The first part of the document contains a decent amount of text so that when overlap is applied, the trailing portion of this chunk is prepended to the next chunk as context.

# Part B

The second part of the document is independent content that should begin with overlapping text from Part A above.`

	sections := splitByHeadings(body)
	cfg := ingest.ChunkConfig{
		MaxTokens:     512,
		OverlapTokens: 32,
		MinTokens:     10,
		Strategy:      ingest.StrategyMarkdown,
		Separators:    ingest.DefaultSeparators,
	}
	chunks := ChunkDocument(body, sections, cfg)
	if len(chunks) < 2 {
		t.Fatalf("expected >= 2 chunks, got %d", len(chunks))
	}

	// The second chunk should contain some text from Part A as overlap
	// prefix.
	partAText := chunks[0].Text
	partBText := chunks[1].Text

	// Extract words from the end of Part A.
	words := strings.Fields(partAText)
	if len(words) < 4 {
		t.Fatalf("Part A too short for overlap test")
	}
	// At least some trailing words from Part A should appear in Part B's
	// text (as the overlap prefix).
	trailingSnippet := words[len(words)-3]
	if !strings.Contains(partBText, trailingSnippet) {
		t.Fatalf("overlap missing: Part B does not contain trailing word %q from Part A\nPart B: %s",
			trailingSnippet, partBText[:min(200, len(partBText))])
	}
}

func TestChunkDocument_min_merge(t *testing.T) {
	body := `# Big Section

This section has enough content to remain standalone because it is well above the minimum token threshold configured for this test. The paragraph continues with more words here.

# Tiny

Hi.`

	sections := splitByHeadings(body)
	cfg := ingest.ChunkConfig{
		MaxTokens:     512,
		OverlapTokens: 0,
		MinTokens:     30,
		Strategy:      ingest.StrategyMarkdown,
		Separators:    ingest.DefaultSeparators,
	}
	chunks := ChunkDocument(body, sections, cfg)

	// "Hi." is 3 chars (~1 token) which is below MinTokens=30, so it
	// should be merged into the preceding chunk.
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk after merge, got %d", len(chunks))
	}
	if !strings.Contains(chunks[0].Text, "Hi.") {
		t.Fatal("merged chunk missing tiny section content")
	}
}

func TestChunkDocument_empty_body(t *testing.T) {
	chunks := ChunkDocument("", nil, ingest.ChunkConfig{})
	if chunks != nil {
		t.Fatalf("expected nil for empty body, got %d chunks", len(chunks))
	}

	chunks = ChunkDocument("   \n\n  ", nil, ingest.ChunkConfig{})
	if chunks != nil {
		t.Fatalf("expected nil for whitespace body, got %d chunks", len(chunks))
	}
}

func TestChunkDocument_single_paragraph(t *testing.T) {
	body := "A single paragraph that fits comfortably within the default max token budget so no splitting occurs."
	sections := splitByHeadings(body)
	chunks := ChunkDocument(body, sections, ingest.ChunkConfig{})
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0].Ordinal != 0 {
		t.Fatalf("ordinal = %d, want 0", chunks[0].Ordinal)
	}
}

// conformanceExpected models the JSON fixture structure.
type conformanceExpected struct {
	Description string `json:"description"`
	Config      struct {
		MaxTokens     int `json:"maxTokens"`
		OverlapTokens int `json:"overlapTokens"`
		MinTokens     int `json:"minTokens"`
	} `json:"config"`
	ChunkCount int `json:"chunkCount"`
	Chunks     []struct {
		Ordinal       int    `json:"ordinal"`
		Heading       string `json:"heading"`
		MinTokens     int    `json:"minTokens"`
		MaxTokens     int    `json:"maxTokens"`
		ContentPrefix string `json:"contentPrefix"`
	} `json:"chunks"`
}

func TestChunkDocument_conformance(t *testing.T) {
	dir := fixturesDir(t)

	mdBytes, err := os.ReadFile(filepath.Join(dir, "chunking-conformance.md"))
	if err != nil {
		t.Fatalf("read conformance md: %v", err)
	}
	jsonBytes, err := os.ReadFile(filepath.Join(dir, "chunking-expected.json"))
	if err != nil {
		t.Fatalf("read expected json: %v", err)
	}

	var expected conformanceExpected
	if err := json.Unmarshal(jsonBytes, &expected); err != nil {
		t.Fatalf("parse expected json: %v", err)
	}

	body := strings.TrimSpace(string(mdBytes))
	sections := splitByHeadings(body)
	cfg := ingest.ChunkConfig{
		MaxTokens:     expected.Config.MaxTokens,
		OverlapTokens: expected.Config.OverlapTokens,
		MinTokens:     expected.Config.MinTokens,
		Strategy:      ingest.StrategyMarkdown,
		Separators:    ingest.DefaultSeparators,
	}
	chunks := ChunkDocument(body, sections, cfg)

	if len(chunks) != expected.ChunkCount {
		for i, c := range chunks {
			t.Logf("chunk %d [%s] tokens=%d text=%q", i, c.Heading, c.Tokens, c.Text[:min(80, len(c.Text))])
		}
		t.Fatalf("chunk count = %d, want %d", len(chunks), expected.ChunkCount)
	}

	for i, ec := range expected.Chunks {
		if i >= len(chunks) {
			break
		}
		c := chunks[i]
		if c.Ordinal != ec.Ordinal {
			t.Errorf("chunk[%d].Ordinal = %d, want %d", i, c.Ordinal, ec.Ordinal)
		}
		if ec.Heading != "" && c.Heading != ec.Heading {
			t.Errorf("chunk[%d].Heading = %q, want %q", i, c.Heading, ec.Heading)
		}
		if ec.MinTokens > 0 && c.Tokens < ec.MinTokens {
			t.Errorf("chunk[%d].Tokens = %d, want >= %d", i, c.Tokens, ec.MinTokens)
		}
		if ec.MaxTokens > 0 && c.Tokens > ec.MaxTokens {
			t.Errorf("chunk[%d].Tokens = %d, want <= %d", i, c.Tokens, ec.MaxTokens)
		}
		if ec.ContentPrefix != "" && !strings.HasPrefix(c.Text, ec.ContentPrefix) {
			t.Errorf("chunk[%d] content does not start with %q, got %q",
				i, ec.ContentPrefix, c.Text[:min(60, len(c.Text))])
		}
	}
}
