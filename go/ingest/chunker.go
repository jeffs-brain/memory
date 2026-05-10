// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// Chunk is a single segment produced by a Chunker. ID is set by the
// registry after the chunker returns; Content is the text payload;
// Metadata carries chunker-specific annotations (heading path, language).
type Chunk struct {
	ID       string
	Content  string
	Metadata map[string]string
}

// Chunker segments content into chunks. Implementations are content-type
// specific; the registry selects the appropriate one based on content type.
type Chunker interface {
	// Chunk splits content into ordered chunks respecting cfg bounds.
	Chunk(ctx context.Context, content string, cfg ChunkConfig) ([]Chunk, error)

	// ContentTypes returns the MIME types this chunker handles.
	ContentTypes() []string

	// Name returns a human-readable identifier for this chunker.
	Name() string
}

// ChunkerFunc adapts a plain function to the Chunker interface. The
// adapted function handles a single content type under a given name.
type ChunkerFunc struct {
	fn           func(ctx context.Context, content string, cfg ChunkConfig) ([]Chunk, error)
	contentTypes []string
	name         string
}

// NewChunkerFunc wraps a function as a Chunker with the given name and
// content types.
func NewChunkerFunc(
	name string,
	contentTypes []string,
	fn func(ctx context.Context, content string, cfg ChunkConfig) ([]Chunk, error),
) *ChunkerFunc {
	return &ChunkerFunc{fn: fn, contentTypes: contentTypes, name: name}
}

func (f *ChunkerFunc) Chunk(ctx context.Context, content string, cfg ChunkConfig) ([]Chunk, error) {
	return f.fn(ctx, content, cfg)
}

func (f *ChunkerFunc) ContentTypes() []string { return f.contentTypes }
func (f *ChunkerFunc) Name() string           { return f.name }

// ChunkerRegistry maps content types to Chunker implementations and
// routes incoming content to the appropriate one.
type ChunkerRegistry struct {
	mu       sync.RWMutex
	routes   map[string]Chunker
	fallback Chunker
}

// NewChunkerRegistry creates a registry pre-loaded with the built-in
// chunkers: MarkdownChunker for text/markdown and RecursiveChunker as
// the fallback for all other types.
func NewChunkerRegistry() *ChunkerRegistry {
	r := &ChunkerRegistry{
		routes:   make(map[string]Chunker, 8),
		fallback: &RecursiveChunker{},
	}
	md := &MarkdownChunker{}
	for _, ct := range md.ContentTypes() {
		r.routes[ct] = md
	}
	return r
}

// Register adds a custom chunker. All content types declared by c are
// mapped to it, overriding any previous registration for those types.
func (r *ChunkerRegistry) Register(c Chunker) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, ct := range c.ContentTypes() {
		r.routes[strings.ToLower(strings.TrimSpace(ct))] = c
	}
}

// Chunk routes content to the registered chunker for contentType and
// invokes it. Falls back to the recursive chunker when no specific
// chunker is registered. Returns an error when the chunker fails or
// the context is cancelled.
func (r *ChunkerRegistry) Chunk(ctx context.Context, content string, contentType string, cfg ChunkConfig) ([]Chunk, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	normalised := strings.ToLower(strings.TrimSpace(contentType))
	if idx := strings.Index(normalised, ";"); idx >= 0 {
		normalised = strings.TrimSpace(normalised[:idx])
	}

	r.mu.RLock()
	chunker, ok := r.routes[normalised]
	if !ok {
		chunker = r.fallback
	}
	r.mu.RUnlock()

	chunks, err := chunker.Chunk(ctx, content, cfg)
	if err != nil {
		return nil, fmt.Errorf("ingest: chunker %s: %w", chunker.Name(), err)
	}
	for i := range chunks {
		if chunks[i].ID == "" {
			chunks[i].ID = fmt.Sprintf("%d", i)
		}
	}
	return chunks, nil
}

// estimateTokens approximates the token count of text using the chars/4
// heuristic. Monotonic in text length.
func estimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return (len(text) + 3) / 4
}
