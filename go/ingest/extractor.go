// SPDX-License-Identifier: Apache-2.0

// Package ingest provides a content-type-routing extractor registry with
// streaming support. Extractors declare the MIME types they handle, and
// the registry routes incoming content to the appropriate extractor chain
// with fallback behaviour for unknown types.
package ingest

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"unicode/utf8"
)

// Security constants for downstream extractors (Phase 4) that decompress
// archives or spawn subprocesses.
const (
	// MaxDecompressionRatio caps the ratio of decompressed-to-compressed
	// size to prevent ZIP bomb attacks.
	MaxDecompressionRatio = 100

	// MaxExtractedFiles limits the number of files extracted from an
	// archive to prevent resource exhaustion.
	MaxExtractedFiles = 1000
)

// sanitizeArgsAllowlist contains flags permitted to pass through to
// subprocess extractors. Anything starting with '-' that is not in this
// set is rejected.
var sanitizeArgsAllowlist = map[string]struct{}{
	"-o":        {},
	"--output":  {},
	"-f":        {},
	"--format":  {},
	"-q":        {},
	"--quiet":   {},
	"-v":        {},
	"--verbose": {},
	"--stdin":   {},
	"--stdout":  {},
	"--no-color": {},
}

// SanitizeArgs filters a slice of command-line arguments, rejecting any
// flag (starting with '-') that is not in the hardcoded allowlist. This
// prevents injection of dangerous flags into subprocess extractors.
func SanitizeArgs(args []string) ([]string, error) {
	sanitized := make([]string, 0, len(args))
	for _, arg := range args {
		if strings.HasPrefix(arg, "-") {
			if _, ok := sanitizeArgsAllowlist[arg]; !ok {
				return nil, fmt.Errorf("ingest: disallowed argument %q", arg)
			}
		}
		sanitized = append(sanitized, arg)
	}
	return sanitized, nil
}

// ExtractOptions provides hints to the extractor about the content being
// processed.
type ExtractOptions struct {
	ContentType string
	FileName    string
	MaxBytes    int64 // 0 = no limit
}

// ExtractResult holds the output of an extraction operation.
type ExtractResult struct {
	Text     string
	Metadata map[string]string
	Skipped  bool
	Reason   string
}

// Extractor defines the contract for content extraction. Implementations
// declare the MIME types they handle and provide both buffered and
// streaming extraction methods.
type Extractor interface {
	// Extract converts buffered raw bytes into text content.
	Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error)
	// ExtractStream processes content from a reader. The default
	// implementation (BaseExtractor) buffers into Extract.
	ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error)
	// ContentTypes returns the MIME types this extractor handles.
	ContentTypes() []string
	// Name returns a human-readable identifier for this extractor.
	Name() string
}

// BaseExtractor provides a default ExtractStream implementation that
// buffers the reader into memory and delegates to Extract. Embed this in
// simple extractors that do not need true streaming.
type BaseExtractor struct {
	ExtractFn      func(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error)
	ContentTypesFn func() []string
	NameFn         func() string
}

// Extract delegates to the embedded ExtractFn.
func (b *BaseExtractor) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	return b.ExtractFn(ctx, raw, opts)
}

// ExtractStream buffers the reader and delegates to Extract.
func (b *BaseExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	var limitReader io.Reader = reader
	if opts.MaxBytes > 0 {
		limitReader = io.LimitReader(reader, opts.MaxBytes)
	}
	raw, err := io.ReadAll(limitReader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading stream: %w", err)
	}
	return b.ExtractFn(ctx, raw, opts)
}

// ContentTypes delegates to the embedded ContentTypesFn.
func (b *BaseExtractor) ContentTypes() []string {
	return b.ContentTypesFn()
}

// Name delegates to the embedded NameFn.
func (b *BaseExtractor) Name() string {
	return b.NameFn()
}

// PlainTextExtractor handles text/* content types by returning the raw
// bytes as UTF-8 text. Non-UTF-8 content is rejected with an error.
type PlainTextExtractor struct{}

// Compile-time interface check.
var _ Extractor = (*PlainTextExtractor)(nil)

// Extract returns the raw bytes as text, validating UTF-8.
func (p *PlainTextExtractor) Extract(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
	if !utf8.Valid(raw) {
		return ExtractResult{}, fmt.Errorf("ingest: content is not valid UTF-8")
	}
	return ExtractResult{
		Text:     string(raw),
		Metadata: map[string]string{},
	}, nil
}

// ExtractStream buffers the reader and delegates to Extract.
func (p *PlainTextExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	var limitReader io.Reader = reader
	if opts.MaxBytes > 0 {
		limitReader = io.LimitReader(reader, opts.MaxBytes)
	}
	raw, err := io.ReadAll(limitReader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading stream: %w", err)
	}
	return p.Extract(ctx, raw, opts)
}

// ContentTypes returns the MIME types handled by the plain text extractor.
func (p *PlainTextExtractor) ContentTypes() []string {
	return []string{
		"text/plain",
		"text/markdown",
		"text/csv",
		"text/x-yaml",
		"application/json",
		"application/x-yaml",
	}
}

// Name returns the extractor identifier.
func (p *PlainTextExtractor) Name() string {
	return "plain-text"
}

// ExtractorRegistry maps content types to extractors and routes incoming
// content to the correct handler. Unknown content types return a skipped
// result rather than an error.
type ExtractorRegistry struct {
	mu         sync.RWMutex
	extractors map[string]Extractor
}

// NewExtractorRegistry creates a registry pre-loaded with the built-in
// PlainTextExtractor for all text/* content types.
func NewExtractorRegistry() *ExtractorRegistry {
	r := &ExtractorRegistry{
		extractors: make(map[string]Extractor, 8),
	}
	plainText := &PlainTextExtractor{}
	r.Register(plainText)
	return r
}

// Register adds an extractor to the registry for all content types it
// declares. Later registrations for the same content type override
// earlier ones.
func (r *ExtractorRegistry) Register(ext Extractor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, ct := range ext.ContentTypes() {
		r.extractors[normaliseContentType(ct)] = ext
	}
}

// Extract routes raw bytes to the appropriate extractor based on the
// content type in opts. Returns a skipped result for unsupported types.
func (r *ExtractorRegistry) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	ext := r.resolve(opts.ContentType)
	if ext == nil {
		return ExtractResult{
			Skipped: true,
			Reason:  fmt.Sprintf("unsupported content type: %s", opts.ContentType),
		}, nil
	}
	return ext.Extract(ctx, raw, opts)
}

// ExtractStream routes a reader to the appropriate extractor based on
// the content type in opts. Returns a skipped result for unsupported
// types.
func (r *ExtractorRegistry) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	ext := r.resolve(opts.ContentType)
	if ext == nil {
		return ExtractResult{
			Skipped: true,
			Reason:  fmt.Sprintf("unsupported content type: %s", opts.ContentType),
		}, nil
	}
	return ext.ExtractStream(ctx, reader, opts)
}

// resolve finds the extractor for a content type. It first tries an
// exact match, then falls back to matching the base type (e.g.
// "text/plain" for "text/plain; charset=utf-8"), then tries the type
// prefix (e.g. "text/" matches any text/* extractor registered as
// "text/plain").
func (r *ExtractorRegistry) resolve(contentType string) Extractor {
	r.mu.RLock()
	defer r.mu.RUnlock()

	normalised := normaliseContentType(contentType)

	// Exact match.
	if ext, ok := r.extractors[normalised]; ok {
		return ext
	}

	// Fallback: match by type prefix for text/* family.
	if strings.HasPrefix(normalised, "text/") {
		if ext, ok := r.extractors["text/plain"]; ok {
			return ext
		}
	}

	return nil
}

// normaliseContentType strips parameters (charset, boundary, etc.) and
// lowercases the media type.
func normaliseContentType(ct string) string {
	base := ct
	if idx := strings.Index(ct, ";"); idx >= 0 {
		base = ct[:idx]
	}
	return strings.TrimSpace(strings.ToLower(base))
}
