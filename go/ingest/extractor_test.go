// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestPlainTextExtractor_Extract(t *testing.T) {
	t.Parallel()
	ext := &PlainTextExtractor{}

	cases := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"valid utf8 text", "hello world", false},
		{"empty input", "", false},
		{"unicode text", "hello 世界", false},
		{"markdown content", "# Title\n\nBody paragraph.", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result, err := ext.Extract(context.Background(), []byte(tc.input), ExtractOptions{})
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.Text != tc.input {
				t.Fatalf("text mismatch: got %q, want %q", result.Text, tc.input)
			}
			if result.Skipped {
				t.Fatal("expected skipped=false")
			}
		})
	}
}

func TestPlainTextExtractor_RejectsInvalidUTF8(t *testing.T) {
	t.Parallel()
	ext := &PlainTextExtractor{}
	invalidUTF8 := []byte{0xff, 0xfe, 0xfd}
	_, err := ext.Extract(context.Background(), invalidUTF8, ExtractOptions{})
	if err == nil {
		t.Fatal("expected error for invalid UTF-8, got nil")
	}
	if !strings.Contains(err.Error(), "not valid UTF-8") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestPlainTextExtractor_ContentTypes(t *testing.T) {
	t.Parallel()
	ext := &PlainTextExtractor{}
	types := ext.ContentTypes()
	if len(types) == 0 {
		t.Fatal("expected non-empty content types")
	}
	found := false
	for _, ct := range types {
		if ct == "text/plain" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected text/plain in content types")
	}
}

func TestPlainTextExtractor_Name(t *testing.T) {
	t.Parallel()
	ext := &PlainTextExtractor{}
	if ext.Name() != "plain-text" {
		t.Fatalf("unexpected name: %s", ext.Name())
	}
}

func TestPlainTextExtractor_ExtractStream(t *testing.T) {
	t.Parallel()
	ext := &PlainTextExtractor{}
	input := "streamed content here"
	reader := strings.NewReader(input)
	result, err := ext.ExtractStream(context.Background(), reader, ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != input {
		t.Fatalf("text mismatch: got %q, want %q", result.Text, input)
	}
}

func TestPlainTextExtractor_ExtractStreamWithMaxBytes(t *testing.T) {
	t.Parallel()
	ext := &PlainTextExtractor{}
	input := "abcdefghij" // 10 bytes
	reader := strings.NewReader(input)
	result, err := ext.ExtractStream(context.Background(), reader, ExtractOptions{MaxBytes: 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "abcde" {
		t.Fatalf("expected truncated text %q, got %q", "abcde", result.Text)
	}
}

func TestExtractorRegistry_RoutesbyContentType(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	input := []byte("plain text content")
	opts := ExtractOptions{ContentType: "text/plain"}

	result, err := registry.Extract(context.Background(), input, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "plain text content" {
		t.Fatalf("text mismatch: got %q", result.Text)
	}
	if result.Skipped {
		t.Fatal("expected skipped=false for text/plain")
	}
}

func TestExtractorRegistry_RoutesMarkdown(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	input := []byte("# Heading\n\nBody")
	opts := ExtractOptions{ContentType: "text/markdown"}

	result, err := registry.Extract(context.Background(), input, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "# Heading\n\nBody" {
		t.Fatalf("text mismatch: got %q", result.Text)
	}
}

func TestExtractorRegistry_FallbackForUnknownTextSubtype(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	input := []byte("some xml content")
	opts := ExtractOptions{ContentType: "text/xml"}

	result, err := registry.Extract(context.Background(), input, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Fatal("expected text/xml to fall back to text/plain extractor")
	}
	if result.Text != "some xml content" {
		t.Fatalf("text mismatch: got %q", result.Text)
	}
}

func TestExtractorRegistry_UnsupportedContentTypeReturnsSkipped(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	input := []byte{0x00, 0x01, 0x02}
	opts := ExtractOptions{ContentType: "application/octet-stream"}

	result, err := registry.Extract(context.Background(), input, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Fatal("expected skipped=true for unsupported content type")
	}
	if !strings.Contains(result.Reason, "unsupported content type") {
		t.Fatalf("unexpected reason: %s", result.Reason)
	}
}

func TestExtractorRegistry_ContentTypeWithCharset(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	input := []byte("charset content")
	opts := ExtractOptions{ContentType: "text/plain; charset=utf-8"}

	result, err := registry.Extract(context.Background(), input, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Fatal("expected charset parameter to be stripped for routing")
	}
	if result.Text != "charset content" {
		t.Fatalf("text mismatch: got %q", result.Text)
	}
}

func TestExtractorRegistry_CustomExtractorOverridesDefault(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()

	custom := &stubExtractor{
		name:         "custom-text",
		contentTypes: []string{"text/plain"},
		extractFn: func(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
			return ExtractResult{
				Text:     "CUSTOM:" + string(raw),
				Metadata: map[string]string{"extractor": "custom"},
			}, nil
		},
	}
	registry.Register(custom)

	input := []byte("hello")
	result, err := registry.Extract(context.Background(), input, ExtractOptions{ContentType: "text/plain"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "CUSTOM:hello" {
		t.Fatalf("expected custom extractor output, got %q", result.Text)
	}
}

func TestExtractorRegistry_ExtractStream(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	input := "streamed plain text"
	reader := strings.NewReader(input)

	result, err := registry.ExtractStream(context.Background(), reader, ExtractOptions{ContentType: "text/plain"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != input {
		t.Fatalf("text mismatch: got %q, want %q", result.Text, input)
	}
}

func TestExtractorRegistry_ExtractStreamUnsupported(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	reader := strings.NewReader("binary data")

	result, err := registry.ExtractStream(context.Background(), reader, ExtractOptions{ContentType: "application/octet-stream"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Fatal("expected skipped=true for unsupported content type in stream mode")
	}
}

func TestExtractorRegistry_DefaultStreamBuffersIntoExtract(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()

	var capturedRaw []byte
	custom := &stubExtractor{
		name:         "capture",
		contentTypes: []string{"application/x-test"},
		extractFn: func(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
			capturedRaw = make([]byte, len(raw))
			copy(capturedRaw, raw)
			return ExtractResult{Text: string(raw), Metadata: map[string]string{}}, nil
		},
	}
	registry.Register(custom)

	input := "buffered via stream"
	reader := strings.NewReader(input)
	result, err := registry.ExtractStream(context.Background(), reader, ExtractOptions{ContentType: "application/x-test"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != input {
		t.Fatalf("text mismatch: got %q, want %q", result.Text, input)
	}
	if string(capturedRaw) != input {
		t.Fatalf("expected stream to buffer into Extract, got %q", string(capturedRaw))
	}
}

func TestSanitizeArgs_AllowlistedFlags(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		args    []string
		want    []string
		wantErr bool
	}{
		{
			name: "no flags",
			args: []string{"input.pdf", "output.txt"},
			want: []string{"input.pdf", "output.txt"},
		},
		{
			name: "allowlisted short flag",
			args: []string{"-o", "output.txt"},
			want: []string{"-o", "output.txt"},
		},
		{
			name: "allowlisted long flag",
			args: []string{"--format", "json"},
			want: []string{"--format", "json"},
		},
		{
			name:    "disallowed flag rejected",
			args:    []string{"--exec", "rm -rf /"},
			wantErr: true,
		},
		{
			name:    "dangerous single dash flag rejected",
			args:    []string{"-e", "evil"},
			wantErr: true,
		},
		{
			name: "mixed allowed and positional",
			args: []string{"file.pdf", "-q", "--output", "out.txt"},
			want: []string{"file.pdf", "-q", "--output", "out.txt"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := SanitizeArgs(tc.args)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("length mismatch: got %d, want %d", len(got), len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("arg[%d] mismatch: got %q, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestSanitizeArgs_EmptySlice(t *testing.T) {
	t.Parallel()
	got, err := SanitizeArgs([]string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("expected empty result, got %v", got)
	}
}

func TestSecurityConstants(t *testing.T) {
	t.Parallel()
	if MaxDecompressionRatio != 100 {
		t.Fatalf("MaxDecompressionRatio: got %d, want 100", MaxDecompressionRatio)
	}
	if MaxExtractedFiles != 1000 {
		t.Fatalf("MaxExtractedFiles: got %d, want 1000", MaxExtractedFiles)
	}
}

func TestBaseExtractor_ExtractStreamBuffers(t *testing.T) {
	t.Parallel()
	var called bool
	base := &BaseExtractor{
		ExtractFn: func(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
			called = true
			return ExtractResult{Text: string(raw), Metadata: map[string]string{}}, nil
		},
		ContentTypesFn: func() []string { return []string{"application/x-base"} },
		NameFn:         func() string { return "base-test" },
	}

	reader := strings.NewReader("base stream test")
	result, err := base.ExtractStream(context.Background(), reader, ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !called {
		t.Fatal("expected ExtractFn to be called via ExtractStream")
	}
	if result.Text != "base stream test" {
		t.Fatalf("text mismatch: got %q", result.Text)
	}
}

func TestBaseExtractor_Methods(t *testing.T) {
	t.Parallel()
	base := &BaseExtractor{
		ExtractFn: func(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
			return ExtractResult{Text: string(raw)}, nil
		},
		ContentTypesFn: func() []string { return []string{"test/type"} },
		NameFn:         func() string { return "test-base" },
	}

	if base.Name() != "test-base" {
		t.Fatalf("Name mismatch: %s", base.Name())
	}
	types := base.ContentTypes()
	if len(types) != 1 || types[0] != "test/type" {
		t.Fatalf("ContentTypes mismatch: %v", types)
	}
	result, err := base.Extract(context.Background(), []byte("hi"), ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Text != "hi" {
		t.Fatalf("text mismatch: %q", result.Text)
	}
}

// stubExtractor is a test double that implements Extractor with
// configurable behavior.
type stubExtractor struct {
	name         string
	contentTypes []string
	extractFn    func(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error)
}

var _ Extractor = (*stubExtractor)(nil)

func (s *stubExtractor) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	return s.extractFn(ctx, raw, opts)
}

func (s *stubExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	raw, err := io.ReadAll(reader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("stub: reading stream: %w", err)
	}
	return s.extractFn(ctx, raw, opts)
}

func (s *stubExtractor) ContentTypes() []string {
	return s.contentTypes
}

func (s *stubExtractor) Name() string {
	return s.name
}

func TestExtractorRegistry_ExtractStreamWithMaxBytes(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()

	// Generate a large input that exceeds MaxBytes
	largeInput := bytes.Repeat([]byte("x"), 100)
	reader := bytes.NewReader(largeInput)

	result, err := registry.ExtractStream(context.Background(), reader, ExtractOptions{
		ContentType: "text/plain",
		MaxBytes:    10,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Text) != 10 {
		t.Fatalf("expected text length 10, got %d", len(result.Text))
	}
}
