// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestImageExtractor_Name(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})
	if ext.Name() != "image-ocr" {
		t.Fatalf("expected name %q, got %q", "image-ocr", ext.Name())
	}
}

func TestImageExtractor_ContentTypes(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})
	types := ext.ContentTypes()

	expected := map[string]struct{}{
		"image/png":  {},
		"image/jpeg": {},
		"image/tiff": {},
		"image/bmp":  {},
		"image/webp": {},
	}

	for _, ct := range types {
		if _, ok := expected[ct]; !ok {
			t.Errorf("unexpected content type: %s", ct)
		}
		delete(expected, ct)
	}

	for ct := range expected {
		t.Errorf("missing content type: %s", ct)
	}
}

func TestImageExtractor_Capability(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})
	cap := ext.Capability()

	// Verify extensions.
	expectedExts := map[string]struct{}{
		".png": {}, ".jpg": {}, ".jpeg": {}, ".tiff": {},
		".tif": {}, ".bmp": {}, ".webp": {},
	}
	for _, e := range cap.Extensions {
		if _, ok := expectedExts[e]; !ok {
			t.Errorf("unexpected extension: %s", e)
		}
		delete(expectedExts, e)
	}
	for e := range expectedExts {
		t.Errorf("missing extension: %s", e)
	}

	// Verify MIME types match content types.
	if len(cap.MIMETypes) != len(ext.ContentTypes()) {
		t.Errorf("MIME types count mismatch: capability=%d, contentTypes=%d", len(cap.MIMETypes), len(ext.ContentTypes()))
	}

	// Verify magic bytes are present.
	if len(cap.MagicBytes) == 0 {
		t.Error("expected magic bytes to be defined")
	}

	// PNG magic: 89 50 4E 47
	foundPNG := false
	for _, sig := range cap.MagicBytes {
		if len(sig.Bytes) >= 4 && sig.Bytes[0] == 0x89 && sig.Bytes[1] == 0x50 && sig.Bytes[2] == 0x4E && sig.Bytes[3] == 0x47 {
			foundPNG = true
		}
	}
	if !foundPNG {
		t.Error("missing PNG magic bytes")
	}

	if !cap.RequiresBinary {
		t.Error("expected RequiresBinary=true")
	}
}

func TestImageExtractor_Available_NoBinaries(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "nonexistent-paddleocr-binary-xyz",
		TesseractBinary: "nonexistent-tesseract-binary-xyz",
	})

	available, err := ext.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if available {
		t.Error("expected Available=false when no OCR binaries exist")
	}
}

func TestImageExtractor_Extract_EmptyInput(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})
	result, err := ext.Extract(context.Background(), nil, ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true for empty input")
	}
	if result.Reason != "empty input" {
		t.Errorf("expected reason %q, got %q", "empty input", result.Reason)
	}
}

func TestImageExtractor_Extract_NoBinaries(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "nonexistent-paddleocr-binary-xyz",
		TesseractBinary: "nonexistent-tesseract-binary-xyz",
	})

	_, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
	})
	if err == nil {
		t.Fatal("expected error when no OCR binaries are available")
	}
	if !strings.Contains(err.Error(), "no OCR engine available") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestImageExtractor_ImplementsExtractorInterface(t *testing.T) {
	t.Parallel()
	var _ Extractor = (*ImageExtractor)(nil)
}

func TestParsePaddleOCROutput_JSONFormat(t *testing.T) {
	t.Parallel()
	// Simulates PaddleOCR JSON output: [["text", confidence], ...]
	output := `[["Hello World", 0.95], ["Second Line", 0.88]]`

	text, confidence := parsePaddleOCROutput(output)

	if !strings.Contains(text, "Hello World") {
		t.Errorf("expected text to contain %q, got %q", "Hello World", text)
	}
	if !strings.Contains(text, "Second Line") {
		t.Errorf("expected text to contain %q, got %q", "Second Line", text)
	}

	expectedConf := (0.95 + 0.88) / 2.0
	if confidence < expectedConf-0.01 || confidence > expectedConf+0.01 {
		t.Errorf("expected confidence ~%.2f, got %.4f", expectedConf, confidence)
	}
}

func TestParsePaddleOCROutput_TupleFormat(t *testing.T) {
	t.Parallel()
	// Simulates PaddleOCR tuple-style output
	output := "('Hello World', 0.95)\n('Second Line', 0.88)\n"

	text, confidence := parsePaddleOCROutput(output)

	if !strings.Contains(text, "Hello World") {
		t.Errorf("expected text to contain %q, got %q", "Hello World", text)
	}
	if !strings.Contains(text, "Second Line") {
		t.Errorf("expected text to contain %q, got %q", "Second Line", text)
	}

	expectedConf := (0.95 + 0.88) / 2.0
	if confidence < expectedConf-0.01 || confidence > expectedConf+0.01 {
		t.Errorf("expected confidence ~%.2f, got %.4f", expectedConf, confidence)
	}
}

func TestParsePaddleOCROutput_EmptyOutput(t *testing.T) {
	t.Parallel()
	text, confidence := parsePaddleOCROutput("")
	if text != "" {
		t.Errorf("expected empty text, got %q", text)
	}
	if confidence != 0.0 {
		t.Errorf("expected 0 confidence, got %f", confidence)
	}
}

func TestMapLanguageToPaddleOCR(t *testing.T) {
	t.Parallel()
	cases := []struct {
		input    string
		expected string
	}{
		{"en", "en"},
		{"zh", "ch"},
		{"ja", "japan"},
		{"de", "german"},
		{"EN", "en"},
		{"unknown", "unknown"},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			t.Parallel()
			result := mapLanguageToPaddleOCR(tc.input)
			if result != tc.expected {
				t.Errorf("mapLanguageToPaddleOCR(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestMapLanguageToTesseract(t *testing.T) {
	t.Parallel()
	cases := []struct {
		input    string
		expected string
	}{
		{"en", "eng"},
		{"zh", "chi_sim"},
		{"ja", "jpn"},
		{"de", "deu"},
		{"unknown", "eng"},
	}
	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			t.Parallel()
			result := mapLanguageToTesseract(tc.input)
			if result != tc.expected {
				t.Errorf("mapLanguageToTesseract(%q) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestExtensionFromOpts(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		opts     ExtractOptions
		expected string
	}{
		{
			"png from content type",
			ExtractOptions{ContentType: "image/png"},
			".png",
		},
		{
			"jpeg from content type",
			ExtractOptions{ContentType: "image/jpeg"},
			".jpg",
		},
		{
			"tiff from content type",
			ExtractOptions{ContentType: "image/tiff"},
			".tiff",
		},
		{
			"default to png for unknown",
			ExtractOptions{ContentType: "application/octet-stream"},
			".png",
		},
		{
			"extension from filename when content type unknown",
			ExtractOptions{ContentType: "application/octet-stream", FileName: "photo.bmp"},
			".bmp",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := extensionFromOpts(tc.opts)
			if result != tc.expected {
				t.Errorf("extensionFromOpts() = %q, want %q", result, tc.expected)
			}
		})
	}
}

func TestStripQuotes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		input    string
		expected string
	}{
		{`'hello'`, "hello"},
		{`"hello"`, "hello"},
		{`hello`, "hello"},
		{`''`, ""},
		{`'`, "'"},
		{``, ""},
	}

	for _, tc := range cases {
		t.Run(tc.input, func(t *testing.T) {
			t.Parallel()
			if got := stripQuotes(tc.input); got != tc.expected {
				t.Errorf("stripQuotes(%q) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestFormatConfidence(t *testing.T) {
	t.Parallel()
	cases := []struct {
		input    float64
		expected string
	}{
		{0.95, "0.9500"},
		{1.0, "1.0000"},
		{0.0, "0.0000"},
		{0.12345, "0.1235"},
	}

	for _, tc := range cases {
		t.Run(tc.expected, func(t *testing.T) {
			t.Parallel()
			if got := formatConfidence(tc.input); got != tc.expected {
				t.Errorf("formatConfidence(%f) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

func TestImageExtractor_ConfigDefaults(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})

	if ext.cfg.PaddleOCRBinary != "paddleocr" {
		t.Errorf("expected default PaddleOCRBinary %q, got %q", "paddleocr", ext.cfg.PaddleOCRBinary)
	}
	if ext.cfg.TesseractBinary != "tesseract" {
		t.Errorf("expected default TesseractBinary %q, got %q", "tesseract", ext.cfg.TesseractBinary)
	}
	if ext.cfg.DefaultLanguage != "en" {
		t.Errorf("expected default language %q, got %q", "en", ext.cfg.DefaultLanguage)
	}
	if ext.cfg.Timeout != DefaultSubprocessTimeout {
		t.Errorf("expected default timeout %v, got %v", DefaultSubprocessTimeout, ext.cfg.Timeout)
	}
}

func TestImageExtractor_ConfigOverrides(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "/usr/local/bin/paddleocr",
		TesseractBinary: "/usr/local/bin/tesseract",
		DefaultLanguage: "de",
		Timeout:         30 * time.Second,
	})

	if ext.cfg.PaddleOCRBinary != "/usr/local/bin/paddleocr" {
		t.Errorf("expected PaddleOCRBinary override, got %q", ext.cfg.PaddleOCRBinary)
	}
	if ext.cfg.TesseractBinary != "/usr/local/bin/tesseract" {
		t.Errorf("expected TesseractBinary override, got %q", ext.cfg.TesseractBinary)
	}
	if ext.cfg.DefaultLanguage != "de" {
		t.Errorf("expected language override, got %q", ext.cfg.DefaultLanguage)
	}
	if ext.cfg.Timeout != 30*time.Second {
		t.Errorf("expected timeout override, got %v", ext.cfg.Timeout)
	}
}

func TestImageExtractor_ResolveLanguage(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{DefaultLanguage: "en"})

	// When encoding field provides a language hint.
	lang := ext.resolveLanguage(ExtractOptions{Encoding: "de"})
	if lang != "de" {
		t.Errorf("expected language %q from opts, got %q", "de", lang)
	}

	// Default when no encoding set.
	lang = ext.resolveLanguage(ExtractOptions{})
	if lang != "en" {
		t.Errorf("expected default language %q, got %q", "en", lang)
	}
}
