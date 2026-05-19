// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
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

	// When Language field provides a language hint.
	lang := ext.resolveLanguage(ExtractOptions{Language: "de"})
	if lang != "de" {
		t.Errorf("expected language %q from opts, got %q", "de", lang)
	}

	// Default when no language set.
	lang = ext.resolveLanguage(ExtractOptions{})
	if lang != "en" {
		t.Errorf("expected default language %q, got %q", "en", lang)
	}
}

// --- C2: Language parameter validation ---

func TestValidateLanguage(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name      string
		input     string
		expectErr bool
	}{
		{"valid language code", "en", false},
		{"valid language code de", "de", false},
		{"valid mapped code", "ch", false},
		{"flag injection single dash", "-malicious", true},
		{"flag injection double dash", "--exploit", true},
		{"empty string", "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			err := validateLanguage(tc.input)
			if tc.expectErr && err == nil {
				t.Errorf("expected error for language %q, got nil", tc.input)
			}
			if !tc.expectErr && err != nil {
				t.Errorf("unexpected error for language %q: %v", tc.input, err)
			}
		})
	}
}

func TestImageExtractor_Extract_RejectsFlagInjection(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})

	_, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
		Language:    "--malicious-flag",
	})
	if err == nil {
		t.Fatal("expected error for flag-injected language parameter")
	}
	if !strings.Contains(err.Error(), "invalid language parameter") {
		t.Errorf("expected flag injection error, got: %v", err)
	}
}

// --- C1: Happy-path mock subprocess tests ---

// mockPaddleOCRScript returns a shell command that emits PaddleOCR-like
// JSON output to stdout without actually running PaddleOCR. This enables
// testing the full extraction pipeline through to output parsing.
const mockPaddleOCRScript = `#!/bin/sh
echo '[["Hello World", 0.95], ["Second Line", 0.88]]'
`

const mockTesseractScript = `#!/bin/sh
echo 'Hello World from Tesseract'
`

func writeMockScript(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("writing mock script %s: %v", name, err)
	}
	return path
}

func TestImageExtractor_ExtractWithPaddleOCR_MockedSubprocess(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", mockPaddleOCRScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: "nonexistent-tesseract-xyz",
		DefaultLanguage: "en",
		Timeout:         10 * time.Second,
	})

	result, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Fatal("expected non-skipped result")
	}
	if !strings.Contains(result.Text, "Hello World") {
		t.Errorf("expected text to contain %q, got %q", "Hello World", result.Text)
	}
	if !strings.Contains(result.Text, "Second Line") {
		t.Errorf("expected text to contain %q, got %q", "Second Line", result.Text)
	}
	if result.Metadata["ocr_engine"] != "paddleocr" {
		t.Errorf("expected ocr_engine=paddleocr, got %q", result.Metadata["ocr_engine"])
	}
	if result.Confidence < 0.8 || result.Confidence > 1.0 {
		t.Errorf("expected confidence in [0.8, 1.0], got %f", result.Confidence)
	}
}

func TestImageExtractor_ExtractWithTesseract_MockedSubprocess(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	tesseractPath := writeMockScript(t, tmpDir, "tesseract", mockTesseractScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "nonexistent-paddleocr-xyz",
		TesseractBinary: tesseractPath,
		DefaultLanguage: "en",
		Timeout:         10 * time.Second,
	})

	result, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Fatal("expected non-skipped result")
	}
	if !strings.Contains(result.Text, "Hello World from Tesseract") {
		t.Errorf("expected text to contain %q, got %q", "Hello World from Tesseract", result.Text)
	}
	if result.Metadata["ocr_engine"] != "tesseract" {
		t.Errorf("expected ocr_engine=tesseract, got %q", result.Metadata["ocr_engine"])
	}
}

func TestImageExtractor_PaddleOCRFallsBackToTesseract_MockedSubprocess(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()

	// PaddleOCR script that exits with error to trigger fallback.
	failScript := "#!/bin/sh\nexit 1\n"
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", failScript)
	tesseractPath := writeMockScript(t, tmpDir, "tesseract", mockTesseractScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: tesseractPath,
		DefaultLanguage: "en",
		Timeout:         10 * time.Second,
	})

	result, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "Hello World from Tesseract") {
		t.Errorf("expected tesseract fallback text, got %q", result.Text)
	}
	if result.Metadata["ocr_engine"] != "tesseract" {
		t.Errorf("expected ocr_engine=tesseract after fallback, got %q", result.Metadata["ocr_engine"])
	}
}

// --- M3: Temp file cleanup verification ---

func TestWriteTempFile_CleansUpAfterCreation(t *testing.T) {
	t.Parallel()
	data := []byte("test image bytes")

	path, err := writeTempFile(data, "memory-ocr-*.png")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify file exists.
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("temp file should exist immediately after creation: %v", err)
	}

	// Remove it (mimicking the defer in extraction).
	if err := os.Remove(path); err != nil {
		t.Fatalf("failed to remove temp file: %v", err)
	}

	// Verify file no longer exists.
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Error("temp file should not exist after removal")
	}
}

func TestImageExtractor_TempFileCleanedUpAfterSuccess(t *testing.T) {
	// Non-parallel: scans system temp directory for leftovers.
	ResetBinaryCache()
	defer ResetBinaryCache()

	// Record existing temp files before extraction.
	beforeFiles := tempFilesMatching("memory-ocr-*")

	tmpDir := t.TempDir()
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", mockPaddleOCRScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: "nonexistent-tesseract-xyz",
		Timeout:         10 * time.Second,
	})

	_, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// After successful extraction, verify no new temp files remain.
	afterFiles := tempFilesMatching("memory-ocr-*")
	newFiles := diffStringSlices(afterFiles, beforeFiles)
	if len(newFiles) > 0 {
		t.Errorf("temp file not cleaned up after successful extraction: %v", newFiles)
	}
}

func TestImageExtractor_TempFileCleanedUpAfterError(t *testing.T) {
	// Non-parallel: scans system temp directory for leftovers.
	ResetBinaryCache()
	defer ResetBinaryCache()

	beforeFiles := tempFilesMatching("memory-ocr-*")

	tmpDir := t.TempDir()
	failScript := "#!/bin/sh\nexit 1\n"
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", failScript)
	tesseractFailPath := writeMockScript(t, tmpDir, "tesseract", failScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: tesseractFailPath,
		Timeout:         10 * time.Second,
	})

	_, _ = ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
	})

	// After failed extraction, verify no new temp files remain.
	afterFiles := tempFilesMatching("memory-ocr-*")
	newFiles := diffStringSlices(afterFiles, beforeFiles)
	if len(newFiles) > 0 {
		t.Errorf("temp file not cleaned up after failed extraction: %v", newFiles)
	}
}

// tempFilesMatching returns paths of files in the system temp dir
// matching the given glob pattern.
func tempFilesMatching(pattern string) []string {
	matches, _ := filepath.Glob(filepath.Join(os.TempDir(), pattern))
	return matches
}

// diffStringSlices returns elements in after that are not in before.
func diffStringSlices(after, before []string) []string {
	beforeSet := make(map[string]struct{}, len(before))
	for _, s := range before {
		beforeSet[s] = struct{}{}
	}
	var diff []string
	for _, s := range after {
		if _, exists := beforeSet[s]; !exists {
			diff = append(diff, s)
		}
	}
	return diff
}

// --- m5: ExtractStream test ---

func TestImageExtractor_ExtractStream_NoBinaries(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "nonexistent-paddleocr-binary-xyz",
		TesseractBinary: "nonexistent-tesseract-binary-xyz",
	})

	reader := bytes.NewReader([]byte{0x89, 0x50, 0x4E, 0x47})
	_, err := ext.ExtractStream(context.Background(), reader, ExtractOptions{
		ContentType: "image/png",
	})
	if err == nil {
		t.Fatal("expected error when no OCR binaries are available")
	}
	if !strings.Contains(err.Error(), "no OCR engine available") {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestImageExtractor_ExtractStream_EmptyInput(t *testing.T) {
	t.Parallel()
	ext := NewImageExtractor(ImageExtractorConfig{})

	reader := bytes.NewReader(nil)
	result, err := ext.ExtractStream(context.Background(), reader, ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true for empty stream input")
	}
}

func TestImageExtractor_ExtractStream_MockedPaddleOCR(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", mockPaddleOCRScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: "nonexistent-tesseract-xyz",
		Timeout:         10 * time.Second,
	})

	reader := bytes.NewReader([]byte{0x89, 0x50, 0x4E, 0x47})
	result, err := ext.ExtractStream(context.Background(), reader, ExtractOptions{
		ContentType: "image/png",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Text, "Hello World") {
		t.Errorf("expected text to contain %q, got %q", "Hello World", result.Text)
	}
}

// --- m6: PaddleOCR output with interleaved garbage/log lines ---

func TestParsePaddleOCROutput_InterleavedLogLines(t *testing.T) {
	t.Parallel()
	output := "Loading model...\nInitializing engine...\n" +
		`[["Hello", 0.95]]` + "\n" +
		"Processing complete.\n"

	text, confidence := parsePaddleOCROutput(output)

	if text != "Hello" {
		t.Errorf("expected text %q, got %q", "Hello", text)
	}
	if confidence < 0.94 || confidence > 0.96 {
		t.Errorf("expected confidence ~0.95, got %f", confidence)
	}
}

// --- M5: Integration test skeleton ---

func TestImageExtractor_Integration(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ctx := context.Background()
	ext := NewImageExtractor(ImageExtractorConfig{})

	available, _ := ext.Available(ctx)
	if !available {
		t.Skip("skipping integration test: no OCR binary available on this system")
	}

	// Create a minimal 1x1 white PNG for testing.
	// PNG header + IHDR + IDAT + IEND.
	pngData := createMinimalPNG()

	result, err := ext.Extract(ctx, pngData, ExtractOptions{
		ContentType: "image/png",
		FileName:    "test.png",
	})
	if err != nil {
		t.Fatalf("integration test extraction failed: %v", err)
	}

	if result.Skipped {
		t.Error("expected non-skipped result from integration test")
	}

	engine := result.Metadata["ocr_engine"]
	if engine != "paddleocr" && engine != "tesseract" {
		t.Errorf("expected ocr_engine to be paddleocr or tesseract, got %q", engine)
	}

	t.Logf("integration test used engine=%s, text_length=%d", engine, len(result.Text))
}

// createMinimalPNG generates a valid 1x1 white PNG file in memory.
func createMinimalPNG() []byte {
	// Rather than importing image/png, use a pre-computed minimal valid PNG.
	// 1x1 pixel, 8-bit RGB, white pixel.
	return []byte{
		0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
		0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk length + type
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1
		0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // 8-bit RGB
		0xDE, // CRC
		0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, // IDAT chunk
		0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, // compressed data
		0x00, 0x02, 0x00, 0x01, 0xE2, 0x21, 0xBC, 0x33, // CRC
		0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, // IEND chunk
		0xAE, 0x42, 0x60, 0x82, // IEND CRC
	}
}

// --- C1: Verify language is passed through to OCR metadata ---

func TestImageExtractor_ExtractWithLanguageOverride(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", mockPaddleOCRScript)

	ext := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: "nonexistent-tesseract-xyz",
		DefaultLanguage: "en",
		Timeout:         10 * time.Second,
	})

	result, err := ext.Extract(context.Background(), []byte{0x89, 0x50, 0x4E, 0x47}, ExtractOptions{
		ContentType: "image/png",
		Language:    "de",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Language != "de" {
		t.Errorf("expected Language=%q, got %q", "de", result.Language)
	}
	if result.Metadata["ocr_language"] != "german" {
		t.Errorf("expected ocr_language=%q, got %q", "german", result.Metadata["ocr_language"])
	}
}

