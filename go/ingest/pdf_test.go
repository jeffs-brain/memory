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

func TestPDFExtractor_Name(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{})
	if ext.Name() != "pdf-ocr" {
		t.Fatalf("expected name %q, got %q", "pdf-ocr", ext.Name())
	}
}

func TestPDFExtractor_ContentTypes(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{})
	types := ext.ContentTypes()
	if len(types) != 1 {
		t.Fatalf("expected 1 content type, got %d", len(types))
	}
	if types[0] != "application/pdf" {
		t.Errorf("expected content type %q, got %q", "application/pdf", types[0])
	}
}

func TestPDFExtractor_Capability(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{})
	cap := ext.Capability()

	if len(cap.Extensions) != 1 || cap.Extensions[0] != ".pdf" {
		t.Errorf("expected extensions [.pdf], got %v", cap.Extensions)
	}

	if len(cap.MIMETypes) != 1 || cap.MIMETypes[0] != "application/pdf" {
		t.Errorf("expected MIME types [application/pdf], got %v", cap.MIMETypes)
	}

	// Verify PDF magic bytes: %PDF (0x25 0x50 0x44 0x46).
	if len(cap.MagicBytes) == 0 {
		t.Fatal("expected magic bytes to be defined")
	}
	sig := cap.MagicBytes[0]
	if sig.Offset != 0 || len(sig.Bytes) != 4 ||
		sig.Bytes[0] != 0x25 || sig.Bytes[1] != 0x50 ||
		sig.Bytes[2] != 0x44 || sig.Bytes[3] != 0x46 {
		t.Error("expected %PDF magic bytes at offset 0")
	}

	if !cap.RequiresBinary {
		t.Error("expected RequiresBinary=true")
	}
}

func TestPDFExtractor_ImplementsExtractorInterface(t *testing.T) {
	t.Parallel()
	var _ Extractor = (*PDFExtractor)(nil)
}

func TestPDFExtractor_ConfigDefaults(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{})
	if ext.cfg.PdftoppmBinary != "pdftoppm" {
		t.Errorf("expected default PdftoppmBinary %q, got %q", "pdftoppm", ext.cfg.PdftoppmBinary)
	}
	if ext.cfg.PdftotextBinary != "pdftotext" {
		t.Errorf("expected default PdftotextBinary %q, got %q", "pdftotext", ext.cfg.PdftotextBinary)
	}
	if ext.cfg.MaxPages != defaultMaxPDFPages {
		t.Errorf("expected default MaxPages %d, got %d", defaultMaxPDFPages, ext.cfg.MaxPages)
	}
	if ext.cfg.Timeout != defaultPDFTimeout {
		t.Errorf("expected default Timeout %v, got %v", defaultPDFTimeout, ext.cfg.Timeout)
	}
}

func TestPDFExtractor_ConfigOverrides(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{
		PdftoppmBinary:  "/usr/local/bin/pdftoppm",
		PdftotextBinary: "/usr/local/bin/pdftotext",
		MaxPages:        50,
		Timeout:         90 * time.Second,
	})
	if ext.cfg.PdftoppmBinary != "/usr/local/bin/pdftoppm" {
		t.Errorf("expected PdftoppmBinary override, got %q", ext.cfg.PdftoppmBinary)
	}
	if ext.cfg.PdftotextBinary != "/usr/local/bin/pdftotext" {
		t.Errorf("expected PdftotextBinary override, got %q", ext.cfg.PdftotextBinary)
	}
	if ext.cfg.MaxPages != 50 {
		t.Errorf("expected MaxPages 50, got %d", ext.cfg.MaxPages)
	}
	if ext.cfg.Timeout != 90*time.Second {
		t.Errorf("expected Timeout 90s, got %v", ext.cfg.Timeout)
	}
}

func TestPDFExtractor_Available_NoBinaries(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ext := NewPDFExtractor(PDFExtractorConfig{
		PdftoppmBinary:  "nonexistent-pdftoppm-xyz",
		PdftotextBinary: "nonexistent-pdftotext-xyz",
	})
	available, err := ext.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if available {
		t.Error("expected Available=false when binaries are absent")
	}
}

func TestPDFExtractor_Available_NoImageExtractor(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{
		ImageExtractor: nil,
	})
	available, err := ext.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if available {
		t.Error("expected Available=false when ImageExtractor is nil")
	}
}

func TestPDFExtractor_Extract_EmptyInput(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{})
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

func TestIsSubstantialText(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		input    string
		expected bool
	}{
		{"empty string", "", false},
		{"whitespace only", "   \n\t  ", false},
		{"short artefact", "  \f  ", false},
		{"just under threshold", strings.Repeat("a", scannedTextThreshold-1), false},
		{"at threshold", strings.Repeat("a", scannedTextThreshold), true},
		{"substantial text", "This is a real paragraph with enough words to pass the threshold.", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := isSubstantialText(tc.input)
			if result != tc.expected {
				t.Errorf("isSubstantialText(%q) = %v, want %v", tc.input, result, tc.expected)
			}
		})
	}
}

// --- Mock-based tests for the PDF extraction pipeline ---

const mockPdftotextScriptEmpty = `#!/bin/sh
# Simulates a scanned PDF with no text layer.
echo ""
`

const mockPdftotextScriptWithText = `#!/bin/sh
# Simulates a text-based PDF with substantial content.
echo "This is a PDF document with a genuine text layer that contains enough characters to pass the scanned threshold test."
`

const mockPdftoppmScript = `#!/bin/sh
# Creates a fake PNG file to simulate page image output.
# pdftoppm args: -png -r 300 -l <maxPages> <pdfPath> <outputPrefix>
# The output prefix is the last argument ($6).
prefix=""
for arg; do
  prefix="$arg"
done
# Create a minimal file that the OCR mock can process.
printf '\x89PNG' > "${prefix}-1.png"
`

func TestPDFExtractor_TextBasedPDF_MockedSubprocess(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	pdftotextPath := writeMockScript(t, tmpDir, "pdftotext", mockPdftotextScriptWithText)

	imgExt := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "nonexistent-paddleocr-xyz",
		TesseractBinary: "nonexistent-tesseract-xyz",
	})

	ext := NewPDFExtractor(PDFExtractorConfig{
		PdftotextBinary: pdftotextPath,
		PdftoppmBinary:  "nonexistent-pdftoppm-xyz",
		ImageExtractor:  imgExt,
		Timeout:         10 * time.Second,
	})

	// Create a dummy PDF (just needs to be non-empty for the test).
	pdfData := []byte("%PDF-1.4 dummy content for testing")
	result, err := ext.Extract(context.Background(), pdfData, ExtractOptions{
		ContentType: "application/pdf",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Fatal("expected non-skipped result for text-based PDF")
	}
	if !strings.Contains(result.Text, "genuine text layer") {
		t.Errorf("expected text from pdftotext, got %q", result.Text)
	}
	if result.Metadata["pdf_method"] != "pdftotext" {
		t.Errorf("expected pdf_method=pdftotext, got %q", result.Metadata["pdf_method"])
	}
}

func TestPDFExtractor_ScannedPDF_MockedSubprocess(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	pdftotextPath := writeMockScript(t, tmpDir, "pdftotext", mockPdftotextScriptEmpty)
	pdftoppmPath := writeMockScript(t, tmpDir, "pdftoppm", mockPdftoppmScript)
	paddlePath := writeMockScript(t, tmpDir, "paddleocr", mockPaddleOCRScript)

	imgExt := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: paddlePath,
		TesseractBinary: "nonexistent-tesseract-xyz",
		Timeout:         10 * time.Second,
	})

	ext := NewPDFExtractor(PDFExtractorConfig{
		PdftotextBinary: pdftotextPath,
		PdftoppmBinary:  pdftoppmPath,
		ImageExtractor:  imgExt,
		Timeout:         10 * time.Second,
	})

	pdfData := []byte("%PDF-1.4 dummy scanned content")
	result, err := ext.Extract(context.Background(), pdfData, ExtractOptions{
		ContentType: "application/pdf",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Skipped {
		t.Fatal("expected non-skipped result")
	}
	if result.Metadata["pdf_method"] != "pdftoppm+ocr" {
		t.Errorf("expected pdf_method=pdftoppm+ocr, got %q", result.Metadata["pdf_method"])
	}
	if result.Metadata["extractor"] != "pdf-ocr" {
		t.Errorf("expected extractor=pdf-ocr, got %q", result.Metadata["extractor"])
	}
}

func TestPDFExtractor_NoImageExtractor_ReturnsError(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	pdftotextPath := writeMockScript(t, tmpDir, "pdftotext", mockPdftotextScriptEmpty)

	ext := NewPDFExtractor(PDFExtractorConfig{
		PdftotextBinary: pdftotextPath,
		PdftoppmBinary:  "nonexistent-pdftoppm-xyz",
		ImageExtractor:  nil,
		Timeout:         10 * time.Second,
	})

	pdfData := []byte("%PDF-1.4 dummy scanned content")
	_, err := ext.Extract(context.Background(), pdfData, ExtractOptions{
		ContentType: "application/pdf",
	})
	if err == nil {
		t.Fatal("expected error when ImageExtractor is nil and PDF is scanned")
	}
	if !strings.Contains(err.Error(), "ImageExtractor") {
		t.Errorf("expected error to mention ImageExtractor: %v", err)
	}
}

func TestPDFExtractor_ExtractStream_EmptyInput(t *testing.T) {
	t.Parallel()
	ext := NewPDFExtractor(PDFExtractorConfig{})
	reader := bytes.NewReader(nil)
	result, err := ext.ExtractStream(context.Background(), reader, ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Error("expected Skipped=true for empty stream")
	}
}

func TestPDFExtractor_TempFileCleanup(t *testing.T) {
	ResetBinaryCache()
	defer ResetBinaryCache()

	beforeFiles := tempFilesMatching("memory-pdf-*")

	tmpDir := t.TempDir()
	pdftotextPath := writeMockScript(t, tmpDir, "pdftotext", mockPdftotextScriptWithText)

	imgExt := NewImageExtractor(ImageExtractorConfig{
		PaddleOCRBinary: "nonexistent-paddleocr-xyz",
		TesseractBinary: "nonexistent-tesseract-xyz",
	})

	ext := NewPDFExtractor(PDFExtractorConfig{
		PdftotextBinary: pdftotextPath,
		PdftoppmBinary:  "nonexistent-pdftoppm-xyz",
		ImageExtractor:  imgExt,
		Timeout:         10 * time.Second,
	})

	pdfData := []byte("%PDF-1.4 dummy content")
	_, _ = ext.Extract(context.Background(), pdfData, ExtractOptions{
		ContentType: "application/pdf",
	})

	afterFiles := tempFilesMatching("memory-pdf-*")
	newFiles := diffStringSlices(afterFiles, beforeFiles)
	if len(newFiles) > 0 {
		t.Errorf("temp files not cleaned up: %v", newFiles)
	}
}

func TestPDFExtractor_EnvVarOverrides(t *testing.T) {
	t.Setenv("MEMORY_PDFTOPPM_PATH", "/custom/pdftoppm")
	t.Setenv("MEMORY_PDFTOTEXT_PATH", "/custom/pdftotext")

	ext := NewPDFExtractor(PDFExtractorConfig{})
	if ext.cfg.PdftoppmBinary != "/custom/pdftoppm" {
		t.Errorf("expected PdftoppmBinary from env, got %q", ext.cfg.PdftoppmBinary)
	}
	if ext.cfg.PdftotextBinary != "/custom/pdftotext" {
		t.Errorf("expected PdftotextBinary from env, got %q", ext.cfg.PdftotextBinary)
	}
}

func TestPDFExtractor_Integration(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ctx := context.Background()
	imgExt := NewImageExtractor(ImageExtractorConfig{})

	ext := NewPDFExtractor(PDFExtractorConfig{
		ImageExtractor: imgExt,
	})

	available, _ := ext.Available(ctx)
	if !available {
		t.Skip("skipping integration test: required binaries not available")
	}

	// Create a minimal valid PDF with text content.
	pdfContent := createMinimalTextPDF()
	result, err := ext.Extract(ctx, pdfContent, ExtractOptions{
		ContentType: "application/pdf",
		FileName:    "test.pdf",
	})
	if err != nil {
		t.Fatalf("integration test extraction failed: %v", err)
	}
	if result.Skipped {
		t.Error("expected non-skipped result from integration test")
	}
	t.Logf("integration test: method=%s, text_length=%d", result.Metadata["pdf_method"], len(result.Text))
}

// createMinimalTextPDF generates a minimal valid PDF with "Hello World" text.
func createMinimalTextPDF() []byte {
	pdf := `%PDF-1.0
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT /F1 12 Tf 100 700 Td (Hello World) Tj ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000360 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
441
%%EOF`

	// Fix the offsets - for testing the exact offsets don't matter since
	// we're testing the extraction pipeline, not PDF parsing.
	return []byte(pdf)
}

// tempDirsMatching returns paths of directories in the system temp dir
// matching the given glob pattern.
func tempDirsMatching(pattern string) []string {
	entries, _ := filepath.Glob(filepath.Join(os.TempDir(), pattern))
	return entries
}
