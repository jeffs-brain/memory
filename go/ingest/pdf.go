// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// PDFExtractorConfig configures the scanned PDF OCR extractor.
type PDFExtractorConfig struct {
	// PdftoppmBinary is the path or name of the pdftoppm binary
	// (from poppler-utils). Default: "pdftoppm".
	// Override via MEMORY_PDFTOPPM_PATH env var.
	PdftoppmBinary string

	// PdftotextBinary is the path or name of the pdftotext binary
	// (from poppler-utils). Default: "pdftotext".
	// Override via MEMORY_PDFTOTEXT_PATH env var.
	PdftotextBinary string

	// ImageExtractor is the delegate used for OCR on converted pages.
	ImageExtractor *ImageExtractor

	// Timeout for the overall PDF extraction process.
	// Default: 120 seconds. Override via MEMORY_EXTRACTOR_TIMEOUT_MS.
	Timeout time.Duration

	// MaxPages limits the number of PDF pages to extract. Default: 100.
	MaxPages int

	// Logger for diagnostic output. Uses slog.Default() when nil.
	Logger *slog.Logger
}

const (
	defaultMaxPDFPages    = 100
	defaultPDFTimeout     = 120 * time.Second
	pdfContentType        = "application/pdf"
	scannedTextThreshold  = 50 // characters below which a PDF page is considered scanned
)

func (c *PDFExtractorConfig) applyDefaults() {
	if c.PdftoppmBinary == "" {
		c.PdftoppmBinary = envOrDefault("MEMORY_PDFTOPPM_PATH", "pdftoppm")
	}
	if c.PdftotextBinary == "" {
		c.PdftotextBinary = envOrDefault("MEMORY_PDFTOTEXT_PATH", "pdftotext")
	}
	if c.Timeout <= 0 {
		if ms := os.Getenv("MEMORY_EXTRACTOR_TIMEOUT_MS"); ms != "" {
			if v, err := strconv.ParseInt(ms, 10, 64); err == nil && v > 0 {
				c.Timeout = time.Duration(v) * time.Millisecond
			}
		}
		if c.Timeout <= 0 {
			c.Timeout = defaultPDFTimeout
		}
	}
	if c.MaxPages == 0 {
		c.MaxPages = defaultMaxPDFPages
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// PDFExtractor extracts text from PDF files. Text-based PDFs are
// handled via pdftotext. Scanned PDFs (no text layer or minimal text)
// are converted page-by-page to images via pdftoppm, then OCR-ed
// via the ImageExtractor delegate.
type PDFExtractor struct {
	cfg PDFExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*PDFExtractor)(nil)

// NewPDFExtractor creates a PDFExtractor with the given config.
// Zero-value fields are populated with sensible defaults.
func NewPDFExtractor(cfg PDFExtractorConfig) *PDFExtractor {
	cfg.applyDefaults()
	return &PDFExtractor{cfg: cfg}
}

// Name returns the extractor identifier.
func (e *PDFExtractor) Name() string {
	return "pdf-ocr"
}

// ContentTypes returns the MIME types this extractor handles.
func (e *PDFExtractor) ContentTypes() []string {
	return []string{pdfContentType}
}

// Capability returns the capability descriptor for PDF extraction.
func (e *PDFExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".pdf"},
		MIMETypes:  []string{pdfContentType},
		MagicBytes: []MagicSignature{
			{Offset: 0, Bytes: []byte{0x25, 0x50, 0x44, 0x46}}, // %PDF
		},
		RequiresBinary: true,
	}
}

// Available reports whether the required binaries are present.
// Both pdftotext (for text extraction) and pdftoppm + an OCR engine
// (for scanned PDF fallback) are checked.
func (e *PDFExtractor) Available(ctx context.Context) (bool, error) {
	hasPdftotext := CheckBinaryAvailable(ctx, e.cfg.PdftotextBinary)
	if !hasPdftotext {
		return false, nil
	}

	hasPdftoppm := CheckBinaryAvailable(ctx, e.cfg.PdftoppmBinary)
	if !hasPdftoppm {
		return false, nil
	}

	if e.cfg.ImageExtractor == nil {
		return false, nil
	}

	return e.cfg.ImageExtractor.Available(ctx)
}

// Extract performs text extraction from a PDF. If the PDF has a
// substantial text layer it uses pdftotext directly. Otherwise it
// converts pages to images and applies OCR.
func (e *PDFExtractor) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	if len(raw) == 0 {
		return ExtractResult{
			Text:     "",
			Metadata: map[string]string{"extractor": "pdf-ocr"},
			Skipped:  true,
			Reason:   "empty input",
		}, nil
	}

	tmpDir, err := os.MkdirTemp("", "memory-pdf-*")
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: creating temp dir for PDF: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	pdfPath := filepath.Join(tmpDir, "input.pdf")
	if err := os.WriteFile(pdfPath, raw, 0o600); err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: writing temp PDF: %w", err)
	}

	// Try text extraction first to detect if scanned.
	textResult, textErr := e.extractWithPdftotext(ctx, pdfPath)
	if textErr == nil && isSubstantialText(textResult) {
		return ExtractResult{
			Text:        textResult,
			ContentType: "text/plain",
			Metadata: map[string]string{
				"extractor":  "pdf-ocr",
				"pdf_method": "pdftotext",
			},
		}, nil
	}

	// PDF is scanned or pdftotext failed; convert to images and OCR.
	e.cfg.Logger.Info("pdf appears scanned or has minimal text layer, falling back to OCR",
		"text_length", len(textResult),
		"file", opts.FileName,
	)

	return e.extractViaOCR(ctx, pdfPath, tmpDir, opts)
}

// ExtractStream buffers the reader and delegates to Extract.
func (e *PDFExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	var limitReader io.Reader = reader
	if opts.MaxBytes > 0 {
		limitReader = io.LimitReader(reader, opts.MaxBytes)
	}
	raw, err := io.ReadAll(limitReader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading PDF stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}

// extractWithPdftotext runs pdftotext to extract the text layer.
func (e *PDFExtractor) extractWithPdftotext(ctx context.Context, pdfPath string) (string, error) {
	result, err := RunSubprocess(ctx, e.cfg.PdftotextBinary, []string{"-layout", pdfPath, "-"}, nil, SubprocessOptions{
		Timeout: e.cfg.Timeout,
	})
	if err != nil {
		return "", fmt.Errorf("ingest: pdftotext: %w", err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("ingest: pdftotext exited with code %d: %s", result.ExitCode, result.Stderr)
	}
	return strings.TrimSpace(string(result.Stdout)), nil
}

// isSubstantialText checks whether extracted text has enough content to
// be considered a genuine text layer rather than artefacts.
func isSubstantialText(text string) bool {
	cleaned := strings.TrimSpace(text)
	return len(cleaned) >= scannedTextThreshold
}

// extractViaOCR converts PDF pages to images via pdftoppm, then runs
// each page through the ImageExtractor for OCR.
func (e *PDFExtractor) extractViaOCR(ctx context.Context, pdfPath, tmpDir string, opts ExtractOptions) (ExtractResult, error) {
	if e.cfg.ImageExtractor == nil {
		return ExtractResult{}, fmt.Errorf("ingest: PDF OCR requires an ImageExtractor dependency")
	}

	// Convert PDF pages to PNG images.
	outputPrefix := filepath.Join(tmpDir, "page")
	maxPagesStr := strconv.Itoa(e.cfg.MaxPages)

	args := []string{
		"-png",
		"-r", "300",
		"-l", maxPagesStr,
		pdfPath,
		outputPrefix,
	}

	result, err := RunSubprocess(ctx, e.cfg.PdftoppmBinary, args, nil, SubprocessOptions{
		Timeout: e.cfg.Timeout,
	})
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: pdftoppm: %w", err)
	}
	if result.ExitCode != 0 {
		return ExtractResult{}, fmt.Errorf("ingest: pdftoppm exited with code %d: %s", result.ExitCode, result.Stderr)
	}

	// Collect generated page images.
	pageImages, err := filepath.Glob(outputPrefix + "*.png")
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: listing page images: %w", err)
	}
	if len(pageImages) == 0 {
		return ExtractResult{}, fmt.Errorf("ingest: pdftoppm produced no page images")
	}

	// OCR each page image.
	var pageTexts []string
	var totalConfidence float64
	var pageCount int

	for _, imagePath := range pageImages {
		imageData, readErr := os.ReadFile(imagePath)
		if readErr != nil {
			e.cfg.Logger.Warn("skipping unreadable page image",
				"path", imagePath,
				"error", readErr,
			)
			continue
		}

		pageOpts := ExtractOptions{
			ContentType: "image/png",
			FileName:    filepath.Base(imagePath),
			Language:    opts.Language,
		}

		pageResult, ocrErr := e.cfg.ImageExtractor.Extract(ctx, imageData, pageOpts)
		if ocrErr != nil {
			e.cfg.Logger.Warn("OCR failed for page",
				"page", filepath.Base(imagePath),
				"error", ocrErr,
			)
			continue
		}

		if !pageResult.Skipped && pageResult.Text != "" {
			pageTexts = append(pageTexts, pageResult.Text)
			totalConfidence += pageResult.Confidence
			pageCount++
		}
	}

	avgConfidence := 0.0
	if pageCount > 0 {
		avgConfidence = totalConfidence / float64(pageCount)
	}

	return ExtractResult{
		Text:        strings.Join(pageTexts, "\n\n"),
		ContentType: "text/plain",
		Metadata: map[string]string{
			"extractor":      "pdf-ocr",
			"pdf_method":     "pdftoppm+ocr",
			"pdf_pages":      strconv.Itoa(len(pageImages)),
			"ocr_pages":      strconv.Itoa(pageCount),
			"ocr_confidence": formatConfidence(avgConfidence),
		},
		Pages:      len(pageImages),
		Language:   opts.Language,
		Confidence: avgConfidence,
	}, nil
}
