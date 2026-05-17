// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// ImageExtractorConfig configures the image OCR extractor.
type ImageExtractorConfig struct {
	// PaddleOCRBinary is the path or name of the PaddleOCR binary.
	// Default: "paddleocr". Override via MEMORY_PADDLEOCR_PATH env var.
	PaddleOCRBinary string

	// TesseractBinary is the path or name of the Tesseract binary.
	// Default: "tesseract". Override via MEMORY_TESSERACT_PATH env var.
	TesseractBinary string

	// DefaultLanguage is the OCR language hint. Default: "en".
	DefaultLanguage string

	// Timeout for each OCR subprocess invocation.
	// Default: 60 seconds. Override via MEMORY_EXTRACTOR_TIMEOUT_MS env var.
	Timeout time.Duration

	// Logger for diagnostic output. Uses slog.Default() when nil.
	Logger *slog.Logger
}

// applyDefaults fills zero-value fields with defaults and environment
// variable overrides.
func (c *ImageExtractorConfig) applyDefaults() {
	if c.PaddleOCRBinary == "" {
		c.PaddleOCRBinary = envOrDefault("MEMORY_PADDLEOCR_PATH", "paddleocr")
	}
	if c.TesseractBinary == "" {
		c.TesseractBinary = envOrDefault("MEMORY_TESSERACT_PATH", "tesseract")
	}
	if c.DefaultLanguage == "" {
		c.DefaultLanguage = "en"
	}
	if c.Timeout <= 0 {
		if ms := os.Getenv("MEMORY_EXTRACTOR_TIMEOUT_MS"); ms != "" {
			if v, err := strconv.ParseInt(ms, 10, 64); err == nil && v > 0 {
				c.Timeout = time.Duration(v) * time.Millisecond
			}
		}
		if c.Timeout <= 0 {
			c.Timeout = DefaultSubprocessTimeout
		}
	}
	if c.Logger == nil {
		c.Logger = slog.Default()
	}
}

// envOrDefault returns the value of the named environment variable, or
// the fallback when the variable is empty or unset.
func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ImageExtractor extracts text from images using PaddleOCR (primary)
// with Tesseract as fallback. Both engines are invoked as subprocesses
// — no native bindings or cgo required.
type ImageExtractor struct {
	cfg ImageExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*ImageExtractor)(nil)

// NewImageExtractor creates an ImageExtractor with the given config.
// Zero-value fields are populated with sensible defaults.
func NewImageExtractor(cfg ImageExtractorConfig) *ImageExtractor {
	cfg.applyDefaults()
	return &ImageExtractor{cfg: cfg}
}

// imageContentTypes lists all MIME types handled by the image extractor.
var imageContentTypes = []string{
	"image/png",
	"image/jpeg",
	"image/tiff",
	"image/bmp",
	"image/webp",
}

// imageExtensions lists all file extensions handled by the image extractor.
var imageExtensions = []string{
	".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
}

// imageMagicBytes contains magic byte signatures for supported image
// formats.
var imageMagicBytes = []MagicSignature{
	{Offset: 0, Bytes: []byte{0x89, 0x50, 0x4E, 0x47}},             // PNG
	{Offset: 0, Bytes: []byte{0xFF, 0xD8, 0xFF}},                    // JPEG
	{Offset: 0, Bytes: []byte{0x49, 0x49, 0x2A, 0x00}},              // TIFF (little-endian)
	{Offset: 0, Bytes: []byte{0x4D, 0x4D, 0x00, 0x2A}},              // TIFF (big-endian)
	{Offset: 0, Bytes: []byte{0x42, 0x4D}},                          // BMP
	{Offset: 0, Bytes: []byte{0x52, 0x49, 0x46, 0x46}},              // WebP (RIFF header)
}

// Name returns the extractor identifier.
func (e *ImageExtractor) Name() string {
	return "image-ocr"
}

// ContentTypes returns the MIME types this extractor handles.
func (e *ImageExtractor) ContentTypes() []string {
	return imageContentTypes
}

// Capability returns the capability descriptor for image extraction.
func (e *ImageExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions:     imageExtensions,
		MIMETypes:      imageContentTypes,
		MagicBytes:     imageMagicBytes,
		RequiresBinary: true,
	}
}

// Available reports whether at least one OCR engine (PaddleOCR or
// Tesseract) is installed on the system. Returns false gracefully
// when neither is available.
func (e *ImageExtractor) Available(ctx context.Context) (bool, error) {
	paddleAvailable := CheckBinaryAvailable(ctx, e.cfg.PaddleOCRBinary)
	if paddleAvailable {
		return true, nil
	}

	tesseractAvailable := CheckBinaryAvailable(ctx, e.cfg.TesseractBinary)
	return tesseractAvailable, nil
}

// Extract performs OCR on the raw image bytes. PaddleOCR is attempted
// first; if unavailable or failing, Tesseract is used as fallback.
// Returns a descriptive error when both engines are unavailable.
func (e *ImageExtractor) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	if len(raw) == 0 {
		return ExtractResult{
			Text:     "",
			Metadata: map[string]string{"ocr_engine": "none"},
			Skipped:  true,
			Reason:   "empty input",
		}, nil
	}

	language := e.resolveLanguage(opts)

	// Try PaddleOCR first.
	if CheckBinaryAvailable(ctx, e.cfg.PaddleOCRBinary) {
		result, err := e.extractWithPaddleOCR(ctx, raw, language, opts)
		if err == nil {
			return result, nil
		}
		e.cfg.Logger.Warn("paddleocr extraction failed, falling back to tesseract",
			"error", err,
			"file", opts.FileName,
		)
	}

	// Fallback to Tesseract.
	if CheckBinaryAvailable(ctx, e.cfg.TesseractBinary) {
		result, err := e.extractWithTesseract(ctx, raw, language, opts)
		if err == nil {
			return result, nil
		}
		e.cfg.Logger.Warn("tesseract extraction failed",
			"error", err,
			"file", opts.FileName,
		)
		return ExtractResult{}, fmt.Errorf("ingest: image extraction failed with both OCR engines: %w", err)
	}

	return ExtractResult{}, fmt.Errorf("ingest: no OCR engine available (tried %s, %s)", e.cfg.PaddleOCRBinary, e.cfg.TesseractBinary)
}

// ExtractStream buffers the reader and delegates to Extract.
func (e *ImageExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	var limitReader io.Reader = reader
	if opts.MaxBytes > 0 {
		limitReader = io.LimitReader(reader, opts.MaxBytes)
	}
	raw, err := io.ReadAll(limitReader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading image stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}

// resolveLanguage determines the OCR language parameter from options
// or falls back to the configured default.
func (e *ImageExtractor) resolveLanguage(opts ExtractOptions) string {
	if opts.Encoding != "" {
		return opts.Encoding
	}
	return e.cfg.DefaultLanguage
}

// writeTempFile writes raw bytes to a secure temporary file and
// returns the file path. The caller is responsible for removing the
// file.
func writeTempFile(raw []byte, pattern string) (string, error) {
	f, err := os.CreateTemp("", pattern)
	if err != nil {
		return "", fmt.Errorf("ingest: creating temp file: %w", err)
	}

	path := f.Name()

	if _, err := f.Write(raw); err != nil {
		f.Close()
		os.Remove(path)
		return "", fmt.Errorf("ingest: writing temp file: %w", err)
	}

	if err := f.Close(); err != nil {
		os.Remove(path)
		return "", fmt.Errorf("ingest: closing temp file: %w", err)
	}

	return path, nil
}

// extractWithPaddleOCR invokes PaddleOCR via subprocess. PaddleOCR
// does not accept stdin, so the image is written to a temporary file.
func (e *ImageExtractor) extractWithPaddleOCR(ctx context.Context, raw []byte, language string, opts ExtractOptions) (ExtractResult, error) {
	ext := extensionFromOpts(opts)
	tmpPath, err := writeTempFile(raw, "memory-ocr-*"+ext)
	if err != nil {
		return ExtractResult{}, err
	}
	defer os.Remove(tmpPath)

	// Resolve to absolute path to prevent path traversal.
	absPath, err := filepath.Abs(tmpPath)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: resolving temp path: %w", err)
	}

	paddleLang := mapLanguageToPaddleOCR(language)

	args := []string{
		"--image_dir", absPath,
		"--use_angle_cls", "true",
		"--lang", paddleLang,
		"--type", "ocr",
	}

	result, err := RunSubprocess(ctx, e.cfg.PaddleOCRBinary, args, nil, SubprocessOptions{
		Timeout: e.cfg.Timeout,
	})
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: paddleocr: %w", err)
	}

	if result.ExitCode != 0 {
		return ExtractResult{}, fmt.Errorf("ingest: paddleocr exited with code %d: %s", result.ExitCode, result.Stderr)
	}

	text, confidence := parsePaddleOCROutput(string(result.Stdout))

	return ExtractResult{
		Text:        text,
		ContentType: opts.ContentType,
		Metadata: map[string]string{
			"ocr_engine":     "paddleocr",
			"ocr_confidence": formatConfidence(confidence),
			"ocr_language":   paddleLang,
		},
		Language:   language,
		Confidence: confidence,
	}, nil
}

// extractWithTesseract invokes Tesseract via subprocess. The image is
// written to a temporary file since Tesseract's stdin support varies
// across versions.
func (e *ImageExtractor) extractWithTesseract(ctx context.Context, raw []byte, language string, opts ExtractOptions) (ExtractResult, error) {
	ext := extensionFromOpts(opts)
	tmpPath, err := writeTempFile(raw, "memory-ocr-*"+ext)
	if err != nil {
		return ExtractResult{}, err
	}
	defer os.Remove(tmpPath)

	absPath, err := filepath.Abs(tmpPath)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: resolving temp path: %w", err)
	}

	tesseractLang := mapLanguageToTesseract(language)

	args := []string{
		absPath,
		"stdout",
		"-l", tesseractLang,
		"--oem", "3",
		"--psm", "3",
	}

	result, err := RunSubprocess(ctx, e.cfg.TesseractBinary, args, nil, SubprocessOptions{
		Timeout: e.cfg.Timeout,
	})
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: tesseract: %w", err)
	}

	if result.ExitCode != 0 {
		return ExtractResult{}, fmt.Errorf("ingest: tesseract exited with code %d: %s", result.ExitCode, result.Stderr)
	}

	text := strings.TrimSpace(string(result.Stdout))

	return ExtractResult{
		Text:        text,
		ContentType: opts.ContentType,
		Metadata: map[string]string{
			"ocr_engine":   "tesseract",
			"ocr_language": tesseractLang,
		},
		Language: language,
	}, nil
}

// parsePaddleOCROutput parses PaddleOCR's JSON output format.
// PaddleOCR outputs an array of [text, confidence] pairs. The text
// blocks are concatenated in reading order and the average confidence
// is computed.
func parsePaddleOCROutput(output string) (string, float64) {
	// PaddleOCR outputs results line by line. The actual text results
	// follow a pattern of JSON-like arrays. Try JSON parsing first,
	// then fall back to line-by-line text extraction.
	lines := strings.Split(output, "\n")
	var textBlocks []string
	var totalConfidence float64
	var blockCount int

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Try parsing as JSON array: [["text", confidence], ...]
		var entries [][]json.RawMessage
		if json.Unmarshal([]byte(line), &entries) == nil {
			for _, entry := range entries {
				if len(entry) < 2 {
					continue
				}
				var text string
				var conf float64
				if json.Unmarshal(entry[0], &text) == nil && json.Unmarshal(entry[1], &conf) == nil {
					textBlocks = append(textBlocks, text)
					totalConfidence += conf
					blockCount++
				}
			}
			continue
		}

		// Try parsing individual result line: ("text", confidence)
		// PaddleOCR CLI sometimes outputs: ('text', 0.95)
		if strings.HasPrefix(line, "(") && strings.HasSuffix(line, ")") {
			inner := line[1 : len(line)-1]
			// Find the last comma to split text from confidence.
			lastComma := strings.LastIndex(inner, ",")
			if lastComma > 0 {
				textPart := strings.TrimSpace(inner[:lastComma])
				confPart := strings.TrimSpace(inner[lastComma+1:])

				// Strip quotes from text part.
				textPart = stripQuotes(textPart)

				if conf, err := strconv.ParseFloat(confPart, 64); err == nil {
					textBlocks = append(textBlocks, textPart)
					totalConfidence += conf
					blockCount++
				}
			}
		}
	}

	text := strings.Join(textBlocks, "\n")
	avgConfidence := 0.0
	if blockCount > 0 {
		avgConfidence = totalConfidence / float64(blockCount)
	}

	return text, avgConfidence
}

// stripQuotes removes surrounding single or double quotes from a
// string.
func stripQuotes(s string) string {
	if len(s) < 2 {
		return s
	}
	first, last := s[0], s[len(s)-1]
	if (first == '\'' && last == '\'') || (first == '"' && last == '"') {
		return s[1 : len(s)-1]
	}
	return s
}

// extensionFromOpts determines the file extension from the extract
// options, defaulting to ".png".
func extensionFromOpts(opts ExtractOptions) string {
	extensionMap := map[string]string{
		"image/png":  ".png",
		"image/jpeg": ".jpg",
		"image/tiff": ".tiff",
		"image/bmp":  ".bmp",
		"image/webp": ".webp",
	}

	if ext, ok := extensionMap[opts.ContentType]; ok {
		return ext
	}

	if opts.FileName != "" {
		ext := filepath.Ext(opts.FileName)
		if ext != "" {
			return ext
		}
	}

	return ".png"
}

// mapLanguageToPaddleOCR maps ISO 639-1 language codes to PaddleOCR's
// language parameter format.
var paddleOCRLanguageMap = map[string]string{
	"en": "en",
	"zh": "ch",
	"ja": "japan",
	"ko": "korean",
	"fr": "fr",
	"de": "german",
	"es": "es",
	"pt": "pt",
	"it": "it",
	"ru": "ru",
	"ar": "ar",
	"hi": "hi",
	"ta": "ta",
	"te": "te",
}

func mapLanguageToPaddleOCR(lang string) string {
	normalised := strings.ToLower(strings.TrimSpace(lang))
	if mapped, ok := paddleOCRLanguageMap[normalised]; ok {
		return mapped
	}
	return normalised
}

// mapLanguageToTesseract maps ISO 639-1 codes to Tesseract's 3-letter
// language codes.
var tesseractLanguageMap = map[string]string{
	"en": "eng",
	"zh": "chi_sim",
	"ja": "jpn",
	"ko": "kor",
	"fr": "fra",
	"de": "deu",
	"es": "spa",
	"pt": "por",
	"it": "ita",
	"ru": "rus",
	"ar": "ara",
	"hi": "hin",
	"ta": "tam",
	"te": "tel",
}

func mapLanguageToTesseract(lang string) string {
	normalised := strings.ToLower(strings.TrimSpace(lang))
	if mapped, ok := tesseractLanguageMap[normalised]; ok {
		return mapped
	}
	return "eng"
}

// formatConfidence formats a float64 confidence value as a string with
// 4 decimal places.
func formatConfidence(conf float64) string {
	return strconv.FormatFloat(conf, 'f', 4, 64)
}
