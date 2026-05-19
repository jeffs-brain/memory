// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// injectionConfidenceThreshold is the score above which content is
// flagged as a potential prompt injection. The ML model returns scores
// in [0.0, 1.0]; 0.5 balances precision and recall.
const injectionConfidenceThreshold = 0.5

// isolationTag is the XML element name used to wrap ingested content,
// providing structural isolation from the host prompt.
const isolationTag = "ingested-document"

// closingTagPattern matches the isolation closing tag in any casing so
// it can be escaped to prevent content breakout.
var closingTagPattern = regexp.MustCompile(`(?i)</ingested-document>`)

// ScanResult holds the output of a prompt injection scan.
type ScanResult struct {
	InjectionDetected bool
	Confidence        float64
	Detections        []string
}

// SafetyMetadata contains the metadata fields added to flagged
// documents. Callers merge these into the document's metadata map.
type SafetyMetadata struct {
	InjectionRisk       bool
	InjectionConfidence float64
}

// Scanner is the interface for ML-based prompt injection detection.
// Implementations wrap specific model runtimes (ONNX via hugot,
// remote API, etc.). The interface allows graceful degradation: when
// no scanner is available the pipeline continues without ML scoring.
type Scanner interface {
	// Scan analyses preprocessed text for prompt injection indicators.
	// Implementations must respect context cancellation.
	Scan(ctx context.Context, text string) (ScanResult, error)

	// Available reports whether the underlying model is loaded and
	// ready to serve predictions.
	Available() bool
}

// ScannerConfig holds construction parameters for a Scanner.
type ScannerConfig struct {
	// ModelPath is the filesystem path to the ONNX model file.
	// Required for hugot-backed implementations.
	ModelPath string

	// ConfidenceThreshold overrides the default injection threshold.
	// Zero means use the package default (0.5).
	ConfidenceThreshold float64
}

// noopScanner is the fallback scanner returned when no ML model is
// available. It never flags content and always reports unavailable.
type noopScanner struct{}

func (noopScanner) Scan(_ context.Context, _ string) (ScanResult, error) {
	return ScanResult{InjectionDetected: false, Confidence: 0, Detections: nil}, nil
}

func (noopScanner) Available() bool { return false }

// NewNoopScanner returns a Scanner that performs no ML inference. Use
// this as the default when the ONNX model is not available.
func NewNoopScanner() Scanner {
	return noopScanner{}
}

// PreprocessText normalises content for reliable scanning. Applies
// Unicode NFKC normalisation and strips zero-width characters commonly
// used to evade text-based detection.
//
// Time: O(n) where n = number of runes.
// Space: O(n) for the normalised copy.
func PreprocessText(text string) string {
	normalised := norm.NFKC.String(text)
	var b strings.Builder
	b.Grow(len(normalised))
	for _, r := range normalised {
		if isZeroWidthChar(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// isZeroWidthChar reports whether a rune is a zero-width or invisible
// formatting character used for evasion.
func isZeroWidthChar(r rune) bool {
	switch r {
	case
		'​', // zero-width space
		'‌', // zero-width non-joiner
		'‍', // zero-width joiner
		'‎', // left-to-right mark
		'‏', // right-to-left mark
		'⁠', // word joiner
		'⁡', // function application
		'⁢', // invisible times
		'⁣', // invisible separator
		'⁤', // invisible plus
		'\ufeff', // zero-width no-break space (BOM)
		'­', // soft hyphen
		' ', // en quad (thin space, often abused)
		' ': // em quad
		return true
	}
	// Catch any remaining characters in the General_Category=Format class
	// that are not common whitespace.
	return unicode.Is(unicode.Cf, r) && !isCommonWhitespace(r)
}

// isCommonWhitespace identifies whitespace characters that should be
// preserved (space, tab, newline, carriage return).
func isCommonWhitespace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return false
}

// WrapInIsolation encloses content in XML isolation delimiters. Any
// occurrences of the closing tag within the content are escaped to
// prevent delimiter breakout.
//
// Time: O(n) where n = content length.
// Space: O(n) for the wrapped string.
func WrapInIsolation(content, source, hash string) string {
	escapedContent := closingTagPattern.ReplaceAllString(content, "&lt;/ingested-document&gt;")
	escapedSource := escapeAttr(source)
	return fmt.Sprintf(
		"<%s source=%q hash=%q>%s</%s>",
		isolationTag, escapedSource, hash, escapedContent, isolationTag,
	)
}

// escapeAttr escapes XML special characters in attribute values.
func escapeAttr(value string) string {
	value = strings.ReplaceAll(value, "&", "&amp;")
	value = strings.ReplaceAll(value, "\"", "&quot;")
	value = strings.ReplaceAll(value, "<", "&lt;")
	value = strings.ReplaceAll(value, ">", "&gt;")
	return value
}

// ContentHash computes a SHA-256 hash of the content, returned as a
// hex string. Used for the isolation delimiter hash attribute.
func ContentHash(content []byte) string {
	sum := sha256.Sum256(content)
	return hex.EncodeToString(sum[:])
}

// BuildSafetyMetadata returns metadata fields for a flagged document.
// Returns nil when no injection was detected so callers can skip the
// merge.
func BuildSafetyMetadata(result ScanResult) *SafetyMetadata {
	if !result.InjectionDetected {
		return nil
	}
	return &SafetyMetadata{
		InjectionRisk:       true,
		InjectionConfidence: result.Confidence,
	}
}

// ScanAndWrap is the primary integration point: preprocess the content,
// run the scanner, wrap in isolation delimiters, and return the wrapped
// content plus any safety metadata.
func ScanAndWrap(
	ctx context.Context,
	scanner Scanner,
	content string,
	source string,
) (wrappedContent string, metadata *SafetyMetadata, err error) {
	preprocessed := PreprocessText(content)
	hash := ContentHash([]byte(preprocessed))

	var scanResult ScanResult
	if scanner.Available() {
		var scanErr error
		scanResult, scanErr = scanner.Scan(ctx, preprocessed)
		if scanErr != nil {
			// Graceful degradation: log via caller, do not block ingestion
			scanResult = ScanResult{InjectionDetected: false, Confidence: 0}
		}
	}

	wrapped := WrapInIsolation(preprocessed, source, hash)
	meta := BuildSafetyMetadata(scanResult)
	return wrapped, meta, nil
}
