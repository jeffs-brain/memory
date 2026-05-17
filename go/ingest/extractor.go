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

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/encoding/korean"
	"golang.org/x/text/encoding/simplifiedchinese"
	"golang.org/x/text/encoding/traditionalchinese"
	"golang.org/x/text/encoding/unicode"
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

// MagicSignature identifies a file format by magic bytes at a given offset.
type MagicSignature struct {
	Offset int
	Bytes  []byte
}

// ExtractorCapability describes what content types an extractor handles.
// Used by the registry for routing and by callers to inspect extractor
// capabilities without instantiating extraction.
type ExtractorCapability struct {
	Extensions     []string
	MIMETypes      []string
	MagicBytes     []MagicSignature
	RequiresBinary bool
}

// ExtractOptions provides hints to the extractor about the content being
// processed.
type ExtractOptions struct {
	ContentType string
	FileName    string
	Encoding    string
	Language    string // ISO 639-1 language hint for OCR extractors (e.g. "en", "de").
	MaxBytes    int64  // 0 = no limit
}

// ExtractResult holds the output of an extraction operation.
type ExtractResult struct {
	Text        string
	ContentType string
	Encoding    string
	Metadata    map[string]string
	Pages       int
	Language    string
	Confidence  float64
	Skipped     bool
	Reason      string
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
	// Available reports whether this extractor's external dependencies
	// are present (e.g. PaddleOCR, FFmpeg). Extractors with no external
	// dependencies always return (true, nil).
	Available(ctx context.Context) (bool, error)
	// Capability describes what content types, file extensions, and
	// magic byte signatures this extractor handles. Used by the registry
	// for routing and by callers for introspection.
	Capability() ExtractorCapability
}

// BaseExtractor provides a default ExtractStream implementation that
// buffers the reader into memory and delegates to Extract. Embed this in
// simple extractors that do not need true streaming.
type BaseExtractor struct {
	ExtractFn      func(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error)
	ContentTypesFn func() []string
	NameFn         func() string
	AvailableFn    func(ctx context.Context) (bool, error)
	CapabilityFn   func() ExtractorCapability
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

// Available delegates to the embedded AvailableFn. Returns (true, nil)
// if no AvailableFn is configured.
func (b *BaseExtractor) Available(ctx context.Context) (bool, error) {
	if b.AvailableFn != nil {
		return b.AvailableFn(ctx)
	}
	return true, nil
}

// Capability delegates to the embedded CapabilityFn. Returns an empty
// capability if no CapabilityFn is configured.
func (b *BaseExtractor) Capability() ExtractorCapability {
	if b.CapabilityFn != nil {
		return b.CapabilityFn()
	}
	return ExtractorCapability{}
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

// Available always returns true since the plain text extractor has no
// external dependencies.
func (p *PlainTextExtractor) Available(_ context.Context) (bool, error) {
	return true, nil
}

// Capability returns the capability descriptor for the plain text extractor.
func (p *PlainTextExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".txt", ".text", ".log", ".md", ".markdown", ".csv", ".yaml", ".yml", ".json"},
		MIMETypes:  p.ContentTypes(),
	}
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

// encodingLookup maps canonical encoding names (lowercase) to their
// golang.org/x/text/encoding implementations. Covers the most commonly
// encountered encodings in real-world content.
var encodingLookup = map[string]encoding.Encoding{
	"utf-8":         unicode.UTF8,
	"iso-8859-1":    charmap.ISO8859_1,
	"iso-8859-2":    charmap.ISO8859_2,
	"iso-8859-3":    charmap.ISO8859_3,
	"iso-8859-4":    charmap.ISO8859_4,
	"iso-8859-5":    charmap.ISO8859_5,
	"iso-8859-6":    charmap.ISO8859_6,
	"iso-8859-7":    charmap.ISO8859_7,
	"iso-8859-8":    charmap.ISO8859_8,
	"iso-8859-9":    charmap.ISO8859_9,
	"iso-8859-10":   charmap.ISO8859_10,
	"iso-8859-13":   charmap.ISO8859_13,
	"iso-8859-14":   charmap.ISO8859_14,
	"iso-8859-15":   charmap.ISO8859_15,
	"iso-8859-16":   charmap.ISO8859_16,
	"windows-1250":  charmap.Windows1250,
	"windows-1251":  charmap.Windows1251,
	"windows-1252":  charmap.Windows1252,
	"windows-1253":  charmap.Windows1253,
	"windows-1254":  charmap.Windows1254,
	"windows-1255":  charmap.Windows1255,
	"windows-1256":  charmap.Windows1256,
	"koi8-r":        charmap.KOI8R,
	"koi8-u":        charmap.KOI8U,
	"shift_jis":     japanese.ShiftJIS,
	"euc-jp":        japanese.EUCJP,
	"iso-2022-jp":   japanese.ISO2022JP,
	"euc-kr":        korean.EUCKR,
	"gb2312":        simplifiedchinese.HZGB2312,
	"gbk":           simplifiedchinese.GBK,
	"gb18030":       simplifiedchinese.GB18030,
	"big5":          traditionalchinese.Big5,
	"utf-16be":      unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM),
	"utf-16le":      unicode.UTF16(unicode.LittleEndian, unicode.IgnoreBOM),
}

// DetectEncoding identifies the character encoding of raw bytes. Returns
// the encoding name (e.g. "UTF-8", "ISO-8859-1", "Shift_JIS"). Uses a
// heuristic approach: checks for BOM markers first, then validates UTF-8,
// and falls back to statistical analysis of byte frequency patterns.
func DetectEncoding(raw []byte) string {
	if len(raw) == 0 {
		return "UTF-8"
	}

	// Check for BOM markers.
	if len(raw) >= 3 && raw[0] == 0xEF && raw[1] == 0xBB && raw[2] == 0xBF {
		return "UTF-8"
	}
	if len(raw) >= 2 && raw[0] == 0xFE && raw[1] == 0xFF {
		return "UTF-16BE"
	}
	if len(raw) >= 2 && raw[0] == 0xFF && raw[1] == 0xFE {
		return "UTF-16LE"
	}

	// Valid UTF-8: the vast majority of modern content.
	if utf8.Valid(raw) {
		return "UTF-8"
	}

	// Statistical heuristics for common single-byte encodings.
	return detectByByteFrequency(raw)
}

// detectByByteFrequency uses byte frequency analysis to distinguish
// between common single-byte encodings when the content is not valid
// UTF-8.
func detectByByteFrequency(raw []byte) string {
	// First pass: count Shift_JIS double-byte pairs. Track which byte
	// positions are consumed as part of a pair so the second pass can
	// count unpaired high bytes accurately.
	paired := make([]bool, len(raw))
	var shiftJISPairs int

	for i := 0; i < len(raw); i++ {
		b := raw[i]
		// Shift_JIS lead byte ranges: 0x81-0x9F, 0xE0-0xEF followed by
		// trail byte: 0x40-0x7E, 0x80-0xFC.
		if (b >= 0x81 && b <= 0x9F) || (b >= 0xE0 && b <= 0xEF) {
			if i+1 < len(raw) {
				trail := raw[i+1]
				if (trail >= 0x40 && trail <= 0x7E) || (trail >= 0x80 && trail <= 0xFC) {
					paired[i] = true
					paired[i+1] = true
					shiftJISPairs++
					i++ // skip trail byte
				}
			}
		}
	}

	// Second pass: count unpaired high bytes and C1 controls.
	var highBytes, unpairedHighBytes int
	var unpairedC1Controls int

	for i := 0; i < len(raw); i++ {
		b := raw[i]
		if b >= 0x80 {
			highBytes++
			if !paired[i] {
				unpairedHighBytes++
				if b <= 0x9F {
					unpairedC1Controls++
				}
			}
		}
	}

	// If all high bytes are consumed by Shift_JIS pairs and there are
	// at least 2 pairs, classify as Shift_JIS. This handles pure
	// Japanese text including bytes in the C1 range (0x81-0x9F) which
	// are valid Shift_JIS lead bytes.
	if shiftJISPairs >= 2 && unpairedHighBytes == 0 {
		return "Shift_JIS"
	}

	// Unpaired C1 control characters (0x80-0x9F) are a strong signal
	// for Windows-1252 which maps printable characters (smart quotes,
	// em dashes) into this range. ISO-8859-1 treats them as control
	// characters which are extremely rare in real text.
	if unpairedC1Controls > 0 {
		return "Windows-1252"
	}

	// Default to ISO-8859-1 for content with high bytes (0xA0-0xFF)
	// that does not match other patterns.
	return "ISO-8859-1"
}

// TranscodeToUTF8 converts raw bytes from the specified encoding to
// UTF-8. Returns the bytes unchanged if the source encoding is already
// UTF-8. Returns an error if the encoding is unsupported or the content
// cannot be transcoded.
func TranscodeToUTF8(raw []byte, fromEncoding string) ([]byte, error) {
	normalised := strings.ToLower(strings.TrimSpace(fromEncoding))
	if normalised == "utf-8" || normalised == "" {
		return raw, nil
	}

	enc, ok := encodingLookup[normalised]
	if !ok {
		return nil, fmt.Errorf("ingest: unsupported encoding %q", fromEncoding)
	}

	decoded, err := enc.NewDecoder().Bytes(raw)
	if err != nil {
		return nil, fmt.Errorf("ingest: transcoding from %s to UTF-8: %w", fromEncoding, err)
	}
	return decoded, nil
}

// -------------------------------------------------------------------
// Structured data extractors (P4-4)
// -------------------------------------------------------------------

// CSVExtractor wraps the CSV extraction function as a canonical
// ingest.Extractor implementation.
type CSVExtractor struct {
	Config CsvExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*CSVExtractor)(nil)

// Name implements Extractor.
func (e *CSVExtractor) Name() string { return "csv" }

// ContentTypes implements Extractor.
func (e *CSVExtractor) ContentTypes() []string {
	return []string{"text/csv", "text/tab-separated-values"}
}

// Capability implements Extractor.
func (e *CSVExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".csv", ".tsv"},
		MIMETypes:  e.ContentTypes(),
	}
}

// Available implements Extractor. Structured data extractors have no
// external dependencies and are always available.
func (e *CSVExtractor) Available(_ context.Context) (bool, error) { return true, nil }

// Extract implements Extractor.
func (e *CSVExtractor) Extract(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
	return ExtractCSV(raw, e.Config)
}

// ExtractStream implements Extractor by buffering the reader and
// delegating to Extract.
func (e *CSVExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	raw, err := io.ReadAll(reader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading csv stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}

// -------------------------------------------------------------------

// JSONExtractor wraps the JSON extraction function as a canonical
// ingest.Extractor implementation.
type JSONExtractor struct {
	Config JsonExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*JSONExtractor)(nil)

// Name implements Extractor.
func (e *JSONExtractor) Name() string { return "json" }

// ContentTypes implements Extractor.
func (e *JSONExtractor) ContentTypes() []string {
	return []string{"application/json"}
}

// Capability implements Extractor.
func (e *JSONExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".json"},
		MIMETypes:  e.ContentTypes(),
	}
}

// Available implements Extractor.
func (e *JSONExtractor) Available(_ context.Context) (bool, error) { return true, nil }

// Extract implements Extractor.
func (e *JSONExtractor) Extract(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
	return ExtractJSON(raw, e.Config)
}

// ExtractStream implements Extractor.
func (e *JSONExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	raw, err := io.ReadAll(reader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading json stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}

// -------------------------------------------------------------------

// JSONLExtractor wraps the JSONL extraction function as a canonical
// ingest.Extractor implementation.
type JSONLExtractor struct {
	Config JsonExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*JSONLExtractor)(nil)

// Name implements Extractor.
func (e *JSONLExtractor) Name() string { return "jsonl" }

// ContentTypes implements Extractor.
func (e *JSONLExtractor) ContentTypes() []string {
	return []string{"application/jsonl", "application/x-ndjson"}
}

// Capability implements Extractor.
func (e *JSONLExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".jsonl", ".ndjson"},
		MIMETypes:  e.ContentTypes(),
	}
}

// Available implements Extractor.
func (e *JSONLExtractor) Available(_ context.Context) (bool, error) { return true, nil }

// Extract implements Extractor.
func (e *JSONLExtractor) Extract(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
	return ExtractJSONL(raw, e.Config)
}

// ExtractStream implements Extractor.
func (e *JSONLExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	raw, err := io.ReadAll(reader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading jsonl stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}

// -------------------------------------------------------------------

// XMLExtractor wraps the XML extraction function as a canonical
// ingest.Extractor implementation.
type XMLExtractor struct {
	Config XmlExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*XMLExtractor)(nil)

// Name implements Extractor.
func (e *XMLExtractor) Name() string { return "xml" }

// ContentTypes implements Extractor.
func (e *XMLExtractor) ContentTypes() []string {
	return []string{"application/xml", "text/xml"}
}

// Capability implements Extractor.
func (e *XMLExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".xml"},
		MIMETypes:  e.ContentTypes(),
	}
}

// Available implements Extractor.
func (e *XMLExtractor) Available(_ context.Context) (bool, error) { return true, nil }

// Extract implements Extractor.
func (e *XMLExtractor) Extract(_ context.Context, raw []byte, _ ExtractOptions) (ExtractResult, error) {
	return ExtractXML(raw, e.Config)
}

// ExtractStream implements Extractor.
func (e *XMLExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	raw, err := io.ReadAll(reader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading xml stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}
