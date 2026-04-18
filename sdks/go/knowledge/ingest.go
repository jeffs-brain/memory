// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	pdfreader "github.com/ledongthuc/pdf"

	"github.com/jeffs-brain/memory/go/brain"
)

// Content-type constants for the supported raw formats. Used by the
// ingest pipeline to route a [IngestRequest] to the correct extractor.
const (
	contentTypeMarkdown = "text/markdown"
	contentTypeText     = "text/plain"
	contentTypeHTML     = "text/html"
	contentTypePDF      = "application/pdf"
	contentTypeJSON     = "application/json"
	contentTypeYAML     = "application/x-yaml"
)

// maxReadBytes caps the body we accept from any single source. Matches
// jeff's text/binary ingest limits rolled into one conservative ceiling
// so an oversized URL or local file cannot exhaust memory.
const maxReadBytes = 50 * 1024 * 1024

// defaultHTTPTimeout is the per-request wall clock budget for
// [defaultFetcher]. Matches jeff's ingest.go default.
const defaultHTTPTimeout = 30 * time.Second

// htmlTagPattern strips every HTML tag from a document so the plain
// extractor never emits markup to the index. Matches both open and
// self-closing tags.
var htmlTagPattern = regexp.MustCompile(`(?s)<[^>]+>`)

// scriptPattern and stylePattern drop <script> and <style> blocks
// wholesale before the generic tag stripper runs so the index never
// captures JavaScript or stylesheet text as content. Go's RE2 does not
// support backreferences, so each block gets its own explicit pattern.
var (
	scriptPattern = regexp.MustCompile(`(?is)<script[^>]*>.*?</\s*script\s*>`)
	stylePattern  = regexp.MustCompile(`(?is)<style[^>]*>.*?</\s*style\s*>`)
)

// defaultFetcher is the production HTTP fetcher used by [Base.IngestURL]
// when the caller does not supply an override.
type defaultFetcher struct{}

// Fetch implements [Fetcher].
func (defaultFetcher) Fetch(ctx context.Context, rawURL string) ([]byte, string, error) {
	client := &http.Client{Timeout: defaultHTTPTimeout}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, "", fmt.Errorf("knowledge: building request: %w", err)
	}
	req.Header.Set("Accept", "text/plain, text/markdown, text/html, application/pdf")
	resp, err := client.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("knowledge: fetching %s: %w", rawURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", fmt.Errorf("knowledge: fetch %s: HTTP %d", rawURL, resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxReadBytes))
	if err != nil {
		return nil, "", fmt.Errorf("knowledge: reading response: %w", err)
	}
	ctype := resp.Header.Get("Content-Type")
	return body, ctype, nil
}

// Ingest implements [Base].
func (k *kbase) Ingest(ctx context.Context, req IngestRequest) (IngestResponse, error) {
	if err := k.requireStore(); err != nil {
		return IngestResponse{}, err
	}
	start := time.Now()

	raw, ctype, sourceLabel, err := k.readRequestBody(req)
	if err != nil {
		return IngestResponse{}, err
	}

	extracted, err := extractPlain(raw, ctype, filepath.Ext(strings.ToLower(req.Path)))
	if err != nil {
		return IngestResponse{}, err
	}

	doc := buildDocument(req, ctype, sourceLabel, extracted, raw)
	if err := k.writeDocument(ctx, doc); err != nil {
		return IngestResponse{}, err
	}

	// Compile inline so the index reflects the ingest immediately.
	chunks, err := k.chunkAndIndex(ctx, doc)
	if err != nil {
		return IngestResponse{}, err
	}

	return IngestResponse{
		DocumentID: doc.ID,
		Path:       doc.Path,
		ChunkCount: chunks,
		Bytes:      doc.Bytes,
		TookMs:     time.Since(start).Milliseconds(),
	}, nil
}

// IngestURL implements [Base]. Fetches the URL through the configured
// [Fetcher], routes the body through the plain-text extractor, and
// persists the result.
func (k *kbase) IngestURL(ctx context.Context, rawURL string) (IngestResponse, error) {
	if err := k.requireStore(); err != nil {
		return IngestResponse{}, err
	}
	parsed, err := normaliseURL(rawURL)
	if err != nil {
		return IngestResponse{}, err
	}
	_, _, fetcher := k.snapshot()
	body, ctype, err := fetcher.Fetch(ctx, parsed)
	if err != nil {
		return IngestResponse{}, err
	}
	return k.Ingest(ctx, IngestRequest{
		BrainID:     k.brainID,
		Path:        parsed,
		ContentType: ctype,
		Content:     bytes.NewReader(body),
	})
}

// readRequestBody loads the request body. When req.Content is set it
// takes precedence; otherwise req.Path is opened from the local
// filesystem. Returns the raw bytes, the detected content type, and a
// source label suitable for frontmatter.
func (k *kbase) readRequestBody(req IngestRequest) ([]byte, string, string, error) {
	ctype := strings.ToLower(strings.TrimSpace(req.ContentType))
	// Normalise the charset suffix (e.g. "text/html; charset=utf-8")
	// so downstream switches work on the base media type alone.
	if idx := strings.Index(ctype, ";"); idx >= 0 {
		ctype = strings.TrimSpace(ctype[:idx])
	}

	if req.Content != nil {
		data, err := io.ReadAll(io.LimitReader(req.Content, maxReadBytes))
		if err != nil {
			return nil, "", "", fmt.Errorf("knowledge: reading content: %w", err)
		}
		if ctype == "" {
			ctype = detectContentType(req.Path, data)
		}
		return data, ctype, firstNonEmpty(req.Path, "inline"), nil
	}

	if strings.TrimSpace(req.Path) == "" {
		return nil, "", "", fmt.Errorf("knowledge: either Content or Path required")
	}

	if _, parseErr := url.ParseRequestURI(req.Path); parseErr == nil && strings.Contains(req.Path, "://") {
		return nil, "", "", fmt.Errorf("knowledge: use IngestURL for %s", req.Path)
	}

	abs, err := filepath.Abs(req.Path)
	if err != nil {
		return nil, "", "", fmt.Errorf("knowledge: resolving path: %w", err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return nil, "", "", fmt.Errorf("knowledge: stat %s: %w", abs, err)
	}
	if info.IsDir() {
		return nil, "", "", fmt.Errorf("knowledge: %s is a directory", abs)
	}
	if info.Size() > maxReadBytes {
		return nil, "", "", fmt.Errorf("knowledge: %s exceeds %d byte limit", abs, maxReadBytes)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, "", "", fmt.Errorf("knowledge: reading %s: %w", abs, err)
	}
	if ctype == "" {
		ctype = detectContentType(abs, data)
	}
	return data, ctype, abs, nil
}

// detectContentType picks a content type from the file extension first,
// falling back to http.DetectContentType for binary blobs without a
// useful suffix.
func detectContentType(p string, data []byte) string {
	ext := strings.ToLower(filepath.Ext(p))
	switch ext {
	case ".md", ".markdown":
		return contentTypeMarkdown
	case ".txt", ".text", ".log":
		return contentTypeText
	case ".html", ".htm":
		return contentTypeHTML
	case ".pdf":
		return contentTypePDF
	case ".json":
		return contentTypeJSON
	case ".yaml", ".yml":
		return contentTypeYAML
	}
	if len(data) > 4 && string(data[:4]) == "%PDF" {
		return contentTypePDF
	}
	sniff := http.DetectContentType(data)
	if strings.HasPrefix(sniff, "text/html") {
		return contentTypeHTML
	}
	if strings.HasPrefix(sniff, "text/") {
		return contentTypeText
	}
	return sniff
}

// extractPlain routes raw bytes to the correct extractor. Markdown and
// plain text pass through untouched; HTML is stripped; PDF is decoded
// via ledongthuc/pdf. Everything else is rejected.
func extractPlain(raw []byte, ctype, ext string) (string, error) {
	base := strings.ToLower(strings.TrimSpace(ctype))
	if idx := strings.Index(base, ";"); idx >= 0 {
		base = strings.TrimSpace(base[:idx])
	}
	switch base {
	case contentTypeMarkdown, contentTypeText, contentTypeJSON, contentTypeYAML:
		if !utf8.Valid(raw) {
			return "", fmt.Errorf("knowledge: content is not valid UTF-8")
		}
		return string(raw), nil
	case contentTypeHTML:
		return stripHTML(raw), nil
	case contentTypePDF:
		return extractPDF(raw)
	}
	if ext == ".pdf" || (len(raw) > 4 && string(raw[:4]) == "%PDF") {
		return extractPDF(raw)
	}
	if strings.HasPrefix(base, "text/") {
		if !utf8.Valid(raw) {
			return "", fmt.Errorf("knowledge: content is not valid UTF-8")
		}
		return string(raw), nil
	}
	return "", fmt.Errorf("knowledge: unsupported content-type %q", ctype)
}

// stripHTML drops scripts, styles, and tags to produce a plain-text
// extract suitable for indexing.
func stripHTML(raw []byte) string {
	s := string(raw)
	s = scriptPattern.ReplaceAllString(s, " ")
	s = stylePattern.ReplaceAllString(s, " ")
	s = htmlTagPattern.ReplaceAllString(s, " ")
	s = strings.ReplaceAll(s, "&nbsp;", " ")
	s = strings.ReplaceAll(s, "&amp;", "&")
	s = strings.ReplaceAll(s, "&lt;", "<")
	s = strings.ReplaceAll(s, "&gt;", ">")
	s = strings.ReplaceAll(s, "&quot;", `"`)
	return collapseWhitespace(s)
}

// extractPDF runs the ledongthuc/pdf GetPlainText path across every
// page and returns the concatenated text.
func extractPDF(raw []byte) (string, error) {
	if len(raw) == 0 {
		return "", fmt.Errorf("knowledge: empty pdf body")
	}
	reader := bytes.NewReader(raw)
	r, err := pdfreader.NewReader(reader, int64(len(raw)))
	if err != nil {
		return "", fmt.Errorf("knowledge: opening pdf: %w", err)
	}
	var b strings.Builder
	pages := r.NumPage()
	for i := 1; i <= pages; i++ {
		p := r.Page(i)
		if p.V.IsNull() {
			continue
		}
		text, err := p.GetPlainText(nil)
		if err != nil {
			// Non-fatal; return what we have so far. PDF extractors
			// commonly trip over malformed embedded fonts mid-document.
			continue
		}
		if text != "" {
			if b.Len() > 0 {
				b.WriteString("\n\n")
			}
			b.WriteString(text)
		}
	}
	return collapseWhitespace(b.String()), nil
}

// buildDocument assembles a [Document] from a request plus the already-
// extracted plain text. Title, summary, and tags prefer frontmatter
// when present, otherwise fall back to the request hints.
func buildDocument(req IngestRequest, ctype, sourceLabel, extracted string, raw []byte) *Document {
	now := time.Now().UTC()
	title := strings.TrimSpace(req.Title)
	var summary string
	tags := append([]string{}, req.Tags...)

	fmParsed, body := ParseFrontmatter(extracted)
	if title == "" {
		switch {
		case fmParsed.Title != "":
			title = fmParsed.Title
		case fmParsed.Name != "":
			title = fmParsed.Name
		default:
			title = deriveTitle(body, sourceLabel)
		}
	}
	summary = fmParsed.Summary
	if summary == "" {
		summary = fmParsed.Description
	}
	if len(fmParsed.Tags) > 0 {
		tags = append(tags, fmParsed.Tags...)
	}

	slug := slugify(title)
	if slug == "" {
		slug = hashSlug(raw)
	}

	docID := hashSlug(append([]byte(slug+":"), raw...))
	logical := brain.RawDocument(slug)

	return &Document{
		ID:          docID,
		Path:        logical,
		Title:       title,
		Source:      sourceLabel,
		ContentType: ctype,
		Tags:        dedupeStrings(tags),
		Summary:     summary,
		Body:        strings.TrimSpace(body),
		Bytes:       len(raw),
		Ingested:    now,
		Modified:    now,
	}
}

// writeDocument persists the document under raw/documents/<slug>.md as a
// canonical markdown file with a regenerated YAML frontmatter header.
func (k *kbase) writeDocument(ctx context.Context, doc *Document) error {
	fm := buildFrontmatterYAML(doc)
	payload := fm + "\n\n" + doc.Body + "\n"
	return k.store.Write(ctx, doc.Path, []byte(payload))
}

// buildFrontmatterYAML emits the canonical frontmatter block written
// alongside every ingested document. British English throughout, no em
// dashes, consistent quoting.
func buildFrontmatterYAML(doc *Document) string {
	var b strings.Builder
	b.WriteString("---\n")
	writeYAMLString(&b, "title", doc.Title)
	if doc.Summary != "" {
		writeYAMLString(&b, "summary", doc.Summary)
	}
	writeYAMLString(&b, "source", doc.Source)
	writeYAMLString(&b, "source_type", routeSourceType(doc.ContentType))
	writeYAMLString(&b, "ingested", doc.Ingested.Format(time.RFC3339))
	writeYAMLString(&b, "modified", doc.Modified.Format(time.RFC3339))
	if len(doc.Tags) > 0 {
		b.WriteString("tags:\n")
		for _, t := range doc.Tags {
			b.WriteString("  - ")
			b.WriteString(t)
			b.WriteString("\n")
		}
	}
	b.WriteString("---")
	return b.String()
}

// writeYAMLString emits a single quoted key/value pair.
func writeYAMLString(b *strings.Builder, key, value string) {
	b.WriteString(key)
	b.WriteString(`: `)
	b.WriteString(quoteYAML(value))
	b.WriteString("\n")
}

// quoteYAML wraps value in double quotes and escapes embedded quotes.
func quoteYAML(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, `"`, `\"`)
	return `"` + value + `"`
}

// routeSourceType maps a content type to the frontmatter source_type
// tag. Used to keep downstream filters consistent across ingest paths.
func routeSourceType(ctype string) string {
	switch ctype {
	case contentTypeMarkdown:
		return "markdown"
	case contentTypeText:
		return "text"
	case contentTypeHTML:
		return "html"
	case contentTypePDF:
		return "pdf"
	case contentTypeJSON:
		return "json"
	case contentTypeYAML:
		return "yaml"
	}
	return "document"
}

// deriveTitle picks a best-effort title from the body when the caller
// provides none. Prefers the first `# heading`, then the first
// non-empty line, then the file basename.
func deriveTitle(body, fallback string) string {
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimSpace(strings.TrimPrefix(line, "# "))
		}
	}
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			if len(line) > 120 {
				line = line[:120]
			}
			return line
		}
	}
	base := filepath.Base(fallback)
	base = strings.TrimSuffix(base, filepath.Ext(base))
	if base == "" || base == "." {
		return "untitled"
	}
	return base
}

// hashSlug is a deterministic fallback slug built from a SHA-256 sum
// truncated to 12 hex characters. Used when slugify collapses to empty.
func hashSlug(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])[:12]
}

// slugify mirrors jeff's helper: lowercase, hyphen-separated, capped at
// 60 characters. Unicode letters and digits survive; everything else
// becomes a hyphen.
func slugify(title string) string {
	s := strings.ToLower(strings.TrimSpace(title))
	var b strings.Builder
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			b.WriteRune('-')
		}
	}
	slug := collapseHyphens(b.String())
	slug = strings.Trim(slug, "-")
	if len(slug) > 60 {
		slug = slug[:60]
		slug = strings.TrimRight(slug, "-")
	}
	return slug
}

// collapseHyphens reduces runs of hyphens to a single hyphen.
func collapseHyphens(s string) string {
	var b strings.Builder
	var prevDash bool
	for _, r := range s {
		if r == '-' {
			if prevDash {
				continue
			}
			prevDash = true
			b.WriteRune(r)
			continue
		}
		prevDash = false
		b.WriteRune(r)
	}
	return b.String()
}

// collapseWhitespace trims and collapses runs of whitespace to a single
// space, preserving paragraph breaks where possible.
func collapseWhitespace(s string) string {
	paragraphs := strings.Split(s, "\n\n")
	for i, p := range paragraphs {
		paragraphs[i] = strings.Join(strings.Fields(p), " ")
	}
	joined := strings.Join(paragraphs, "\n\n")
	return strings.TrimSpace(joined)
}

// normaliseURL ensures a URL has a scheme and is well-formed.
func normaliseURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", fmt.Errorf("knowledge: empty URL")
	}
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("knowledge: parsing URL: %w", err)
	}
	if parsed.Host == "" {
		return "", fmt.Errorf("knowledge: URL missing host")
	}
	return parsed.String(), nil
}

// firstNonEmpty returns the first non-empty argument.
func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

// dedupeStrings removes duplicate entries while preserving order.
func dedupeStrings(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		key := strings.TrimSpace(s)
		if key == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, key)
	}
	return out
}
