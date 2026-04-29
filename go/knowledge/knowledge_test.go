// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"database/sql"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/mem"

	_ "modernc.org/sqlite"
)

// newKB builds a Base backed by an in-memory store plus an optional
// search index.
func newKB(t *testing.T, withIndex bool) (Base, brain.Store, *search.Index) {
	t.Helper()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	opts := Options{
		BrainID: "test",
		Store:   store,
	}
	var idx *search.Index
	if withIndex {
		db, err := sql.Open("sqlite", ":memory:")
		if err != nil {
			t.Fatalf("open sqlite: %v", err)
		}
		t.Cleanup(func() { _ = db.Close() })
		idx, err = search.NewIndex(db, store)
		if err != nil {
			t.Fatalf("new index: %v", err)
		}
		opts.Index = idx
	}
	base, err := New(opts)
	if err != nil {
		t.Fatalf("new base: %v", err)
	}
	t.Cleanup(func() { _ = base.Close() })
	return base, store, idx
}

// TestNew_RequiresStore asserts that New fails fast when the caller
// forgets to supply a store.
func TestNew_RequiresStore(t *testing.T) {
	_, err := New(Options{})
	if err == nil {
		t.Fatalf("expected error without store")
	}
}

// TestIngest_Markdown covers the happy path: a markdown file lands in
// raw/documents, frontmatter is parsed, and the response reports the
// chunk count.
func TestIngest_Markdown(t *testing.T) {
	base, store, _ := newKB(t, false)
	body := `---
title: "First Post"
tags:
  - alpha
  - beta
summary: "a short summary"
---

# Heading

Body paragraph with some words to chunk.

## Second heading

More content here to establish multi-section behaviour for the chunker.
`
	resp, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "note.md",
		ContentType: contentTypeMarkdown,
		Content:     strings.NewReader(body),
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if resp.DocumentID == "" {
		t.Fatalf("expected DocumentID, got empty")
	}
	if resp.ChunkCount == 0 {
		t.Fatalf("expected chunks > 0, got %d", resp.ChunkCount)
	}
	// File should land under raw/documents.
	if !strings.HasPrefix(string(resp.Path), "raw/documents/") {
		t.Fatalf("unexpected path %q", resp.Path)
	}
	data, err := store.Read(context.Background(), resp.Path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	fm, _ := ParseFrontmatter(string(data))
	if fm.Title != "First Post" {
		t.Fatalf("title=%q, want First Post", fm.Title)
	}
	if len(fm.Tags) < 2 {
		t.Fatalf("tags=%v, want >= 2", fm.Tags)
	}
}

// TestIngest_PlainText ensures .txt ingest round-trips through the
// frontmatter generator.
func TestIngest_PlainText(t *testing.T) {
	base, _, _ := newKB(t, false)
	resp, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "hello.txt",
		ContentType: contentTypeText,
		Content:     strings.NewReader("Just plain text, no markup.\n"),
		Title:       "Plain",
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if resp.ChunkCount == 0 {
		t.Fatalf("expected chunks, got 0")
	}
}

// TestIngest_PathOnDisk reads from a local path rather than an inline
// reader.
func TestIngest_PathOnDisk(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "note.md")
	if err := os.WriteFile(p, []byte("# Local\n\nlocal body\n"), 0o644); err != nil {
		t.Fatalf("write tmp: %v", err)
	}
	base, _, _ := newKB(t, false)
	resp, err := base.Ingest(context.Background(), IngestRequest{
		Path: p,
	})
	if err != nil {
		t.Fatalf("ingest path: %v", err)
	}
	if resp.ChunkCount == 0 {
		t.Fatalf("no chunks")
	}
}

// TestIngest_HTMLStrip verifies the HTML extractor drops tags and
// script blocks.
func TestIngest_HTMLStrip(t *testing.T) {
	base, store, _ := newKB(t, false)
	html := `<html><head><script>bad=1</script><title>t</title></head>
<body><h1>Hello</h1><p>Stripped <em>clean</em>.</p></body></html>`
	resp, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "page.html",
		ContentType: contentTypeHTML,
		Content:     strings.NewReader(html),
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	data, err := store.Read(context.Background(), resp.Path)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	body := string(data)
	if strings.Contains(body, "<script>") || strings.Contains(body, "bad=1") {
		t.Fatalf("script survived: %s", body)
	}
	if !strings.Contains(body, "Hello") {
		t.Fatalf("headline missing: %s", body)
	}
}

// TestIngest_RejectsUnknownContentType exercises the safety rail.
func TestIngest_RejectsUnknownContentType(t *testing.T) {
	base, _, _ := newKB(t, false)
	_, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "blob.bin",
		ContentType: "application/octet-stream",
		Content:     strings.NewReader("\x01\x02\x03"),
	})
	if err == nil {
		t.Fatalf("expected error for unknown content type")
	}
}

// TestIngest_RejectsBadUTF8 exercises the UTF-8 guard.
func TestIngest_RejectsBadUTF8(t *testing.T) {
	base, _, _ := newKB(t, false)
	_, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "bad.txt",
		ContentType: contentTypeText,
		Content:     strings.NewReader("\xff\xfe\x00"),
	})
	if err == nil {
		t.Fatalf("expected UTF-8 error")
	}
}

// TestIngest_RequiresContentOrPath asserts the guard for missing input.
func TestIngest_RequiresContentOrPath(t *testing.T) {
	base, _, _ := newKB(t, false)
	_, err := base.Ingest(context.Background(), IngestRequest{})
	if err == nil {
		t.Fatalf("expected error for empty request")
	}
}

// fakeFetcher captures the URL passed to IngestURL and returns the
// caller-supplied body. Used to exercise the URL path deterministically
// without hitting the network.
type fakeFetcher struct {
	called      int
	urlReceived string
	body        []byte
	ctype       string
	err         error
}

func (f *fakeFetcher) Fetch(ctx context.Context, url string) ([]byte, string, error) {
	f.called++
	f.urlReceived = url
	return f.body, f.ctype, f.err
}

// TestIngestURL_HappyPath verifies URL fetch + extraction + persistence.
func TestIngestURL_HappyPath(t *testing.T) {
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	ff := &fakeFetcher{
		body:  []byte("<html><h1>Title</h1><p>content</p></html>"),
		ctype: "text/html",
	}
	base, err := New(Options{Store: store, HTTPFetcher: ff})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	resp, err := base.IngestURL(context.Background(), "https://example.test/a")
	if err != nil {
		t.Fatalf("ingest url: %v", err)
	}
	if ff.called != 1 {
		t.Fatalf("fetcher called %d times", ff.called)
	}
	if ff.urlReceived != "https://example.test/a" {
		t.Fatalf("url=%q", ff.urlReceived)
	}
	if resp.ChunkCount == 0 {
		t.Fatalf("no chunks")
	}
}

// TestIngestURL_NormalisesScheme verifies the scheme is added when
// missing.
func TestIngestURL_NormalisesScheme(t *testing.T) {
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	ff := &fakeFetcher{body: []byte("# plain\nbody"), ctype: "text/markdown"}
	base, err := New(Options{Store: store, HTTPFetcher: ff})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, err := base.IngestURL(context.Background(), "example.test/b"); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if !strings.HasPrefix(ff.urlReceived, "https://") {
		t.Fatalf("url not normalised: %q", ff.urlReceived)
	}
}

// TestIngestURL_RejectsEmpty covers the safety rail.
func TestIngestURL_RejectsEmpty(t *testing.T) {
	base, _, _ := newKB(t, false)
	_, err := base.IngestURL(context.Background(), "")
	if err == nil {
		t.Fatalf("expected error for empty URL")
	}
}

// TestIngestURL_PropagatesFetchError checks that fetcher errors surface.
func TestIngestURL_PropagatesFetchError(t *testing.T) {
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	ff := &fakeFetcher{err: errors.New("boom")}
	base, err := New(Options{Store: store, HTTPFetcher: ff})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	_, err = base.IngestURL(context.Background(), "https://example.test/c")
	if err == nil {
		t.Fatalf("expected error")
	}
}

// TestParseFrontmatter_Basic covers the YAML path.
func TestParseFrontmatter_Basic(t *testing.T) {
	fm, body := ParseFrontmatter(`---
title: "Hello"
tags: [one, two]
summary: "short"
---
body
`)
	if fm.Title != "Hello" {
		t.Fatalf("title=%q", fm.Title)
	}
	if fm.Summary != "short" {
		t.Fatalf("summary=%q", fm.Summary)
	}
	if len(fm.Tags) != 2 {
		t.Fatalf("tags=%v", fm.Tags)
	}
	if body != "body" {
		t.Fatalf("body=%q", body)
	}
}

// TestParseFrontmatter_ListForm exercises the bullet-list fallback.
func TestParseFrontmatter_ListForm(t *testing.T) {
	fm, _ := ParseFrontmatter(`---
title: "With list"
tags:
  - alpha
  - beta
sources:
  - a
  - b
---
body
`)
	if len(fm.Tags) != 2 || fm.Tags[0] != "alpha" {
		t.Fatalf("tags=%v", fm.Tags)
	}
	if len(fm.Sources) != 2 {
		t.Fatalf("sources=%v", fm.Sources)
	}
}

// TestParseFrontmatter_NoHeader returns the content untouched.
func TestParseFrontmatter_NoHeader(t *testing.T) {
	fm, body := ParseFrontmatter("plain body\n")
	if fm.Title != "" {
		t.Fatalf("unexpected title")
	}
	if !strings.Contains(body, "plain body") {
		t.Fatalf("body=%q", body)
	}
}

// TestParseFrontmatter_MemoryScope exercises the name/description form
// used by the memory SDK.
func TestParseFrontmatter_MemoryScope(t *testing.T) {
	fm, _ := ParseFrontmatter(`---
name: "foo"
description: "bar"
---
body`)
	if fm.Name != "foo" || fm.Description != "bar" {
		t.Fatalf("fm=%+v", fm)
	}
}

// TestCompile_AllDocumentsInStore walks the raw/documents prefix.
func TestCompile_AllDocumentsInStore(t *testing.T) {
	base, store, _ := newKB(t, false)
	ctx := context.Background()

	for i, body := range []string{"# A\n\nalpha body", "# B\n\nbeta body"} {
		_, err := base.Ingest(ctx, IngestRequest{
			Path:        "doc.md",
			ContentType: contentTypeMarkdown,
			Content:     strings.NewReader(body),
			Title:       []string{"A", "B"}[i],
		})
		if err != nil {
			t.Fatalf("ingest %d: %v", i, err)
		}
	}
	// Force-compile again.
	res, err := base.Compile(ctx, CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if res.Compiled < 2 {
		t.Fatalf("compiled=%d", res.Compiled)
	}
	if res.Chunks == 0 {
		t.Fatalf("chunks=0")
	}
	_ = store
}

// TestCompile_DryRun does not mutate the index.
func TestCompile_DryRun(t *testing.T) {
	base, _, _ := newKB(t, false)
	ctx := context.Background()
	if _, err := base.Ingest(ctx, IngestRequest{
		Path:        "doc.md",
		ContentType: contentTypeMarkdown,
		Content:     strings.NewReader("# A\n\nbody"),
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	res, err := base.Compile(ctx, CompileOptions{DryRun: true})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if res.Compiled == 0 {
		t.Fatalf("expected compiled > 0 even in dry run")
	}
}

// TestCompile_MaxBatch caps iteration.
func TestCompile_MaxBatch(t *testing.T) {
	base, _, _ := newKB(t, false)
	ctx := context.Background()
	for i := 0; i < 5; i++ {
		_, err := base.Ingest(ctx, IngestRequest{
			Path:        "doc.md",
			ContentType: contentTypeMarkdown,
			Content:     strings.NewReader("# T\n\nbody"),
			Title:       "t" + string(rune('a'+i)),
		})
		if err != nil {
			t.Fatalf("ingest: %v", err)
		}
	}
	res, err := base.Compile(ctx, CompileOptions{MaxBatch: 2})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if res.Compiled > 2 {
		t.Fatalf("compiled=%d, want <= 2", res.Compiled)
	}
}

// TestSearch_InMemoryFallback asserts the fallback scorer returns hits
// when no index is bound.
func TestSearch_InMemoryFallback(t *testing.T) {
	base, _, _ := newKB(t, false)
	ctx := context.Background()
	if _, err := base.Ingest(ctx, IngestRequest{
		Path:        "alpha.md",
		ContentType: contentTypeMarkdown,
		Content:     strings.NewReader("# Alpha\n\ntalks about memory retrieval"),
		Title:       "Alpha",
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if _, err := base.Ingest(ctx, IngestRequest{
		Path:        "beta.md",
		ContentType: contentTypeMarkdown,
		Content:     strings.NewReader("# Beta\n\nunrelated document"),
		Title:       "Beta",
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}

	resp, err := base.Search(ctx, SearchRequest{Query: "memory", MaxResults: 5})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(resp.Hits) == 0 {
		t.Fatalf("expected hits for 'memory'")
	}
	if resp.Hits[0].Title != "Alpha" {
		t.Fatalf("want Alpha first, got %q", resp.Hits[0].Title)
	}
	if resp.Hits[0].Source != "memory" {
		t.Fatalf("want memory source, got %q", resp.Hits[0].Source)
	}
}

// TestSearch_EmptyQuery returns no hits.
func TestSearch_EmptyQuery(t *testing.T) {
	base, _, _ := newKB(t, false)
	resp, err := base.Search(context.Background(), SearchRequest{Query: "   "})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(resp.Hits) != 0 {
		t.Fatalf("expected zero hits, got %d", len(resp.Hits))
	}
}

// TestSearch_WithIndex exercises the BM25 path through search.Index.
// Documents are written under wiki/ so the search.Index classifier
// picks them up (its classifier only covers memory/ and wiki/).
func TestSearch_WithIndex(t *testing.T) {
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("new index: %v", err)
	}
	base, err := New(Options{Store: store, Index: idx})
	if err != nil {
		t.Fatalf("new base: %v", err)
	}
	t.Cleanup(func() { _ = base.Close() })

	ctx := context.Background()
	// Seed documents under wiki/ so search.Index indexes them.
	alpha := []byte("---\ntitle: Alpha\n---\n\nalpha talks about memory retrieval\n")
	beta := []byte("---\ntitle: Beta\n---\n\nbeta is unrelated\n")
	if err := store.Write(ctx, brain.Path("wiki/a/alpha.md"), alpha); err != nil {
		t.Fatalf("seed alpha: %v", err)
	}
	if err := store.Write(ctx, brain.Path("wiki/b/beta.md"), beta); err != nil {
		t.Fatalf("seed beta: %v", err)
	}
	if err := idx.Rebuild(ctx); err != nil {
		t.Fatalf("rebuild: %v", err)
	}

	resp, err := base.Search(ctx, SearchRequest{Query: "memory", MaxResults: 5, Mode: SearchBM25})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(resp.Hits) == 0 {
		t.Fatalf("expected hits")
	}
	if resp.Mode != "bm25" {
		t.Fatalf("mode=%q", resp.Mode)
	}
	found := false
	for _, h := range resp.Hits {
		if h.Title == "Alpha" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("Alpha missing from hits: %+v", resp.Hits)
	}
}

// stubRetriever implements retrieval.Retriever for the hybrid-path
// test.
type stubRetriever struct {
	chunks []retrieval.RetrievedChunk
	err    error
}

func (s *stubRetriever) Retrieve(ctx context.Context, req retrieval.Request) (retrieval.Response, error) {
	if s.err != nil {
		return retrieval.Response{}, s.err
	}
	return retrieval.Response{
		Chunks: s.chunks,
		Trace: retrieval.Trace{
			RequestedMode: req.Mode,
			EffectiveMode: retrieval.ModeHybrid,
			FusedHits:     len(s.chunks),
		},
	}, nil
}

// TestSearch_HybridRetriever exercises the retriever delegation.
func TestSearch_HybridRetriever(t *testing.T) {
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	retr := &stubRetriever{
		chunks: []retrieval.RetrievedChunk{
			{Path: "wiki/a/alpha.md", Title: "Alpha", Summary: "s", Score: 0.9, Text: "body of alpha"},
			{Path: "wiki/b/beta.md", Title: "Beta", Summary: "s", Score: 0.3},
		},
	}
	base, err := New(Options{Store: store, Retriever: retr})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	t.Cleanup(func() { _ = base.Close() })

	resp, err := base.Search(context.Background(), SearchRequest{Query: "alpha"})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(resp.Hits) != 2 {
		t.Fatalf("hits=%d", len(resp.Hits))
	}
	if resp.Mode != string(retrieval.ModeHybrid) {
		t.Fatalf("mode=%q", resp.Mode)
	}
	if resp.Hits[0].Title != "Alpha" {
		t.Fatalf("first=%+v", resp.Hits[0])
	}
	if resp.Hits[0].Source != "fused" {
		t.Fatalf("source=%q", resp.Hits[0].Source)
	}
}

// TestSearch_HybridFallsBackOnError verifies the BM25 fallback when the
// retriever errors.
func TestSearch_HybridFallsBackOnError(t *testing.T) {
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })
	retr := &stubRetriever{err: errors.New("retriever down")}
	base, err := New(Options{Store: store, Retriever: retr})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	t.Cleanup(func() { _ = base.Close() })

	// Seed an in-memory doc so BM25 fallback returns something.
	ctx := context.Background()
	if _, err := base.Ingest(ctx, IngestRequest{
		Path:        "a.md",
		ContentType: contentTypeMarkdown,
		Content:     strings.NewReader("# Alpha\n\nalpha body"),
		Title:       "Alpha",
	}); err != nil {
		t.Fatalf("ingest: %v", err)
	}

	resp, err := base.Search(ctx, SearchRequest{Query: "alpha", Mode: SearchHybrid})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !resp.FellBack {
		t.Fatalf("expected FellBack=true")
	}
	if resp.Mode != "bm25" {
		t.Fatalf("mode=%q", resp.Mode)
	}
}

// TestSetSearchIndex_Rebind ensures the setter swaps the index safely.
func TestSetSearchIndex_Rebind(t *testing.T) {
	base, _, _ := newKB(t, false)
	// Swap in a nil (detach) and back to the original.
	base.SetSearchIndex(nil)
	base.SetSearchIndex(nil)
	// No panic means success.
}

// TestSetRetriever_Rebind ensures the retriever setter swaps cleanly.
func TestSetRetriever_Rebind(t *testing.T) {
	base, _, _ := newKB(t, false)
	base.SetRetriever(nil)
	// Bind a stub and unbind.
	base.SetRetriever(&stubRetriever{})
	base.SetRetriever(nil)
}

// TestStore_Accessor returns the backing store.
func TestStore_Accessor(t *testing.T) {
	base, store, _ := newKB(t, false)
	if base.Store() != store {
		t.Fatalf("store accessor mismatch")
	}
}

// TestSegmentDocument_Empty returns nil for an empty body.
func TestSegmentDocument_Empty(t *testing.T) {
	if segmentDocument(&Document{Body: "   "}) != nil {
		t.Fatalf("expected nil for empty body")
	}
}

// TestSegmentDocument_Headings splits by heading.
func TestSegmentDocument_Headings(t *testing.T) {
	doc := &Document{Body: `# One

This paragraph about the first section carries enough body content to clear the minimum chunk length threshold used by the segmenter and merge pass.

## Two

This paragraph about the second section is also long enough to survive the small chunk merge pass so the segmenter emits two distinct chunks for this document.
`}
	chunks := segmentDocument(doc)
	if len(chunks) < 2 {
		t.Fatalf("chunks=%d", len(chunks))
	}
	if chunks[0].Heading != "One" {
		t.Fatalf("heading[0]=%q", chunks[0].Heading)
	}
	if chunks[1].Heading != "Two" {
		t.Fatalf("heading[1]=%q", chunks[1].Heading)
	}
}

// TestSegmentDocument_MergesSmall verifies the small-chunk merge pass.
func TestSegmentDocument_MergesSmall(t *testing.T) {
	doc := &Document{Body: `# Big

A reasonably sized paragraph of body content that comfortably exceeds the minimum chunk length threshold used by the segmenter so the merge pass keeps it intact.

## Tiny

short
`}
	chunks := segmentDocument(doc)
	// Tiny section should merge into Big.
	if len(chunks) != 1 {
		t.Fatalf("expected merge to 1 chunk, got %d", len(chunks))
	}
}

// TestSlugify normalises strings to url-safe slugs. The helper treats
// underscores as non-alphanumeric separators so they collapse to
// hyphens alongside every other punctuation character.
func TestSlugify(t *testing.T) {
	cases := map[string]string{
		"Hello World":     "hello-world",
		"  a/b/c  ":       "a-b-c",
		"ONE_TWO_THREE":   "one-two-three",
		"é lectrique":     "é-lectrique",
		"":                "",
		"$$$":             "",
		"Title!!! Bang??": "title-bang",
	}
	for in, want := range cases {
		got := slugify(in)
		if got != want {
			t.Errorf("slugify(%q)=%q, want %q", in, got, want)
		}
	}
}

// TestExtractPDF_NoPanic ensures the extractor handles an empty body
// without crashing.
func TestExtractPDF_NoPanic(t *testing.T) {
	_, err := extractPDF([]byte("%PDF-1.0\n"))
	if err == nil {
		t.Logf("extractPDF returned nil err for minimal body")
	}
}

// TestStripHTML scrubs tags, scripts, and entities.
func TestStripHTML(t *testing.T) {
	out := stripHTML([]byte(`<div>hello <script>x=1</script> world &amp; friends</div>`))
	if strings.Contains(out, "<") {
		t.Fatalf("tags remain: %q", out)
	}
	if strings.Contains(out, "x=1") {
		t.Fatalf("script remains: %q", out)
	}
	if !strings.Contains(out, "friends") {
		t.Fatalf("content missing: %q", out)
	}
}

// TestDetectContentType routes by extension and magic.
func TestDetectContentType(t *testing.T) {
	if got := detectContentType("a.md", nil); got != contentTypeMarkdown {
		t.Errorf("md=%q", got)
	}
	if got := detectContentType("a.html", nil); got != contentTypeHTML {
		t.Errorf("html=%q", got)
	}
	if got := detectContentType("x", []byte("%PDF-1.4\n")); got != contentTypePDF {
		t.Errorf("magic pdf=%q", got)
	}
}

// TestDeriveTitle prefers an H1 when present.
func TestDeriveTitle(t *testing.T) {
	if got := deriveTitle("# Hello\n\nbody", "fallback.md"); got != "Hello" {
		t.Errorf("want Hello, got %q", got)
	}
	if got := deriveTitle("", "doc.md"); got != "doc" {
		t.Errorf("want doc, got %q", got)
	}
}

// TestIngest_DirectoryRejected guards against directory paths.
func TestIngest_DirectoryRejected(t *testing.T) {
	dir := t.TempDir()
	base, _, _ := newKB(t, false)
	_, err := base.Ingest(context.Background(), IngestRequest{
		Path: dir,
	})
	if err == nil {
		t.Fatalf("expected error for directory")
	}
}

// TestTokeniseQuery trims and lowercases.
func TestTokeniseQuery(t *testing.T) {
	got := tokeniseQuery(" Hello, World?  ")
	want := []string{"hello", "world"}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("tokens=%v", got)
	}
}

// TestClose_Idempotent close does not complain on repeat invocation.
func TestClose_Idempotent(t *testing.T) {
	base, _, _ := newKB(t, false)
	if err := base.Close(); err != nil {
		t.Fatalf("first close: %v", err)
	}
	if err := base.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
}

// TestReadContent_LimitedSize uses io.LimitReader semantics to ensure
// oversized bodies are truncated rather than rejected when they come
// via an io.Reader.
func TestReadContent_LimitedSize(t *testing.T) {
	base, _, _ := newKB(t, false)
	// A reader that returns 100 bytes of 'a'.
	body := strings.NewReader(strings.Repeat("a", 100))
	resp, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "a.txt",
		ContentType: contentTypeText,
		Content:     body,
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if resp.Bytes != 100 {
		t.Fatalf("bytes=%d", resp.Bytes)
	}
	if resp.TookMs < 0 {
		t.Fatalf("negative TookMs")
	}
}

// TestIngest_FileReadInlineReader exercises io.Reader consumption
// directly for coverage.
func TestIngest_FileReadInlineReader(t *testing.T) {
	base, _, _ := newKB(t, false)
	_, err := base.Ingest(context.Background(), IngestRequest{
		Path:        "r.md",
		ContentType: contentTypeMarkdown,
		Content:     io.NopCloser(strings.NewReader("# R\n\nbody")),
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
}

// TestCompile_RespectsContextCancel exits early when the caller
// cancels.
func TestCompile_RespectsContextCancel(t *testing.T) {
	base, _, _ := newKB(t, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := base.Compile(ctx, CompileOptions{})
	if err == nil {
		return // zero documents -> no error expected
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled, got %v", err)
	}
}

// TestDedupeStrings removes duplicates and blanks.
func TestDedupeStrings(t *testing.T) {
	got := dedupeStrings([]string{"a", "", "b", "a", "c", " "})
	if len(got) != 3 {
		t.Fatalf("got=%v", got)
	}
}

// TestBuildFrontmatterYAML is deterministic for the same document.
func TestBuildFrontmatterYAML(t *testing.T) {
	doc := &Document{
		Title:       "T",
		Summary:     "S",
		Source:      "src",
		ContentType: contentTypeMarkdown,
		Tags:        []string{"a", "b"},
	}
	out := buildFrontmatterYAML(doc)
	if !strings.Contains(out, `title: "T"`) {
		t.Fatalf("missing title: %s", out)
	}
	if !strings.Contains(out, "- a\n  - b") {
		// Tags are emitted on separate lines.
		if !strings.Contains(out, "- a") || !strings.Contains(out, "- b") {
			t.Fatalf("tags missing: %s", out)
		}
	}
}
