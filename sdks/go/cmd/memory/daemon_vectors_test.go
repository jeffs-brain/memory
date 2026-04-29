// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"io"
	"log/slog"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/mem"

	_ "modernc.org/sqlite"
)

const daemonVectorTestModel = "daemon-vector-test-model"

type recordingEmbedder struct {
	dims  int
	texts []string
}

func (e *recordingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	e.texts = append(e.texts, texts...)
	out := make([][]float32, len(texts))
	for i, text := range texts {
		sum := sha256.Sum256([]byte(text))
		out[i] = make([]float32, e.dims)
		for j := 0; j < e.dims; j++ {
			offset := (j * 4) % len(sum)
			raw := binary.BigEndian.Uint32(sum[offset : offset+4])
			out[i][j] = float32(raw%10000 + 1)
		}
	}
	return out, nil
}

func (e *recordingEmbedder) Dimensions() int { return e.dims }

func (e *recordingEmbedder) Close() error { return nil }

type vectorBackfillHarness struct {
	store *mem.Store
	idx   *search.Index
	vec   *search.VectorIndex
}

func newVectorBackfillHarness(t *testing.T) vectorBackfillHarness {
	t.Helper()

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("opening sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("NewIndex: %v", err)
	}
	vec, err := search.NewVectorIndex(db)
	if err != nil {
		t.Fatalf("NewVectorIndex: %v", err)
	}

	return vectorBackfillHarness{
		store: store,
		idx:   idx,
		vec:   vec,
	}
}

func TestBackfillVectorsRegeneratesWhenChecksumChanges(t *testing.T) {
	ctx := context.Background()
	h := newVectorBackfillHarness(t)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	const path = "wiki/vector.md"

	writeWikiVectorDoc(t, h.store, path, "first indexed body")
	if err := h.idx.Update(ctx); err != nil {
		t.Fatalf("first index update: %v", err)
	}

	firstEmbedder := &recordingEmbedder{dims: 4}
	backfillVectors(ctx, "test-brain", h.store, h.idx, h.vec, firstEmbedder, daemonVectorTestModel, log)
	if len(firstEmbedder.texts) != 1 {
		t.Fatalf("first backfill embedded %d texts, want 1", len(firstEmbedder.texts))
	}
	firstChecksum := indexedChecksum(t, h.idx, path)
	firstEntry := vectorEntry(t, h.vec, path)
	if firstEntry.Checksum != vectorBackfillChecksum(firstChecksum) {
		t.Fatalf("first vector checksum = %q, want %q", firstEntry.Checksum, vectorBackfillChecksum(firstChecksum))
	}
	if firstEntry.Title != "Vector Test" {
		t.Fatalf("first vector title = %q, want Vector Test", firstEntry.Title)
	}
	firstVector := append([]float32(nil), firstEntry.Vector...)

	writeWikiVectorDoc(t, h.store, path, "second indexed body with a fresh checksum")
	if err := h.idx.Update(ctx); err != nil {
		t.Fatalf("second index update: %v", err)
	}

	secondEmbedder := &recordingEmbedder{dims: 4}
	backfillVectors(ctx, "test-brain", h.store, h.idx, h.vec, secondEmbedder, daemonVectorTestModel, log)
	if len(secondEmbedder.texts) != 1 {
		t.Fatalf("second backfill embedded %d texts, want 1", len(secondEmbedder.texts))
	}
	if !strings.Contains(secondEmbedder.texts[0], "second indexed body") {
		t.Fatalf("second backfill embedded stale content: %q", secondEmbedder.texts[0])
	}

	secondChecksum := indexedChecksum(t, h.idx, path)
	if secondChecksum == firstChecksum {
		t.Fatalf("index checksum did not change: %q", secondChecksum)
	}
	secondEntry := vectorEntry(t, h.vec, path)
	if secondEntry.Checksum != vectorBackfillChecksum(secondChecksum) {
		t.Fatalf("second vector checksum = %q, want %q", secondEntry.Checksum, vectorBackfillChecksum(secondChecksum))
	}
	if equalFloat32(firstVector, secondEntry.Vector) {
		t.Fatalf("vector was not regenerated after checksum change")
	}
}

func TestBackfillVectorsNoOpsWhenChecksumMatches(t *testing.T) {
	ctx := context.Background()
	h := newVectorBackfillHarness(t)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	const path = "wiki/stable.md"

	writeWikiVectorDoc(t, h.store, path, "stable indexed body")
	if err := h.idx.Update(ctx); err != nil {
		t.Fatalf("index update: %v", err)
	}

	embedder := &recordingEmbedder{dims: 4}
	backfillVectors(ctx, "test-brain", h.store, h.idx, h.vec, embedder, daemonVectorTestModel, log)
	if len(embedder.texts) != 1 {
		t.Fatalf("first backfill embedded %d texts, want 1", len(embedder.texts))
	}

	embedder.texts = nil
	backfillVectors(ctx, "test-brain", h.store, h.idx, h.vec, embedder, daemonVectorTestModel, log)
	if len(embedder.texts) != 0 {
		t.Fatalf("second backfill embedded %d texts, want 0", len(embedder.texts))
	}
	count, err := h.vec.Count(ctx, daemonVectorTestModel)
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if count != 1 {
		t.Fatalf("vector count = %d, want 1", count)
	}
}

func TestBackfillVectorsDeletesUnindexedPaths(t *testing.T) {
	ctx := context.Background()
	h := newVectorBackfillHarness(t)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	writeWikiVectorDoc(t, h.store, "wiki/keep.md", "keep indexed body")
	writeWikiVectorDoc(t, h.store, "wiki/drop.md", "drop indexed body")
	if err := h.idx.Update(ctx); err != nil {
		t.Fatalf("first index update: %v", err)
	}

	embedder := &recordingEmbedder{dims: 4}
	backfillVectors(ctx, "test-brain", h.store, h.idx, h.vec, embedder, daemonVectorTestModel, log)
	if len(embedder.texts) != 2 {
		t.Fatalf("first backfill embedded %d texts, want 2", len(embedder.texts))
	}

	if err := h.store.Delete(ctx, brain.Path("wiki/drop.md")); err != nil {
		t.Fatalf("delete dropped doc: %v", err)
	}
	if err := h.idx.Update(ctx); err != nil {
		t.Fatalf("second index update: %v", err)
	}

	embedder.texts = nil
	backfillVectors(ctx, "test-brain", h.store, h.idx, h.vec, embedder, daemonVectorTestModel, log)
	if len(embedder.texts) != 0 {
		t.Fatalf("stale cleanup embedded %d texts, want 0", len(embedder.texts))
	}

	entries, err := h.vec.LoadAll(ctx, daemonVectorTestModel)
	if err != nil {
		t.Fatalf("LoadAll: %v", err)
	}
	if len(entries) != 1 || entries[0].Path != "wiki/keep.md" {
		t.Fatalf("vectors after stale cleanup = %+v, want only wiki/keep.md", vectorPaths(entries))
	}
}

func TestTruncateVectorBackfillTextPreservesHeadAndTail(t *testing.T) {
	text := "front-" + strings.Repeat("middle", 20) + "-tail"

	got := truncateVectorBackfillText(text, 64)

	if len(got) != 64 {
		t.Fatalf("truncated length = %d, want 64", len(got))
	}
	if !strings.HasPrefix(got, "front-") {
		t.Fatalf("truncated text lost head: %q", got)
	}
	if !strings.HasSuffix(got, "-tail") {
		t.Fatalf("truncated text lost tail: %q", got)
	}
	if !strings.Contains(got, "[...middle truncated for embedding...]") {
		t.Fatalf("truncated text missing marker: %q", got)
	}
}

func TestVectorBackfillTextUsesIndexedContentNotRawFrontmatter(t *testing.T) {
	row := search.IndexedRow{
		Title:       "Vector Test",
		Summary:     "A concise summary",
		Scope:       "project_memory",
		ProjectSlug: "eval-lme",
		SessionDate: "2024-03-10",
		Tags:        "memory replay",
		Content:     "The useful body is here.",
	}

	got := vectorBackfillText(row)

	for _, want := range []string{
		"Title: Vector Test",
		"Summary: A concise summary",
		"Scope: project_memory",
		"Project: eval-lme",
		"Session date: 2024-03-10",
		"Tags: memory replay",
		"Content: The useful body is here.",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("vector backfill text missing %q: %q", want, got)
		}
	}
	if strings.Contains(got, "---") || strings.Contains(got, "title:") {
		t.Fatalf("vector backfill text contains raw frontmatter noise: %q", got)
	}
}

func writeWikiVectorDoc(t *testing.T, store *mem.Store, path, body string) {
	t.Helper()
	content := "---\ntitle: Vector Test\n---\n" + body + "\n"
	if err := store.Write(context.Background(), brain.Path(path), []byte(content)); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func indexedChecksum(t *testing.T, idx *search.Index, path string) string {
	t.Helper()
	checksums, err := idx.IndexedChecksums()
	if err != nil {
		t.Fatalf("IndexedChecksums: %v", err)
	}
	checksum, ok := checksums[path]
	if !ok {
		t.Fatalf("indexed checksum for %s missing", path)
	}
	return checksum
}

func vectorEntry(t *testing.T, vec *search.VectorIndex, path string) search.VectorEntry {
	t.Helper()
	entry, err := vec.LoadByPath(context.Background(), path)
	if err != nil {
		t.Fatalf("LoadByPath %s: %v", path, err)
	}
	if entry == nil {
		t.Fatalf("vector entry for %s missing", path)
	}
	return *entry
}

func equalFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func vectorPaths(entries []search.VectorEntry) []string {
	paths := make([]string, len(entries))
	for i, entry := range entries {
		paths[i] = entry.Path
	}
	return paths
}
