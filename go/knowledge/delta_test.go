// SPDX-License-Identifier: Apache-2.0
package knowledge

import (
	"context"
	"fmt"
	"testing"

	"github.com/jeffs-brain/memory/go/store/mem"
)

func TestComputeChunkDeltas_first_ingest(t *testing.T) {
	t.Parallel()
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "alpha content"},
		{DocumentID: "doc1", Ordinal: 1, Text: "beta content"},
		{DocumentID: "doc1", Ordinal: 2, Text: "gamma content"},
	}

	deltas := ComputeChunkDeltas(chunks, nil)

	if len(deltas) != 3 {
		t.Fatalf("expected 3 deltas, got %d", len(deltas))
	}
	for i, d := range deltas {
		if d.Category != DeltaNew {
			t.Errorf("delta[%d] expected DeltaNew, got %d", i, d.Category)
		}
		if d.Hash == "" {
			t.Errorf("delta[%d] has empty hash", i)
		}
	}
}

func TestComputeChunkDeltas_no_change(t *testing.T) {
	t.Parallel()
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "alpha content"},
		{DocumentID: "doc1", Ordinal: 1, Text: "beta content"},
	}

	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("alpha content"), ChunkID: "doc1hash_0"},
			{Hash: HashChunk("beta content"), ChunkID: "doc1hash_1"},
		},
	}

	deltas := ComputeChunkDeltas(chunks, manifest)

	var newCount, unchangedCount, removedCount int
	for _, d := range deltas {
		switch d.Category {
		case DeltaNew:
			newCount++
		case DeltaUnchanged:
			unchangedCount++
		case DeltaRemoved:
			removedCount++
		}
	}

	if newCount != 0 {
		t.Errorf("expected 0 new, got %d", newCount)
	}
	if unchangedCount != 2 {
		t.Errorf("expected 2 unchanged, got %d", unchangedCount)
	}
	if removedCount != 0 {
		t.Errorf("expected 0 removed, got %d", removedCount)
	}
}

func TestComputeChunkDeltas_chunk_added(t *testing.T) {
	t.Parallel()
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "alpha content"},
		{DocumentID: "doc1", Ordinal: 1, Text: "beta content"},
		{DocumentID: "doc1", Ordinal: 2, Text: "new paragraph"},
	}

	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("alpha content"), ChunkID: "doc1hash_0"},
			{Hash: HashChunk("beta content"), ChunkID: "doc1hash_1"},
		},
	}

	deltas := ComputeChunkDeltas(chunks, manifest)

	var newCount, unchangedCount int
	for _, d := range deltas {
		switch d.Category {
		case DeltaNew:
			newCount++
			if d.Chunk.Text != "new paragraph" {
				t.Errorf("expected new chunk text 'new paragraph', got %q", d.Chunk.Text)
			}
		case DeltaUnchanged:
			unchangedCount++
		case DeltaRemoved:
			t.Errorf("unexpected removed chunk: %+v", d)
		}
	}

	if newCount != 1 {
		t.Errorf("expected 1 new, got %d", newCount)
	}
	if unchangedCount != 2 {
		t.Errorf("expected 2 unchanged, got %d", unchangedCount)
	}
}

func TestComputeChunkDeltas_chunk_removed(t *testing.T) {
	t.Parallel()
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "alpha content"},
	}

	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("alpha content"), ChunkID: "doc1hash_0"},
			{Hash: HashChunk("beta content"), ChunkID: "doc1hash_1"},
		},
	}

	deltas := ComputeChunkDeltas(chunks, manifest)

	var unchangedCount, removedCount int
	for _, d := range deltas {
		switch d.Category {
		case DeltaNew:
			t.Errorf("unexpected new chunk: %+v", d)
		case DeltaUnchanged:
			unchangedCount++
		case DeltaRemoved:
			removedCount++
			if d.Hash != HashChunk("beta content") {
				t.Errorf("expected removed hash for 'beta content', got %q", d.Hash)
			}
		}
	}

	if unchangedCount != 1 {
		t.Errorf("expected 1 unchanged, got %d", unchangedCount)
	}
	if removedCount != 1 {
		t.Errorf("expected 1 removed, got %d", removedCount)
	}
}

func TestComputeChunkDeltas_reordered(t *testing.T) {
	t.Parallel()
	// Same content in different positions should all be DeltaUnchanged.
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "beta content"},
		{DocumentID: "doc1", Ordinal: 1, Text: "alpha content"},
		{DocumentID: "doc1", Ordinal: 2, Text: "gamma content"},
	}

	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("alpha content"), ChunkID: "doc1hash_0"},
			{Hash: HashChunk("beta content"), ChunkID: "doc1hash_1"},
			{Hash: HashChunk("gamma content"), ChunkID: "doc1hash_2"},
		},
	}

	deltas := ComputeChunkDeltas(chunks, manifest)

	for i, d := range deltas {
		if d.Category != DeltaUnchanged {
			t.Errorf("delta[%d] expected DeltaUnchanged (reorder), got %d for text %q", i, d.Category, d.Chunk.Text)
		}
	}
}

func TestComputeChunkDeltas_one_chunk_changed(t *testing.T) {
	t.Parallel()
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "alpha content"},
		{DocumentID: "doc1", Ordinal: 1, Text: "beta MODIFIED content"},
		{DocumentID: "doc1", Ordinal: 2, Text: "gamma content"},
	}

	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("alpha content"), ChunkID: "doc1hash_0"},
			{Hash: HashChunk("beta content"), ChunkID: "doc1hash_1"},
			{Hash: HashChunk("gamma content"), ChunkID: "doc1hash_2"},
		},
	}

	deltas := ComputeChunkDeltas(chunks, manifest)

	var newCount, unchangedCount, removedCount int
	for _, d := range deltas {
		switch d.Category {
		case DeltaNew:
			newCount++
			if d.Chunk.Text != "beta MODIFIED content" {
				t.Errorf("expected new chunk 'beta MODIFIED content', got %q", d.Chunk.Text)
			}
		case DeltaUnchanged:
			unchangedCount++
		case DeltaRemoved:
			removedCount++
			if d.Hash != HashChunk("beta content") {
				t.Errorf("expected removed hash for 'beta content', got %q", d.Hash)
			}
		}
	}

	if newCount != 1 {
		t.Errorf("expected 1 new, got %d", newCount)
	}
	if unchangedCount != 2 {
		t.Errorf("expected 2 unchanged, got %d", unchangedCount)
	}
	if removedCount != 1 {
		t.Errorf("expected 1 removed, got %d", removedCount)
	}
}

func TestComputeChunkDeltas_empty_new_chunks(t *testing.T) {
	t.Parallel()
	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("alpha content"), ChunkID: "doc1hash_0"},
			{Hash: HashChunk("beta content"), ChunkID: "doc1hash_1"},
		},
	}

	deltas := ComputeChunkDeltas(nil, manifest)

	var removedCount int
	for _, d := range deltas {
		if d.Category != DeltaRemoved {
			t.Errorf("expected DeltaRemoved, got %d", d.Category)
		}
		removedCount++
	}
	if removedCount != 2 {
		t.Errorf("expected 2 removed, got %d", removedCount)
	}
}

func TestComputeChunkDeltas_duplicate_content(t *testing.T) {
	t.Parallel()
	// Two chunks with identical content in new set, one in old manifest.
	// One should be unchanged, one should be new.
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "repeated"},
		{DocumentID: "doc1", Ordinal: 1, Text: "repeated"},
	}

	manifest := &ChunkManifest{
		DocumentHash: "doc1hash",
		Generation:   1,
		Chunks: []ChunkManifestEntry{
			{Hash: HashChunk("repeated"), ChunkID: "doc1hash_0"},
		},
	}

	deltas := ComputeChunkDeltas(chunks, manifest)

	var newCount, unchangedCount int
	for _, d := range deltas {
		switch d.Category {
		case DeltaNew:
			newCount++
		case DeltaUnchanged:
			unchangedCount++
		case DeltaRemoved:
			t.Errorf("unexpected removed delta")
		}
	}

	if unchangedCount != 1 {
		t.Errorf("expected 1 unchanged, got %d", unchangedCount)
	}
	if newCount != 1 {
		t.Errorf("expected 1 new, got %d", newCount)
	}
}

func TestReadWriteChunkManifest_round_trip(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "first chunk"},
		{DocumentID: "doc1", Ordinal: 1, Text: "second chunk"},
		{DocumentID: "doc1", Ordinal: 2, Text: "third chunk"},
	}

	manifest := BuildChunkManifest("abc123def456", chunks)

	if err := WriteChunkManifest(ctx, store, manifest); err != nil {
		t.Fatalf("write manifest: %v", err)
	}

	loaded, err := ReadChunkManifest(ctx, store, "abc123def456")
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	if loaded == nil {
		t.Fatal("expected non-nil manifest after write")
	}

	if loaded.DocumentHash != "abc123def456" {
		t.Errorf("expected document hash 'abc123def456', got %q", loaded.DocumentHash)
	}
	if loaded.Generation != 1 {
		t.Errorf("expected generation 1, got %d", loaded.Generation)
	}
	if len(loaded.Chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(loaded.Chunks))
	}

	for i, entry := range loaded.Chunks {
		expectedHash := HashChunk(chunks[i].Text)
		if entry.Hash != expectedHash {
			t.Errorf("chunk[%d] hash mismatch: got %q, want %q", i, entry.Hash, expectedHash)
		}
		expectedID := fmt.Sprintf("abc123def456_%d", i)
		if entry.ChunkID != expectedID {
			t.Errorf("chunk[%d] id mismatch: got %q, want %q", i, entry.ChunkID, expectedID)
		}
	}

	if loaded.UpdatedAt.IsZero() {
		t.Error("expected non-zero UpdatedAt")
	}
}

func TestReadChunkManifest_missing(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	loaded, err := ReadChunkManifest(ctx, store, "nonexistent")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if loaded != nil {
		t.Fatalf("expected nil manifest for missing document, got %+v", loaded)
	}
}

func TestWriteChunkManifest_increments_generation(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "content v1"},
	}

	manifest := BuildChunkManifest("docXYZ", chunks)

	if err := WriteChunkManifest(ctx, store, manifest); err != nil {
		t.Fatalf("first write: %v", err)
	}

	loaded, err := ReadChunkManifest(ctx, store, "docXYZ")
	if err != nil {
		t.Fatalf("first read: %v", err)
	}
	if loaded.Generation != 1 {
		t.Errorf("expected generation 1 after first write, got %d", loaded.Generation)
	}

	// Write again with different content.
	chunks2 := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "content v2"},
	}
	manifest2 := BuildChunkManifest("docXYZ", chunks2)

	if err := WriteChunkManifest(ctx, store, manifest2); err != nil {
		t.Fatalf("second write: %v", err)
	}

	loaded2, err := ReadChunkManifest(ctx, store, "docXYZ")
	if err != nil {
		t.Fatalf("second read: %v", err)
	}
	if loaded2.Generation != 2 {
		t.Errorf("expected generation 2 after second write, got %d", loaded2.Generation)
	}
}

func TestHashChunk_deterministic(t *testing.T) {
	t.Parallel()
	h1 := HashChunk("hello world")
	h2 := HashChunk("hello world")
	if h1 != h2 {
		t.Errorf("HashChunk not deterministic: %q != %q", h1, h2)
	}
	h3 := HashChunk("different content")
	if h1 == h3 {
		t.Error("HashChunk collision on different content")
	}
}

func TestBuildChunkManifest(t *testing.T) {
	t.Parallel()
	chunks := []Chunk{
		{DocumentID: "doc1", Ordinal: 0, Text: "chunk A"},
		{DocumentID: "doc1", Ordinal: 1, Text: "chunk B"},
	}

	manifest := BuildChunkManifest("hash123", chunks)

	if manifest.DocumentHash != "hash123" {
		t.Errorf("expected document hash 'hash123', got %q", manifest.DocumentHash)
	}
	if len(manifest.Chunks) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(manifest.Chunks))
	}
	if manifest.Chunks[0].ChunkID != "hash123_0" {
		t.Errorf("expected chunk id 'hash123_0', got %q", manifest.Chunks[0].ChunkID)
	}
	if manifest.Chunks[1].ChunkID != "hash123_1" {
		t.Errorf("expected chunk id 'hash123_1', got %q", manifest.Chunks[1].ChunkID)
	}
	if manifest.Chunks[0].Hash != HashChunk("chunk A") {
		t.Errorf("chunk[0] hash mismatch")
	}
	if manifest.Chunks[1].Hash != HashChunk("chunk B") {
		t.Errorf("chunk[1] hash mismatch")
	}
}
