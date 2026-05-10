// SPDX-License-Identifier: Apache-2.0
package search

import (
	"context"
	"testing"
)

func TestUpsertChunks_InsertsCorrectly(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	chunks := []Chunk{
		{
			ID:      "doc_abc_0",
			Path:    "raw/documents/test.md",
			Ordinal: 0,
			Content: "First chunk about quantum computing and entanglement.",
			Title:   "Quantum Physics",
			Tags:    []string{"physics", "quantum"},
		},
		{
			ID:      "doc_abc_1",
			Path:    "raw/documents/test.md",
			Ordinal: 1,
			Content: "Second chunk about wave functions and measurements.",
			Title:   "Quantum Physics",
			Tags:    []string{"physics", "measurement"},
		},
	}

	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}

	var count int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_fts WHERE scope = 'chunk'").Scan(&count); err != nil {
		t.Fatalf("counting chunk rows: %v", err)
	}
	if count != 2 {
		t.Fatalf("chunk row count = %d, want 2", count)
	}

	results, err := idx.Search("quantum entanglement", SearchOpts{Scope: "chunk"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one chunk result for 'quantum entanglement'")
	}
	if results[0].Path != "doc_abc_0" {
		t.Errorf("expected path = %q, got %q", "doc_abc_0", results[0].Path)
	}
}

func TestUpsertChunks_UpsertSemantics(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	chunk := Chunk{
		ID:      "doc_abc_0",
		Path:    "raw/documents/test.md",
		Ordinal: 0,
		Content: "Original content about cats.",
		Title:   "Animals",
		Tags:    []string{"cats"},
	}
	if err := idx.UpsertChunks(ctx, []Chunk{chunk}); err != nil {
		t.Fatalf("UpsertChunks first: %v", err)
	}

	chunk.Content = "Updated content about dogs and canines."
	chunk.Tags = []string{"dogs"}
	if err := idx.UpsertChunks(ctx, []Chunk{chunk}); err != nil {
		t.Fatalf("UpsertChunks second: %v", err)
	}

	var count int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_fts WHERE path = 'doc_abc_0'").Scan(&count); err != nil {
		t.Fatalf("counting: %v", err)
	}
	if count != 1 {
		t.Fatalf("expected 1 row for chunk ID, got %d", count)
	}

	results, err := idx.Search("canines", SearchOpts{Scope: "chunk"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected result for updated chunk content")
	}

	oldResults, err := idx.Search("cats", SearchOpts{Scope: "chunk"})
	if err != nil {
		t.Fatalf("Search old content: %v", err)
	}
	if len(oldResults) != 0 {
		t.Errorf("old chunk content should not be searchable, got %d results", len(oldResults))
	}
}

func TestDeleteChunks_RemovesSpecific(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	chunks := []Chunk{
		{ID: "doc_abc_0", Path: "raw/documents/test.md", Ordinal: 0, Content: "chunk zero content alpha"},
		{ID: "doc_abc_1", Path: "raw/documents/test.md", Ordinal: 1, Content: "chunk one content beta"},
		{ID: "doc_abc_2", Path: "raw/documents/test.md", Ordinal: 2, Content: "chunk two content gamma"},
	}
	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}

	if err := idx.DeleteChunks(ctx, []string{"doc_abc_1"}); err != nil {
		t.Fatalf("DeleteChunks: %v", err)
	}

	var count int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_fts WHERE scope = 'chunk'").Scan(&count); err != nil {
		t.Fatalf("counting: %v", err)
	}
	if count != 2 {
		t.Fatalf("chunk count = %d, want 2 after deleting one", count)
	}

	results, err := idx.Search("beta", SearchOpts{Scope: "chunk"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("deleted chunk should not be searchable, got %d results", len(results))
	}
}

func TestDeleteByDocPath_RemovesAllChunks(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	chunks := []Chunk{
		{ID: "doc_abc_0", Path: "raw/documents/test.md", Ordinal: 0, Content: "first chunk content"},
		{ID: "doc_abc_1", Path: "raw/documents/test.md", Ordinal: 1, Content: "second chunk content"},
		{ID: "doc_xyz_0", Path: "raw/documents/other.md", Ordinal: 0, Content: "other document chunk"},
	}
	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}

	if err := idx.DeleteByDocPath(ctx, "raw/documents/test.md"); err != nil {
		t.Fatalf("DeleteByDocPath: %v", err)
	}

	var count int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_fts WHERE scope = 'chunk'").Scan(&count); err != nil {
		t.Fatalf("counting: %v", err)
	}
	if count != 1 {
		t.Fatalf("chunk count = %d, want 1 (only 'other.md' chunk)", count)
	}

	var remaining string
	if err := db.QueryRow("SELECT path FROM knowledge_fts WHERE scope = 'chunk'").Scan(&remaining); err != nil {
		t.Fatalf("querying remaining: %v", err)
	}
	if remaining != "doc_xyz_0" {
		t.Errorf("remaining chunk ID = %q, want %q", remaining, "doc_xyz_0")
	}
}

func TestDeleteByDocPath_CleansUpMetadata(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	// Use hash-based IDs that do NOT contain the doc path as a prefix.
	// This verifies the fix: chunk IDs like "a1b2c3d4_0" should still be
	// found and deleted when the parent doc path is "raw/documents/test.md".
	chunks := []Chunk{
		{ID: "a1b2c3d4e5f6_0", Path: "raw/documents/test.md", Ordinal: 0, Content: "first chunk"},
		{ID: "a1b2c3d4e5f6_1", Path: "raw/documents/test.md", Ordinal: 1, Content: "second chunk"},
		{ID: "ffee00112233_0", Path: "raw/documents/keep.md", Ordinal: 0, Content: "keep this chunk"},
	}
	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}

	// Attach metadata to chunks belonging to the target document.
	if err := idx.SetChunkMetadata(ctx, "a1b2c3d4e5f6_0", map[string]string{
		"ontology_type": "factual",
		"confidence":    "0.9",
	}); err != nil {
		t.Fatalf("SetChunkMetadata chunk 0: %v", err)
	}
	if err := idx.SetChunkMetadata(ctx, "a1b2c3d4e5f6_1", map[string]string{
		"ontology_type": "procedural",
	}); err != nil {
		t.Fatalf("SetChunkMetadata chunk 1: %v", err)
	}
	// Metadata for the unrelated chunk should survive.
	if err := idx.SetChunkMetadata(ctx, "ffee00112233_0", map[string]string{
		"ontology_type": "reference",
	}); err != nil {
		t.Fatalf("SetChunkMetadata keep chunk: %v", err)
	}

	// Delete by document path.
	if err := idx.DeleteByDocPath(ctx, "raw/documents/test.md"); err != nil {
		t.Fatalf("DeleteByDocPath: %v", err)
	}

	// Verify FTS: only the unrelated chunk remains.
	var ftsCount int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_fts WHERE scope = 'chunk'").Scan(&ftsCount); err != nil {
		t.Fatalf("counting FTS chunks: %v", err)
	}
	if ftsCount != 1 {
		t.Fatalf("FTS chunk count = %d, want 1", ftsCount)
	}
	var remainingID string
	if err := db.QueryRow("SELECT path FROM knowledge_fts WHERE scope = 'chunk'").Scan(&remainingID); err != nil {
		t.Fatalf("querying remaining FTS chunk: %v", err)
	}
	if remainingID != "ffee00112233_0" {
		t.Errorf("remaining FTS chunk = %q, want %q", remainingID, "ffee00112233_0")
	}

	// Verify metadata for deleted chunks is gone.
	meta0, err := idx.GetChunkMetadata(ctx, "a1b2c3d4e5f6_0")
	if err != nil {
		t.Fatalf("GetChunkMetadata deleted chunk 0: %v", err)
	}
	if len(meta0) != 0 {
		t.Errorf("expected metadata for deleted chunk 0 to be empty, got %d entries", len(meta0))
	}

	meta1, err := idx.GetChunkMetadata(ctx, "a1b2c3d4e5f6_1")
	if err != nil {
		t.Fatalf("GetChunkMetadata deleted chunk 1: %v", err)
	}
	if len(meta1) != 0 {
		t.Errorf("expected metadata for deleted chunk 1 to be empty, got %d entries", len(meta1))
	}

	// Verify metadata for the kept chunk is intact.
	metaKeep, err := idx.GetChunkMetadata(ctx, "ffee00112233_0")
	if err != nil {
		t.Fatalf("GetChunkMetadata kept chunk: %v", err)
	}
	if metaKeep["ontology_type"] != "reference" {
		t.Errorf("kept chunk metadata ontology_type = %q, want %q", metaKeep["ontology_type"], "reference")
	}
}

func TestDeleteByDocPath_PathWithSpecialChars(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	// Paths with characters that would be problematic in LIKE patterns.
	specialPath := "raw/documents/100%_complete_file.md"
	chunks := []Chunk{
		{ID: "special_chunk_0", Path: specialPath, Ordinal: 0, Content: "special content"},
		{ID: "other_chunk_0", Path: "raw/documents/normal.md", Ordinal: 0, Content: "normal content"},
	}
	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}

	if err := idx.DeleteByDocPath(ctx, specialPath); err != nil {
		t.Fatalf("DeleteByDocPath: %v", err)
	}

	var count int
	if err := db.QueryRow("SELECT count(*) FROM knowledge_fts WHERE scope = 'chunk'").Scan(&count); err != nil {
		t.Fatalf("counting: %v", err)
	}
	if count != 1 {
		t.Fatalf("chunk count = %d, want 1", count)
	}
}

func TestUpsertChunks_EmptySlice(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	if err := idx.UpsertChunks(ctx, nil); err != nil {
		t.Fatalf("UpsertChunks(nil): %v", err)
	}
	if err := idx.UpsertChunks(ctx, []Chunk{}); err != nil {
		t.Fatalf("UpsertChunks(empty): %v", err)
	}
}

func TestUpsertChunks_ValidationErrors(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	cases := []struct {
		name  string
		chunk Chunk
	}{
		{"empty ID", Chunk{ID: "", Path: "some/path.md", Content: "data"}},
		{"empty path", Chunk{ID: "chunk_1", Path: "", Content: "data"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := idx.UpsertChunks(ctx, []Chunk{tc.chunk})
			if err == nil {
				t.Fatal("expected error for invalid chunk")
			}
		})
	}
}

func TestUpsertChunks_CoexistsWithFileLevel(t *testing.T) {
	t.Parallel()
	db, idx := newIndexEmpty(t)
	ctx := context.Background()

	if _, err := db.Exec(
		`INSERT INTO knowledge_fts (path, title, summary, tags, content, scope, project_slug, session_date)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
		"wiki/quantum.md", "Quantum Guide", "A guide to quantum physics",
		"quantum physics", "Introduction to quantum mechanics and wave-particle duality.",
		"wiki", "", "",
	); err != nil {
		t.Fatalf("inserting file-level doc: %v", err)
	}

	chunks := []Chunk{
		{
			ID:      "doc_abc_0",
			Path:    "raw/documents/quantum-deep.md",
			Ordinal: 0,
			Content: "Detailed chunk about quantum field theory.",
			Title:   "QFT",
			Tags:    []string{"quantum", "field_theory"},
		},
	}
	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}

	wikiResults, err := idx.Search("quantum", SearchOpts{Scope: "wiki"})
	if err != nil {
		t.Fatalf("Search wiki: %v", err)
	}
	if len(wikiResults) == 0 {
		t.Fatal("file-level wiki search should still return results")
	}
	if wikiResults[0].Path != "wiki/quantum.md" {
		t.Errorf("wiki result path = %q, want %q", wikiResults[0].Path, "wiki/quantum.md")
	}

	chunkResults, err := idx.Search("quantum field theory", SearchOpts{Scope: "chunk"})
	if err != nil {
		t.Fatalf("Search chunk: %v", err)
	}
	if len(chunkResults) == 0 {
		t.Fatal("chunk search should return results")
	}
	if chunkResults[0].Path != "doc_abc_0" {
		t.Errorf("chunk result path = %q, want %q", chunkResults[0].Path, "doc_abc_0")
	}
}

func TestDeleteChunks_EmptySlice(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	if err := idx.DeleteChunks(ctx, nil); err != nil {
		t.Fatalf("DeleteChunks(nil): %v", err)
	}
	if err := idx.DeleteChunks(ctx, []string{}); err != nil {
		t.Fatalf("DeleteChunks(empty): %v", err)
	}
}

func TestDeleteChunks_EmptyIDError(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	err := idx.DeleteChunks(ctx, []string{"valid_id", ""})
	if err == nil {
		t.Fatal("expected error for empty chunk ID")
	}
}
