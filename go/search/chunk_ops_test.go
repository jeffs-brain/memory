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
