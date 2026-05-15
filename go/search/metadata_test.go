// SPDX-License-Identifier: Apache-2.0
package search

import (
	"context"
	"testing"
)

func TestSetGetChunkMetadata_RoundTrip(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	meta := map[string]string{
		"ontology_type": "factual",
		"confidence":    "0.95",
		"source_model":  "gpt-4o",
	}
	if err := idx.SetChunkMetadata(ctx, "chunk_001", meta); err != nil {
		t.Fatalf("SetChunkMetadata: %v", err)
	}

	got, err := idx.GetChunkMetadata(ctx, "chunk_001")
	if err != nil {
		t.Fatalf("GetChunkMetadata: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("metadata entries = %d, want 3", len(got))
	}
	if got["ontology_type"] != "factual" {
		t.Errorf("ontology_type = %q, want %q", got["ontology_type"], "factual")
	}
	if got["confidence"] != "0.95" {
		t.Errorf("confidence = %q, want %q", got["confidence"], "0.95")
	}
	if got["source_model"] != "gpt-4o" {
		t.Errorf("source_model = %q, want %q", got["source_model"], "gpt-4o")
	}
}

func TestSetChunkMetadata_OverwritesExisting(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	if err := idx.SetChunkMetadata(ctx, "chunk_002", map[string]string{
		"confidence": "0.5",
	}); err != nil {
		t.Fatalf("SetChunkMetadata first: %v", err)
	}

	if err := idx.SetChunkMetadata(ctx, "chunk_002", map[string]string{
		"confidence": "0.99",
	}); err != nil {
		t.Fatalf("SetChunkMetadata second: %v", err)
	}

	got, err := idx.GetChunkMetadata(ctx, "chunk_002")
	if err != nil {
		t.Fatalf("GetChunkMetadata: %v", err)
	}
	if got["confidence"] != "0.99" {
		t.Errorf("confidence = %q, want %q (overwritten)", got["confidence"], "0.99")
	}
}

func TestGetChunkMetadata_NonExistentChunk(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	got, err := idx.GetChunkMetadata(ctx, "nonexistent_chunk")
	if err != nil {
		t.Fatalf("GetChunkMetadata: %v", err)
	}
	if got == nil {
		t.Fatal("expected non-nil empty map")
	}
	if len(got) != 0 {
		t.Errorf("metadata for non-existent chunk should be empty, got %d entries", len(got))
	}
}

func TestQueryByMetadata_ReturnsMatchingChunkIDs(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	if err := idx.SetChunkMetadata(ctx, "chunk_a", map[string]string{
		"ontology_type": "factual",
		"topic":         "physics",
	}); err != nil {
		t.Fatalf("SetChunkMetadata a: %v", err)
	}
	if err := idx.SetChunkMetadata(ctx, "chunk_b", map[string]string{
		"ontology_type": "opinion",
		"topic":         "physics",
	}); err != nil {
		t.Fatalf("SetChunkMetadata b: %v", err)
	}
	if err := idx.SetChunkMetadata(ctx, "chunk_c", map[string]string{
		"ontology_type": "factual",
		"topic":         "chemistry",
	}); err != nil {
		t.Fatalf("SetChunkMetadata c: %v", err)
	}

	ids, err := idx.QueryByMetadata(ctx, "ontology_type", "factual", 10)
	if err != nil {
		t.Fatalf("QueryByMetadata: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 factual chunks, got %d", len(ids))
	}
	if ids[0] != "chunk_a" || ids[1] != "chunk_c" {
		t.Errorf("expected [chunk_a, chunk_c], got %v", ids)
	}

	topicIDs, err := idx.QueryByMetadata(ctx, "topic", "physics", 10)
	if err != nil {
		t.Fatalf("QueryByMetadata topic: %v", err)
	}
	if len(topicIDs) != 2 {
		t.Fatalf("expected 2 physics chunks, got %d", len(topicIDs))
	}
}

func TestQueryByMetadata_RespectsLimit(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	for _, id := range []string{"chunk_01", "chunk_02", "chunk_03", "chunk_04", "chunk_05"} {
		if err := idx.SetChunkMetadata(ctx, id, map[string]string{"type": "fact"}); err != nil {
			t.Fatalf("SetChunkMetadata %s: %v", id, err)
		}
	}

	ids, err := idx.QueryByMetadata(ctx, "type", "fact", 3)
	if err != nil {
		t.Fatalf("QueryByMetadata: %v", err)
	}
	if len(ids) != 3 {
		t.Fatalf("expected 3 results with limit, got %d", len(ids))
	}
}

func TestQueryByMetadata_NoMatches(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	ids, err := idx.QueryByMetadata(ctx, "nonexistent_key", "value", 10)
	if err != nil {
		t.Fatalf("QueryByMetadata: %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("expected 0 results for non-existent key, got %d", len(ids))
	}
}

func TestSetChunkMetadata_ValidationErrors(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	cases := []struct {
		name    string
		chunkID string
		meta    map[string]string
	}{
		{"empty chunk ID", "", map[string]string{"key": "val"}},
		{"empty key", "chunk_1", map[string]string{"": "val"}},
		{"empty value", "chunk_1", map[string]string{"key": ""}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := idx.SetChunkMetadata(ctx, tc.chunkID, tc.meta)
			if err == nil {
				t.Fatal("expected error for invalid input")
			}
		})
	}
}

func TestSetChunkMetadata_EmptyMap(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	if err := idx.SetChunkMetadata(ctx, "chunk_1", nil); err != nil {
		t.Fatalf("SetChunkMetadata(nil): %v", err)
	}
	if err := idx.SetChunkMetadata(ctx, "chunk_1", map[string]string{}); err != nil {
		t.Fatalf("SetChunkMetadata(empty): %v", err)
	}
}

func TestQueryByMetadata_EmptyKeyError(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	_, err := idx.QueryByMetadata(ctx, "", "value", 10)
	if err == nil {
		t.Fatal("expected error for empty key")
	}
}

func TestDeleteChunks_CleansUpMetadata(t *testing.T) {
	t.Parallel()
	_, idx := newIndexEmpty(t)
	ctx := context.Background()

	chunks := []Chunk{
		{ID: "chunk_with_meta", Path: "raw/documents/doc.md", Ordinal: 0, Content: "some content"},
	}
	if err := idx.UpsertChunks(ctx, chunks); err != nil {
		t.Fatalf("UpsertChunks: %v", err)
	}
	if err := idx.SetChunkMetadata(ctx, "chunk_with_meta", map[string]string{
		"type":       "factual",
		"confidence": "0.9",
	}); err != nil {
		t.Fatalf("SetChunkMetadata: %v", err)
	}

	if err := idx.DeleteChunks(ctx, []string{"chunk_with_meta"}); err != nil {
		t.Fatalf("DeleteChunks: %v", err)
	}

	got, err := idx.GetChunkMetadata(ctx, "chunk_with_meta")
	if err != nil {
		t.Fatalf("GetChunkMetadata after delete: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("expected metadata to be cleaned up, got %d entries", len(got))
	}
}
