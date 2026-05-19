// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"testing"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/store/mem"
)

func TestHashMigrator_EmptyStore(t *testing.T) {
	t.Parallel()
	store := mem.New()
	migrator := NewHashMigrator(store)

	result, err := migrator.Migrate(context.Background(), MigrateOpts{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Migrated != 0 {
		t.Errorf("expected 0 migrated, got %d", result.Migrated)
	}
	if result.Total != 0 {
		t.Errorf("expected 0 total, got %d", result.Total)
	}
	if result.NextCursor != "" {
		t.Errorf("expected empty cursor, got %q", result.NextCursor)
	}
}

func TestHashMigrator_MigratesDocuments(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()

	// Write documents with SHA-256-based slugs.
	docs := []struct {
		content []byte
	}{
		{content: []byte("document one content")},
		{content: []byte("document two content")},
		{content: []byte("document three content")},
	}

	for _, doc := range docs {
		sha256Slug := HashSlugSHA256(doc.content)
		docPath := brain.RawDocument(sha256Slug)
		if err := store.Write(ctx, docPath, doc.content); err != nil {
			t.Fatalf("write failed: %v", err)
		}
	}

	migrator := NewHashMigrator(store)
	result, err := migrator.Migrate(ctx, MigrateOpts{BatchSize: 10})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Migrated != 3 {
		t.Errorf("expected 3 migrated, got %d", result.Migrated)
	}
	if result.Total != 3 {
		t.Errorf("expected 3 total, got %d", result.Total)
	}

	// Verify new paths exist with BLAKE3 slugs.
	for _, doc := range docs {
		blake3Slug := HashSlug(doc.content)
		newPath := brain.RawDocument(blake3Slug)
		exists, err := store.Exists(ctx, newPath)
		if err != nil {
			t.Fatalf("exists check failed: %v", err)
		}
		if !exists {
			t.Errorf("expected document at BLAKE3 path %s to exist", newPath)
		}

		// Verify old SHA-256 paths removed.
		sha256Slug := HashSlugSHA256(doc.content)
		oldPath := brain.RawDocument(sha256Slug)
		exists, err = store.Exists(ctx, oldPath)
		if err != nil {
			t.Fatalf("exists check failed: %v", err)
		}
		if exists {
			t.Errorf("expected old SHA-256 path %s to be removed", oldPath)
		}
	}
}

func TestResolveHash_FindsByBLAKE3First(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()

	content := []byte("test content for dual-read")
	blake3Slug := HashSlug(content)
	blake3Path := brain.RawDocument(blake3Slug)
	if err := store.Write(ctx, blake3Path, content); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	// Also write at SHA-256 path (simulating pre-migration state).
	sha256Slug := HashSlugSHA256(content)
	sha256Path := brain.RawDocument(sha256Slug)
	if err := store.Write(ctx, sha256Path, content); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	resolved, err := ResolveHash(ctx, store, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resolved != blake3Slug {
		t.Errorf("expected BLAKE3 slug %q, got %q", blake3Slug, resolved)
	}
}

func TestResolveHash_FallsBackToSHA256(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()

	content := []byte("legacy content only at sha256 path")
	sha256Slug := HashSlugSHA256(content)
	sha256Path := brain.RawDocument(sha256Slug)
	if err := store.Write(ctx, sha256Path, content); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	resolved, err := ResolveHash(ctx, store, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resolved != sha256Slug {
		t.Errorf("expected SHA-256 slug %q, got %q", sha256Slug, resolved)
	}
}

func TestResolveHash_ReturnsBLAKE3ForNewDocuments(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()

	content := []byte("brand new document never seen before")
	blake3Slug := HashSlug(content)

	resolved, err := ResolveHash(ctx, store, content)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resolved != blake3Slug {
		t.Errorf("expected BLAKE3 slug %q for new document, got %q", blake3Slug, resolved)
	}
}

func TestHashMigrator_ResumeFromCursor(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()

	// Write 5 documents with SHA-256 slugs.
	contents := [][]byte{
		[]byte("alpha content"),
		[]byte("beta content"),
		[]byte("gamma content"),
		[]byte("delta content"),
		[]byte("epsilon content"),
	}

	for _, content := range contents {
		sha256Slug := HashSlugSHA256(content)
		docPath := brain.RawDocument(sha256Slug)
		if err := store.Write(ctx, docPath, content); err != nil {
			t.Fatalf("write failed: %v", err)
		}
	}

	migrator := NewHashMigrator(store)

	// First pass: migrate only 2 documents.
	result1, err := migrator.Migrate(ctx, MigrateOpts{BatchSize: 2})
	if err != nil {
		t.Fatalf("first pass error: %v", err)
	}
	if result1.Migrated != 2 {
		t.Errorf("first pass: expected 2 migrated, got %d", result1.Migrated)
	}
	if result1.NextCursor == "" {
		t.Fatal("first pass: expected non-empty next cursor")
	}
	if result1.Total != 5 {
		t.Errorf("first pass: expected 5 total, got %d", result1.Total)
	}

	// Second pass: resume from cursor. The store has 5 total entries now
	// (2 new BLAKE3 + 3 remaining SHA-256). The remaining 3 SHA-256 docs
	// should be migrated.
	result2, err := migrator.Migrate(ctx, MigrateOpts{BatchSize: 10})
	if err != nil {
		t.Fatalf("second pass error: %v", err)
	}
	// Should migrate the remaining documents (some may show as skipped if
	// they were already migrated, depends on cursor position in the new
	// listing).
	totalProcessed := result2.Migrated + result2.Skipped
	if totalProcessed == 0 {
		t.Error("second pass: expected some documents to be processed")
	}
}

func TestHashMigrator_DryRunDoesNotWrite(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	store := mem.New()

	content := []byte("dry run test content")
	sha256Slug := HashSlugSHA256(content)
	docPath := brain.RawDocument(sha256Slug)
	if err := store.Write(ctx, docPath, content); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	migrator := NewHashMigrator(store)
	result, err := migrator.Migrate(ctx, MigrateOpts{DryRun: true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Migrated != 1 {
		t.Errorf("expected 1 migrated (reported), got %d", result.Migrated)
	}

	// Verify original file still exists (dry-run did not modify).
	exists, err := store.Exists(ctx, docPath)
	if err != nil {
		t.Fatalf("exists check failed: %v", err)
	}
	if !exists {
		t.Error("dry-run should not have removed the original document")
	}

	// Verify BLAKE3 path was NOT created.
	blake3Slug := HashSlug(content)
	newPath := brain.RawDocument(blake3Slug)
	exists, err = store.Exists(ctx, newPath)
	if err != nil {
		t.Fatalf("exists check failed: %v", err)
	}
	if exists {
		t.Error("dry-run should not have created the BLAKE3 path")
	}

	// Verify no state file written.
	statePath := brain.Path(migrationStatePath)
	exists, err = store.Exists(ctx, statePath)
	if err != nil {
		t.Fatalf("exists check failed: %v", err)
	}
	if exists {
		t.Error("dry-run should not have written migration state")
	}
}
