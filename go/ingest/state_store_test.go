// SPDX-License-Identifier: Apache-2.0
package ingest_test

import (
	"context"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/ingest"
	"github.com/jeffs-brain/memory/go/store/fs"
)

const testBrainID = "test-brain"

func makeEntry(overrides ...func(*ingest.PipelineStateEntry)) ingest.PipelineStateEntry {
	entry := ingest.PipelineStateEntry{
		DocumentHash: "abc123def456",
		BrainID:      testBrainID,
		Stage:        ingest.StageStored,
		RetryCount:   0,
		CreatedAt:    time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
		UpdatedAt:    time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC),
	}
	for _, fn := range overrides {
		fn(&entry)
	}
	return entry
}

// Factory creates a fresh PipelineStateStore for each subtest.
type Factory func(t *testing.T) ingest.PipelineStateStore

// runContractSuite executes the full contract test suite against any
// PipelineStateStore implementation.
func runContractSuite(t *testing.T, factory Factory) {
	t.Helper()

	cases := []struct {
		name string
		fn   func(t *testing.T, store ingest.PipelineStateStore)
	}{
		{"Get returns nil when no state exists", testGetNonExistent},
		{"Set then Get round-trips", testSetGetRoundTrip},
		{"Set overwrites existing entry", testSetOverwrite},
		{"ListIncomplete excludes completed and failed", testListIncompleteExcludes},
		{"ListIncomplete filters by brainID", testListIncompleteFiltersBrain},
		{"Delete removes entry", testDelete},
		{"Delete non-existent does not error", testDeleteNonExistent},
		{"Preserves lastError field", testPreservesLastError},
		{"Preserves completedAt field", testPreservesCompletedAt},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			store := factory(t)
			tc.fn(t, store)
		})
	}
}

func testGetNonExistent(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	entry, err := store.Get(ctx, "nonexistent-hash")
	if err != nil {
		t.Fatalf("Get returned unexpected error: %v", err)
	}
	if entry != nil {
		t.Fatalf("expected nil entry, got %+v", entry)
	}
}

func testSetGetRoundTrip(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	entry := makeEntry()

	if err := store.Set(ctx, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, err := store.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("Get returned nil after Set")
	}
	if got.DocumentHash != entry.DocumentHash {
		t.Errorf("DocumentHash = %q, want %q", got.DocumentHash, entry.DocumentHash)
	}
	if got.BrainID != entry.BrainID {
		t.Errorf("BrainID = %q, want %q", got.BrainID, entry.BrainID)
	}
	if got.Stage != entry.Stage {
		t.Errorf("Stage = %q, want %q", got.Stage, entry.Stage)
	}
	if got.RetryCount != entry.RetryCount {
		t.Errorf("RetryCount = %d, want %d", got.RetryCount, entry.RetryCount)
	}
	if !got.CreatedAt.Equal(entry.CreatedAt) {
		t.Errorf("CreatedAt = %v, want %v", got.CreatedAt, entry.CreatedAt)
	}
	if !got.UpdatedAt.Equal(entry.UpdatedAt) {
		t.Errorf("UpdatedAt = %v, want %v", got.UpdatedAt, entry.UpdatedAt)
	}
}

func testSetOverwrite(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	entry := makeEntry()

	if err := store.Set(ctx, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}

	updated := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.Stage = ingest.StageChunked
		e.UpdatedAt = time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC)
	})
	if err := store.Set(ctx, updated); err != nil {
		t.Fatalf("Set overwrite: %v", err)
	}

	got, err := store.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("Get returned nil after overwrite")
	}
	if got.Stage != ingest.StageChunked {
		t.Errorf("Stage = %q, want %q", got.Stage, ingest.StageChunked)
	}
	if !got.UpdatedAt.Equal(time.Date(2026, 1, 2, 0, 0, 0, 0, time.UTC)) {
		t.Errorf("UpdatedAt not updated: %v", got.UpdatedAt)
	}
}

func testListIncompleteExcludes(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()

	entries := []ingest.PipelineStateEntry{
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = "hash-received"; e.Stage = ingest.StageReceived }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = "hash-stored"; e.Stage = ingest.StageStored }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = "hash-completed"; e.Stage = ingest.StageCompleted }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = "hash-failed"; e.Stage = ingest.StageFailed }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = "hash-embedded"; e.Stage = ingest.StageEmbedded }),
	}
	for _, entry := range entries {
		if err := store.Set(ctx, entry); err != nil {
			t.Fatalf("Set %s: %v", entry.DocumentHash, err)
		}
	}

	incomplete, err := store.ListIncomplete(ctx, testBrainID)
	if err != nil {
		t.Fatalf("ListIncomplete: %v", err)
	}

	hashes := make(map[string]bool, len(incomplete))
	for _, e := range incomplete {
		hashes[e.DocumentHash] = true
	}

	if !hashes["hash-received"] {
		t.Error("expected hash-received in incomplete list")
	}
	if !hashes["hash-stored"] {
		t.Error("expected hash-stored in incomplete list")
	}
	if !hashes["hash-embedded"] {
		t.Error("expected hash-embedded in incomplete list")
	}
	if hashes["hash-completed"] {
		t.Error("hash-completed should not be in incomplete list")
	}
	if hashes["hash-failed"] {
		t.Error("hash-failed should not be in incomplete list")
	}
	if len(incomplete) != 3 {
		t.Errorf("expected 3 incomplete entries, got %d", len(incomplete))
	}
}

func testListIncompleteFiltersBrain(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()

	ours := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.DocumentHash = "hash-ours"
		e.BrainID = testBrainID
		e.Stage = ingest.StageStored
	})
	theirs := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.DocumentHash = "hash-theirs"
		e.BrainID = "other-brain"
		e.Stage = ingest.StageStored
	})

	if err := store.Set(ctx, ours); err != nil {
		t.Fatalf("Set ours: %v", err)
	}
	if err := store.Set(ctx, theirs); err != nil {
		t.Fatalf("Set theirs: %v", err)
	}

	incomplete, err := store.ListIncomplete(ctx, testBrainID)
	if err != nil {
		t.Fatalf("ListIncomplete: %v", err)
	}
	if len(incomplete) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(incomplete))
	}
	if incomplete[0].DocumentHash != "hash-ours" {
		t.Errorf("expected hash-ours, got %s", incomplete[0].DocumentHash)
	}
}

func testDelete(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	entry := makeEntry()

	if err := store.Set(ctx, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := store.Delete(ctx, entry.DocumentHash); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	got, err := store.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get after Delete: %v", err)
	}
	if got != nil {
		t.Fatalf("expected nil after Delete, got %+v", got)
	}
}

func testDeleteNonExistent(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	if err := store.Delete(ctx, "nonexistent-hash"); err != nil {
		t.Fatalf("Delete non-existent returned error: %v", err)
	}
}

func testPreservesLastError(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	entry := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.Stage = ingest.StageFailed
		e.LastError = "embedder timeout after 30s"
		e.RetryCount = 3
	})

	if err := store.Set(ctx, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, err := store.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("Get returned nil")
	}
	if got.LastError != "embedder timeout after 30s" {
		t.Errorf("LastError = %q, want %q", got.LastError, "embedder timeout after 30s")
	}
	if got.RetryCount != 3 {
		t.Errorf("RetryCount = %d, want 3", got.RetryCount)
	}
}

func testPreservesCompletedAt(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()
	completedAt := time.Date(2026, 1, 5, 12, 0, 0, 0, time.UTC)
	entry := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.Stage = ingest.StageCompleted
		e.CompletedAt = &completedAt
	})

	if err := store.Set(ctx, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, err := store.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("Get returned nil")
	}
	if got.CompletedAt == nil {
		t.Fatal("CompletedAt is nil after round-trip")
	}
	if !got.CompletedAt.Equal(completedAt) {
		t.Errorf("CompletedAt = %v, want %v", got.CompletedAt, completedAt)
	}
}

// TestFilePipelineStateStore runs the full contract suite against the
// file-based implementation using a real filesystem-backed brain.Store.
func TestFilePipelineStateStore(t *testing.T) {
	runContractSuite(t, func(t *testing.T) ingest.PipelineStateStore {
		t.Helper()
		store, err := fs.New(t.TempDir())
		if err != nil {
			t.Fatalf("fs.New: %v", err)
		}
		t.Cleanup(func() { _ = store.Close() })
		return ingest.NewFilePipelineStateStore(store)
	})
}
