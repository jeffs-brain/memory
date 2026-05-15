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

// defaultHash is a valid 64-character lowercase hex string (SHA-256).
const defaultHash = "abc123def4560000000000000000000000000000000000000000000000000000"

func makeEntry(overrides ...func(*ingest.PipelineStateEntry)) ingest.PipelineStateEntry {
	entry := ingest.PipelineStateEntry{
		DocumentHash: defaultHash,
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
	entry, err := store.Get(ctx, "0000000000000000000000000000000000000000000000000000000000000000")
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

	hashReceived := "1111111111111111111111111111111111111111111111111111111111111111"
	hashStored := "2222222222222222222222222222222222222222222222222222222222222222"
	hashCompleted := "3333333333333333333333333333333333333333333333333333333333333333"
	hashFailed := "4444444444444444444444444444444444444444444444444444444444444444"
	hashEmbedded := "5555555555555555555555555555555555555555555555555555555555555555"

	entries := []ingest.PipelineStateEntry{
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = hashReceived; e.Stage = ingest.StageReceived }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = hashStored; e.Stage = ingest.StageStored }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = hashCompleted; e.Stage = ingest.StageCompleted }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = hashFailed; e.Stage = ingest.StageFailed }),
		makeEntry(func(e *ingest.PipelineStateEntry) { e.DocumentHash = hashEmbedded; e.Stage = ingest.StageEmbedded }),
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

	if !hashes[hashReceived] {
		t.Error("expected hashReceived in incomplete list")
	}
	if !hashes[hashStored] {
		t.Error("expected hashStored in incomplete list")
	}
	if !hashes[hashEmbedded] {
		t.Error("expected hashEmbedded in incomplete list")
	}
	if hashes[hashCompleted] {
		t.Error("hashCompleted should not be in incomplete list")
	}
	if hashes[hashFailed] {
		t.Error("hashFailed should not be in incomplete list")
	}
	if len(incomplete) != 3 {
		t.Errorf("expected 3 incomplete entries, got %d", len(incomplete))
	}
}

func testListIncompleteFiltersBrain(t *testing.T, store ingest.PipelineStateStore) {
	t.Helper()
	ctx := context.Background()

	hashOurs := "aaaa000000000000000000000000000000000000000000000000000000000000"
	hashTheirs := "bbbb000000000000000000000000000000000000000000000000000000000000"

	ours := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.DocumentHash = hashOurs
		e.BrainID = testBrainID
		e.Stage = ingest.StageStored
	})
	theirs := makeEntry(func(e *ingest.PipelineStateEntry) {
		e.DocumentHash = hashTheirs
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
	if incomplete[0].DocumentHash != hashOurs {
		t.Errorf("expected %s, got %s", hashOurs, incomplete[0].DocumentHash)
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
	if err := store.Delete(ctx, "0000000000000000000000000000000000000000000000000000000000000000"); err != nil {
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

// TestFilePipelineStateStore_CustomPrefix verifies that a custom path prefix
// is honoured by the state store.
func TestFilePipelineStateStore_CustomPrefix(t *testing.T) {
	store, err := fs.New(t.TempDir())
	if err != nil {
		t.Fatalf("fs.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	customPrefix := "raw/.custom-state-dir"
	ss := ingest.NewFilePipelineStateStoreWithConfig(ingest.FilePipelineStateStoreConfig{
		Store:  store,
		Prefix: customPrefix,
	})

	ctx := context.Background()
	entry := makeEntry()

	if err := ss.Set(ctx, entry); err != nil {
		t.Fatalf("Set: %v", err)
	}

	got, err := ss.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got == nil {
		t.Fatal("Get returned nil after Set with custom prefix")
	}
	if got.DocumentHash != entry.DocumentHash {
		t.Errorf("DocumentHash = %q, want %q", got.DocumentHash, entry.DocumentHash)
	}

	// Verify the file is stored under the custom prefix, not the default.
	defaultStore := ingest.NewFilePipelineStateStore(store)
	fromDefault, err := defaultStore.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get from default store: %v", err)
	}
	if fromDefault != nil {
		t.Error("entry should not be found under the default prefix")
	}
}

// TestFilePipelineStateStore_DefaultPrefix verifies that omitting the
// prefix in config uses the default "raw/.pipeline-state".
func TestFilePipelineStateStore_DefaultPrefix(t *testing.T) {
	store, err := fs.New(t.TempDir())
	if err != nil {
		t.Fatalf("fs.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	ssConfig := ingest.NewFilePipelineStateStoreWithConfig(ingest.FilePipelineStateStoreConfig{
		Store: store,
	})
	ssDefault := ingest.NewFilePipelineStateStore(store)

	ctx := context.Background()
	entry := makeEntry()

	if err := ssConfig.Set(ctx, entry); err != nil {
		t.Fatalf("Set via config: %v", err)
	}

	got, err := ssDefault.Get(ctx, entry.DocumentHash)
	if err != nil {
		t.Fatalf("Get via default: %v", err)
	}
	if got == nil {
		t.Fatal("default prefix should find entry written with empty config prefix")
	}
}

// TestFilePipelineStateStore_RejectsInvalidHash verifies that path-traversal
// or otherwise invalid document hashes are rejected.
func TestFilePipelineStateStore_RejectsInvalidHash(t *testing.T) {
	store, err := fs.New(t.TempDir())
	if err != nil {
		t.Fatalf("fs.New: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	ss := ingest.NewFilePipelineStateStore(store)

	ctx := context.Background()
	invalidHashes := []string{
		"../../../etc/passwd",
		"short",
		"ABCD000000000000000000000000000000000000000000000000000000000000", // uppercase
		"abc123def456",    // too short
		"abc/123/def/456", // path separators
	}

	for _, hash := range invalidHashes {
		_, err := ss.Get(ctx, hash)
		if err == nil {
			t.Errorf("Get(%q) should have returned error for invalid hash", hash)
		}

		err = ss.Delete(ctx, hash)
		if err == nil {
			t.Errorf("Delete(%q) should have returned error for invalid hash", hash)
		}

		err = ss.Set(ctx, ingest.PipelineStateEntry{
			DocumentHash: hash,
			BrainID:      testBrainID,
			Stage:        ingest.StageStored,
		})
		if err == nil {
			t.Errorf("Set with hash %q should have returned error for invalid hash", hash)
		}
	}
}

// TestPostgresPipelineStateStore_RejectsNilDB verifies that the constructor
// rejects a nil DB connection.
func TestPostgresPipelineStateStore_RejectsNilDB(t *testing.T) {
	_, err := ingest.NewPostgresPipelineStateStore(ingest.PostgresStateStoreConfig{
		DB: nil,
	})
	if err == nil {
		t.Fatal("expected error for nil DB, got nil")
	}
}
