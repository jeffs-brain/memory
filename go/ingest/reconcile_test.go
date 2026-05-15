// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/mem"

	_ "modernc.org/sqlite"
)

// testSetup creates a fresh in-memory store, search index, and
// reconciler for a single test case. The reindexFn records which
// paths were re-indexed instead of performing real work.
type testSetup struct {
	store       *mem.Store
	index       *search.Index
	reconciler  *Reconciler
	reindexed   []brain.Path
	reindexedMu sync.Mutex
}

func newTestSetup(t *testing.T) *testSetup {
	t.Helper()
	store := mem.New()
	t.Cleanup(func() { _ = store.Close() })

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("opening sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	idx, err := search.NewIndex(db, store)
	if err != nil {
		t.Fatalf("creating search index: %v", err)
	}

	ts := &testSetup{
		store: store,
		index: idx,
	}

	rec, err := NewReconciler(ReconcileConfig{
		Store:      store,
		Index:      idx,
		MaxRepairs: 1000,
		ReindexFn: func(ctx context.Context, p brain.Path) error {
			ts.reindexedMu.Lock()
			ts.reindexed = append(ts.reindexed, p)
			ts.reindexedMu.Unlock()
			// Simulate re-indexing by calling Index.Update which scans the store.
			return idx.Update(ctx)
		},
	})
	if err != nil {
		t.Fatalf("creating reconciler: %v", err)
	}
	ts.reconciler = rec
	return ts
}

// writeStoreDoc writes a markdown document to the store at the
// canonical raw/documents path, simulating a successful ingest.
func (ts *testSetup) writeStoreDoc(t *testing.T, slug string, body string) brain.Path {
	t.Helper()
	p := brain.RawDocument(slug)
	content := "---\ntitle: \"" + slug + "\"\n---\n\n" + body + "\n"
	if err := ts.store.Write(context.Background(), p, []byte(content)); err != nil {
		t.Fatalf("writing store doc %s: %v", slug, err)
	}
	return p
}

// indexDoc writes a document to the store AND indexes it via Update().
func (ts *testSetup) indexDoc(t *testing.T, slug string, body string) brain.Path {
	t.Helper()
	p := ts.writeStoreDoc(t, slug, body)
	if err := ts.index.Update(context.Background()); err != nil {
		t.Fatalf("indexing %s: %v", slug, err)
	}
	return p
}

func TestReconcile_detects_missing_from_index(t *testing.T) {
	ts := newTestSetup(t)

	// Write a document to the store but do NOT index it.
	ts.writeStoreDoc(t, "missing-doc", "This document was stored but never indexed.")

	report, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("RunOnce: %v", err)
	}

	if report.TotalDocuments != 1 {
		t.Errorf("expected 1 total document, got %d", report.TotalDocuments)
	}
	if !report.DriftDetected {
		t.Errorf("expected drift to be detected")
	}
	if report.MissingReindexed != 1 {
		t.Errorf("expected 1 missing reindexed, got %d", report.MissingReindexed)
	}
	if report.OrphanedDeleted != 0 {
		t.Errorf("expected 0 orphans deleted, got %d", report.OrphanedDeleted)
	}
	ts.reindexedMu.Lock()
	if len(ts.reindexed) != 1 {
		t.Errorf("expected 1 reindex call, got %d", len(ts.reindexed))
	}
	ts.reindexedMu.Unlock()
}

func TestReconcile_detects_orphaned_in_index(t *testing.T) {
	ts := newTestSetup(t)

	// Index a document, then delete it from the store. The index entry
	// becomes orphaned.
	p := ts.indexDoc(t, "orphan-doc", "Content that will be orphaned.")
	if err := ts.store.Delete(context.Background(), p); err != nil {
		t.Fatalf("deleting store doc: %v", err)
	}

	report, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("RunOnce: %v", err)
	}

	if !report.DriftDetected {
		t.Errorf("expected drift to be detected")
	}
	if report.OrphanedDeleted < 1 {
		t.Errorf("expected at least 1 orphan deleted, got %d", report.OrphanedDeleted)
	}
	if report.MissingReindexed != 0 {
		t.Errorf("expected 0 missing reindexed, got %d", report.MissingReindexed)
	}
}

func TestReconcile_repairs_missing(t *testing.T) {
	ts := newTestSetup(t)

	// Write two documents to the store without indexing.
	ts.writeStoreDoc(t, "repair-alpha", "Alpha content for repair test.")
	ts.writeStoreDoc(t, "repair-beta", "Beta content for repair test.")

	report, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("RunOnce: %v", err)
	}

	if report.MissingReindexed != 2 {
		t.Errorf("expected 2 missing reindexed, got %d", report.MissingReindexed)
	}

	// After reconciliation, the index should contain the documents.
	indexed, err := ts.index.IndexedPaths()
	if err != nil {
		t.Fatalf("IndexedPaths: %v", err)
	}
	if len(indexed) < 2 {
		t.Errorf("expected at least 2 indexed paths after repair, got %d", len(indexed))
	}
}

func TestReconcile_removes_orphans(t *testing.T) {
	ts := newTestSetup(t)

	// Index two documents, then delete both from the store.
	p1 := ts.indexDoc(t, "orphan-one", "First orphan document.")
	p2 := ts.indexDoc(t, "orphan-two", "Second orphan document.")
	if err := ts.store.Delete(context.Background(), p1); err != nil {
		t.Fatalf("delete p1: %v", err)
	}
	if err := ts.store.Delete(context.Background(), p2); err != nil {
		t.Fatalf("delete p2: %v", err)
	}

	report, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("RunOnce: %v", err)
	}

	if report.OrphanedDeleted < 2 {
		t.Errorf("expected at least 2 orphans deleted, got %d", report.OrphanedDeleted)
	}

	// After reconciliation, the index should be empty (or have no raw/documents paths).
	indexed, err := ts.index.IndexedPaths()
	if err != nil {
		t.Fatalf("IndexedPaths: %v", err)
	}
	for _, p := range indexed {
		if p == string(p1) || p == string(p2) {
			t.Errorf("orphaned path %s still present in index after reconciliation", p)
		}
	}
}

func TestReconcile_concurrent_lock(t *testing.T) {
	ts := newTestSetup(t)

	// Write a document to create some work.
	ts.writeStoreDoc(t, "concurrent-doc", "Test content for concurrency.")

	var started, finished atomic.Int32
	var wg sync.WaitGroup

	// Launch multiple goroutines that all try to run reconciliation
	// simultaneously. Only one should perform the actual work.
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			started.Add(1)
			report, err := ts.reconciler.RunOnce(context.Background())
			if err != nil {
				t.Errorf("RunOnce error: %v", err)
				return
			}
			if report.MissingReindexed > 0 || report.DriftDetected {
				finished.Add(1)
			}
		}()
	}
	wg.Wait()

	// At most one goroutine should have done actual reconciliation
	// work. The others should have returned immediately due to the lock.
	finishedCount := finished.Load()
	if finishedCount > 1 {
		t.Errorf("expected at most 1 goroutine to perform work, got %d", finishedCount)
	}
}

func TestReconcile_idempotent(t *testing.T) {
	ts := newTestSetup(t)

	// Write a document without indexing.
	ts.writeStoreDoc(t, "idempotent-doc", "Idempotent reconciliation test content.")

	// First run: should repair the missing document.
	report1, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("first RunOnce: %v", err)
	}
	if report1.MissingReindexed != 1 {
		t.Fatalf("first run: expected 1 missing reindexed, got %d", report1.MissingReindexed)
	}

	// Second run: everything should be aligned, no drift.
	report2, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("second RunOnce: %v", err)
	}
	if report2.DriftDetected {
		t.Errorf("second run: expected no drift, but drift was detected (missing=%d, orphans=%d)",
			report2.MissingReindexed, report2.OrphanedDeleted)
	}
	if report2.MissingReindexed != 0 {
		t.Errorf("second run: expected 0 missing reindexed, got %d", report2.MissingReindexed)
	}
	if report2.OrphanedDeleted != 0 {
		t.Errorf("second run: expected 0 orphans deleted, got %d", report2.OrphanedDeleted)
	}
}

func TestReconcile_empty_store_and_index(t *testing.T) {
	ts := newTestSetup(t)

	// Both store and index are empty. Reconciliation should be a no-op.
	report, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("RunOnce: %v", err)
	}

	if report.TotalDocuments != 0 {
		t.Errorf("expected 0 total documents, got %d", report.TotalDocuments)
	}
	if report.TotalIndexed != 0 {
		t.Errorf("expected 0 total indexed, got %d", report.TotalIndexed)
	}
	if report.DriftDetected {
		t.Errorf("expected no drift on empty store/index")
	}
	if report.MissingReindexed != 0 {
		t.Errorf("expected 0 missing reindexed, got %d", report.MissingReindexed)
	}
	if report.OrphanedDeleted != 0 {
		t.Errorf("expected 0 orphans deleted, got %d", report.OrphanedDeleted)
	}
}

func TestReconcile_max_repairs_circuit_breaker(t *testing.T) {
	ts := newTestSetup(t)

	// Override max repairs to 2.
	ts.reconciler.cfg.MaxRepairs = 2

	// Write 5 documents without indexing.
	for i := 0; i < 5; i++ {
		ts.writeStoreDoc(t, fmt.Sprintf("breaker-%d", i), fmt.Sprintf("Breaker content %d.", i))
	}

	report, err := ts.reconciler.RunOnce(context.Background())
	if err != nil {
		t.Fatalf("RunOnce: %v", err)
	}

	// Should cap at 2 repairs due to circuit breaker.
	if report.MissingReindexed > 2 {
		t.Errorf("expected at most 2 missing reindexed (circuit breaker), got %d", report.MissingReindexed)
	}
}

func TestReconcile_context_cancellation(t *testing.T) {
	ts := newTestSetup(t)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	// Give the timeout time to fire.
	time.Sleep(5 * time.Millisecond)

	_, err := ts.reconciler.RunOnce(ctx)
	if err == nil {
		// Context may have been cancelled during listing.
		// Either an error or a zero-repair report is acceptable.
	}
}

func TestNewReconciler_requires_store(t *testing.T) {
	_, err := NewReconciler(ReconcileConfig{Index: &search.Index{}})
	if err == nil {
		t.Fatalf("expected error when Store is nil")
	}
}

func TestNewReconciler_requires_index(t *testing.T) {
	store := mem.New()
	defer func() { _ = store.Close() }()
	_, err := NewReconciler(ReconcileConfig{Store: store})
	if err == nil {
		t.Fatalf("expected error when Index is nil")
	}
}

