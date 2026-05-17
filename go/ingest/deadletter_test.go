// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"database/sql"
	"fmt"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

func setupTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open test db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func setupAdapter(t *testing.T) *SqliteDeadLetterAdapter {
	t.Helper()
	db := setupTestDB(t)
	adapter := NewSqliteDeadLetterAdapter(db)
	if err := adapter.EnsureTable(context.Background()); err != nil {
		t.Fatalf("ensure table: %v", err)
	}
	return adapter
}

func makeTestEntry(overrides ...func(*DeadLetterEntry)) DeadLetterEntry {
	entry := DeadLetterEntry{
		ID:            "dlq-001",
		OriginalJobID: "job-001",
		BrainID:       "brain-1",
		Payload: JobPayload{
			DocumentHash: "abc123def4560000000000000000000000000000000000000000000000000000",
			BrainID:      "brain-1",
			Source:       "file:///test.md",
			ContentType:  "text/markdown",
		},
		FailureReason: "embedding provider returned 500",
		LastError:     "context deadline exceeded",
		RetryCount:    3,
		MovedAt:       time.Date(2026, 5, 1, 12, 0, 0, 0, time.UTC),
	}
	for _, fn := range overrides {
		fn(&entry)
	}
	return entry
}

func TestSqliteDeadLetter_Move(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry()
	result, err := adapter.Move(ctx, entry)
	if err != nil {
		t.Fatalf("move: %v", err)
	}

	if result.ID != entry.ID {
		t.Errorf("expected id %q, got %q", entry.ID, result.ID)
	}
	if result.BrainID != entry.BrainID {
		t.Errorf("expected brainId %q, got %q", entry.BrainID, result.BrainID)
	}
	if result.FailureReason != entry.FailureReason {
		t.Errorf("expected failure reason %q, got %q", entry.FailureReason, result.FailureReason)
	}
}

func TestSqliteDeadLetter_MoveGeneratesIDWhenEmpty(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = ""
	})
	result, err := adapter.Move(ctx, entry)
	if err != nil {
		t.Fatalf("move: %v", err)
	}
	if result.ID == "" {
		t.Error("expected generated ID, got empty")
	}
}

func TestSqliteDeadLetter_Get(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry()
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}

	got, err := adapter.Get(ctx, entry.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}

	if got.ID != entry.ID {
		t.Errorf("id: expected %q, got %q", entry.ID, got.ID)
	}
	if got.OriginalJobID != entry.OriginalJobID {
		t.Errorf("originalJobId: expected %q, got %q", entry.OriginalJobID, got.OriginalJobID)
	}
	if got.BrainID != entry.BrainID {
		t.Errorf("brainId: expected %q, got %q", entry.BrainID, got.BrainID)
	}
	if got.Payload.DocumentHash != entry.Payload.DocumentHash {
		t.Errorf("payload.documentHash: expected %q, got %q", entry.Payload.DocumentHash, got.Payload.DocumentHash)
	}
	if got.FailureReason != entry.FailureReason {
		t.Errorf("failureReason: expected %q, got %q", entry.FailureReason, got.FailureReason)
	}
	if got.LastError != entry.LastError {
		t.Errorf("lastError: expected %q, got %q", entry.LastError, got.LastError)
	}
	if got.RetryCount != entry.RetryCount {
		t.Errorf("retryCount: expected %d, got %d", entry.RetryCount, got.RetryCount)
	}
}

func TestSqliteDeadLetter_GetNotFound(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	_, err := adapter.Get(ctx, "nonexistent")
	if err != ErrDeadLetterNotFound {
		t.Errorf("expected ErrDeadLetterNotFound, got %v", err)
	}
}

func TestSqliteDeadLetter_ListUnresolvedByDefault(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	// Insert two entries: one unresolved, one resolved.
	unresolved := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-unresolved"
	})
	if _, err := adapter.Move(ctx, unresolved); err != nil {
		t.Fatalf("move unresolved: %v", err)
	}

	resolved := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-resolved"
		now := time.Now().UTC()
		e.ResolvedAt = &now
		e.ResolvedBy = "operator"
	})
	if _, err := adapter.Move(ctx, resolved); err != nil {
		t.Fatalf("move resolved: %v", err)
	}

	result, err := adapter.List(ctx, DeadLetterListOptions{})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if result.Total != 1 {
		t.Errorf("expected total 1 (unresolved only), got %d", result.Total)
	}
	if len(result.Entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(result.Entries))
	}
	if result.Entries[0].ID != "dlq-unresolved" {
		t.Errorf("expected unresolved entry, got %q", result.Entries[0].ID)
	}
}

func TestSqliteDeadLetter_ListIncludeResolved(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	unresolved := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-a"
	})
	if _, err := adapter.Move(ctx, unresolved); err != nil {
		t.Fatalf("move: %v", err)
	}

	resolved := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-b"
		now := time.Now().UTC()
		e.ResolvedAt = &now
		e.ResolvedBy = "operator"
	})
	if _, err := adapter.Move(ctx, resolved); err != nil {
		t.Fatalf("move: %v", err)
	}

	result, err := adapter.List(ctx, DeadLetterListOptions{IncludeResolved: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if result.Total != 2 {
		t.Errorf("expected total 2, got %d", result.Total)
	}
}

func TestSqliteDeadLetter_ListFilterByBrain(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry1 := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-brain1"
		e.BrainID = "brain-alpha"
	})
	if _, err := adapter.Move(ctx, entry1); err != nil {
		t.Fatalf("move: %v", err)
	}

	entry2 := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-brain2"
		e.BrainID = "brain-beta"
	})
	if _, err := adapter.Move(ctx, entry2); err != nil {
		t.Fatalf("move: %v", err)
	}

	result, err := adapter.List(ctx, DeadLetterListOptions{BrainID: "brain-alpha"})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if result.Total != 1 {
		t.Errorf("expected total 1, got %d", result.Total)
	}
	if len(result.Entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(result.Entries))
	}
	if result.Entries[0].BrainID != "brain-alpha" {
		t.Errorf("expected brain-alpha, got %q", result.Entries[0].BrainID)
	}
}

func TestSqliteDeadLetter_ListPagination(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	for i := 0; i < 5; i++ {
		entry := makeTestEntry(func(e *DeadLetterEntry) {
			e.ID = fmt.Sprintf("dlq-page-%d", i)
			e.MovedAt = time.Date(2026, 5, 1, 12, 0, i, 0, time.UTC)
		})
		if _, err := adapter.Move(ctx, entry); err != nil {
			t.Fatalf("move %d: %v", i, err)
		}
	}

	// Page 1: limit 2, offset 0.
	result, err := adapter.List(ctx, DeadLetterListOptions{Limit: 2, Offset: 0})
	if err != nil {
		t.Fatalf("list page 1: %v", err)
	}
	if result.Total != 5 {
		t.Errorf("expected total 5, got %d", result.Total)
	}
	if len(result.Entries) != 2 {
		t.Errorf("expected 2 entries on page 1, got %d", len(result.Entries))
	}

	// Page 2: limit 2, offset 2.
	result2, err := adapter.List(ctx, DeadLetterListOptions{Limit: 2, Offset: 2})
	if err != nil {
		t.Fatalf("list page 2: %v", err)
	}
	if len(result2.Entries) != 2 {
		t.Errorf("expected 2 entries on page 2, got %d", len(result2.Entries))
	}

	// Page 3: limit 2, offset 4.
	result3, err := adapter.List(ctx, DeadLetterListOptions{Limit: 2, Offset: 4})
	if err != nil {
		t.Fatalf("list page 3: %v", err)
	}
	if len(result3.Entries) != 1 {
		t.Errorf("expected 1 entry on page 3, got %d", len(result3.Entries))
	}
}

func TestSqliteDeadLetter_Retry(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry()
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}

	resolved, err := adapter.Retry(ctx, entry.ID, "admin@example.com")
	if err != nil {
		t.Fatalf("retry: %v", err)
	}
	if resolved.ResolvedAt == nil {
		t.Fatal("expected resolvedAt to be set")
	}
	if resolved.ResolvedBy != "admin@example.com" {
		t.Errorf("expected resolvedBy admin@example.com, got %q", resolved.ResolvedBy)
	}

	// Verify stored state.
	got, err := adapter.Get(ctx, entry.ID)
	if err != nil {
		t.Fatalf("get after retry: %v", err)
	}
	if got.ResolvedAt == nil {
		t.Error("expected resolvedAt to be persisted")
	}
}

func TestSqliteDeadLetter_RetryNonExistent(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	_, err := adapter.Retry(ctx, "nonexistent", "operator")
	if err != ErrDeadLetterNotFound {
		t.Errorf("expected ErrDeadLetterNotFound, got %v", err)
	}
}

func TestSqliteDeadLetter_DoubleRetryFails(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry()
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}
	if _, err := adapter.Retry(ctx, entry.ID, "operator-1"); err != nil {
		t.Fatalf("first retry: %v", err)
	}

	_, err := adapter.Retry(ctx, entry.ID, "operator-2")
	if err != ErrDeadLetterAlreadyResolved {
		t.Errorf("expected ErrDeadLetterAlreadyResolved, got %v", err)
	}
}

func TestSqliteDeadLetter_PurgeByID(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry()
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}

	count, err := adapter.Purge(ctx, PurgeOptions{Kind: PurgeByID, ID: entry.ID})
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 purged, got %d", count)
	}

	_, err = adapter.Get(ctx, entry.ID)
	if err != ErrDeadLetterNotFound {
		t.Errorf("expected not found after purge, got %v", err)
	}
}

func TestSqliteDeadLetter_PurgeByBrain(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		entry := makeTestEntry(func(e *DeadLetterEntry) {
			e.ID = fmt.Sprintf("dlq-purge-brain-%d", i)
			e.BrainID = "brain-to-purge"
		})
		if _, err := adapter.Move(ctx, entry); err != nil {
			t.Fatalf("move %d: %v", i, err)
		}
	}

	other := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-other-brain"
		e.BrainID = "brain-keep"
	})
	if _, err := adapter.Move(ctx, other); err != nil {
		t.Fatalf("move other: %v", err)
	}

	count, err := adapter.Purge(ctx, PurgeOptions{Kind: PurgeByBrain, BrainID: "brain-to-purge"})
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3 purged, got %d", count)
	}

	// Verify the other brain's entry survives.
	result, err := adapter.List(ctx, DeadLetterListOptions{IncludeResolved: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if result.Total != 1 {
		t.Errorf("expected 1 remaining, got %d", result.Total)
	}
}

func TestSqliteDeadLetter_PurgeOlderThan(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	old := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-old"
		e.MovedAt = time.Now().UTC().AddDate(0, 0, -60)
	})
	if _, err := adapter.Move(ctx, old); err != nil {
		t.Fatalf("move old: %v", err)
	}

	recent := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-recent"
		e.MovedAt = time.Now().UTC()
	})
	if _, err := adapter.Move(ctx, recent); err != nil {
		t.Fatalf("move recent: %v", err)
	}

	count, err := adapter.Purge(ctx, PurgeOptions{Kind: PurgeOlderThan, Days: 30})
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 purged, got %d", count)
	}

	result, err := adapter.List(ctx, DeadLetterListOptions{IncludeResolved: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if result.Total != 1 {
		t.Errorf("expected 1 remaining, got %d", result.Total)
	}
}

func TestSqliteDeadLetter_PurgeAllResolved(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	unresolved := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-unres"
	})
	if _, err := adapter.Move(ctx, unresolved); err != nil {
		t.Fatalf("move: %v", err)
	}

	resolved := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-res"
		now := time.Now().UTC()
		e.ResolvedAt = &now
		e.ResolvedBy = "operator"
	})
	if _, err := adapter.Move(ctx, resolved); err != nil {
		t.Fatalf("move: %v", err)
	}

	count, err := adapter.Purge(ctx, PurgeOptions{Kind: PurgeAllResolved})
	if err != nil {
		t.Fatalf("purge: %v", err)
	}
	if count != 1 {
		t.Errorf("expected 1 purged, got %d", count)
	}

	result, err := adapter.List(ctx, DeadLetterListOptions{IncludeResolved: true})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if result.Total != 1 {
		t.Errorf("expected 1 remaining, got %d", result.Total)
	}
	if result.Entries[0].ID != "dlq-unres" {
		t.Errorf("expected unresolved entry to survive, got %q", result.Entries[0].ID)
	}
}

func TestSqliteDeadLetter_Count(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	// Empty state.
	count, err := adapter.Count(ctx, "")
	if err != nil {
		t.Fatalf("count empty: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0, got %d", count)
	}

	// Add entries for two brains.
	for i := 0; i < 3; i++ {
		entry := makeTestEntry(func(e *DeadLetterEntry) {
			e.ID = fmt.Sprintf("dlq-count-a-%d", i)
			e.BrainID = "brain-a"
		})
		if _, err := adapter.Move(ctx, entry); err != nil {
			t.Fatalf("move: %v", err)
		}
	}

	entry := makeTestEntry(func(e *DeadLetterEntry) {
		e.ID = "dlq-count-b"
		e.BrainID = "brain-b"
	})
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}

	// Resolve one entry.
	if _, err := adapter.Retry(ctx, "dlq-count-a-0", "operator"); err != nil {
		t.Fatalf("retry: %v", err)
	}

	// Global count (unresolved only).
	count, err = adapter.Count(ctx, "")
	if err != nil {
		t.Fatalf("count global: %v", err)
	}
	if count != 3 {
		t.Errorf("expected 3 unresolved globally, got %d", count)
	}

	// Per-brain count.
	countA, err := adapter.Count(ctx, "brain-a")
	if err != nil {
		t.Fatalf("count brain-a: %v", err)
	}
	if countA != 2 {
		t.Errorf("expected 2 unresolved for brain-a, got %d", countA)
	}

	countB, err := adapter.Count(ctx, "brain-b")
	if err != nil {
		t.Fatalf("count brain-b: %v", err)
	}
	if countB != 1 {
		t.Errorf("expected 1 unresolved for brain-b, got %d", countB)
	}
}

func TestSqliteDeadLetter_MetadataRoundTrip(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry(func(e *DeadLetterEntry) {
		e.Metadata = map[string]string{
			"source":    "webhook",
			"requestId": "req-12345",
		}
		e.GroupID = "batch-001"
	})
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}

	got, err := adapter.Get(ctx, entry.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got.Metadata) != 2 {
		t.Fatalf("expected 2 metadata entries, got %d", len(got.Metadata))
	}
	if got.Metadata["source"] != "webhook" {
		t.Errorf("expected metadata source 'webhook', got %q", got.Metadata["source"])
	}
	if got.Metadata["requestId"] != "req-12345" {
		t.Errorf("expected metadata requestId 'req-12345', got %q", got.Metadata["requestId"])
	}
	if got.GroupID != "batch-001" {
		t.Errorf("expected groupId 'batch-001', got %q", got.GroupID)
	}
}

func TestSqliteDeadLetter_PayloadRoundTrip(t *testing.T) {
	t.Parallel()
	adapter := setupAdapter(t)
	ctx := context.Background()

	entry := makeTestEntry(func(e *DeadLetterEntry) {
		e.Payload = JobPayload{
			DocumentHash: "abc123def4560000000000000000000000000000000000000000000000000000",
			BrainID:      "brain-1",
			Source:       "https://example.com/doc.pdf",
			ContentType:  "application/pdf",
		}
	})
	if _, err := adapter.Move(ctx, entry); err != nil {
		t.Fatalf("move: %v", err)
	}

	got, err := adapter.Get(ctx, entry.ID)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if got.Payload.DocumentHash != entry.Payload.DocumentHash {
		t.Errorf("payload.documentHash: expected %q, got %q", entry.Payload.DocumentHash, got.Payload.DocumentHash)
	}
	if got.Payload.Source != entry.Payload.Source {
		t.Errorf("payload.source: expected %q, got %q", entry.Payload.Source, got.Payload.Source)
	}
	if got.Payload.ContentType != entry.Payload.ContentType {
		t.Errorf("payload.contentType: expected %q, got %q", entry.Payload.ContentType, got.Payload.ContentType)
	}
}

