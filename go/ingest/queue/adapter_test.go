// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"strings"
	"testing"
	"time"
)

// Adapter-level tests exercise the full Enqueue/Claim/Heartbeat/
// Complete/Fail/RecoverStale/CountByStatus workflow using the mock
// database driver defined in mock_test.go.

func TestAdapterEnqueue(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	job, err := q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "file", Path: "/data/doc.md"},
	})
	if err != nil {
		t.Fatalf("Enqueue: %v", err)
	}
	if job.Status != StatusPending {
		t.Errorf("got status %s, want pending", job.Status)
	}
	if job.BrainID != "brain-1" {
		t.Errorf("got brainID %s, want brain-1", job.BrainID)
	}
	if job.MaxRetries != defaultMaxRetries {
		t.Errorf("got maxRetries %d, want %d", job.MaxRetries, defaultMaxRetries)
	}
}

func TestAdapterEnqueueIdempotency(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	first, err := q.Enqueue(ctx, EnqueueInput{
		BrainID:        "brain-1",
		Payload:        JobPayload{Kind: "raw", Content: "hello"},
		IdempotencyKey: "key-1",
	})
	if err != nil {
		t.Fatalf("first enqueue: %v", err)
	}
	second, err := q.Enqueue(ctx, EnqueueInput{
		BrainID:        "brain-1",
		Payload:        JobPayload{Kind: "raw", Content: "hello"},
		IdempotencyKey: "key-1",
	})
	if err != nil {
		t.Fatalf("second enqueue: %v", err)
	}
	if first.ID != second.ID {
		t.Errorf("idempotent enqueue returned different IDs: %s vs %s", first.ID, second.ID)
	}
}

func TestAdapterClaim(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	_, err = q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "file", Path: "/a.md"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "file", Path: "/b.md"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	jobs, err := q.Claim(ctx, ClaimOptions{BatchSize: 5, WorkerID: "worker-a"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if len(jobs) != 2 {
		t.Fatalf("got %d claimed jobs, want 2", len(jobs))
	}
	for _, j := range jobs {
		if j.Status != StatusProcessing {
			t.Errorf("claimed job status = %s, want processing", j.Status)
		}
		if j.ClaimedBy != "worker-a" {
			t.Errorf("claimedBy = %s, want worker-a", j.ClaimedBy)
		}
	}

	// Second claim should return empty.
	second, err := q.Claim(ctx, ClaimOptions{BatchSize: 5, WorkerID: "worker-b"})
	if err != nil {
		t.Fatalf("second claim: %v", err)
	}
	if len(second) != 0 {
		t.Errorf("second claim got %d jobs, want 0", len(second))
	}
}

func TestAdapterClaimBatchSize(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	for i := 0; i < 5; i++ {
		_, err = q.Enqueue(ctx, EnqueueInput{
			BrainID: "brain-1",
			Payload: JobPayload{Kind: "raw"},
		})
		if err != nil {
			t.Fatalf("enqueue %d: %v", i, err)
		}
	}

	jobs, err := q.Claim(ctx, ClaimOptions{BatchSize: 2, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	if len(jobs) != 2 {
		t.Errorf("got %d jobs, want 2", len(jobs))
	}
}

func TestAdapterClaimRequiresWorkerID(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	_, err = q.Claim(context.Background(), ClaimOptions{BatchSize: 1, WorkerID: ""})
	if err == nil {
		t.Error("expected error for empty worker ID")
	}
}

func TestAdapterHeartbeat(t *testing.T) {
	t.Parallel()
	db, store := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	job, err := q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "raw"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}

	store.mu.Lock()
	oldHB := store.jobs[0].lastHeartbeat
	store.mu.Unlock()

	time.Sleep(5 * time.Millisecond)
	err = q.Heartbeat(ctx, job.ID)
	if err != nil {
		t.Fatalf("heartbeat: %v", err)
	}

	store.mu.Lock()
	newHB := store.jobs[0].lastHeartbeat
	store.mu.Unlock()

	if newHB == nil || !newHB.After(*oldHB) {
		t.Error("heartbeat did not advance the timestamp")
	}
}

func TestAdapterHeartbeat_NonProcessing(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	err = q.Heartbeat(context.Background(), "nonexistent")
	if err == nil {
		t.Error("expected error for non-processing heartbeat")
	}
}

func TestAdapterComplete(t *testing.T) {
	t.Parallel()
	db, store := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	job, err := q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "raw"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	err = q.Complete(ctx, job.ID, map[string]string{"chunks": "3"})
	if err != nil {
		t.Fatalf("complete: %v", err)
	}

	store.mu.Lock()
	j := store.jobs[0]
	store.mu.Unlock()

	if j.status != "completed" {
		t.Errorf("status = %s, want completed", j.status)
	}
	if j.completedAt == nil {
		t.Error("completedAt should be set")
	}
}

func TestAdapterFail_Retry(t *testing.T) {
	t.Parallel()
	db, store := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	job, err := q.Enqueue(ctx, EnqueueInput{
		BrainID:    "brain-1",
		Payload:    JobPayload{Kind: "raw"},
		MaxRetries: 3,
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	err = q.Fail(ctx, job.ID, "temporary error", true)
	if err != nil {
		t.Fatalf("fail: %v", err)
	}

	store.mu.Lock()
	j := store.jobs[0]
	store.mu.Unlock()

	if j.status != "pending" {
		t.Errorf("status = %s, want pending", j.status)
	}
	if j.retryCount != 1 {
		t.Errorf("retryCount = %d, want 1", j.retryCount)
	}
	if j.nextRetryAt == nil {
		t.Error("nextRetryAt should be set for retryable failure")
	}
	if j.claimedBy != nil {
		t.Error("claimedBy should be nil after fail")
	}
}

func TestAdapterFail_DeadLetter(t *testing.T) {
	t.Parallel()
	db, store := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	job, err := q.Enqueue(ctx, EnqueueInput{
		BrainID:    "brain-1",
		Payload:    JobPayload{Kind: "raw"},
		MaxRetries: 1,
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	err = q.Fail(ctx, job.ID, "fatal error", true)
	if err != nil {
		t.Fatalf("fail: %v", err)
	}

	store.mu.Lock()
	j := store.jobs[0]
	store.mu.Unlock()

	if j.status != "dead_letter" {
		t.Errorf("status = %s, want dead_letter", j.status)
	}
}

func TestAdapterFail_NotRetryable(t *testing.T) {
	t.Parallel()
	db, store := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	job, err := q.Enqueue(ctx, EnqueueInput{
		BrainID:    "brain-1",
		Payload:    JobPayload{Kind: "raw"},
		MaxRetries: 5,
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	err = q.Fail(ctx, job.ID, "permanent error", false)
	if err != nil {
		t.Fatalf("fail: %v", err)
	}

	store.mu.Lock()
	j := store.jobs[0]
	store.mu.Unlock()

	if j.status != "dead_letter" {
		t.Errorf("status = %s, want dead_letter", j.status)
	}
}

func TestAdapterRecoverStale(t *testing.T) {
	t.Parallel()
	db, store := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	_, err = q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "raw"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}

	// Backdate the heartbeat.
	store.mu.Lock()
	stale := time.Now().Add(-10 * time.Minute)
	store.jobs[0].lastHeartbeat = &stale
	store.mu.Unlock()

	recovered, err := q.RecoverStale(ctx, 5*time.Minute)
	if err != nil {
		t.Fatalf("recoverStale: %v", err)
	}
	if recovered != 1 {
		t.Errorf("recovered = %d, want 1", recovered)
	}

	store.mu.Lock()
	if store.jobs[0].status != "pending" {
		t.Errorf("status = %s, want pending", store.jobs[0].status)
	}
	store.mu.Unlock()

	// Fresh jobs should not be recovered.
	_, err = q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-2",
		Payload: JobPayload{Kind: "raw"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}
	_, err = q.Claim(ctx, ClaimOptions{BatchSize: 1, WorkerID: "w"})
	if err != nil {
		t.Fatalf("claim: %v", err)
	}
	recovered2, err := q.RecoverStale(ctx, 5*time.Minute)
	if err != nil {
		t.Fatalf("recoverStale: %v", err)
	}
	if recovered2 != 0 {
		t.Errorf("recovered = %d, want 0 for fresh jobs", recovered2)
	}
}

func TestAdapterCountByStatus(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}
	defer q.Close()

	ctx := context.Background()
	for i := 0; i < 3; i++ {
		_, err = q.Enqueue(ctx, EnqueueInput{
			BrainID: "brain-1",
			Payload: JobPayload{Kind: "raw"},
		})
		if err != nil {
			t.Fatalf("enqueue %d: %v", i, err)
		}
	}
	_, err = q.Enqueue(ctx, EnqueueInput{
		BrainID: "brain-2",
		Payload: JobPayload{Kind: "raw"},
	})
	if err != nil {
		t.Fatalf("enqueue: %v", err)
	}

	counts, err := q.CountByStatus(ctx, "")
	if err != nil {
		t.Fatalf("countByStatus: %v", err)
	}
	if counts[StatusPending] != 4 {
		t.Errorf("pending = %d, want 4", counts[StatusPending])
	}

	filteredCounts, err := q.CountByStatus(ctx, "brain-1")
	if err != nil {
		t.Fatalf("countByStatus filtered: %v", err)
	}
	if filteredCounts[StatusPending] != 3 {
		t.Errorf("brain-1 pending = %d, want 3", filteredCounts[StatusPending])
	}
}

func TestAdapterClose(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	q, err := NewPostgresQueue(PostgresOptions{DB: db})
	if err != nil {
		t.Fatalf("NewPostgresQueue: %v", err)
	}

	err = q.Close()
	if err != nil {
		t.Fatalf("close: %v", err)
	}

	// Operations after close should fail.
	_, err = q.Enqueue(context.Background(), EnqueueInput{
		BrainID: "brain-1",
		Payload: JobPayload{Kind: "raw"},
	})
	if err == nil {
		t.Error("expected error after close")
	}

	// Second close should be idempotent.
	err = q.Close()
	if err != nil {
		t.Fatalf("second close: %v", err)
	}
}

func TestNewPostgresQueue_InvalidSchemaWithMockDB(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	_, err := NewPostgresQueue(PostgresOptions{
		DB:     db,
		Schema: "invalid-schema",
	})
	if err == nil {
		t.Fatal("expected error for invalid schema")
	}
	if !strings.Contains(err.Error(), "invalid schema name") {
		t.Errorf("error message should mention schema: %v", err)
	}
}

func TestNewPostgresQueue_InvalidTableWithMockDB(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	_, err := NewPostgresQueue(PostgresOptions{
		DB:        db,
		TableName: "table;DROP",
	})
	if err == nil {
		t.Fatal("expected error for invalid table name")
	}
	if !strings.Contains(err.Error(), "invalid table name") {
		t.Errorf("error message should mention table: %v", err)
	}
}

func TestNewPostgresQueue_InvalidNotifyChannel(t *testing.T) {
	t.Parallel()
	db, _ := newMockDB()
	defer db.Close()

	_, err := NewPostgresQueue(PostgresOptions{
		DB:            db,
		NotifyChannel: "channel;DROP",
	})
	if err == nil {
		t.Fatal("expected error for invalid notify channel")
	}
	if !strings.Contains(err.Error(), "invalid notify channel") {
		t.Errorf("error message should mention notify channel: %v", err)
	}
}
