// SPDX-License-Identifier: Apache-2.0

// Package queue defines the ingest queue abstraction and its PostgreSQL
// adapter. Workers claim jobs via FOR UPDATE SKIP LOCKED for safe
// concurrent access across multiple processes.
package queue

import (
	"context"
	"time"
)

// JobStatus represents the lifecycle state of an ingest queue job.
type JobStatus string

const (
	// StatusPending indicates a job is awaiting claim by a worker.
	StatusPending JobStatus = "pending"
	// StatusProcessing indicates a worker has claimed the job and is actively processing it.
	StatusProcessing JobStatus = "processing"
	// StatusCompleted indicates the job finished successfully.
	StatusCompleted JobStatus = "completed"
	// StatusFailed indicates the job failed but may be retried.
	StatusFailed JobStatus = "failed"
	// StatusDeadLetter indicates the job exhausted all retries or was marked non-retryable.
	StatusDeadLetter JobStatus = "dead_letter"
)

// validStatuses maps each valid status string to its typed constant for
// O(1) validation without if/else chains.
var validStatuses = map[JobStatus]struct{}{
	StatusPending:    {},
	StatusProcessing: {},
	StatusCompleted:  {},
	StatusFailed:     {},
	StatusDeadLetter: {},
}

// IsValid reports whether s is a recognised job status.
func (s JobStatus) IsValid() bool {
	_, ok := validStatuses[s]
	return ok
}

// JobPayload describes what the queue job should process.
type JobPayload struct {
	Kind    string `json:"kind"`    // "file", "url", or "raw"
	Path    string `json:"path"`    // filesystem path (kind=file)
	URL     string `json:"url"`     // remote URL (kind=url)
	Content string `json:"content"` // inline content (kind=raw)
	Title   string `json:"title"`   // human-readable title
	Mime    string `json:"mime"`    // MIME type hint
}

// Job represents a single ingest queue entry with its full metadata.
type Job struct {
	ID             string
	BrainID        string
	Status         JobStatus
	Payload        JobPayload
	RetryCount     int
	MaxRetries     int
	Error          string
	ClaimedBy      string
	ClaimedAt      *time.Time
	LastHeartbeat  *time.Time
	NextRetryAt    *time.Time
	CreatedAt      time.Time
	UpdatedAt      time.Time
	CompletedAt    *time.Time
	Metadata       map[string]string
	GroupID        string
	IdempotencyKey string
}

// EnqueueInput holds the parameters for creating a new queue job.
type EnqueueInput struct {
	BrainID        string
	Payload        JobPayload
	MaxRetries     int               // default 3 when zero
	IdempotencyKey string
	GroupID        string
	Metadata       map[string]string
}

// ClaimOptions configures how a worker claims jobs from the queue.
type ClaimOptions struct {
	BatchSize int    // number of jobs to claim in one round; default 1 when zero
	WorkerID  string // unique identifier for the claiming worker
}

// EnvPostgresURL is the environment variable name for the PostgreSQL
// connection URL used by NewPostgresQueueFromEnv.
const EnvPostgresURL = "MEMORY_POSTGRES_URL"

// EnvIngestWorkerIntervalMS is the environment variable name for the
// worker poll interval in milliseconds.
const EnvIngestWorkerIntervalMS = "MEMORY_INGEST_WORKER_INTERVAL_MS"

// defaultMaxRetries is the retry ceiling when the caller does not specify one.
const defaultMaxRetries = 3

// defaultBatchSize is the claim batch size when the caller does not specify one.
const defaultBatchSize = 1

// Adapter defines the contract for an ingest queue backend. Both the
// PostgreSQL and SQLite implementations satisfy this interface so the
// pipeline code is storage-agnostic.
type Adapter interface {
	// Enqueue adds a new job to the queue. Returns the created job.
	// When an idempotency key is provided and a matching active job
	// exists, the existing job is returned unchanged.
	Enqueue(ctx context.Context, input EnqueueInput) (Job, error)

	// Claim atomically locks up to opts.BatchSize pending jobs and
	// assigns them to the specified worker. Returns the claimed jobs
	// (may be empty when no work is available).
	Claim(ctx context.Context, opts ClaimOptions) ([]Job, error)

	// Heartbeat refreshes the liveness timestamp for a processing job
	// so the stale-recovery process does not reclaim it.
	Heartbeat(ctx context.Context, jobID string) error

	// Complete marks a job as successfully finished.
	Complete(ctx context.Context, jobID string, result map[string]string) error

	// Fail records an error against a job. When retryable is true and
	// the retry count has not reached max retries, the job is returned
	// to pending with an exponential backoff delay. Otherwise the job
	// transitions to dead_letter.
	Fail(ctx context.Context, jobID string, errMsg string, retryable bool) error

	// Requeue returns a claimed job to pending status WITHOUT
	// incrementing the retry count. Use this when a job cannot be
	// processed due to transient conditions (e.g. per-brain concurrency
	// limit reached) rather than an actual processing failure.
	Requeue(ctx context.Context, jobID string) error

	// RecoverStale finds processing jobs whose last heartbeat is older
	// than staleThreshold and resets them to pending. Returns the count
	// of recovered jobs.
	RecoverStale(ctx context.Context, staleThreshold time.Duration) (int, error)

	// CountByStatus returns a count of jobs grouped by status,
	// optionally filtered by brain ID (empty string means all brains).
	CountByStatus(ctx context.Context, brainID string) (map[JobStatus]int, error)

	// Close releases resources held by the adapter (connections, timers,
	// listeners).
	Close() error
}

// Logger is the structured logging interface accepted by queue adapters.
// It mirrors log/slog semantics without importing the concrete type, so
// callers can inject their own implementation.
type Logger interface {
	Debug(msg string, args ...any)
	Info(msg string, args ...any)
	Warn(msg string, args ...any)
	Error(msg string, args ...any)
}

// noopLogger silently discards all log output. Used as the default when
// no logger is supplied.
type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
