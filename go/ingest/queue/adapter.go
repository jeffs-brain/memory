// SPDX-License-Identifier: Apache-2.0

// Package queue provides a concurrent worker pool for processing ingestion
// pipeline jobs claimed from a queue backend. The pool manages configurable
// parallelism, per-brain concurrency limits, backpressure detection, and
// graceful shutdown semantics.
//
// The queue.Adapter interface is defined here so the worker pool can be
// developed independently of the PostgreSQL queue implementation (P3-1).
// When P3-1 lands, its concrete adapter satisfies this interface.
package queue

import (
	"context"
	"time"
)

// JobStatus represents the processing state of a queued ingestion job.
type JobStatus string

const (
	// JobStatusPending indicates the job is waiting to be claimed.
	JobStatusPending JobStatus = "pending"
	// JobStatusRunning indicates a worker has claimed and is processing the job.
	JobStatusRunning JobStatus = "running"
	// JobStatusCompleted indicates the job finished successfully.
	JobStatusCompleted JobStatus = "completed"
	// JobStatusFailed indicates the job encountered an unrecoverable error.
	JobStatusFailed JobStatus = "failed"
)

// Job represents a single ingestion pipeline job in the queue. The BrainID
// field is used for per-brain concurrency limiting.
type Job struct {
	ID        string
	BrainID   string
	Payload   []byte
	Status    JobStatus
	Attempts  int
	CreatedAt time.Time
	ClaimedAt time.Time
}

// Adapter abstracts the queue storage backend. P3-1 will provide a
// PostgreSQL-backed implementation using FOR UPDATE SKIP LOCKED.
type Adapter interface {
	// Claim atomically selects and locks the next available job for
	// processing. Returns nil, nil when no jobs are available.
	Claim(ctx context.Context, workerID string) (*Job, error)

	// Complete marks a job as successfully processed.
	Complete(ctx context.Context, jobID string) error

	// Fail marks a job as failed with the given reason.
	Fail(ctx context.Context, jobID string, reason string) error

	// Heartbeat extends the claim lease for an in-progress job so stale
	// detection does not reclaim it prematurely.
	Heartbeat(ctx context.Context, jobID string) error

	// Depth returns the number of pending jobs in the queue, optionally
	// scoped to a specific brain. Pass an empty brainID for the global count.
	Depth(ctx context.Context, brainID string) (int64, error)
}

// Logger is the logging contract for the pool. Callers inject a concrete
// implementation; the pool never writes to stdout/stderr directly.
type Logger interface {
	Debug(msg string, keysAndValues ...any)
	Info(msg string, keysAndValues ...any)
	Warn(msg string, keysAndValues ...any)
	Error(msg string, keysAndValues ...any)
}

// noopLogger discards all log messages. Used when the caller does not
// provide a logger.
type noopLogger struct{}

func (noopLogger) Debug(string, ...any) {}
func (noopLogger) Info(string, ...any)  {}
func (noopLogger) Warn(string, ...any)  {}
func (noopLogger) Error(string, ...any) {}
