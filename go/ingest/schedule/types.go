// SPDX-License-Identifier: Apache-2.0

// Package schedule provides cron-based scheduling for periodic ingestion
// jobs. Jobs persist to SQLite (local) or PostgreSQL (hosted) via the
// Store interface. A Scheduler polls the store for due jobs and fires
// ingestion events.
package schedule

import (
	"context"
	"time"
)

// Job represents a scheduled ingestion job.
type Job struct {
	ID             string            `json:"id"`
	BrainID        string            `json:"brainId"`
	Name           string            `json:"name"`
	CronExpression string            `json:"cronExpression"`
	Target         Target            `json:"target"`
	Enabled        bool              `json:"enabled"`
	LastRunAt      *time.Time        `json:"lastRunAt,omitempty"`
	NextRunAt      time.Time         `json:"nextRunAt"`
	CreatedAt      time.Time         `json:"createdAt"`
	UpdatedAt      time.Time         `json:"updatedAt"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// Target describes what the scheduled job ingests.
type Target struct {
	Kind string `json:"kind"` // "url", "file", "directory"
	URL  string `json:"url,omitempty"`
	Path string `json:"path,omitempty"`
	Glob string `json:"glob,omitempty"`
}

// CreateInput is the input for creating a new scheduled job.
type CreateInput struct {
	BrainID        string
	Name           string
	CronExpression string
	Target         Target
	Metadata       map[string]string
}

// UpdatePatch allows partial updates to a job.
type UpdatePatch struct {
	Name           *string
	CronExpression *string
	Target         *Target
	Enabled        *bool
	Metadata       *map[string]string
}

// Store persists scheduled jobs. Implementations must be safe for
// sequential use within a single scheduler run.
type Store interface {
	Create(ctx context.Context, input CreateInput) (Job, error)
	Get(ctx context.Context, id string) (Job, error)
	List(ctx context.Context, brainID string) ([]Job, error)
	Update(ctx context.Context, id string, patch UpdatePatch) (Job, error)
	Delete(ctx context.Context, id string) error
	FindDue(ctx context.Context, now time.Time) ([]Job, error)
	MarkRun(ctx context.Context, id string, ranAt, nextRunAt time.Time) error
}

// Logger mirrors the logging contract used across the memory SDK.
type Logger interface {
	Debug(msg string, ctx ...map[string]string)
	Info(msg string, ctx ...map[string]string)
	Warn(msg string, ctx ...map[string]string)
	Error(msg string, ctx ...map[string]string)
}

// CronEngine abstracts cron expression parsing and next-occurrence
// computation. The default implementation uses the built-in parser.
// Callers can supply a custom engine to integrate third-party cron
// libraries.
type CronEngine interface {
	// NextOccurrence returns the next time the expression fires after the
	// given reference time. Returns an error if the expression is invalid.
	NextOccurrence(expression string, after time.Time) (time.Time, error)

	// IsValid reports whether expression is a syntactically valid cron
	// expression understood by this engine.
	IsValid(expression string) bool
}

// DispatchFunc is called when a due job should trigger ingestion.
// The context allows callers to enforce per-job timeouts.
type DispatchFunc func(ctx context.Context, job Job) error
