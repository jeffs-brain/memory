// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// DefaultMaxAttempts is the default number of failed attempts before a job
// is moved to the dead letter queue.
const DefaultMaxAttempts = 3

// DefaultRetentionDays is the default number of days dead letter entries
// are retained before they become eligible for age-based purging.
const DefaultRetentionDays = 30

// ErrDeadLetterNotFound signals that the requested dead letter entry does
// not exist.
var ErrDeadLetterNotFound = errors.New("ingest: dead letter entry not found")

// ErrDeadLetterAlreadyResolved signals that the entry has already been
// retried or otherwise resolved.
var ErrDeadLetterAlreadyResolved = errors.New("ingest: dead letter entry already resolved")

// JobPayload holds the serialised job data that was being processed when
// the job failed permanently. This is the data needed to re-enqueue the
// job for a retry from the dead letter queue.
type JobPayload struct {
	DocumentHash string `json:"documentHash"`
	BrainID      string `json:"brainId"`
	Source       string `json:"source,omitempty"`
	ContentType  string `json:"contentType,omitempty"`
}

// DeadLetterEntry represents a job that has been moved to the dead letter
// queue after exhausting its retry budget. It preserves the full error
// history and original job context for operator inspection.
type DeadLetterEntry struct {
	ID            string            `json:"id"`
	OriginalJobID string            `json:"originalJobId"`
	BrainID       string            `json:"brainId"`
	Payload       JobPayload        `json:"payload"`
	FailureReason string            `json:"failureReason"`
	LastError     string            `json:"lastError"`
	ErrorHistory  []string          `json:"errorHistory,omitempty"`
	RetryCount    int               `json:"retryCount"`
	Metadata      map[string]string `json:"metadata,omitempty"`
	GroupID       string            `json:"groupId,omitempty"`
	MovedAt       time.Time         `json:"movedAt"`
	ResolvedAt    *time.Time        `json:"resolvedAt,omitempty"`
	ResolvedBy    string            `json:"resolvedBy,omitempty"`
}

// DeadLetterListOptions controls filtering and pagination when listing
// dead letter entries.
type DeadLetterListOptions struct {
	BrainID         string
	Limit           int
	Offset          int
	IncludeResolved bool
}

// DeadLetterListResult holds a page of dead letter entries and the total
// count matching the filter.
type DeadLetterListResult struct {
	Entries []DeadLetterEntry
	Total   int
}

// PurgeKind discriminates the type of purge operation.
type PurgeKind string

const (
	PurgeByID        PurgeKind = "by-id"
	PurgeByBrain     PurgeKind = "by-brain"
	PurgeOlderThan   PurgeKind = "older-than"
	PurgeAllResolved PurgeKind = "all-resolved"
)

// PurgeOptions specifies which dead letter entries should be removed.
type PurgeOptions struct {
	Kind    PurgeKind
	ID      string
	BrainID string
	Days    int
}

// ReEnqueueFunc is a callback that creates a new queue job in the main
// queue from a dead letter entry. The implementation is responsible for
// resetting the retry count and assigning a new job ID. Returning an
// error prevents the DLQ entry from being marked as resolved.
type ReEnqueueFunc func(ctx context.Context, entry DeadLetterEntry) error

// DeadLetterAdapter defines the operations available on the dead letter
// queue. Both SQLite and PostgreSQL implementations satisfy this
// interface.
type DeadLetterAdapter interface {
	// Move transfers a failed job to the dead letter queue with a reason.
	Move(ctx context.Context, entry DeadLetterEntry) (DeadLetterEntry, error)

	// List returns a paginated, filtered view of dead letter entries.
	List(ctx context.Context, opts DeadLetterListOptions) (DeadLetterListResult, error)

	// Get retrieves a single entry by ID. Returns ErrDeadLetterNotFound
	// when no entry exists.
	Get(ctx context.Context, id string) (DeadLetterEntry, error)

	// Retry marks an entry as resolved and invokes the re-enqueue
	// callback to create a new queue job in the main queue. Returns
	// ErrDeadLetterAlreadyResolved if the entry was already retried.
	// If reEnqueue is nil, the entry is marked as resolved without
	// re-enqueueing (caller takes responsibility).
	Retry(ctx context.Context, id string, resolvedBy string, reEnqueue ReEnqueueFunc) (DeadLetterEntry, error)

	// Purge removes entries matching the given options and returns the
	// count of entries removed.
	Purge(ctx context.Context, opts PurgeOptions) (int, error)

	// Count returns the number of dead letter entries, optionally
	// filtered by brain ID.
	Count(ctx context.Context, brainID string) (int, error)
}

// deadLetterListCap is the initial slice capacity for list results.
const deadLetterListCap = 32

// deadLetterColumns is the column list used in SELECT queries for dead
// letter entries. Both adapters reference the same columns.
const deadLetterColumns = `id, original_job_id, brain_id, payload, failure_reason, last_error,
	       error_history, retry_count, metadata, group_id, moved_at, resolved_at, resolved_by`

// --- Shared helpers ----------------------------------------------------------

func nullableString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func nullableBytes(b []byte) *string {
	if len(b) == 0 {
		return nil
	}
	s := string(b)
	return &s
}

func nullableTime(t *time.Time) *string {
	if t == nil {
		return nil
	}
	s := t.Format(time.RFC3339)
	return &s
}

// scanner is the common interface satisfied by both *sql.Row and *sql.Rows,
// allowing a single scan function to handle both cases per dialect.
type scanner interface {
	Scan(dest ...any) error
}

// populateCommonFields deserialises JSON fields and assigns nullable
// columns that are common to both SQLite and PostgreSQL scan paths.
func populateCommonFields(
	entry *DeadLetterEntry,
	payloadJSON string,
	lastError sql.NullString,
	errorHistoryJSON sql.NullString,
	metadataJSON sql.NullString,
	groupID sql.NullString,
	resolvedBy sql.NullString,
) error {
	if err := json.Unmarshal([]byte(payloadJSON), &entry.Payload); err != nil {
		return fmt.Errorf("ingest: unmarshal dead letter payload: %w", err)
	}

	if lastError.Valid {
		entry.LastError = lastError.String
	}

	if errorHistoryJSON.Valid && errorHistoryJSON.String != "" {
		var history []string
		if unmarshalErr := json.Unmarshal([]byte(errorHistoryJSON.String), &history); unmarshalErr != nil {
			return fmt.Errorf("ingest: unmarshal dead letter error history: %w", unmarshalErr)
		}
		entry.ErrorHistory = history
	}

	if metadataJSON.Valid {
		meta := make(map[string]string)
		if unmarshalErr := json.Unmarshal([]byte(metadataJSON.String), &meta); unmarshalErr != nil {
			return fmt.Errorf("ingest: unmarshal dead letter metadata: %w", unmarshalErr)
		}
		entry.Metadata = meta
	}

	if groupID.Valid {
		entry.GroupID = groupID.String
	}
	if resolvedBy.Valid {
		entry.ResolvedBy = resolvedBy.String
	}

	return nil
}

// buildListWhere constructs a WHERE clause and parameter list for the
// list query. The placeholder argument controls the style: use "?" for
// SQLite or "$" for PostgreSQL (numbered placeholders).
func buildListWhere(opts DeadLetterListOptions, placeholder string) (string, []any) {
	clauses := make([]string, 0, 2)
	args := make([]any, 0, 2)
	paramIdx := 1

	if opts.BrainID != "" {
		ph := "?"
		if placeholder == "$" {
			ph = fmt.Sprintf("$%d", paramIdx)
		}
		clauses = append(clauses, "brain_id = "+ph)
		args = append(args, opts.BrainID)
		paramIdx++
	}
	if !opts.IncludeResolved {
		clauses = append(clauses, "resolved_at IS NULL")
	}

	_ = paramIdx
	if len(clauses) == 0 {
		return "", nil
	}
	return "WHERE " + strings.Join(clauses, " AND "), args
}

// Compile-time interface assertions.
var _ DeadLetterAdapter = (*SqliteDeadLetterAdapter)(nil)
var _ DeadLetterAdapter = (*PostgresDeadLetterAdapter)(nil)
