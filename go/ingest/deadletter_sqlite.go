// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// SqliteDeadLetterAdapter implements DeadLetterAdapter backed by a SQLite
// database. Used for local/embedded deployments.
type SqliteDeadLetterAdapter struct {
	db *sql.DB
}

// NewSqliteDeadLetterAdapter creates a dead letter adapter backed by
// SQLite. The caller must ensure the ingest_dead_letter table exists.
func NewSqliteDeadLetterAdapter(db *sql.DB) *SqliteDeadLetterAdapter {
	return &SqliteDeadLetterAdapter{db: db}
}

// EnsureTable creates the dead letter table and indices if they do not
// already exist.
func (a *SqliteDeadLetterAdapter) EnsureTable(ctx context.Context) error {
	_, err := a.db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS ingest_dead_letter (
			id              TEXT PRIMARY KEY,
			original_job_id TEXT NOT NULL,
			brain_id        TEXT NOT NULL,
			payload         TEXT NOT NULL,
			failure_reason  TEXT NOT NULL,
			last_error      TEXT,
			error_history   TEXT,
			retry_count     INTEGER NOT NULL DEFAULT 0,
			metadata        TEXT,
			group_id        TEXT,
			moved_at        TEXT NOT NULL DEFAULT (datetime('now')),
			resolved_at     TEXT,
			resolved_by     TEXT
		)`)
	if err != nil {
		return fmt.Errorf("ingest: create dead letter table: %w", err)
	}
	_, err = a.db.ExecContext(ctx, `CREATE INDEX IF NOT EXISTS idx_dlq_brain ON ingest_dead_letter(brain_id)`)
	if err != nil {
		return fmt.Errorf("ingest: create dead letter brain index: %w", err)
	}
	_, err = a.db.ExecContext(ctx, `CREATE INDEX IF NOT EXISTS idx_dlq_moved ON ingest_dead_letter(moved_at)`)
	if err != nil {
		return fmt.Errorf("ingest: create dead letter moved index: %w", err)
	}
	return nil
}

func (a *SqliteDeadLetterAdapter) Move(ctx context.Context, entry DeadLetterEntry) (DeadLetterEntry, error) {
	if entry.ID == "" {
		entry.ID = uuid.New().String()
	}
	if entry.MovedAt.IsZero() {
		entry.MovedAt = time.Now().UTC()
	}

	payloadJSON, err := json.Marshal(entry.Payload)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: marshal dead letter payload: %w", err)
	}

	var metadataJSON []byte
	if len(entry.Metadata) > 0 {
		metadataJSON, err = json.Marshal(entry.Metadata)
		if err != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: marshal dead letter metadata: %w", err)
		}
	}

	var errorHistoryJSON []byte
	if len(entry.ErrorHistory) > 0 {
		errorHistoryJSON, err = json.Marshal(entry.ErrorHistory)
		if err != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: marshal dead letter error history: %w", err)
		}
	}

	_, err = a.db.ExecContext(ctx, `
		INSERT INTO ingest_dead_letter
			(id, original_job_id, brain_id, payload, failure_reason, last_error,
			 error_history, retry_count, metadata, group_id, moved_at, resolved_at, resolved_by)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		entry.ID,
		entry.OriginalJobID,
		entry.BrainID,
		string(payloadJSON),
		entry.FailureReason,
		nullableString(entry.LastError),
		nullableBytes(errorHistoryJSON),
		entry.RetryCount,
		nullableBytes(metadataJSON),
		nullableString(entry.GroupID),
		entry.MovedAt.Format(time.RFC3339),
		nullableTime(entry.ResolvedAt),
		nullableString(entry.ResolvedBy),
	)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: move to dead letter: %w", err)
	}
	return entry, nil
}

func (a *SqliteDeadLetterAdapter) List(ctx context.Context, opts DeadLetterListOptions) (DeadLetterListResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	whereClause, args := buildListWhere(opts, "?")

	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM ingest_dead_letter %s", whereClause)
	var total int
	if err := a.db.QueryRowContext(ctx, countQuery, args...).Scan(&total); err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: dead letter count: %w", err)
	}

	selectQuery := fmt.Sprintf(`
		SELECT %s
		  FROM ingest_dead_letter
		  %s
		 ORDER BY moved_at DESC
		 LIMIT ? OFFSET ?`, deadLetterColumns, whereClause)

	selectArgs := append(args, limit, opts.Offset) //nolint:gocritic // append to copy is intentional
	rows, err := a.db.QueryContext(ctx, selectQuery, selectArgs...)
	if err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: dead letter list: %w", err)
	}
	defer func() { _ = rows.Close() }()

	entries := make([]DeadLetterEntry, 0, deadLetterListCap)
	for rows.Next() {
		entry, scanErr := scanSqliteRow(rows)
		if scanErr != nil {
			return DeadLetterListResult{}, fmt.Errorf("ingest: dead letter list scan: %w", scanErr)
		}
		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: dead letter list iterate: %w", err)
	}

	return DeadLetterListResult{Entries: entries, Total: total}, nil
}

func (a *SqliteDeadLetterAdapter) Get(ctx context.Context, id string) (DeadLetterEntry, error) {
	row := a.db.QueryRowContext(ctx, fmt.Sprintf(`
		SELECT %s
		  FROM ingest_dead_letter
		 WHERE id = ?`, deadLetterColumns), id)

	entry, err := scanSqliteRow(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return DeadLetterEntry{}, ErrDeadLetterNotFound
		}
		return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter get: %w", err)
	}
	return entry, nil
}

func (a *SqliteDeadLetterAdapter) Retry(ctx context.Context, id string, resolvedBy string, reEnqueue ReEnqueueFunc) (DeadLetterEntry, error) {
	tx, err := a.db.BeginTx(ctx, nil)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter retry begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // rollback after commit is a no-op

	now := time.Now().UTC()
	nowStr := now.Format(time.RFC3339)

	// Atomic update: only update if not already resolved.
	row := tx.QueryRowContext(ctx, fmt.Sprintf(`
		UPDATE ingest_dead_letter
		   SET resolved_at = ?, resolved_by = ?
		 WHERE id = ? AND resolved_at IS NULL
		 RETURNING %s`, deadLetterColumns),
		nowStr, resolvedBy, id)

	entry, scanErr := scanSqliteRow(row)
	if scanErr != nil {
		if scanErr == sql.ErrNoRows {
			// Distinguish between not-found and already-resolved.
			var exists int
			checkErr := tx.QueryRowContext(ctx, "SELECT 1 FROM ingest_dead_letter WHERE id = ?", id).Scan(&exists)
			if checkErr != nil {
				return DeadLetterEntry{}, ErrDeadLetterNotFound
			}
			return DeadLetterEntry{}, ErrDeadLetterAlreadyResolved
		}
		return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter retry scan: %w", scanErr)
	}

	if reEnqueue != nil {
		if enqueueErr := reEnqueue(ctx, entry); enqueueErr != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter retry re-enqueue: %w", enqueueErr)
		}
	}

	if commitErr := tx.Commit(); commitErr != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter retry commit: %w", commitErr)
	}

	return entry, nil
}

func (a *SqliteDeadLetterAdapter) Purge(ctx context.Context, opts PurgeOptions) (int, error) {
	query, args := buildSqlitePurgeQuery(opts)
	result, err := a.db.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, fmt.Errorf("ingest: dead letter purge: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("ingest: dead letter purge rows affected: %w", err)
	}
	return int(affected), nil
}

func (a *SqliteDeadLetterAdapter) Count(ctx context.Context, brainID string) (int, error) {
	query := "SELECT COUNT(*) FROM ingest_dead_letter WHERE resolved_at IS NULL"
	args := make([]any, 0, 1)
	if brainID != "" {
		query = "SELECT COUNT(*) FROM ingest_dead_letter WHERE brain_id = ? AND resolved_at IS NULL"
		args = append(args, brainID)
	}

	var count int
	if err := a.db.QueryRowContext(ctx, query, args...).Scan(&count); err != nil {
		return 0, fmt.Errorf("ingest: dead letter count: %w", err)
	}
	return count, nil
}

// --- SQLite scan & query helpers ---------------------------------------------

// scanSqliteRow scans a dead letter row from SQLite. It handles
// timestamps stored as RFC 3339 strings and nullable columns stored as
// sql.NullString.
func scanSqliteRow(s scanner) (DeadLetterEntry, error) {
	var entry DeadLetterEntry
	var payloadJSON string
	var lastError sql.NullString
	var errorHistoryJSON sql.NullString
	var metadataJSON sql.NullString
	var groupID sql.NullString
	var movedAt string
	var resolvedAt sql.NullString
	var resolvedBy sql.NullString

	err := s.Scan(
		&entry.ID,
		&entry.OriginalJobID,
		&entry.BrainID,
		&payloadJSON,
		&entry.FailureReason,
		&lastError,
		&errorHistoryJSON,
		&entry.RetryCount,
		&metadataJSON,
		&groupID,
		&movedAt,
		&resolvedAt,
		&resolvedBy,
	)
	if err != nil {
		return DeadLetterEntry{}, err
	}

	if populateErr := populateCommonFields(&entry, payloadJSON, lastError, errorHistoryJSON, metadataJSON, groupID, resolvedBy); populateErr != nil {
		return DeadLetterEntry{}, populateErr
	}

	parsed, parseErr := time.Parse(time.RFC3339, movedAt)
	if parseErr != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: parse dead letter moved_at: %w", parseErr)
	}
	entry.MovedAt = parsed

	if resolvedAt.Valid {
		parsedResolved, parseResolvedErr := time.Parse(time.RFC3339, resolvedAt.String)
		if parseResolvedErr != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: parse dead letter resolved_at: %w", parseResolvedErr)
		}
		entry.ResolvedAt = &parsedResolved
	}

	return entry, nil
}

func buildSqlitePurgeQuery(opts PurgeOptions) (string, []any) {
	purgeQueries := map[PurgeKind]struct {
		query string
		args  func() []any
	}{
		PurgeByID: {
			query: "DELETE FROM ingest_dead_letter WHERE id = ?",
			args:  func() []any { return []any{opts.ID} },
		},
		PurgeByBrain: {
			query: "DELETE FROM ingest_dead_letter WHERE brain_id = ?",
			args:  func() []any { return []any{opts.BrainID} },
		},
		PurgeOlderThan: {
			query: "DELETE FROM ingest_dead_letter WHERE moved_at < datetime('now', ?)",
			args: func() []any {
				return []any{fmt.Sprintf("-%d days", opts.Days)}
			},
		},
		PurgeAllResolved: {
			query: "DELETE FROM ingest_dead_letter WHERE resolved_at IS NOT NULL",
			args:  func() []any { return nil },
		},
	}

	entry, ok := purgeQueries[opts.Kind]
	if !ok {
		return "SELECT 0", nil
	}
	return entry.query, entry.args()
}
