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

// PostgresDeadLetterAdapter implements DeadLetterAdapter backed by
// PostgreSQL. Used for hosted/production deployments.
type PostgresDeadLetterAdapter struct {
	db     *sql.DB
	schema string
}

// PostgresDeadLetterConfig configures the PostgreSQL dead letter adapter.
type PostgresDeadLetterConfig struct {
	DB     *sql.DB
	Schema string
}

// NewPostgresDeadLetterAdapter creates a dead letter adapter backed by
// PostgreSQL. Returns an error if the schema name is invalid.
func NewPostgresDeadLetterAdapter(cfg PostgresDeadLetterConfig) (*PostgresDeadLetterAdapter, error) {
	if cfg.DB == nil {
		return nil, fmt.Errorf("ingest: postgres DB is required for dead letter adapter")
	}
	schema := cfg.Schema
	if schema == "" {
		schema = "memory"
	}
	if !schemaNamePattern.MatchString(schema) {
		return nil, fmt.Errorf("ingest: invalid schema name %q for dead letter adapter", schema)
	}
	return &PostgresDeadLetterAdapter{db: cfg.DB, schema: schema}, nil
}

func (a *PostgresDeadLetterAdapter) table() string {
	return fmt.Sprintf("%s.ingest_dead_letter", a.schema)
}

func (a *PostgresDeadLetterAdapter) Move(ctx context.Context, entry DeadLetterEntry) (DeadLetterEntry, error) {
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

	query := fmt.Sprintf(`
		INSERT INTO %s
			(id, original_job_id, brain_id, payload, failure_reason, last_error,
			 error_history, retry_count, metadata, group_id, moved_at, resolved_at, resolved_by)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)`, a.table())

	_, err = a.db.ExecContext(ctx, query,
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
		entry.MovedAt,
		nullableTime(entry.ResolvedAt),
		nullableString(entry.ResolvedBy),
	)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg move to dead letter: %w", err)
	}
	return entry, nil
}

func (a *PostgresDeadLetterAdapter) List(ctx context.Context, opts DeadLetterListOptions) (DeadLetterListResult, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	}

	whereClause, args := buildListWhere(opts, "$")

	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM %s %s", a.table(), whereClause)
	var total int
	if err := a.db.QueryRowContext(ctx, countQuery, args...).Scan(&total); err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: pg dead letter count: %w", err)
	}

	nextParam := len(args) + 1
	selectQuery := fmt.Sprintf(`
		SELECT %s
		  FROM %s
		  %s
		 ORDER BY moved_at DESC
		 LIMIT $%d OFFSET $%d`, deadLetterColumns, a.table(), whereClause, nextParam, nextParam+1)

	selectArgs := append(args, limit, opts.Offset) //nolint:gocritic // append to copy is intentional
	rows, err := a.db.QueryContext(ctx, selectQuery, selectArgs...)
	if err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: pg dead letter list: %w", err)
	}
	defer rows.Close()

	entries := make([]DeadLetterEntry, 0, deadLetterListCap)
	for rows.Next() {
		entry, scanErr := scanPgRow(rows)
		if scanErr != nil {
			return DeadLetterListResult{}, fmt.Errorf("ingest: pg dead letter list scan: %w", scanErr)
		}
		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: pg dead letter list iterate: %w", err)
	}

	return DeadLetterListResult{Entries: entries, Total: total}, nil
}

func (a *PostgresDeadLetterAdapter) Get(ctx context.Context, id string) (DeadLetterEntry, error) {
	query := fmt.Sprintf(`
		SELECT %s
		  FROM %s
		 WHERE id = $1`, deadLetterColumns, a.table())

	row := a.db.QueryRowContext(ctx, query, id)
	entry, err := scanPgRow(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return DeadLetterEntry{}, ErrDeadLetterNotFound
		}
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter get: %w", err)
	}
	return entry, nil
}

func (a *PostgresDeadLetterAdapter) Retry(ctx context.Context, id string, resolvedBy string, reEnqueue ReEnqueueFunc) (DeadLetterEntry, error) {
	tx, err := a.db.BeginTx(ctx, nil)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter retry begin tx: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck // rollback after commit is a no-op

	now := time.Now().UTC()

	// Atomic update: only update if not already resolved.
	query := fmt.Sprintf(`
		UPDATE %s
		   SET resolved_at = $1, resolved_by = $2
		 WHERE id = $3 AND resolved_at IS NULL
		 RETURNING %s`, a.table(), deadLetterColumns)

	row := tx.QueryRowContext(ctx, query, now, resolvedBy, id)
	entry, scanErr := scanPgRow(row)
	if scanErr != nil {
		if scanErr == sql.ErrNoRows {
			// Distinguish between not-found and already-resolved.
			var exists int
			checkQuery := fmt.Sprintf("SELECT 1 FROM %s WHERE id = $1", a.table())
			checkErr := a.db.QueryRowContext(ctx, checkQuery, id).Scan(&exists)
			if checkErr != nil {
				return DeadLetterEntry{}, ErrDeadLetterNotFound
			}
			return DeadLetterEntry{}, ErrDeadLetterAlreadyResolved
		}
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter retry scan: %w", scanErr)
	}

	if reEnqueue != nil {
		if enqueueErr := reEnqueue(ctx, entry); enqueueErr != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter retry re-enqueue: %w", enqueueErr)
		}
	}

	if commitErr := tx.Commit(); commitErr != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter retry commit: %w", commitErr)
	}

	return entry, nil
}

func (a *PostgresDeadLetterAdapter) Purge(ctx context.Context, opts PurgeOptions) (int, error) {
	query, args := buildPgPurgeQuery(opts, a.table())
	result, err := a.db.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, fmt.Errorf("ingest: pg dead letter purge: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("ingest: pg dead letter purge rows affected: %w", err)
	}
	return int(affected), nil
}

func (a *PostgresDeadLetterAdapter) Count(ctx context.Context, brainID string) (int, error) {
	query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE resolved_at IS NULL", a.table())
	args := make([]any, 0, 1)
	if brainID != "" {
		query = fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE brain_id = $1 AND resolved_at IS NULL", a.table())
		args = append(args, brainID)
	}

	var count int
	if err := a.db.QueryRowContext(ctx, query, args...).Scan(&count); err != nil {
		return 0, fmt.Errorf("ingest: pg dead letter count: %w", err)
	}
	return count, nil
}

// --- PG scan & query helpers -------------------------------------------------

// scanPgRow scans a dead letter row from PostgreSQL. It uses native
// time.Time/sql.NullTime for TIMESTAMPTZ columns, unlike the SQLite
// scanner which reads timestamps as strings.
func scanPgRow(s scanner) (DeadLetterEntry, error) {
	var entry DeadLetterEntry
	var payloadJSON string
	var lastError sql.NullString
	var errorHistoryJSON sql.NullString
	var metadataJSON sql.NullString
	var groupID sql.NullString
	var movedAt time.Time
	var resolvedAt sql.NullTime
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

	entry.MovedAt = movedAt
	if resolvedAt.Valid {
		entry.ResolvedAt = &resolvedAt.Time
	}

	return entry, nil
}

func buildPgPurgeQuery(opts PurgeOptions, table string) (string, []any) {
	purgeQueries := map[PurgeKind]struct {
		query string
		args  func() []any
	}{
		PurgeByID: {
			query: fmt.Sprintf("DELETE FROM %s WHERE id = $1", table),
			args:  func() []any { return []any{opts.ID} },
		},
		PurgeByBrain: {
			query: fmt.Sprintf("DELETE FROM %s WHERE brain_id = $1", table),
			args:  func() []any { return []any{opts.BrainID} },
		},
		PurgeOlderThan: {
			query: fmt.Sprintf("DELETE FROM %s WHERE moved_at < NOW() - INTERVAL '1 day' * $1", table),
			args: func() []any {
				return []any{opts.Days}
			},
		},
		PurgeAllResolved: {
			query: fmt.Sprintf("DELETE FROM %s WHERE resolved_at IS NOT NULL", table),
			args:  func() []any { return nil },
		},
	}

	entry, ok := purgeQueries[opts.Kind]
	if !ok {
		return "SELECT 0", nil
	}
	return entry.query, entry.args()
}
