// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"regexp"
	"time"
)

// listIncompleteInitialCap is the initial capacity for the result slice in
// ListIncomplete, chosen to avoid small re-allocations for typical workloads.
const listIncompleteInitialCap = 16

// schemaNamePattern validates PostgreSQL schema names: lowercase letters or
// underscore start, followed by lowercase alphanumerics or underscores.
var schemaNamePattern = regexp.MustCompile(`^[a-z_][a-z0-9_]*$`)

// PostgresPipelineStateStore implements PipelineStateStore backed by a
// PostgreSQL pipeline_state table. Uses prepared statements for safe
// parameterised queries.
type PostgresPipelineStateStore struct {
	db     *sql.DB
	schema string
}

// PostgresStateStoreConfig configures the PostgreSQL state store.
type PostgresStateStoreConfig struct {
	// DB is the database connection pool.
	DB *sql.DB
	// Schema is the PostgreSQL schema containing the pipeline_state table.
	// Defaults to "memory" when empty.
	Schema string
}

// NewPostgresPipelineStateStore creates a PostgreSQL-backed state store.
// Returns an error if the schema name is invalid.
func NewPostgresPipelineStateStore(cfg PostgresStateStoreConfig) (*PostgresPipelineStateStore, error) {
	schema := cfg.Schema
	if schema == "" {
		schema = "memory"
	}
	if !schemaNamePattern.MatchString(schema) {
		return nil, fmt.Errorf("ingest: invalid schema name %q: must match ^[a-z_][a-z0-9_]*$", schema)
	}
	return &PostgresPipelineStateStore{db: cfg.DB, schema: schema}, nil
}

func (s *PostgresPipelineStateStore) table() string {
	return fmt.Sprintf("%s.pipeline_state", s.schema)
}

func (s *PostgresPipelineStateStore) Get(ctx context.Context, documentHash string) (*PipelineStateEntry, error) {
	query := fmt.Sprintf(`
		SELECT document_hash, brain_id, stage, retry_count, last_error,
		       created_at, updated_at, completed_at
		  FROM %s
		 WHERE document_hash = $1
		 LIMIT 1`, s.table())

	row := s.db.QueryRowContext(ctx, query, documentHash)
	entry, err := scanEntry(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("ingest: pg state get %s: %w", documentHash, err)
	}
	return entry, nil
}

func (s *PostgresPipelineStateStore) Set(ctx context.Context, entry PipelineStateEntry) error {
	query := fmt.Sprintf(`
		INSERT INTO %s
		  (document_hash, brain_id, stage, retry_count, last_error, created_at, updated_at, completed_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (document_hash) DO UPDATE SET
		  brain_id = EXCLUDED.brain_id,
		  stage = EXCLUDED.stage,
		  retry_count = EXCLUDED.retry_count,
		  last_error = EXCLUDED.last_error,
		  updated_at = EXCLUDED.updated_at,
		  completed_at = EXCLUDED.completed_at`, s.table())

	var lastError *string
	if entry.LastError != "" {
		lastError = &entry.LastError
	}

	var completedAt *time.Time
	if entry.CompletedAt != nil {
		completedAt = entry.CompletedAt
	}

	_, err := s.db.ExecContext(ctx, query,
		entry.DocumentHash,
		entry.BrainID,
		string(entry.Stage),
		entry.RetryCount,
		lastError,
		entry.CreatedAt,
		entry.UpdatedAt,
		completedAt,
	)
	if err != nil {
		return fmt.Errorf("ingest: pg state set %s: %w", entry.DocumentHash, err)
	}
	return nil
}

func (s *PostgresPipelineStateStore) ListIncomplete(ctx context.Context, brainID string) ([]PipelineStateEntry, error) {
	query := fmt.Sprintf(`
		SELECT document_hash, brain_id, stage, retry_count, last_error,
		       created_at, updated_at, completed_at
		  FROM %s
		 WHERE brain_id = $1
		   AND stage NOT IN ('completed', 'failed')
		 ORDER BY created_at ASC`, s.table())

	rows, err := s.db.QueryContext(ctx, query, brainID)
	if err != nil {
		return nil, fmt.Errorf("ingest: pg state list incomplete: %w", err)
	}
	defer rows.Close()

	entries := make([]PipelineStateEntry, 0, listIncompleteInitialCap)
	for rows.Next() {
		entry, scanErr := scanRows(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("ingest: pg state list scan: %w", scanErr)
		}
		entries = append(entries, *entry)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("ingest: pg state list iterate: %w", err)
	}
	return entries, nil
}

func (s *PostgresPipelineStateStore) Delete(ctx context.Context, documentHash string) error {
	query := fmt.Sprintf(`DELETE FROM %s WHERE document_hash = $1`, s.table())
	_, err := s.db.ExecContext(ctx, query, documentHash)
	if err != nil {
		return fmt.Errorf("ingest: pg state delete %s: %w", documentHash, err)
	}
	return nil
}

func scanEntry(row *sql.Row) (*PipelineStateEntry, error) {
	var entry PipelineStateEntry
	var lastError sql.NullString
	var completedAt sql.NullTime
	var stage string

	err := row.Scan(
		&entry.DocumentHash,
		&entry.BrainID,
		&stage,
		&entry.RetryCount,
		&lastError,
		&entry.CreatedAt,
		&entry.UpdatedAt,
		&completedAt,
	)
	if err != nil {
		return nil, err
	}

	entry.Stage = PipelineStage(stage)
	if lastError.Valid {
		entry.LastError = lastError.String
	}
	if completedAt.Valid {
		entry.CompletedAt = &completedAt.Time
	}
	return &entry, nil
}

func scanRows(rows *sql.Rows) (*PipelineStateEntry, error) {
	var entry PipelineStateEntry
	var lastError sql.NullString
	var completedAt sql.NullTime
	var stage string

	err := rows.Scan(
		&entry.DocumentHash,
		&entry.BrainID,
		&stage,
		&entry.RetryCount,
		&lastError,
		&entry.CreatedAt,
		&entry.UpdatedAt,
		&completedAt,
	)
	if err != nil {
		return nil, err
	}

	entry.Stage = PipelineStage(stage)
	if lastError.Valid {
		entry.LastError = lastError.String
	}
	if completedAt.Valid {
		entry.CompletedAt = &completedAt.Time
	}
	return &entry, nil
}

var _ PipelineStateStore = (*PostgresPipelineStateStore)(nil)
