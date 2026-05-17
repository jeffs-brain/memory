// SPDX-License-Identifier: Apache-2.0
package ingest

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"
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
	PurgeByID       PurgeKind = "by-id"
	PurgeByBrain    PurgeKind = "by-brain"
	PurgeOlderThan  PurgeKind = "older-than"
	PurgeAllResolved PurgeKind = "all-resolved"
)

// PurgeOptions specifies which dead letter entries should be removed.
type PurgeOptions struct {
	Kind    PurgeKind
	ID      string
	BrainID string
	Days    int
}

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

	// Retry marks an entry as resolved and returns a new JobPayload that
	// can be re-enqueued. Returns ErrDeadLetterAlreadyResolved if the
	// entry was already retried.
	Retry(ctx context.Context, id string, resolvedBy string) (DeadLetterEntry, error)

	// Purge removes entries matching the given options and returns the
	// count of entries removed.
	Purge(ctx context.Context, opts PurgeOptions) (int, error)

	// Count returns the number of dead letter entries, optionally
	// filtered by brain ID.
	Count(ctx context.Context, brainID string) (int, error)
}

// deadLetterListCap is the initial slice capacity for list results.
const deadLetterListCap = 32

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

	_, err = a.db.ExecContext(ctx, `
		INSERT INTO ingest_dead_letter
			(id, original_job_id, brain_id, payload, failure_reason, last_error,
			 retry_count, metadata, group_id, moved_at, resolved_at, resolved_by)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		entry.ID,
		entry.OriginalJobID,
		entry.BrainID,
		string(payloadJSON),
		entry.FailureReason,
		nullableString(entry.LastError),
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

	whereClause, args := buildListWhere(opts)

	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM ingest_dead_letter %s", whereClause)
	var total int
	if err := a.db.QueryRowContext(ctx, countQuery, args...).Scan(&total); err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: dead letter count: %w", err)
	}

	selectQuery := fmt.Sprintf(`
		SELECT id, original_job_id, brain_id, payload, failure_reason, last_error,
		       retry_count, metadata, group_id, moved_at, resolved_at, resolved_by
		  FROM ingest_dead_letter
		  %s
		 ORDER BY moved_at DESC
		 LIMIT ? OFFSET ?`, whereClause)

	selectArgs := append(args, limit, opts.Offset) //nolint:gocritic // append to copy is intentional
	rows, err := a.db.QueryContext(ctx, selectQuery, selectArgs...)
	if err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: dead letter list: %w", err)
	}
	defer rows.Close()

	entries := make([]DeadLetterEntry, 0, deadLetterListCap)
	for rows.Next() {
		entry, scanErr := scanDeadLetterRow(rows)
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
	row := a.db.QueryRowContext(ctx, `
		SELECT id, original_job_id, brain_id, payload, failure_reason, last_error,
		       retry_count, metadata, group_id, moved_at, resolved_at, resolved_by
		  FROM ingest_dead_letter
		 WHERE id = ?`, id)

	entry, err := scanDeadLetterSingleRow(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return DeadLetterEntry{}, ErrDeadLetterNotFound
		}
		return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter get: %w", err)
	}
	return entry, nil
}

func (a *SqliteDeadLetterAdapter) Retry(ctx context.Context, id string, resolvedBy string) (DeadLetterEntry, error) {
	entry, err := a.Get(ctx, id)
	if err != nil {
		return DeadLetterEntry{}, err
	}
	if entry.ResolvedAt != nil {
		return DeadLetterEntry{}, ErrDeadLetterAlreadyResolved
	}

	now := time.Now().UTC()
	_, err = a.db.ExecContext(ctx, `
		UPDATE ingest_dead_letter
		   SET resolved_at = ?, resolved_by = ?
		 WHERE id = ?`,
		now.Format(time.RFC3339), resolvedBy, id)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: dead letter retry update: %w", err)
	}

	entry.ResolvedAt = &now
	entry.ResolvedBy = resolvedBy
	return entry, nil
}

func (a *SqliteDeadLetterAdapter) Purge(ctx context.Context, opts PurgeOptions) (int, error) {
	query, args := buildPurgeQuery(opts)
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
	var count int
	var err error
	if brainID == "" {
		err = a.db.QueryRowContext(ctx,
			"SELECT COUNT(*) FROM ingest_dead_letter WHERE resolved_at IS NULL").Scan(&count)
	} else {
		err = a.db.QueryRowContext(ctx,
			"SELECT COUNT(*) FROM ingest_dead_letter WHERE brain_id = ? AND resolved_at IS NULL",
			brainID).Scan(&count)
	}
	if err != nil {
		return 0, fmt.Errorf("ingest: dead letter count: %w", err)
	}
	return count, nil
}

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

	query := fmt.Sprintf(`
		INSERT INTO %s
			(id, original_job_id, brain_id, payload, failure_reason, last_error,
			 retry_count, metadata, group_id, moved_at, resolved_at, resolved_by)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)`, a.table())

	_, err = a.db.ExecContext(ctx, query,
		entry.ID,
		entry.OriginalJobID,
		entry.BrainID,
		string(payloadJSON),
		entry.FailureReason,
		nullableString(entry.LastError),
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

	whereClause, args := buildListWherePg(opts)

	countQuery := fmt.Sprintf("SELECT COUNT(*) FROM %s %s", a.table(), whereClause)
	var total int
	if err := a.db.QueryRowContext(ctx, countQuery, args...).Scan(&total); err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: pg dead letter count: %w", err)
	}

	nextParam := len(args) + 1
	selectQuery := fmt.Sprintf(`
		SELECT id, original_job_id, brain_id, payload, failure_reason, last_error,
		       retry_count, metadata, group_id, moved_at, resolved_at, resolved_by
		  FROM %s
		  %s
		 ORDER BY moved_at DESC
		 LIMIT $%d OFFSET $%d`, a.table(), whereClause, nextParam, nextParam+1)

	selectArgs := append(args, limit, opts.Offset) //nolint:gocritic // append to copy is intentional
	rows, err := a.db.QueryContext(ctx, selectQuery, selectArgs...)
	if err != nil {
		return DeadLetterListResult{}, fmt.Errorf("ingest: pg dead letter list: %w", err)
	}
	defer rows.Close()

	entries := make([]DeadLetterEntry, 0, deadLetterListCap)
	for rows.Next() {
		entry, scanErr := scanDeadLetterRows(rows)
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
		SELECT id, original_job_id, brain_id, payload, failure_reason, last_error,
		       retry_count, metadata, group_id, moved_at, resolved_at, resolved_by
		  FROM %s
		 WHERE id = $1`, a.table())

	row := a.db.QueryRowContext(ctx, query, id)
	entry, err := scanDeadLetterSingleRow(row)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return DeadLetterEntry{}, ErrDeadLetterNotFound
		}
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter get: %w", err)
	}
	return entry, nil
}

func (a *PostgresDeadLetterAdapter) Retry(ctx context.Context, id string, resolvedBy string) (DeadLetterEntry, error) {
	entry, err := a.Get(ctx, id)
	if err != nil {
		return DeadLetterEntry{}, err
	}
	if entry.ResolvedAt != nil {
		return DeadLetterEntry{}, ErrDeadLetterAlreadyResolved
	}

	now := time.Now().UTC()
	query := fmt.Sprintf(`
		UPDATE %s
		   SET resolved_at = $1, resolved_by = $2
		 WHERE id = $3`, a.table())

	_, err = a.db.ExecContext(ctx, query, now, resolvedBy, id)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: pg dead letter retry update: %w", err)
	}

	entry.ResolvedAt = &now
	entry.ResolvedBy = resolvedBy
	return entry, nil
}

func (a *PostgresDeadLetterAdapter) Purge(ctx context.Context, opts PurgeOptions) (int, error) {
	query, args := buildPurgeQueryPg(opts, a.table())
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
	var count int
	var err error
	if brainID == "" {
		query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE resolved_at IS NULL", a.table())
		err = a.db.QueryRowContext(ctx, query).Scan(&count)
	} else {
		query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE brain_id = $1 AND resolved_at IS NULL", a.table())
		err = a.db.QueryRowContext(ctx, query, brainID).Scan(&count)
	}
	if err != nil {
		return 0, fmt.Errorf("ingest: pg dead letter count: %w", err)
	}
	return count, nil
}

// --- Helpers ----------------------------------------------------------------

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

func buildListWhere(opts DeadLetterListOptions) (string, []interface{}) {
	clauses := make([]string, 0, 2)
	args := make([]interface{}, 0, 2)
	paramIdx := 1

	if opts.BrainID != "" {
		clauses = append(clauses, fmt.Sprintf("brain_id = ?"))
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
	where := "WHERE " + clauses[0]
	for i := 1; i < len(clauses); i++ {
		where += " AND " + clauses[i]
	}
	return where, args
}

func buildListWherePg(opts DeadLetterListOptions) (string, []interface{}) {
	clauses := make([]string, 0, 2)
	args := make([]interface{}, 0, 2)
	paramIdx := 1

	if opts.BrainID != "" {
		clauses = append(clauses, fmt.Sprintf("brain_id = $%d", paramIdx))
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
	where := "WHERE " + clauses[0]
	for i := 1; i < len(clauses); i++ {
		where += " AND " + clauses[i]
	}
	return where, args
}

func buildPurgeQuery(opts PurgeOptions) (string, []interface{}) {
	purgeQueries := map[PurgeKind]struct {
		query string
		args  func() []interface{}
	}{
		PurgeByID: {
			query: "DELETE FROM ingest_dead_letter WHERE id = ?",
			args:  func() []interface{} { return []interface{}{opts.ID} },
		},
		PurgeByBrain: {
			query: "DELETE FROM ingest_dead_letter WHERE brain_id = ?",
			args:  func() []interface{} { return []interface{}{opts.BrainID} },
		},
		PurgeOlderThan: {
			query: "DELETE FROM ingest_dead_letter WHERE moved_at < datetime('now', ?)",
			args: func() []interface{} {
				return []interface{}{fmt.Sprintf("-%d days", opts.Days)}
			},
		},
		PurgeAllResolved: {
			query: "DELETE FROM ingest_dead_letter WHERE resolved_at IS NOT NULL",
			args:  func() []interface{} { return nil },
		},
	}

	entry, ok := purgeQueries[opts.Kind]
	if !ok {
		return "SELECT 0", nil
	}
	return entry.query, entry.args()
}

func buildPurgeQueryPg(opts PurgeOptions, table string) (string, []interface{}) {
	purgeQueries := map[PurgeKind]struct {
		query string
		args  func() []interface{}
	}{
		PurgeByID: {
			query: fmt.Sprintf("DELETE FROM %s WHERE id = $1", table),
			args:  func() []interface{} { return []interface{}{opts.ID} },
		},
		PurgeByBrain: {
			query: fmt.Sprintf("DELETE FROM %s WHERE brain_id = $1", table),
			args:  func() []interface{} { return []interface{}{opts.BrainID} },
		},
		PurgeOlderThan: {
			query: fmt.Sprintf("DELETE FROM %s WHERE moved_at < NOW() - INTERVAL '1 day' * $1", table),
			args: func() []interface{} {
				return []interface{}{opts.Days}
			},
		},
		PurgeAllResolved: {
			query: fmt.Sprintf("DELETE FROM %s WHERE resolved_at IS NOT NULL", table),
			args:  func() []interface{} { return nil },
		},
	}

	entry, ok := purgeQueries[opts.Kind]
	if !ok {
		return "SELECT 0", nil
	}
	return entry.query, entry.args()
}

func scanDeadLetterRow(rows *sql.Rows) (DeadLetterEntry, error) {
	var entry DeadLetterEntry
	var payloadJSON string
	var lastError sql.NullString
	var metadataJSON sql.NullString
	var groupID sql.NullString
	var movedAt string
	var resolvedAt sql.NullString
	var resolvedBy sql.NullString

	err := rows.Scan(
		&entry.ID,
		&entry.OriginalJobID,
		&entry.BrainID,
		&payloadJSON,
		&entry.FailureReason,
		&lastError,
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

	return populateDeadLetterEntry(entry, payloadJSON, lastError, metadataJSON, groupID, movedAt, resolvedAt, resolvedBy)
}

func scanDeadLetterRows(rows *sql.Rows) (DeadLetterEntry, error) {
	var entry DeadLetterEntry
	var payloadJSON string
	var lastError sql.NullString
	var metadataJSON sql.NullString
	var groupID sql.NullString
	var movedAt time.Time
	var resolvedAt sql.NullTime
	var resolvedBy sql.NullString

	err := rows.Scan(
		&entry.ID,
		&entry.OriginalJobID,
		&entry.BrainID,
		&payloadJSON,
		&entry.FailureReason,
		&lastError,
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

	if err := json.Unmarshal([]byte(payloadJSON), &entry.Payload); err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: unmarshal dead letter payload: %w", err)
	}
	if lastError.Valid {
		entry.LastError = lastError.String
	}
	if metadataJSON.Valid {
		meta := make(map[string]string)
		if unmarshalErr := json.Unmarshal([]byte(metadataJSON.String), &meta); unmarshalErr != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: unmarshal dead letter metadata: %w", unmarshalErr)
		}
		entry.Metadata = meta
	}
	if groupID.Valid {
		entry.GroupID = groupID.String
	}
	entry.MovedAt = movedAt
	if resolvedAt.Valid {
		entry.ResolvedAt = &resolvedAt.Time
	}
	if resolvedBy.Valid {
		entry.ResolvedBy = resolvedBy.String
	}

	return entry, nil
}

func scanDeadLetterSingleRow(row *sql.Row) (DeadLetterEntry, error) {
	var entry DeadLetterEntry
	var payloadJSON string
	var lastError sql.NullString
	var metadataJSON sql.NullString
	var groupID sql.NullString
	var movedAt string
	var resolvedAt sql.NullString
	var resolvedBy sql.NullString

	err := row.Scan(
		&entry.ID,
		&entry.OriginalJobID,
		&entry.BrainID,
		&payloadJSON,
		&entry.FailureReason,
		&lastError,
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

	return populateDeadLetterEntry(entry, payloadJSON, lastError, metadataJSON, groupID, movedAt, resolvedAt, resolvedBy)
}

func populateDeadLetterEntry(
	entry DeadLetterEntry,
	payloadJSON string,
	lastError sql.NullString,
	metadataJSON sql.NullString,
	groupID sql.NullString,
	movedAt string,
	resolvedAt sql.NullString,
	resolvedBy sql.NullString,
) (DeadLetterEntry, error) {
	if err := json.Unmarshal([]byte(payloadJSON), &entry.Payload); err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: unmarshal dead letter payload: %w", err)
	}

	if lastError.Valid {
		entry.LastError = lastError.String
	}
	if metadataJSON.Valid {
		meta := make(map[string]string)
		if unmarshalErr := json.Unmarshal([]byte(metadataJSON.String), &meta); unmarshalErr != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: unmarshal dead letter metadata: %w", unmarshalErr)
		}
		entry.Metadata = meta
	}
	if groupID.Valid {
		entry.GroupID = groupID.String
	}

	parsed, err := time.Parse(time.RFC3339, movedAt)
	if err != nil {
		return DeadLetterEntry{}, fmt.Errorf("ingest: parse dead letter moved_at: %w", err)
	}
	entry.MovedAt = parsed

	if resolvedAt.Valid {
		parsedResolved, parseErr := time.Parse(time.RFC3339, resolvedAt.String)
		if parseErr != nil {
			return DeadLetterEntry{}, fmt.Errorf("ingest: parse dead letter resolved_at: %w", parseErr)
		}
		entry.ResolvedAt = &parsedResolved
	}
	if resolvedBy.Valid {
		entry.ResolvedBy = resolvedBy.String
	}

	return entry, nil
}

// Compile-time interface assertions.
var _ DeadLetterAdapter = (*SqliteDeadLetterAdapter)(nil)
var _ DeadLetterAdapter = (*PostgresDeadLetterAdapter)(nil)
