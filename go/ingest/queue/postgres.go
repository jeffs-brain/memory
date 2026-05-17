// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"time"
)

// PostgresOptions configures the PostgreSQL queue adapter.
type PostgresOptions struct {
	// DB is the *sql.DB connection pool. Required.
	DB *sql.DB
	// Schema is the PostgreSQL schema name. Defaults to "public".
	Schema string
	// TableName overrides the queue table name. Defaults to "ingest_queue".
	TableName string
	// HeartbeatInterval controls how often the heartbeat goroutine
	// refreshes liveness for claimed jobs. Defaults to 30 seconds.
	HeartbeatInterval time.Duration
	// StaleThreshold is the duration after which a processing job
	// without a heartbeat is considered stale. Defaults to 5 minutes.
	StaleThreshold time.Duration
	// Logger receives structured log output. Defaults to silent.
	Logger Logger
	// NotifyChannel is the LISTEN/NOTIFY channel name.
	// Defaults to "ingest_queue_new_job".
	NotifyChannel string
}

// defaultHeartbeatInterval is the heartbeat refresh cadence.
const defaultHeartbeatInterval = 30 * time.Second

// defaultStaleThreshold is how long a heartbeat can be missing before
// the job is considered abandoned.
const defaultStaleThreshold = 5 * time.Minute

// defaultNotifyChannel is the PostgreSQL LISTEN channel.
const defaultNotifyChannel = "ingest_queue_new_job"

// backoffBaseDelay is the base delay for exponential retry backoff.
const backoffBaseDelay = 1 * time.Second

// backoffJitterMin is the lower jitter multiplier bound (inclusive).
const backoffJitterMin = 0.5

// backoffJitterMax is the upper jitter multiplier bound (exclusive).
const backoffJitterMax = 1.5

// PostgresQueue implements Adapter backed by a PostgreSQL table using
// FOR UPDATE SKIP LOCKED for safe multi-worker claiming.
type PostgresQueue struct {
	db            *sql.DB
	table         string
	schema        string
	heartbeatInt  time.Duration
	staleThresh   time.Duration
	log           Logger
	notifyChannel string

	mu       sync.Mutex
	closed   bool
	stopCh   chan struct{}
	activeWg sync.WaitGroup

	// claimedJobs tracks job IDs actively held by this adapter instance
	// so the heartbeat goroutine can refresh them.
	claimedMu   sync.Mutex
	claimedJobs map[string]struct{}
}

// NewPostgresQueue constructs a PostgreSQL-backed queue adapter.
// It validates that the connection is reachable and starts the
// heartbeat goroutine.
func NewPostgresQueue(opts PostgresOptions) (*PostgresQueue, error) {
	if opts.DB == nil {
		return nil, fmt.Errorf("ingest: queue requires a non-nil *sql.DB")
	}

	schema := opts.Schema
	if schema == "" {
		schema = "public"
	}
	table := opts.TableName
	if table == "" {
		table = "ingest_queue"
	}
	if err := validateIdentifier(schema); err != nil {
		return nil, fmt.Errorf("ingest: invalid schema name: %w", err)
	}
	if err := validateIdentifier(table); err != nil {
		return nil, fmt.Errorf("ingest: invalid table name: %w", err)
	}
	heartbeatInt := opts.HeartbeatInterval
	if heartbeatInt <= 0 {
		heartbeatInt = defaultHeartbeatInterval
	}
	staleThresh := opts.StaleThreshold
	if staleThresh <= 0 {
		staleThresh = defaultStaleThreshold
	}
	log := opts.Logger
	if log == nil {
		log = noopLogger{}
	}
	notifyCh := opts.NotifyChannel
	if notifyCh == "" {
		notifyCh = defaultNotifyChannel
	}

	q := &PostgresQueue{
		db:            opts.DB,
		table:         table,
		schema:        schema,
		heartbeatInt:  heartbeatInt,
		staleThresh:   staleThresh,
		log:           log,
		notifyChannel: notifyCh,
		stopCh:        make(chan struct{}),
		claimedJobs:   make(map[string]struct{}),
	}

	q.activeWg.Add(1)
	go q.heartbeatLoop()

	return q, nil
}

// qualifiedTable returns the schema-qualified table name.
func (q *PostgresQueue) qualifiedTable() string {
	return q.schema + "." + q.table
}

// Enqueue inserts a new job into the queue. If an idempotency key is
// provided and a matching active job exists, the existing job is returned.
func (q *PostgresQueue) Enqueue(ctx context.Context, input EnqueueInput) (Job, error) {
	if err := q.ensureOpen(); err != nil {
		return Job{}, err
	}

	maxRetries := input.MaxRetries
	if maxRetries <= 0 {
		maxRetries = defaultMaxRetries
	}

	payloadJSON, err := json.Marshal(input.Payload)
	if err != nil {
		return Job{}, fmt.Errorf("ingest: marshalling payload: %w", err)
	}

	var metaJSON []byte
	if len(input.Metadata) > 0 {
		metaJSON, err = json.Marshal(input.Metadata)
		if err != nil {
			return Job{}, fmt.Errorf("ingest: marshalling metadata: %w", err)
		}
	}

	tbl := q.qualifiedTable()

	// When an idempotency key is provided, try to find an existing active job first.
	if input.IdempotencyKey != "" {
		existing, findErr := q.findByIdempotencyKey(ctx, input.IdempotencyKey)
		if findErr != nil {
			return Job{}, findErr
		}
		if existing != nil {
			q.log.Debug("ingest: idempotent enqueue returned existing job",
				"job_id", existing.ID, "idempotency_key", input.IdempotencyKey)
			return *existing, nil
		}
	}

	query := fmt.Sprintf(`
		INSERT INTO %s (brain_id, status, payload, max_retries, metadata, group_id, idempotency_key)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING id, brain_id, status, payload, retry_count, max_retries, error,
			claimed_by, claimed_at, last_heartbeat, next_retry_at,
			created_at, updated_at, completed_at, metadata, group_id, idempotency_key`,
		tbl)

	var groupID, idempKey *string
	if input.GroupID != "" {
		groupID = &input.GroupID
	}
	if input.IdempotencyKey != "" {
		idempKey = &input.IdempotencyKey
	}

	row := q.db.QueryRowContext(ctx, query,
		input.BrainID,
		string(StatusPending),
		payloadJSON,
		maxRetries,
		nullableBytes(metaJSON),
		groupID,
		idempKey,
	)

	job, err := scanJob(row)
	if err != nil {
		// Handle idempotency constraint violation as a race condition:
		// another enqueue won the race, so look up the existing job.
		if strings.Contains(err.Error(), "idx_ingest_queue_idempotency") {
			if input.IdempotencyKey != "" {
				existing, findErr := q.findByIdempotencyKey(ctx, input.IdempotencyKey)
				if findErr != nil {
					return Job{}, findErr
				}
				if existing != nil {
					return *existing, nil
				}
			}
		}
		return Job{}, fmt.Errorf("ingest: enqueue failed: %w", err)
	}

	q.log.Info("ingest: job enqueued", "job_id", job.ID, "brain_id", job.BrainID)
	return job, nil
}

// Claim atomically locks pending jobs using FOR UPDATE SKIP LOCKED and
// assigns them to the specified worker. Advisory locks per brain prevent
// concurrent processing of the same brain.
func (q *PostgresQueue) Claim(ctx context.Context, opts ClaimOptions) ([]Job, error) {
	if err := q.ensureOpen(); err != nil {
		return nil, err
	}

	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = defaultBatchSize
	}
	if opts.WorkerID == "" {
		return nil, fmt.Errorf("ingest: claim requires a non-empty worker ID")
	}

	tbl := q.qualifiedTable()
	query := fmt.Sprintf(`
		UPDATE %s
		SET status = $1,
			claimed_by = $2,
			claimed_at = NOW(),
			last_heartbeat = NOW(),
			updated_at = NOW()
		WHERE id IN (
			SELECT id FROM %s
			WHERE status = 'pending'
				AND (next_retry_at IS NULL OR next_retry_at <= NOW())
			ORDER BY created_at ASC
			LIMIT $3
			FOR UPDATE SKIP LOCKED
		)
		RETURNING id, brain_id, status, payload, retry_count, max_retries, error,
			claimed_by, claimed_at, last_heartbeat, next_retry_at,
			created_at, updated_at, completed_at, metadata, group_id, idempotency_key`,
		tbl, tbl)

	rows, err := q.db.QueryContext(ctx, query,
		string(StatusProcessing),
		opts.WorkerID,
		batchSize,
	)
	if err != nil {
		return nil, fmt.Errorf("ingest: claim query failed: %w", err)
	}
	defer rows.Close()

	jobs, err := scanJobs(rows)
	if err != nil {
		return nil, fmt.Errorf("ingest: scanning claimed jobs: %w", err)
	}

	// Track claimed jobs for heartbeat refresh and attempt advisory locks.
	for i := range jobs {
		q.trackClaimed(jobs[i].ID)
		lockKey := advisoryLockKey(jobs[i].BrainID)
		q.tryAdvisoryLock(ctx, lockKey)
	}

	if len(jobs) > 0 {
		q.log.Info("ingest: claimed jobs",
			"count", len(jobs), "worker_id", opts.WorkerID)
	}

	return jobs, nil
}

// Heartbeat refreshes the liveness timestamp for a processing job.
func (q *PostgresQueue) Heartbeat(ctx context.Context, jobID string) error {
	if err := q.ensureOpen(); err != nil {
		return err
	}

	tbl := q.qualifiedTable()
	query := fmt.Sprintf(`
		UPDATE %s
		SET last_heartbeat = NOW(), updated_at = NOW()
		WHERE id = $1 AND status = $2`,
		tbl)

	result, err := q.db.ExecContext(ctx, query, jobID, string(StatusProcessing))
	if err != nil {
		return fmt.Errorf("ingest: heartbeat update failed: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("ingest: heartbeat rows affected: %w", err)
	}
	if affected == 0 {
		return fmt.Errorf("ingest: heartbeat found no processing job with id %s", jobID)
	}
	return nil
}

// Complete marks a job as successfully finished and releases its
// advisory lock.
func (q *PostgresQueue) Complete(ctx context.Context, jobID string, result map[string]string) error {
	if err := q.ensureOpen(); err != nil {
		return err
	}

	var resultJSON []byte
	var marshalErr error
	if len(result) > 0 {
		resultJSON, marshalErr = json.Marshal(result)
		if marshalErr != nil {
			return fmt.Errorf("ingest: marshalling result: %w", marshalErr)
		}
	}

	tbl := q.qualifiedTable()
	query := fmt.Sprintf(`
		UPDATE %s
		SET status = $1,
			completed_at = NOW(),
			updated_at = NOW(),
			metadata = COALESCE($3::jsonb, metadata)
		WHERE id = $2 AND status = $4
		RETURNING brain_id`,
		tbl)

	var brainID string
	err := q.db.QueryRowContext(ctx, query,
		string(StatusCompleted),
		jobID,
		nullableBytes(resultJSON),
		string(StatusProcessing),
	).Scan(&brainID)
	if err != nil {
		return fmt.Errorf("ingest: complete job %s: %w", jobID, err)
	}

	q.untrackClaimed(jobID)
	q.tryAdvisoryUnlock(ctx, advisoryLockKey(brainID))

	q.log.Info("ingest: job completed", "job_id", jobID, "brain_id", brainID)
	return nil
}

// Fail records an error against a job. When retryable is true and the
// retry ceiling has not been reached, the job returns to pending with
// exponential backoff and jitter. Otherwise the job moves to dead_letter.
func (q *PostgresQueue) Fail(ctx context.Context, jobID string, errMsg string, retryable bool) error {
	if err := q.ensureOpen(); err != nil {
		return err
	}

	tbl := q.qualifiedTable()

	// Fetch current state to decide next transition.
	fetchQuery := fmt.Sprintf(`
		SELECT retry_count, max_retries, brain_id
		FROM %s WHERE id = $1 AND status = $2
		FOR UPDATE`, tbl)

	tx, err := q.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("ingest: begin tx for fail: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	var retryCount, maxRetries int
	var brainID string
	err = tx.QueryRowContext(ctx, fetchQuery, jobID, string(StatusProcessing)).
		Scan(&retryCount, &maxRetries, &brainID)
	if err != nil {
		return fmt.Errorf("ingest: fail lookup job %s: %w", jobID, err)
	}

	newRetry := retryCount + 1
	canRetry := retryable && newRetry < maxRetries

	var nextStatus JobStatus
	var nextRetryAt *time.Time
	switch {
	case canRetry:
		nextStatus = StatusPending
		t := computeBackoff(newRetry)
		nextRetryAt = &t
	default:
		nextStatus = StatusDeadLetter
	}

	updateQuery := fmt.Sprintf(`
		UPDATE %s
		SET status = $1,
			retry_count = $2,
			error = $3,
			next_retry_at = $4,
			claimed_by = NULL,
			claimed_at = NULL,
			last_heartbeat = NULL,
			updated_at = NOW()
		WHERE id = $5`, tbl)

	_, err = tx.ExecContext(ctx, updateQuery,
		string(nextStatus),
		newRetry,
		errMsg,
		nextRetryAt,
		jobID,
	)
	if err != nil {
		return fmt.Errorf("ingest: fail update job %s: %w", jobID, err)
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("ingest: fail commit for job %s: %w", jobID, err)
	}

	q.untrackClaimed(jobID)
	q.tryAdvisoryUnlock(ctx, advisoryLockKey(brainID))

	q.log.Info("ingest: job failed",
		"job_id", jobID, "status", string(nextStatus),
		"retry_count", newRetry, "retryable", canRetry)

	return nil
}

// RecoverStale finds processing jobs whose last heartbeat is older than
// staleThreshold and resets them to pending. Returns the count recovered.
func (q *PostgresQueue) RecoverStale(ctx context.Context, staleThreshold time.Duration) (int, error) {
	if err := q.ensureOpen(); err != nil {
		return 0, err
	}

	tbl := q.qualifiedTable()
	query := fmt.Sprintf(`
		UPDATE %s
		SET status = $1,
			claimed_by = NULL,
			claimed_at = NULL,
			last_heartbeat = NULL,
			updated_at = NOW()
		WHERE status = $2
			AND last_heartbeat < NOW() - $3::interval`,
		tbl)

	intervalStr := fmt.Sprintf("%d seconds", int(staleThreshold.Seconds()))
	result, err := q.db.ExecContext(ctx, query,
		string(StatusPending),
		string(StatusProcessing),
		intervalStr,
	)
	if err != nil {
		return 0, fmt.Errorf("ingest: recover stale failed: %w", err)
	}

	count, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("ingest: recover stale rows affected: %w", err)
	}

	if count > 0 {
		q.log.Warn("ingest: recovered stale jobs", "count", count)
	}

	return int(count), nil
}

// CountByStatus returns the number of jobs grouped by status. When
// brainID is non-empty, results are filtered to that brain.
func (q *PostgresQueue) CountByStatus(ctx context.Context, brainID string) (map[JobStatus]int, error) {
	if err := q.ensureOpen(); err != nil {
		return nil, err
	}

	tbl := q.qualifiedTable()
	var query string
	var args []any

	switch {
	case brainID != "":
		query = fmt.Sprintf(`
			SELECT status, COUNT(*) FROM %s
			WHERE brain_id = $1
			GROUP BY status`, tbl)
		args = []any{brainID}
	default:
		query = fmt.Sprintf(`
			SELECT status, COUNT(*) FROM %s
			GROUP BY status`, tbl)
	}

	rows, err := q.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("ingest: count by status failed: %w", err)
	}
	defer rows.Close()

	counts := make(map[JobStatus]int)
	for rows.Next() {
		var status string
		var count int
		if err := rows.Scan(&status, &count); err != nil {
			return nil, fmt.Errorf("ingest: scanning count row: %w", err)
		}
		counts[JobStatus(status)] = count
	}
	return counts, rows.Err()
}

// Close stops the heartbeat goroutine and releases resources.
func (q *PostgresQueue) Close() error {
	q.mu.Lock()
	if q.closed {
		q.mu.Unlock()
		return nil
	}
	q.closed = true
	close(q.stopCh)
	q.mu.Unlock()

	q.activeWg.Wait()
	q.log.Info("ingest: queue adapter closed")
	return nil
}

// --- internal helpers ---

// ensureOpen returns an error if the adapter has been closed.
func (q *PostgresQueue) ensureOpen() error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return fmt.Errorf("ingest: queue adapter is closed")
	}
	return nil
}

// trackClaimed adds a job ID to the set of claimed jobs that the
// heartbeat goroutine refreshes.
func (q *PostgresQueue) trackClaimed(jobID string) {
	q.claimedMu.Lock()
	q.claimedJobs[jobID] = struct{}{}
	q.claimedMu.Unlock()
}

// untrackClaimed removes a job ID from heartbeat tracking.
func (q *PostgresQueue) untrackClaimed(jobID string) {
	q.claimedMu.Lock()
	delete(q.claimedJobs, jobID)
	q.claimedMu.Unlock()
}

// claimedIDs returns a snapshot of currently claimed job IDs.
func (q *PostgresQueue) claimedIDs() []string {
	q.claimedMu.Lock()
	defer q.claimedMu.Unlock()
	ids := make([]string, 0, len(q.claimedJobs))
	for id := range q.claimedJobs {
		ids = append(ids, id)
	}
	return ids
}

// heartbeatLoop periodically refreshes the heartbeat for all claimed
// jobs until the adapter is closed.
func (q *PostgresQueue) heartbeatLoop() {
	defer q.activeWg.Done()
	ticker := time.NewTicker(q.heartbeatInt)
	defer ticker.Stop()

	for {
		select {
		case <-q.stopCh:
			return
		case <-ticker.C:
			ids := q.claimedIDs()
			for _, id := range ids {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				if err := q.Heartbeat(ctx, id); err != nil {
					q.log.Warn("ingest: heartbeat refresh failed",
						"job_id", id, "error", err.Error())
				}
				cancel()
			}
		}
	}
}

// findByIdempotencyKey looks up an active job with the given key.
func (q *PostgresQueue) findByIdempotencyKey(ctx context.Context, key string) (*Job, error) {
	tbl := q.qualifiedTable()
	query := fmt.Sprintf(`
		SELECT id, brain_id, status, payload, retry_count, max_retries, error,
			claimed_by, claimed_at, last_heartbeat, next_retry_at,
			created_at, updated_at, completed_at, metadata, group_id, idempotency_key
		FROM %s
		WHERE idempotency_key = $1
			AND status NOT IN ('dead_letter', 'completed', 'failed')
		LIMIT 1`, tbl)

	row := q.db.QueryRowContext(ctx, query, key)
	job, err := scanJob(row)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("ingest: idempotency lookup: %w", err)
	}
	return &job, nil
}

// tryAdvisoryLock attempts to acquire a session-level advisory lock for
// the given key. Non-blocking; returns silently on failure.
func (q *PostgresQueue) tryAdvisoryLock(ctx context.Context, key int64) {
	_, err := q.db.ExecContext(ctx, "SELECT pg_try_advisory_lock($1)", key)
	if err != nil {
		q.log.Debug("ingest: advisory lock acquisition failed", "key", key, "error", err.Error())
	}
}

// tryAdvisoryUnlock releases a session-level advisory lock.
func (q *PostgresQueue) tryAdvisoryUnlock(ctx context.Context, key int64) {
	_, err := q.db.ExecContext(ctx, "SELECT pg_advisory_unlock($1)", key)
	if err != nil {
		q.log.Debug("ingest: advisory lock release failed", "key", key, "error", err.Error())
	}
}

// advisoryLockKey computes a stable int64 lock key from a brain ID
// using FNV-1a hashing.
func advisoryLockKey(brainID string) int64 {
	h := fnv.New64a()
	h.Write([]byte(brainID))
	return int64(h.Sum64())
}

// computeBackoff calculates the next retry time using exponential
// backoff with jitter: baseDelay * 2^retryCount * random(0.5, 1.5).
func computeBackoff(retryCount int) time.Time {
	multiplier := math.Pow(2, float64(retryCount))
	jitter := backoffJitterMin + rand.Float64()*(backoffJitterMax-backoffJitterMin)
	delay := time.Duration(float64(backoffBaseDelay) * multiplier * jitter)
	return time.Now().Add(delay)
}

// validateIdentifier rejects SQL identifiers that contain characters
// outside the safe set [a-zA-Z0-9_]. This is a path traversal defence
// for schema and table name injection.
func validateIdentifier(s string) error {
	if s == "" {
		return fmt.Errorf("identifier must not be empty")
	}
	for _, c := range s {
		valid := (c >= 'a' && c <= 'z') ||
			(c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') ||
			c == '_'
		if !valid {
			return fmt.Errorf("identifier contains invalid character %q", c)
		}
	}
	return nil
}

// nullableBytes returns nil when b is empty, otherwise returns b.
// Used for optional JSONB columns.
func nullableBytes(b []byte) *[]byte {
	if len(b) == 0 {
		return nil
	}
	return &b
}

// scanJob reads a single job row from a *sql.Row.
type rowScanner interface {
	Scan(dest ...any) error
}

func scanJob(row rowScanner) (Job, error) {
	var j Job
	var payloadJSON, metadataJSON []byte
	var status string
	var errStr, claimedBy, groupID, idempKey sql.NullString
	var claimedAt, lastHB, nextRetry, completedAt sql.NullTime

	err := row.Scan(
		&j.ID,
		&j.BrainID,
		&status,
		&payloadJSON,
		&j.RetryCount,
		&j.MaxRetries,
		&errStr,
		&claimedBy,
		&claimedAt,
		&lastHB,
		&nextRetry,
		&j.CreatedAt,
		&j.UpdatedAt,
		&completedAt,
		&metadataJSON,
		&groupID,
		&idempKey,
	)
	if err != nil {
		return Job{}, err
	}

	j.Status = JobStatus(status)
	if errStr.Valid {
		j.Error = errStr.String
	}
	if claimedBy.Valid {
		j.ClaimedBy = claimedBy.String
	}
	if claimedAt.Valid {
		t := claimedAt.Time
		j.ClaimedAt = &t
	}
	if lastHB.Valid {
		t := lastHB.Time
		j.LastHeartbeat = &t
	}
	if nextRetry.Valid {
		t := nextRetry.Time
		j.NextRetryAt = &t
	}
	if completedAt.Valid {
		t := completedAt.Time
		j.CompletedAt = &t
	}
	if groupID.Valid {
		j.GroupID = groupID.String
	}
	if idempKey.Valid {
		j.IdempotencyKey = idempKey.String
	}

	if err := json.Unmarshal(payloadJSON, &j.Payload); err != nil {
		return Job{}, fmt.Errorf("ingest: unmarshalling payload: %w", err)
	}
	if len(metadataJSON) > 0 {
		j.Metadata = make(map[string]string)
		if err := json.Unmarshal(metadataJSON, &j.Metadata); err != nil {
			return Job{}, fmt.Errorf("ingest: unmarshalling metadata: %w", err)
		}
	}

	return j, nil
}

// scanJobs reads multiple job rows from *sql.Rows.
func scanJobs(rows *sql.Rows) ([]Job, error) {
	var jobs []Job
	for rows.Next() {
		j, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}
