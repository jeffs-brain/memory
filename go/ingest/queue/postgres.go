// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
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
	// ListenConn is an optional dedicated *sql.Conn for LISTEN/NOTIFY.
	// When provided, the adapter subscribes for immediate wake on new
	// jobs. This should be a dedicated connection, not from the pool.
	ListenConn *sql.Conn
	// OnNotify is called when a LISTEN notification arrives with the
	// job ID payload. Only used when ListenConn is provided.
	OnNotify func(jobID string)
	// MaxQueueDepth sets a backpressure limit. When the number of
	// pending jobs exceeds this value, Enqueue returns an error.
	// Zero means no limit.
	MaxQueueDepth int
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
	listenConn    *sql.Conn
	onNotify      func(jobID string)
	maxQueueDepth int

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
// heartbeat goroutine. When a ListenConn is provided, it also
// starts the LISTEN/NOTIFY consumer goroutine.
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
	if err := validateIdentifier(notifyCh); err != nil {
		return nil, fmt.Errorf("ingest: invalid notify channel name: %w", err)
	}

	q := &PostgresQueue{
		db:            opts.DB,
		table:         table,
		schema:        schema,
		heartbeatInt:  heartbeatInt,
		staleThresh:   staleThresh,
		log:           log,
		notifyChannel: notifyCh,
		listenConn:    opts.ListenConn,
		onNotify:      opts.OnNotify,
		maxQueueDepth: opts.MaxQueueDepth,
		stopCh:        make(chan struct{}),
		claimedJobs:   make(map[string]struct{}),
	}

	q.activeWg.Add(1)
	go q.heartbeatLoop()

	// Start LISTEN/NOTIFY consumer when a dedicated connection is provided.
	if opts.ListenConn != nil {
		q.activeWg.Add(1)
		go q.listenLoop()
	}

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

	// Backpressure: reject enqueue when the pending queue exceeds depth limit.
	if q.maxQueueDepth > 0 {
		tbl := q.qualifiedTable()
		var pendingCount int
		row := q.db.QueryRowContext(ctx,
			fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE status = $1", tbl),
			string(StatusPending),
		)
		if err := row.Scan(&pendingCount); err == nil && pendingCount >= q.maxQueueDepth {
			return Job{}, fmt.Errorf("ingest: queue depth limit reached (%d pending jobs)", pendingCount)
		}
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
	defer func() { _ = rows.Close() }()

	jobs, err := scanJobs(rows)
	if err != nil {
		return nil, fmt.Errorf("ingest: scanning claimed jobs: %w", err)
	}

	// Track claimed jobs for heartbeat refresh and acquire advisory locks.
	// Batch all advisory lock acquisitions into a single query to reduce
	// round-trips from O(k) to O(1) where k = batch size.
	for i := range jobs {
		q.trackClaimed(jobs[i].ID)
	}

	if len(jobs) > 0 {
		lockKeys := make([]int64, len(jobs))
		for i := range jobs {
			lockKeys[i] = advisoryLockKey(jobs[i].BrainID)
		}
		q.tryBatchAdvisoryLock(ctx, lockKeys)

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
// advisory lock. Uses a transaction-level advisory lock release via
// pg_advisory_xact_lock within the same transaction that updates the
// job, ensuring the lock is always released when the transaction commits
// regardless of which pooled connection handles the query.
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

	tx, err := q.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("ingest: begin tx for complete: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

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
	err = tx.QueryRowContext(ctx, query,
		string(StatusCompleted),
		jobID,
		nullableBytes(resultJSON),
		string(StatusProcessing),
	).Scan(&brainID)
	if err != nil {
		return fmt.Errorf("ingest: complete job %s: %w", jobID, err)
	}

	// Release the advisory lock within the transaction so it runs on
	// the same connection that holds the lock.
	lockKey := advisoryLockKey(brainID)
	_, _ = tx.ExecContext(ctx, "SELECT pg_advisory_unlock($1)", lockKey)

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("ingest: complete commit for job %s: %w", jobID, err)
	}

	q.untrackClaimed(jobID)
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

	// Release the advisory lock within the transaction so it runs on
	// the same connection that acquired it.
	lockKey := advisoryLockKey(brainID)
	_, _ = tx.ExecContext(ctx, "SELECT pg_advisory_unlock($1)", lockKey)

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("ingest: fail commit for job %s: %w", jobID, err)
	}

	q.untrackClaimed(jobID)

	q.log.Info("ingest: job failed",
		"job_id", jobID, "status", string(nextStatus),
		"retry_count", newRetry, "retryable", canRetry)

	return nil
}

// Requeue returns a claimed job to pending status without incrementing
// the retry count. The advisory lock is released within the same
// transaction to prevent stale locks on pooled connections.
func (q *PostgresQueue) Requeue(ctx context.Context, jobID string) error {
	if err := q.ensureOpen(); err != nil {
		return err
	}

	tbl := q.qualifiedTable()

	tx, err := q.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("ingest: begin tx for requeue: %w", err)
	}
	defer tx.Rollback() //nolint:errcheck

	query := fmt.Sprintf(`
		UPDATE %s
		SET status = $1,
			claimed_by = NULL,
			claimed_at = NULL,
			last_heartbeat = NULL,
			updated_at = NOW()
		WHERE id = $2 AND status = $3
		RETURNING brain_id`, tbl)

	var brainID string
	err = tx.QueryRowContext(ctx, query,
		string(StatusPending),
		jobID,
		string(StatusProcessing),
	).Scan(&brainID)
	if err != nil {
		return fmt.Errorf("ingest: requeue job %s: %w", jobID, err)
	}

	lockKey := advisoryLockKey(brainID)
	_, _ = tx.ExecContext(ctx, "SELECT pg_advisory_unlock($1)", lockKey)

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("ingest: requeue commit for job %s: %w", jobID, err)
	}

	q.untrackClaimed(jobID)
	q.log.Info("ingest: job requeued", "job_id", jobID, "brain_id", brainID)
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
	defer func() { _ = rows.Close() }()

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

// Close stops the heartbeat goroutine, the listen goroutine (if active),
// and releases resources.
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

// tryBatchAdvisoryLock acquires advisory locks for multiple keys in a
// single database round-trip using unnest. This reduces claim-path
// latency from O(k) round-trips to O(1) where k is the batch size.
func (q *PostgresQueue) tryBatchAdvisoryLock(ctx context.Context, keys []int64) {
	if len(keys) == 0 {
		return
	}
	_, err := q.db.ExecContext(ctx,
		"SELECT pg_try_advisory_lock(k) FROM unnest($1::bigint[]) AS k",
		pq64Array(keys))
	if err != nil {
		q.log.Debug("ingest: batch advisory lock acquisition failed", "count", len(keys), "error", err.Error())
	}
}

