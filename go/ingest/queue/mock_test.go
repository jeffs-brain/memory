// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// mockDB implements database/sql/driver interfaces to simulate
// PostgreSQL operations in-memory for adapter-level unit tests.

// --- mock job store ---

type mockJob struct {
	id             string
	brainID        string
	status         string
	payload        string
	retryCount     int
	maxRetries     int
	errMsg         *string
	claimedBy      *string
	claimedAt      *time.Time
	lastHeartbeat  *time.Time
	nextRetryAt    *time.Time
	createdAt      time.Time
	updatedAt      time.Time
	completedAt    *time.Time
	metadata       *string
	groupID        *string
	idempotencyKey *string
}

type mockStore struct {
	mu        sync.Mutex
	jobs      []*mockJob
	idCounter int
}

func newMockStore() *mockStore {
	return &mockStore{jobs: make([]*mockJob, 0)}
}

func (s *mockStore) nextID() string {
	s.idCounter++
	return fmt.Sprintf("mock-%d", s.idCounter)
}

// --- mock driver ---

type mockDriver struct {
	store *mockStore
}

type mockConnector struct {
	driver *mockDriver
}

func (c *mockConnector) Connect(_ context.Context) (driver.Conn, error) {
	return &mockConn{store: c.driver.store}, nil
}

func (c *mockConnector) Driver() driver.Driver {
	return c.driver
}

func (d *mockDriver) Open(_ string) (driver.Conn, error) {
	return &mockConn{store: d.store}, nil
}

type mockConn struct {
	store *mockStore
}

func (c *mockConn) Prepare(query string) (driver.Stmt, error) {
	return &mockStmt{store: c.store, query: query}, nil
}

func (c *mockConn) Close() error { return nil }

func (c *mockConn) Begin() (driver.Tx, error) {
	return &mockTx{store: c.store}, nil
}

type mockTx struct {
	store *mockStore
}

func (tx *mockTx) Commit() error   { return nil }
func (tx *mockTx) Rollback() error { return nil }

type mockStmt struct {
	store *mockStore
	query string
}

func (s *mockStmt) Close() error { return nil }

func (s *mockStmt) NumInput() int { return -1 }

func (s *mockStmt) Exec(args []driver.Value) (driver.Result, error) {
	return s.execMock(args)
}

func (s *mockStmt) Query(args []driver.Value) (driver.Rows, error) {
	return s.queryMock(args)
}

func (s *mockStmt) execMock(args []driver.Value) (driver.Result, error) {
	normalised := normaliseMockQuery(s.query)

	s.store.mu.Lock()
	defer s.store.mu.Unlock()

	// Advisory lock operations (no-op) -- includes batch unnest locks
	if strings.Contains(normalised, "pg_try_advisory_lock") ||
		strings.Contains(normalised, "pg_advisory_unlock") {
		return &mockExecResult{affected: 0}, nil
	}

	// Heartbeat update (single)
	if strings.Contains(normalised, "last_heartbeat = now()") &&
		strings.Contains(normalised, "where id =") &&
		!strings.Contains(normalised, "in (") {
		jobID := mockValStr(args, 0)
		status := mockValStr(args, 1)
		now := time.Now()
		var affected int64
		for _, j := range s.store.jobs {
			if j.id == jobID && j.status == status {
				j.lastHeartbeat = &now
				j.updatedAt = now
				affected++
			}
		}
		return &mockExecResult{affected: affected}, nil
	}

	// Bulk heartbeat (ANY($1::uuid[]))
	if strings.Contains(normalised, "last_heartbeat = now()") &&
		strings.Contains(normalised, "any($1") {
		now := time.Now()
		// First arg is the array literal, second arg is the status.
		arrayStr := mockValStr(args, 0)
		status := mockValStr(args, 1)
		// Parse the PostgreSQL array literal: {"id1","id2",...}
		jobIDs := parsePgArray(arrayStr)
		var affected int64
		for _, jobID := range jobIDs {
			for _, j := range s.store.jobs {
				if j.id == jobID && j.status == status {
					j.lastHeartbeat = &now
					j.updatedAt = now
					affected++
				}
			}
		}
		return &mockExecResult{affected: affected}, nil
	}

	// Recover stale
	if strings.Contains(normalised, "last_heartbeat <") {
		thresholdStr := mockValStr(args, 2)
		seconds := 0
		fmt.Sscanf(thresholdStr, "%d seconds", &seconds)
		cutoff := time.Now().Add(-time.Duration(seconds) * time.Second)
		var affected int64
		for _, j := range s.store.jobs {
			if j.status == "processing" && j.lastHeartbeat != nil && j.lastHeartbeat.Before(cutoff) {
				j.status = "pending"
				j.claimedBy = nil
				j.claimedAt = nil
				j.lastHeartbeat = nil
				j.updatedAt = time.Now()
				affected++
			}
		}
		return &mockExecResult{affected: affected}, nil
	}

	// Fail update
	if strings.Contains(normalised, "retry_count = $") &&
		strings.Contains(normalised, "error = $") {
		newStatus := mockValStr(args, 0)
		retryCount := mockValInt(args, 1)
		errMsg := mockValStr(args, 2)
		jobID := mockValStr(args, 4)
		now := time.Now()
		for _, j := range s.store.jobs {
			if j.id == jobID {
				j.status = newStatus
				j.retryCount = retryCount
				j.errMsg = &errMsg
				j.claimedBy = nil
				j.claimedAt = nil
				j.lastHeartbeat = nil
				j.updatedAt = now
				if args[3] != nil {
					t := args[3].(time.Time)
					j.nextRetryAt = &t
				} else {
					j.nextRetryAt = nil
				}
			}
		}
		return &mockExecResult{affected: 1}, nil
	}

	// LISTEN (no-op)
	if strings.HasPrefix(normalised, "listen ") {
		return &mockExecResult{affected: 0}, nil
	}

	// SELECT 1 (no-op for poll)
	if normalised == "select 1" {
		return &mockExecResult{affected: 0}, nil
	}

	return &mockExecResult{affected: 0}, nil
}

func (s *mockStmt) queryMock(args []driver.Value) (driver.Rows, error) {
	normalised := normaliseMockQuery(s.query)

	s.store.mu.Lock()
	defer s.store.mu.Unlock()

	// INSERT (enqueue)
	if strings.Contains(normalised, "insert into") {
		now := time.Now()
		id := s.store.nextID()

		newJob := &mockJob{
			id:         id,
			brainID:    mockValStr(args, 0),
			status:     mockValStr(args, 1),
			payload:    mockValStr(args, 2),
			maxRetries: mockValInt(args, 3),
			createdAt:  now,
			updatedAt:  now,
		}
		if args[4] != nil {
			v := mockValStr(args, 4)
			newJob.metadata = &v
		}
		if args[5] != nil {
			v := mockValStr(args, 5)
			newJob.groupID = &v
		}
		if args[6] != nil {
			v := mockValStr(args, 6)
			newJob.idempotencyKey = &v

			// Check idempotency constraint
			for _, j := range s.store.jobs {
				if j.idempotencyKey != nil && *j.idempotencyKey == v &&
					j.status != "dead_letter" && j.status != "completed" && j.status != "failed" {
					return nil, fmt.Errorf("idx_ingest_queue_idempotency")
				}
			}
		}

		s.store.jobs = append(s.store.jobs, newJob)
		return mockJobToRows([]*mockJob{newJob}), nil
	}

	// SELECT for idempotency lookup
	if strings.Contains(normalised, "idempotency_key = $1") &&
		strings.Contains(normalised, "select") {
		key := mockValStr(args, 0)
		for _, j := range s.store.jobs {
			if j.idempotencyKey != nil && *j.idempotencyKey == key &&
				j.status != "dead_letter" && j.status != "completed" && j.status != "failed" {
				return mockJobToRows([]*mockJob{j}), nil
			}
		}
		return mockJobToRows(nil), nil
	}

	// SELECT retry_count (fail lookup)
	if strings.Contains(normalised, "select retry_count") {
		jobID := mockValStr(args, 0)
		status := mockValStr(args, 1)
		for _, j := range s.store.jobs {
			if j.id == jobID && j.status == status {
				cols := []string{"retry_count", "max_retries", "brain_id"}
				return &mockDriverRows{
					cols: cols,
					data: [][]driver.Value{
						{int64(j.retryCount), int64(j.maxRetries), j.brainID},
					},
				}, nil
			}
		}
		return &mockDriverRows{
			cols: []string{"retry_count", "max_retries", "brain_id"},
			data: [][]driver.Value{},
		}, nil
	}

	// UPDATE ... FOR UPDATE SKIP LOCKED (claim)
	if strings.Contains(normalised, "for update skip locked") {
		workerID := mockValStr(args, 1)
		batchSize := mockValInt(args, 2)
		now := time.Now()

		var pending []*mockJob
		for _, j := range s.store.jobs {
			if j.status == "pending" && (j.nextRetryAt == nil || !j.nextRetryAt.After(now)) {
				pending = append(pending, j)
			}
		}
		if len(pending) > batchSize {
			pending = pending[:batchSize]
		}

		for _, j := range pending {
			j.status = "processing"
			j.claimedBy = &workerID
			j.claimedAt = &now
			j.lastHeartbeat = &now
			j.updatedAt = now
		}

		return mockJobToRows(pending), nil
	}

	// UPDATE requeue (returns to pending without incrementing retry count)
	if strings.Contains(normalised, "claimed_by = null") &&
		strings.Contains(normalised, "returning brain_id") &&
		!strings.Contains(normalised, "completed_at") &&
		!strings.Contains(normalised, "retry_count") {
		newStatus := mockValStr(args, 0)
		jobID := mockValStr(args, 1)
		oldStatus := mockValStr(args, 2)
		now := time.Now()
		for _, j := range s.store.jobs {
			if j.id == jobID && j.status == oldStatus {
				j.status = newStatus
				j.claimedBy = nil
				j.claimedAt = nil
				j.lastHeartbeat = nil
				j.updatedAt = now
				cols := []string{"brain_id"}
				return &mockDriverRows{
					cols: cols,
					data: [][]driver.Value{{j.brainID}},
				}, nil
			}
		}
		return &mockDriverRows{
			cols: []string{"brain_id"},
			data: [][]driver.Value{},
		}, nil
	}

	// UPDATE complete
	if strings.Contains(normalised, "completed_at = now()") {
		jobID := mockValStr(args, 1)
		status := mockValStr(args, 3)
		now := time.Now()
		for _, j := range s.store.jobs {
			if j.id == jobID && j.status == status {
				j.status = "completed"
				j.completedAt = &now
				j.updatedAt = now
				cols := []string{"brain_id"}
				return &mockDriverRows{
					cols: cols,
					data: [][]driver.Value{{j.brainID}},
				}, nil
			}
		}
		return &mockDriverRows{
			cols: []string{"brain_id"},
			data: [][]driver.Value{},
		}, nil
	}

	// COUNT by status
	if strings.Contains(normalised, "count(*)") {
		counts := map[string]int{}
		brainFilter := ""
		if len(args) > 0 {
			brainFilter = mockValStr(args, 0)
		}
		for _, j := range s.store.jobs {
			if brainFilter != "" && j.brainID != brainFilter {
				continue
			}
			counts[j.status]++
		}
		var data [][]driver.Value
		for status, count := range counts {
			data = append(data, []driver.Value{status, int64(count)})
		}
		return &mockDriverRows{
			cols: []string{"status", "count"},
			data: data,
		}, nil
	}

	return &mockDriverRows{cols: []string{}, data: [][]driver.Value{}}, nil
}

// --- helper functions ---

func normaliseMockQuery(s string) string {
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

func mockValStr(args []driver.Value, i int) string {
	if i >= len(args) || args[i] == nil {
		return ""
	}
	switch v := args[i].(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func mockValInt(args []driver.Value, i int) int {
	if i >= len(args) || args[i] == nil {
		return 0
	}
	switch v := args[i].(type) {
	case int64:
		return int(v)
	case int:
		return v
	case float64:
		return int(v)
	default:
		return 0
	}
}

// --- mock rows ---

func mockJobToRows(jobs []*mockJob) *mockDriverRows {
	cols := []string{
		"id", "brain_id", "status", "payload", "retry_count", "max_retries",
		"error", "claimed_by", "claimed_at", "last_heartbeat", "next_retry_at",
		"created_at", "updated_at", "completed_at", "metadata", "group_id",
		"idempotency_key",
	}
	var data [][]driver.Value
	for _, j := range jobs {
		row := []driver.Value{
			j.id,
			j.brainID,
			j.status,
			[]byte(j.payload),
			int64(j.retryCount),
			int64(j.maxRetries),
			mockNullStr(j.errMsg),
			mockNullStr(j.claimedBy),
			mockNullTime(j.claimedAt),
			mockNullTime(j.lastHeartbeat),
			mockNullTime(j.nextRetryAt),
			j.createdAt,
			j.updatedAt,
			mockNullTime(j.completedAt),
			mockNullBytesFromStr(j.metadata),
			mockNullStr(j.groupID),
			mockNullStr(j.idempotencyKey),
		}
		data = append(data, row)
	}
	return &mockDriverRows{cols: cols, data: data}
}

func mockNullStr(s *string) driver.Value {
	if s == nil {
		return nil
	}
	return *s
}

func mockNullTime(t *time.Time) driver.Value {
	if t == nil {
		return nil
	}
	return *t
}

func mockNullBytesFromStr(s *string) driver.Value {
	if s == nil {
		return nil
	}
	return []byte(*s)
}

type mockDriverRows struct {
	cols    []string
	data    [][]driver.Value
	current int
}

func (r *mockDriverRows) Columns() []string { return r.cols }

func (r *mockDriverRows) Close() error { return nil }

func (r *mockDriverRows) Next(dest []driver.Value) error {
	if r.current >= len(r.data) {
		return io.EOF
	}
	copy(dest, r.data[r.current])
	r.current++
	return nil
}

type mockExecResult struct {
	affected int64
}

func (r *mockExecResult) LastInsertId() (int64, error) { return 0, nil }
func (r *mockExecResult) RowsAffected() (int64, error) { return r.affected, nil }

// parsePgArray extracts string elements from a PostgreSQL array literal
// such as {"id-1","id-2"}. Returns nil for empty arrays.
func parsePgArray(s string) []string {
	s = strings.TrimPrefix(s, "{")
	s = strings.TrimSuffix(s, "}")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	result := make([]string, 0, len(parts))
	for _, p := range parts {
		result = append(result, strings.Trim(p, `"`))
	}
	return result
}

// newMockDB creates a mock *sql.DB backed by an in-memory store.
func newMockDB() (*sql.DB, *mockStore) {
	store := newMockStore()
	d := &mockDriver{store: store}
	db := sql.OpenDB(&mockConnector{driver: d})
	return db, store
}
