// SPDX-License-Identifier: Apache-2.0
package schedule

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

const createTableSQL = `
CREATE TABLE IF NOT EXISTS ingest_schedules (
  id TEXT PRIMARY KEY,
  brain_id TEXT NOT NULL,
  name TEXT NOT NULL,
  cron_expression TEXT NOT NULL,
  target_kind TEXT NOT NULL,
  target_url TEXT,
  target_path TEXT,
  target_glob TEXT,
  enabled INTEGER NOT NULL DEFAULT 1,
  last_run_at TEXT,
  next_run_at TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_schedules_brain ON ingest_schedules(brain_id);
CREATE INDEX IF NOT EXISTS idx_schedules_due ON ingest_schedules(enabled, next_run_at);
`

// SQLiteStore implements Store using SQLite.
type SQLiteStore struct {
	db *sql.DB
}

// NewSQLiteStore creates a new SQLite-backed schedule store and ensures
// the table exists.
func NewSQLiteStore(db *sql.DB) (*SQLiteStore, error) {
	if _, err := db.Exec(createTableSQL); err != nil {
		return nil, fmt.Errorf("schedule: create table: %w", err)
	}
	return &SQLiteStore{db: db}, nil
}

func (s *SQLiteStore) Create(ctx context.Context, input CreateInput) (Job, error) {
	if !IsValid(input.CronExpression) {
		return Job{}, fmt.Errorf("schedule: invalid cron expression: %q", input.CronExpression)
	}

	sched, _ := ParseCron(input.CronExpression)
	now := time.Now().UTC()
	nextRun := NextOccurrence(sched, now)

	id := uuid.New().String()
	var metadataJSON *string
	if input.Metadata != nil {
		data, marshalErr := json.Marshal(input.Metadata)
		if marshalErr != nil {
			return Job{}, fmt.Errorf("schedule: marshal metadata: %w", marshalErr)
		}
		str := string(data)
		metadataJSON = &str
	}

	_, err := s.db.ExecContext(ctx,
		`INSERT INTO ingest_schedules (id, brain_id, name, cron_expression, target_kind, target_url, target_path, target_glob, enabled, next_run_at, created_at, updated_at, metadata)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)`,
		id, input.BrainID, input.Name, input.CronExpression,
		input.Target.Kind, nullStr(input.Target.URL), nullStr(input.Target.Path), nullStr(input.Target.Glob),
		nextRun.Format(time.RFC3339), now.Format(time.RFC3339), now.Format(time.RFC3339),
		metadataJSON,
	)
	if err != nil {
		return Job{}, fmt.Errorf("schedule: insert: %w", err)
	}

	return Job{
		ID:             id,
		BrainID:        input.BrainID,
		Name:           input.Name,
		CronExpression: input.CronExpression,
		Target:         input.Target,
		Enabled:        true,
		NextRunAt:      nextRun,
		CreatedAt:      now,
		UpdatedAt:      now,
		Metadata:       input.Metadata,
	}, nil
}

func (s *SQLiteStore) Get(ctx context.Context, id string) (Job, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, brain_id, name, cron_expression, target_kind, target_url, target_path, target_glob, enabled, last_run_at, next_run_at, created_at, updated_at, metadata
		 FROM ingest_schedules WHERE id = ?`, id)
	return scanJob(row)
}

func (s *SQLiteStore) List(ctx context.Context, brainID string) ([]Job, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, brain_id, name, cron_expression, target_kind, target_url, target_path, target_glob, enabled, last_run_at, next_run_at, created_at, updated_at, metadata
		 FROM ingest_schedules WHERE brain_id = ? ORDER BY name`, brainID)
	if err != nil {
		return nil, fmt.Errorf("schedule: list: %w", err)
	}
	defer rows.Close()

	var jobs []Job
	for rows.Next() {
		job, err := scanJobRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, job)
	}
	return jobs, rows.Err()
}

func (s *SQLiteStore) Update(ctx context.Context, id string, patch UpdatePatch) (Job, error) {
	job, err := s.Get(ctx, id)
	if err != nil {
		return Job{}, err
	}

	if patch.Name != nil {
		job.Name = *patch.Name
	}
	if patch.CronExpression != nil {
		if !IsValid(*patch.CronExpression) {
			return Job{}, fmt.Errorf("schedule: invalid cron expression: %q", *patch.CronExpression)
		}
		job.CronExpression = *patch.CronExpression
		sched, _ := ParseCron(job.CronExpression)
		job.NextRunAt = NextOccurrence(sched, time.Now().UTC())
	}
	if patch.Target != nil {
		job.Target = *patch.Target
	}
	if patch.Enabled != nil {
		job.Enabled = *patch.Enabled
	}
	if patch.Metadata != nil {
		job.Metadata = *patch.Metadata
	}

	now := time.Now().UTC()
	job.UpdatedAt = now

	var metadataJSON *string
	if job.Metadata != nil {
		data, marshalErr := json.Marshal(job.Metadata)
		if marshalErr != nil {
			return Job{}, fmt.Errorf("schedule: marshal metadata: %w", marshalErr)
		}
		str := string(data)
		metadataJSON = &str
	}

	_, err = s.db.ExecContext(ctx,
		`UPDATE ingest_schedules SET name=?, cron_expression=?, target_kind=?, target_url=?, target_path=?, target_glob=?, enabled=?, next_run_at=?, updated_at=?, metadata=? WHERE id=?`,
		job.Name, job.CronExpression, job.Target.Kind,
		nullStr(job.Target.URL), nullStr(job.Target.Path), nullStr(job.Target.Glob),
		boolToInt(job.Enabled), job.NextRunAt.Format(time.RFC3339), now.Format(time.RFC3339),
		metadataJSON, id,
	)
	if err != nil {
		return Job{}, fmt.Errorf("schedule: update: %w", err)
	}

	return job, nil
}

func (s *SQLiteStore) Delete(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM ingest_schedules WHERE id = ?`, id)
	if err != nil {
		return fmt.Errorf("schedule: delete: %w", err)
	}
	return nil
}

func (s *SQLiteStore) FindDue(ctx context.Context, now time.Time) ([]Job, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, brain_id, name, cron_expression, target_kind, target_url, target_path, target_glob, enabled, last_run_at, next_run_at, created_at, updated_at, metadata
		 FROM ingest_schedules WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at`,
		now.Format(time.RFC3339))
	if err != nil {
		return nil, fmt.Errorf("schedule: findDue: %w", err)
	}
	defer rows.Close()

	var jobs []Job
	for rows.Next() {
		job, err := scanJobRows(rows)
		if err != nil {
			return nil, err
		}
		jobs = append(jobs, job)
	}
	return jobs, rows.Err()
}

func (s *SQLiteStore) MarkRun(ctx context.Context, id string, ranAt, nextRunAt time.Time) error {
	now := time.Now().UTC()
	_, err := s.db.ExecContext(ctx,
		`UPDATE ingest_schedules SET last_run_at=?, next_run_at=?, updated_at=? WHERE id=?`,
		ranAt.Format(time.RFC3339), nextRunAt.Format(time.RFC3339), now.Format(time.RFC3339), id)
	if err != nil {
		return fmt.Errorf("schedule: markRun: %w", err)
	}
	return nil
}

// scanner abstracts the common Scan method shared by *sql.Row and
// *sql.Rows so both can use the same hydration logic.
type scanner interface {
	Scan(dest ...any) error
}

// scanJobFrom hydrates a Job from any scanner (Row or Rows).
func scanJobFrom(s scanner) (Job, error) {
	var j Job
	var enabled int
	var lastRunAt, nextRunAt, createdAt, updatedAt sql.NullString
	var targetURL, targetPath, targetGlob sql.NullString
	var metadataJSON sql.NullString

	err := s.Scan(&j.ID, &j.BrainID, &j.Name, &j.CronExpression,
		&j.Target.Kind, &targetURL, &targetPath, &targetGlob,
		&enabled, &lastRunAt, &nextRunAt, &createdAt, &updatedAt, &metadataJSON)
	if err != nil {
		if err == sql.ErrNoRows {
			return Job{}, fmt.Errorf("schedule: job not found")
		}
		return Job{}, fmt.Errorf("schedule: scan: %w", err)
	}

	j.Target.URL = targetURL.String
	j.Target.Path = targetPath.String
	j.Target.Glob = targetGlob.String
	j.Enabled = enabled == 1
	if lastRunAt.Valid {
		t, _ := time.Parse(time.RFC3339, lastRunAt.String)
		j.LastRunAt = &t
	}
	if nextRunAt.Valid {
		j.NextRunAt, _ = time.Parse(time.RFC3339, nextRunAt.String)
	}
	if createdAt.Valid {
		j.CreatedAt, _ = time.Parse(time.RFC3339, createdAt.String)
	}
	if updatedAt.Valid {
		j.UpdatedAt, _ = time.Parse(time.RFC3339, updatedAt.String)
	}
	if metadataJSON.Valid {
		_ = json.Unmarshal([]byte(metadataJSON.String), &j.Metadata)
	}

	return j, nil
}

// scanJob scans a single row into a Job.
func scanJob(row *sql.Row) (Job, error) {
	return scanJobFrom(row)
}

// scanJobRows scans a Rows iterator into a Job.
func scanJobRows(rows *sql.Rows) (Job, error) {
	return scanJobFrom(rows)
}

func nullStr(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
