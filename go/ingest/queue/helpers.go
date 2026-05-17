// SPDX-License-Identifier: Apache-2.0

package queue

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"time"
)

// heartbeatTimeoutSeconds is the context deadline for each heartbeat
// refresh operation.
const heartbeatTimeoutSeconds = 5

// rowScanner abstracts *sql.Row and *sql.Rows so scanJob can accept
// either.
type rowScanner interface {
	Scan(dest ...any) error
}

// scanJob reads a single job row into a Job struct.
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

// computeBackoff calculates the next retry time using exponential
// backoff with jitter: baseDelay * 2^retryCount * random(0.5, 1.5).
func computeBackoff(retryCount int) time.Time {
	multiplier := math.Pow(2, float64(retryCount))
	jitter := backoffJitterMin + rand.Float64()*(backoffJitterMax-backoffJitterMin)
	delay := time.Duration(float64(backoffBaseDelay) * multiplier * jitter)
	return time.Now().Add(delay)
}

// advisoryLockKey computes a stable int64 lock key from a brain ID
// using FNV-1a hashing. The uint64 hash is cast to int64, which is
// the same representation PostgreSQL's bigint uses. The TS adapter
// must replicate this signed conversion for cross-language parity.
func advisoryLockKey(brainID string) int64 {
	h := fnv.New64a()
	h.Write([]byte(brainID))
	return int64(h.Sum64())
}

// nullableBytes returns nil when b is empty, otherwise returns b.
// Used for optional JSONB columns.
func nullableBytes(b []byte) *[]byte {
	if len(b) == 0 {
		return nil
	}
	return &b
}

// pq64Array implements the driver.Valuer interface for a []int64
// slice so it is transmitted as a PostgreSQL bigint[] array literal.
// This enables single-parameter array binding for batch advisory locks.
type pq64Array []int64

// Value returns the PostgreSQL array literal representation.
func (a pq64Array) Value() (interface{}, error) {
	if len(a) == 0 {
		return "{}", nil
	}
	b := make([]byte, 0, len(a)*20)
	b = append(b, '{')
	for i, v := range a {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, []byte(fmt.Sprintf("%d", v))...)
	}
	b = append(b, '}')
	return string(b), nil
}

// pqStringArray implements the driver.Valuer interface for a []string
// slice so it is transmitted as a PostgreSQL uuid[] array literal.
// This enables single-parameter array binding for bulk heartbeat.
type pqStringArray []string

// Value returns the PostgreSQL array literal representation.
func (a pqStringArray) Value() (interface{}, error) {
	if len(a) == 0 {
		return "{}", nil
	}
	b := make([]byte, 0, len(a)*40)
	b = append(b, '{')
	for i, v := range a {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, '"')
		b = append(b, []byte(v)...)
		b = append(b, '"')
	}
	b = append(b, '}')
	return string(b), nil
}
