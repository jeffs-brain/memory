// SPDX-License-Identifier: Apache-2.0

/**
 * Mock PgClient for unit testing the PostgreSQL queue adapter.
 * Simulates core PostgreSQL operations in-memory by pattern-matching
 * normalised query strings.
 */

import type { PgClient } from './postgres.js'

/** Row store entry for the mock database. */
export type MockRow = {
  id: string
  brain_id: string
  status: string
  payload: string
  retry_count: number
  max_retries: number
  error: string | null
  claimed_by: string | null
  claimed_at: Date | null
  last_heartbeat: Date | null
  next_retry_at: Date | null
  created_at: Date
  updated_at: Date
  completed_at: Date | null
  metadata: string | null
  group_id: string | null
  idempotency_key: string | null
}

/**
 * Create a mock PgClient that simulates core PostgreSQL operations
 * in-memory. This exercises the adapter's SQL generation and row
 * mapping without requiring a real database.
 */
export const createMockPgClient = (): PgClient & { rows: MockRow[] } => {
  let idCounter = 0
  const rows: MockRow[] = []

  const query = async <R = Record<string, unknown>>(
    text: string,
    values?: ReadonlyArray<unknown>,
  ): Promise<{ readonly rows: readonly R[]; readonly rowCount: number }> => {
    const normalised = text.replace(/\s+/g, ' ').trim()
    const vals = values ?? []

    // INSERT INTO ... RETURNING
    if (normalised.includes('INSERT INTO')) {
      idCounter++
      const now = new Date()
      const newRow: MockRow = {
        id: `mock-${idCounter}`,
        brain_id: vals[0] as string,
        status: vals[1] as string,
        payload: vals[2] as string,
        retry_count: 0,
        max_retries: vals[3] as number,
        error: null,
        claimed_by: null,
        claimed_at: null,
        last_heartbeat: null,
        next_retry_at: null,
        created_at: now,
        updated_at: now,
        completed_at: null,
        metadata: vals[4] as string | null,
        group_id: vals[5] as string | null,
        idempotency_key: vals[6] as string | null,
      }

      // Check idempotency constraint.
      const existing = rows.find(
        (r) =>
          r.idempotency_key === newRow.idempotency_key &&
          newRow.idempotency_key !== null &&
          !['dead_letter', 'completed', 'failed'].includes(r.status),
      )
      if (existing !== undefined) {
        throw new Error('idx_ingest_queue_idempotency')
      }

      rows.push(newRow)
      return { rows: [newRow as unknown as R], rowCount: 1 }
    }

    // SELECT for idempotency lookup
    if (normalised.includes('idempotency_key = $1') && normalised.includes('SELECT')) {
      const key = vals[0] as string
      const found = rows.filter(
        (r) =>
          r.idempotency_key === key &&
          !['dead_letter', 'completed', 'failed'].includes(r.status),
      )
      return { rows: found as unknown as R[], rowCount: found.length }
    }

    // SELECT for fail lookup (retry_count, max_retries, brain_id)
    if (normalised.includes('SELECT retry_count')) {
      const jobId = vals[0] as string
      const status = vals[1] as string
      const found = rows.filter((r) => r.id === jobId && r.status === status)
      return { rows: found as unknown as R[], rowCount: found.length }
    }

    // UPDATE ... SET status ... FOR UPDATE SKIP LOCKED (claim)
    if (normalised.includes('FOR UPDATE SKIP LOCKED')) {
      const workerName = vals[1] as string
      const batchSize = vals[2] as number
      const now = new Date()
      const pending = rows
        .filter(
          (r) =>
            r.status === 'pending' &&
            (r.next_retry_at === null || r.next_retry_at <= now),
        )
        .sort((a, b) => a.created_at.getTime() - b.created_at.getTime())
        .slice(0, batchSize)

      for (const row of pending) {
        row.status = 'processing'
        row.claimed_by = workerName
        row.claimed_at = now
        row.last_heartbeat = now
        row.updated_at = now
      }

      return { rows: pending as unknown as R[], rowCount: pending.length }
    }

    // UPDATE heartbeat
    if (normalised.includes('last_heartbeat = NOW()') && normalised.includes('WHERE id = $1')) {
      const jobId = vals[0] as string
      const status = vals[1] as string
      const now = new Date()
      let count = 0
      for (const row of rows) {
        if (row.id === jobId && row.status === status) {
          row.last_heartbeat = now
          row.updated_at = now
          count++
        }
      }
      return { rows: [] as unknown as R[], rowCount: count }
    }

    // UPDATE requeue (returns to pending without incrementing retry count)
    if (
      normalised.includes('claimed_by = NULL') &&
      normalised.includes('RETURNING brain_id') &&
      !normalised.includes('completed_at') &&
      !normalised.includes('retry_count')
    ) {
      const newStatus = vals[0] as string
      const jobId = vals[1] as string
      const oldStatus = vals[2] as string
      const found: MockRow[] = []
      for (const row of rows) {
        if (row.id === jobId && row.status === oldStatus) {
          row.status = newStatus
          row.claimed_by = null
          row.claimed_at = null
          row.last_heartbeat = null
          row.updated_at = new Date()
          found.push(row)
        }
      }
      return { rows: found as unknown as R[], rowCount: found.length }
    }

    // UPDATE complete
    if (normalised.includes('completed_at = NOW()')) {
      const jobId = vals[1] as string
      const status = vals[3] as string
      const now = new Date()
      const found: MockRow[] = []
      for (const row of rows) {
        if (row.id === jobId && row.status === status) {
          row.status = 'completed'
          row.completed_at = now
          row.updated_at = now
          if (vals[2] !== null) {
            row.metadata = vals[2] as string
          }
          found.push(row)
        }
      }
      return { rows: found as unknown as R[], rowCount: found.length }
    }

    // UPDATE fail
    if (normalised.includes('retry_count = $2') && normalised.includes('error = $3')) {
      const newStatus = vals[0] as string
      const retryCount = vals[1] as number
      const errorMsg = vals[2] as string
      const nextRetry = vals[3] as Date | null
      const jobId = vals[4] as string
      for (const row of rows) {
        if (row.id === jobId) {
          row.status = newStatus
          row.retry_count = retryCount
          row.error = errorMsg
          row.next_retry_at = nextRetry
          row.claimed_by = null
          row.claimed_at = null
          row.last_heartbeat = null
          row.updated_at = new Date()
        }
      }
      return { rows: [] as unknown as R[], rowCount: 1 }
    }

    // UPDATE recover stale
    if (normalised.includes('last_heartbeat <')) {
      const threshold = parseInt((vals[2] as string).split(' ')[0]!, 10)
      const cutoff = new Date(Date.now() - threshold * 1000)
      let count = 0
      for (const row of rows) {
        if (
          row.status === 'processing' &&
          row.last_heartbeat !== null &&
          row.last_heartbeat < cutoff
        ) {
          row.status = 'pending'
          row.claimed_by = null
          row.claimed_at = null
          row.last_heartbeat = null
          row.updated_at = new Date()
          count++
        }
      }
      return { rows: [] as unknown as R[], rowCount: count }
    }

    // SELECT count by status
    if (normalised.includes('COUNT(*)')) {
      const statusCounts: Record<string, number> = {}
      const brainFilter = vals.length > 0 ? (vals[0] as string) : undefined
      for (const row of rows) {
        if (brainFilter !== undefined && row.brain_id !== brainFilter) continue
        statusCounts[row.status] = (statusCounts[row.status] ?? 0) + 1
      }
      const resultRows = Object.entries(statusCounts).map(([status, count]) => ({
        status,
        count: String(count),
      }))
      return { rows: resultRows as unknown as R[], rowCount: resultRows.length }
    }

    // Transaction control (no-op in mock -- single-threaded tests)
    if (normalised === 'BEGIN' || normalised === 'COMMIT' || normalised === 'ROLLBACK') {
      return { rows: [] as unknown as R[], rowCount: 0 }
    }

    // Bulk heartbeat (ANY)
    if (normalised.includes('id = ANY($1)') && normalised.includes('last_heartbeat = NOW()')) {
      const ids = vals[0] as string[]
      const status = vals[1] as string
      const now = new Date()
      let count = 0
      for (const row of rows) {
        if (ids.includes(row.id) && row.status === status) {
          row.last_heartbeat = now
          row.updated_at = now
          count++
        }
      }
      return { rows: [] as unknown as R[], rowCount: count }
    }

    // Advisory lock operations (no-op in mock)
    if (normalised.includes('pg_try_advisory_lock') || normalised.includes('pg_advisory_unlock')) {
      return { rows: [] as unknown as R[], rowCount: 0 }
    }

    // LISTEN (no-op in mock)
    if (normalised.startsWith('LISTEN')) {
      return { rows: [] as unknown as R[], rowCount: 0 }
    }

    return { rows: [] as unknown as R[], rowCount: 0 }
  }

  return { query, rows }
}
