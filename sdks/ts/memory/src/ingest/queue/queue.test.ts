// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the ingest queue types, helpers, and PostgreSQL adapter.
 * Uses a mock PgClient so tests run without a real database.
 */

import { afterEach, describe, expect, it } from 'vitest'
import {
  BACKOFF_BASE_DELAY_MS,
  DEFAULT_MAX_RETRIES,
  VALID_STATUSES,
  advisoryLockKey,
  computeBackoff,
  validateIdentifier,
} from './types.js'
import { type PgClient, createPostgresQueue } from './postgres.js'
import type { QueueAdapter, QueueJobPayload } from './types.js'
import { type MockRow, createMockPgClient } from './queue.mock.js'

// --- Pure function tests ---

describe('validateIdentifier', () => {
  it('accepts valid identifiers', () => {
    expect(() => validateIdentifier('ingest_queue')).not.toThrow()
    expect(() => validateIdentifier('MyTable')).not.toThrow()
    expect(() => validateIdentifier('table123')).not.toThrow()
  })

  it('rejects empty string', () => {
    expect(() => validateIdentifier('')).toThrow('must not be empty')
  })

  it('rejects special characters', () => {
    expect(() => validateIdentifier('ingest-queue')).toThrow('invalid characters')
    expect(() => validateIdentifier('ingest queue')).toThrow('invalid characters')
    expect(() => validateIdentifier('table;DROP')).toThrow('invalid characters')
    expect(() => validateIdentifier("table'")).toThrow('invalid characters')
  })
})

describe('advisoryLockKey', () => {
  it('returns deterministic values', () => {
    const key1 = advisoryLockKey('brain-abc')
    const key2 = advisoryLockKey('brain-abc')
    expect(key1).toBe(key2)
  })

  it('produces different keys for different brains', () => {
    const keyA = advisoryLockKey('brain-alpha')
    const keyB = advisoryLockKey('brain-beta')
    expect(keyA).not.toBe(keyB)
  })

  it('returns a bigint', () => {
    const key = advisoryLockKey('test-brain')
    expect(typeof key).toBe('bigint')
  })
})

describe('computeBackoff', () => {
  it('returns a future date', () => {
    const before = Date.now()
    const result = computeBackoff(0)
    expect(result.getTime()).toBeGreaterThan(before)
  })

  it('increases delay with retry count', () => {
    // Run multiple samples to account for jitter.
    const samples = 20
    let avgDelay0 = 0
    let avgDelay2 = 0
    for (let i = 0; i < samples; i++) {
      const before = Date.now()
      avgDelay0 += computeBackoff(0).getTime() - before
      avgDelay2 += computeBackoff(2).getTime() - before
    }
    avgDelay0 /= samples
    avgDelay2 /= samples
    // Retry 2 should be roughly 4x retry 0 (2^2 = 4).
    expect(avgDelay2).toBeGreaterThan(avgDelay0 * 2)
  })

  it('stays within jitter bounds', () => {
    for (let i = 0; i < 50; i++) {
      const before = Date.now()
      const result = computeBackoff(1)
      const delayMs = result.getTime() - before
      const baseMs = BACKOFF_BASE_DELAY_MS * 2 // 2^1
      // Minimum: baseMs * 0.5, Maximum: baseMs * 1.5
      expect(delayMs).toBeGreaterThanOrEqual(baseMs * 0.4) // small tolerance for timing
      expect(delayMs).toBeLessThanOrEqual(baseMs * 1.6) // small tolerance for timing
    }
  })
})

describe('VALID_STATUSES', () => {
  it('contains all five status values', () => {
    expect(VALID_STATUSES.size).toBe(5)
    expect(VALID_STATUSES.has('pending')).toBe(true)
    expect(VALID_STATUSES.has('processing')).toBe(true)
    expect(VALID_STATUSES.has('completed')).toBe(true)
    expect(VALID_STATUSES.has('failed')).toBe(true)
    expect(VALID_STATUSES.has('dead_letter')).toBe(true)
  })
})

// --- PostgresQueue adapter tests (mock-backed) ---

describe('createPostgresQueue', () => {
  const adapters: QueueAdapter[] = []

  const freshAdapter = (): { adapter: QueueAdapter; mockClient: PgClient & { rows: MockRow[] } } => {
    const mockClient = createMockPgClient()
    const adapter = createPostgresQueue({
      client: mockClient,
      heartbeatIntervalMs: 60_000, // long interval to avoid timer noise
    })
    adapters.push(adapter)
    return { adapter, mockClient }
  }

  afterEach(async () => {
    for (const a of adapters) {
      await a.close()
    }
    adapters.length = 0
  })

  const samplePayload: QueueJobPayload = {
    kind: 'file',
    path: '/data/document.md',
    title: 'Test Document',
    mime: 'text/markdown',
  }

  describe('enqueue', () => {
    it('creates a job with status=pending', async () => {
      const { adapter } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
      })
      expect(job.status).toBe('pending')
      expect(job.brainId).toBe('brain-1')
      expect(job.retryCount).toBe(0)
      expect(job.maxRetries).toBe(DEFAULT_MAX_RETRIES)
      expect(job.payload.kind).toBe('file')
      expect(job.payload.path).toBe('/data/document.md')
    })

    it('sets custom max retries', async () => {
      const { adapter } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        maxRetries: 5,
      })
      expect(job.maxRetries).toBe(5)
    })

    it('preserves metadata', async () => {
      const { adapter } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        metadata: { source: 'upload', user: 'alice' },
      })
      expect(job.metadata).toEqual({ source: 'upload', user: 'alice' })
    })

    it('preserves group id', async () => {
      const { adapter } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        groupId: 'batch-42',
      })
      expect(job.groupId).toBe('batch-42')
    })

    it('returns existing job for duplicate idempotency key', async () => {
      const { adapter } = freshAdapter()
      const first = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        idempotencyKey: 'unique-op-1',
      })
      const second = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        idempotencyKey: 'unique-op-1',
      })
      expect(second.id).toBe(first.id)
    })

    it('handles idempotency race condition via constraint violation fallback', async () => {
      // Simulate the race condition where the pre-check finds nothing
      // but the INSERT hits the unique constraint because another
      // process inserted between the check and the INSERT.
      const mockClient = createMockPgClient()

      // Pre-insert a row directly into the mock store to simulate
      // the race: the idempotency lookup will find nothing (because
      // we insert *after* the lookup intercept), but the INSERT will
      // hit the constraint. Then the fallback lookup succeeds.
      let lookupCount = 0
      const originalQuery = mockClient.query.bind(mockClient)
      const racyQuery = async <R = Record<string, unknown>>(
        text: string,
        values?: ReadonlyArray<unknown>,
      ): Promise<{ readonly rows: readonly R[]; readonly rowCount: number }> => {
        const normalised = text.replace(/\s+/g, ' ').trim()
        // On the FIRST idempotency lookup, return empty (simulating the
        // race window). The INSERT will then throw the constraint error,
        // and the fallback lookup will find the pre-existing row.
        if (normalised.includes('idempotency_key = $1') && normalised.includes('SELECT') && lookupCount === 0) {
          lookupCount++
          // Inject a row directly so the fallback lookup finds it.
          mockClient.rows.push({
            id: 'race-winner',
            brain_id: 'brain-1',
            status: 'pending',
            payload: JSON.stringify(samplePayload),
            retry_count: 0,
            max_retries: 3,
            error: null,
            claimed_by: null,
            claimed_at: null,
            last_heartbeat: null,
            next_retry_at: null,
            created_at: new Date(),
            updated_at: new Date(),
            completed_at: null,
            metadata: null,
            group_id: null,
            idempotency_key: 'race-key-1',
          })
          return { rows: [] as unknown as R[], rowCount: 0 }
        }
        return originalQuery(text, values)
      }

      const racyClient = { ...mockClient, query: racyQuery }
      const adapter = createPostgresQueue({
        client: racyClient,
        heartbeatIntervalMs: 60_000,
      })
      adapters.push(adapter)

      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        idempotencyKey: 'race-key-1',
      })

      // The adapter should have fallen back to the existing row
      // inserted by the "race winner".
      expect(job.id).toBe('race-winner')
    })
  })

  describe('claim', () => {
    it('returns pending jobs and sets status=processing', async () => {
      const { adapter } = freshAdapter()
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })

      const claimed = await adapter.claim({ batchSize: 5, workerId: 'worker-a' })
      expect(claimed.length).toBe(2)
      for (const job of claimed) {
        expect(job.status).toBe('processing')
        expect(job.claimedBy).toBe('worker-a')
        expect(job.claimedAt).toBeInstanceOf(Date)
        expect(job.lastHeartbeat).toBeInstanceOf(Date)
      }
    })

    it('does not double-claim jobs', async () => {
      const { adapter } = freshAdapter()
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })

      const first = await adapter.claim({ batchSize: 5, workerId: 'worker-a' })
      const second = await adapter.claim({ batchSize: 5, workerId: 'worker-b' })
      expect(first.length).toBe(1)
      expect(second.length).toBe(0)
    })

    it('respects batch size limit', async () => {
      const { adapter } = freshAdapter()
      for (let i = 0; i < 5; i++) {
        await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      }

      const claimed = await adapter.claim({ batchSize: 2, workerId: 'worker-a' })
      expect(claimed.length).toBe(2)
    })

    it('requires non-empty worker ID', async () => {
      const { adapter } = freshAdapter()
      await expect(
        adapter.claim({ batchSize: 1, workerId: '' }),
      ).rejects.toThrow('non-empty worker ID')
    })
  })

  describe('heartbeat', () => {
    it('updates last_heartbeat for processing jobs', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      const [claimed] = await adapter.claim({ batchSize: 1, workerId: 'worker-a' })
      expect(claimed).toBeDefined()

      const oldHB = mockClient.rows.find((r) => r.id === job.id)!.last_heartbeat!
      // Simulate slight time passage.
      await new Promise((resolve) => setTimeout(resolve, 5))
      await adapter.heartbeat(job.id)

      const row = mockClient.rows.find((r) => r.id === job.id)!
      expect(row.last_heartbeat!.getTime()).toBeGreaterThanOrEqual(oldHB.getTime())
    })

    it('throws for non-processing jobs', async () => {
      const { adapter } = freshAdapter()
      await expect(adapter.heartbeat('nonexistent')).rejects.toThrow('no processing job')
    })
  })

  describe('complete', () => {
    it('sets status=completed with completedAt', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })
      await adapter.complete(job.id)

      const row = mockClient.rows.find((r) => r.id === job.id)!
      expect(row.status).toBe('completed')
      expect(row.completed_at).toBeInstanceOf(Date)
    })

    it('merges result metadata', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })
      await adapter.complete(job.id, { chunks: '5', duration: '120ms' })

      const row = mockClient.rows.find((r) => r.id === job.id)!
      const meta = JSON.parse(row.metadata!)
      expect(meta.chunks).toBe('5')
    })
  })

  describe('fail', () => {
    it('retries with exponential backoff when retryable', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        maxRetries: 3,
      })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })
      await adapter.fail(job.id, 'temporary error', true)

      const row = mockClient.rows.find((r) => r.id === job.id)!
      expect(row.status).toBe('pending')
      expect(row.retry_count).toBe(1)
      expect(row.error).toBe('temporary error')
      expect(row.next_retry_at).toBeInstanceOf(Date)
      expect(row.claimed_by).toBeNull()
    })

    it('moves to dead_letter when retries exhausted', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        maxRetries: 1,
      })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })
      await adapter.fail(job.id, 'fatal error', true)

      const row = mockClient.rows.find((r) => r.id === job.id)!
      // retry_count (1) >= max_retries (1) => dead_letter
      expect(row.status).toBe('dead_letter')
    })

    it('moves to dead_letter when not retryable', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({
        brainId: 'brain-1',
        payload: samplePayload,
        maxRetries: 5,
      })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })
      await adapter.fail(job.id, 'permanent error', false)

      const row = mockClient.rows.find((r) => r.id === job.id)!
      expect(row.status).toBe('dead_letter')
    })
  })

  describe('recoverStale', () => {
    it('resets stale processing jobs to pending', async () => {
      const { adapter, mockClient } = freshAdapter()
      const job = await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })

      // Simulate stale heartbeat by backdating.
      const row = mockClient.rows.find((r) => r.id === job.id)!
      row.last_heartbeat = new Date(Date.now() - 600_000) // 10 minutes ago

      const recovered = await adapter.recoverStale(300_000) // 5 min threshold
      expect(recovered).toBe(1)
      expect(row.status).toBe('pending')
      expect(row.claimed_by).toBeNull()
    })

    it('does not recover fresh jobs', async () => {
      const { adapter } = freshAdapter()
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.claim({ batchSize: 1, workerId: 'worker-a' })

      const recovered = await adapter.recoverStale(300_000)
      expect(recovered).toBe(0)
    })
  })

  describe('countByStatus', () => {
    it('returns counts grouped by status', async () => {
      const { adapter } = freshAdapter()
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.enqueue({ brainId: 'brain-2', payload: samplePayload })

      const counts = await adapter.countByStatus()
      expect(counts.pending).toBe(3)
    })

    it('filters by brain when specified', async () => {
      const { adapter } = freshAdapter()
      await adapter.enqueue({ brainId: 'brain-1', payload: samplePayload })
      await adapter.enqueue({ brainId: 'brain-2', payload: samplePayload })
      await adapter.enqueue({ brainId: 'brain-2', payload: samplePayload })

      const counts = await adapter.countByStatus('brain-2')
      expect(counts.pending).toBe(2)
    })
  })

  describe('close', () => {
    it('prevents further operations after close', async () => {
      const { adapter } = freshAdapter()
      await adapter.close()
      await expect(
        adapter.enqueue({ brainId: 'brain-1', payload: samplePayload }),
      ).rejects.toThrow('closed')
    })

    it('is idempotent', async () => {
      const { adapter } = freshAdapter()
      await adapter.close()
      await adapter.close() // no error
    })
  })
})
