// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the dead letter queue adapter. Covers the in-memory adapter
 * which validates the full interface contract. PostgreSQL and SQLite
 * adapters are tested separately with real database connections.
 */

import { describe, expect, it, beforeEach } from 'vitest'
import {
  createInMemoryDeadLetterAdapter,
  DeadLetterNotFoundError,
  DeadLetterAlreadyResolvedError,
  type DeadLetterAdapter,
  type DeadLetterEntry,
  type ReEnqueueFn,
} from './dead-letter.js'

const makeEntry = (overrides: Partial<DeadLetterEntry> = {}): DeadLetterEntry => ({
  id: 'dlq-001',
  originalJobId: 'job-001',
  brainId: 'brain-1',
  payload: {
    documentHash: 'abc123def4560000000000000000000000000000000000000000000000000000',
    brainId: 'brain-1',
    source: 'file:///test.md',
    contentType: 'text/markdown',
  },
  failureReason: 'embedding provider returned 500',
  lastError: 'context deadline exceeded',
  retryCount: 3,
  movedAt: new Date('2026-05-01T12:00:00Z'),
  ...overrides,
})

describe('InMemoryDeadLetterAdapter', () => {
  let adapter: DeadLetterAdapter

  beforeEach(() => {
    adapter = createInMemoryDeadLetterAdapter()
  })

  describe('move', () => {
    it('stores an entry and returns it', async () => {
      const entry = makeEntry()
      const result = await adapter.move(entry)

      expect(result.id).toBe(entry.id)
      expect(result.brainId).toBe(entry.brainId)
      expect(result.failureReason).toBe(entry.failureReason)
      expect(result.lastError).toBe(entry.lastError)
      expect(result.retryCount).toBe(entry.retryCount)
    })

    it('generates an ID when empty', async () => {
      const entry = makeEntry({ id: '' })
      const result = await adapter.move(entry)
      expect(result.id).toBeTruthy()
      expect(result.id).not.toBe('')
    })

    it('preserves full payload', async () => {
      const entry = makeEntry()
      const result = await adapter.move(entry)

      expect(result.payload.documentHash).toBe(entry.payload.documentHash)
      expect(result.payload.brainId).toBe(entry.payload.brainId)
      expect(result.payload.source).toBe(entry.payload.source)
      expect(result.payload.contentType).toBe(entry.payload.contentType)
    })
  })

  describe('get', () => {
    it('returns the entry by ID', async () => {
      const entry = makeEntry()
      await adapter.move(entry)

      const got = await adapter.get(entry.id)
      expect(got).toBeDefined()
      expect(got?.id).toBe(entry.id)
      expect(got?.originalJobId).toBe(entry.originalJobId)
      expect(got?.failureReason).toBe(entry.failureReason)
    })

    it('returns undefined for non-existent ID', async () => {
      const got = await adapter.get('nonexistent')
      expect(got).toBeUndefined()
    })
  })

  describe('list', () => {
    it('returns unresolved entries by default', async () => {
      await adapter.move(makeEntry({ id: 'dlq-unresolved' }))
      await adapter.move(makeEntry({
        id: 'dlq-resolved',
        resolvedAt: new Date(),
        resolvedBy: 'operator',
      }))

      const result = await adapter.list()
      expect(result.total).toBe(1)
      expect(result.entries).toHaveLength(1)
      expect(result.entries[0].id).toBe('dlq-unresolved')
    })

    it('includes resolved entries when requested', async () => {
      await adapter.move(makeEntry({ id: 'dlq-a' }))
      await adapter.move(makeEntry({
        id: 'dlq-b',
        resolvedAt: new Date(),
        resolvedBy: 'operator',
      }))

      const result = await adapter.list({ includeResolved: true })
      expect(result.total).toBe(2)
      expect(result.entries).toHaveLength(2)
    })

    it('filters by brainId', async () => {
      await adapter.move(makeEntry({ id: 'dlq-alpha', brainId: 'brain-alpha' }))
      await adapter.move(makeEntry({ id: 'dlq-beta', brainId: 'brain-beta' }))

      const result = await adapter.list({ brainId: 'brain-alpha' })
      expect(result.total).toBe(1)
      expect(result.entries[0].brainId).toBe('brain-alpha')
    })

    it('paginates results', async () => {
      for (let i = 0; i < 5; i++) {
        await adapter.move(makeEntry({
          id: `dlq-page-${i}`,
          movedAt: new Date(Date.UTC(2026, 4, 1, 12, 0, i)),
        }))
      }

      const page1 = await adapter.list({ limit: 2, offset: 0 })
      expect(page1.total).toBe(5)
      expect(page1.entries).toHaveLength(2)

      const page2 = await adapter.list({ limit: 2, offset: 2 })
      expect(page2.entries).toHaveLength(2)

      const page3 = await adapter.list({ limit: 2, offset: 4 })
      expect(page3.entries).toHaveLength(1)
    })

    it('returns entries sorted by movedAt descending', async () => {
      await adapter.move(makeEntry({
        id: 'dlq-oldest',
        movedAt: new Date('2026-01-01T00:00:00Z'),
      }))
      await adapter.move(makeEntry({
        id: 'dlq-newest',
        movedAt: new Date('2026-06-01T00:00:00Z'),
      }))
      await adapter.move(makeEntry({
        id: 'dlq-middle',
        movedAt: new Date('2026-03-01T00:00:00Z'),
      }))

      const result = await adapter.list()
      expect(result.entries[0].id).toBe('dlq-newest')
      expect(result.entries[1].id).toBe('dlq-middle')
      expect(result.entries[2].id).toBe('dlq-oldest')
    })
  })

  describe('retry', () => {
    it('marks an entry as resolved and returns it', async () => {
      await adapter.move(makeEntry())

      const resolved = await adapter.retry('dlq-001', 'admin@example.com')
      expect(resolved.resolvedAt).toBeDefined()
      expect(resolved.resolvedBy).toBe('admin@example.com')
    })

    it('persists the resolved state', async () => {
      await adapter.move(makeEntry())
      await adapter.retry('dlq-001', 'admin@example.com')

      const got = await adapter.get('dlq-001')
      expect(got?.resolvedAt).toBeDefined()
      expect(got?.resolvedBy).toBe('admin@example.com')
    })

    it('throws DeadLetterNotFoundError for non-existent ID', async () => {
      await expect(adapter.retry('nonexistent', 'operator'))
        .rejects.toThrow(DeadLetterNotFoundError)
    })

    it('throws DeadLetterAlreadyResolvedError on double retry', async () => {
      await adapter.move(makeEntry())
      await adapter.retry('dlq-001', 'operator-1')

      await expect(adapter.retry('dlq-001', 'operator-2'))
        .rejects.toThrow(DeadLetterAlreadyResolvedError)
    })

    it('calls reEnqueue callback with resolved entry', async () => {
      await adapter.move(makeEntry())

      let enqueuedEntry: DeadLetterEntry | undefined
      const reEnqueue: ReEnqueueFn = async (entry) => {
        enqueuedEntry = entry
      }

      const resolved = await adapter.retry('dlq-001', 'operator', reEnqueue)
      expect(resolved.resolvedAt).toBeDefined()
      expect(enqueuedEntry).toBeDefined()
      expect(enqueuedEntry?.id).toBe('dlq-001')
      expect(enqueuedEntry?.payload.documentHash).toBe(makeEntry().payload.documentHash)
    })

    it('does not resolve entry when reEnqueue throws', async () => {
      await adapter.move(makeEntry())

      const failingReEnqueue: ReEnqueueFn = async () => {
        throw new Error('queue is full')
      }

      await expect(adapter.retry('dlq-001', 'operator', failingReEnqueue))
        .rejects.toThrow('queue is full')

      const got = await adapter.get('dlq-001')
      expect(got?.resolvedAt).toBeUndefined()
    })

    it('defaults resolvedBy to system when omitted', async () => {
      await adapter.move(makeEntry())
      const resolved = await adapter.retry('dlq-001')
      expect(resolved.resolvedBy).toBe('system')
    })
  })

  describe('purge', () => {
    it('removes entry by ID', async () => {
      await adapter.move(makeEntry())

      const removed = await adapter.purge({ kind: 'by-id', id: 'dlq-001' })
      expect(removed).toBe(1)

      const got = await adapter.get('dlq-001')
      expect(got).toBeUndefined()
    })

    it('returns 0 when purging non-existent ID', async () => {
      const removed = await adapter.purge({ kind: 'by-id', id: 'nonexistent' })
      expect(removed).toBe(0)
    })

    it('removes all entries for a brain', async () => {
      for (let i = 0; i < 3; i++) {
        await adapter.move(makeEntry({ id: `dlq-purge-${i}`, brainId: 'brain-to-purge' }))
      }
      await adapter.move(makeEntry({ id: 'dlq-keep', brainId: 'brain-keep' }))

      const removed = await adapter.purge({ kind: 'by-brain', brainId: 'brain-to-purge' })
      expect(removed).toBe(3)

      const result = await adapter.list({ includeResolved: true })
      expect(result.total).toBe(1)
      expect(result.entries[0].brainId).toBe('brain-keep')
    })

    it('removes entries older than threshold', async () => {
      await adapter.move(makeEntry({
        id: 'dlq-old',
        movedAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000),
      }))
      await adapter.move(makeEntry({
        id: 'dlq-recent',
        movedAt: new Date(),
      }))

      const removed = await adapter.purge({ kind: 'older-than', days: 30 })
      expect(removed).toBe(1)

      const result = await adapter.list({ includeResolved: true })
      expect(result.total).toBe(1)
      expect(result.entries[0].id).toBe('dlq-recent')
    })

    it('removes all resolved entries', async () => {
      await adapter.move(makeEntry({ id: 'dlq-unres' }))
      await adapter.move(makeEntry({
        id: 'dlq-res',
        resolvedAt: new Date(),
        resolvedBy: 'operator',
      }))

      const removed = await adapter.purge({ kind: 'all-resolved' })
      expect(removed).toBe(1)

      const result = await adapter.list({ includeResolved: true })
      expect(result.total).toBe(1)
      expect(result.entries[0].id).toBe('dlq-unres')
    })
  })

  describe('count', () => {
    it('returns 0 when empty', async () => {
      const total = await adapter.count()
      expect(total).toBe(0)
    })

    it('counts unresolved entries globally', async () => {
      for (let i = 0; i < 3; i++) {
        await adapter.move(makeEntry({ id: `dlq-count-${i}`, brainId: 'brain-a' }))
      }
      await adapter.move(makeEntry({ id: 'dlq-count-b', brainId: 'brain-b' }))
      await adapter.retry('dlq-count-0', 'operator')

      const total = await adapter.count()
      expect(total).toBe(3)
    })

    it('counts unresolved entries per brain', async () => {
      for (let i = 0; i < 3; i++) {
        await adapter.move(makeEntry({ id: `dlq-ca-${i}`, brainId: 'brain-a' }))
      }
      await adapter.move(makeEntry({ id: 'dlq-cb', brainId: 'brain-b' }))
      await adapter.retry('dlq-ca-0', 'operator')

      expect(await adapter.count('brain-a')).toBe(2)
      expect(await adapter.count('brain-b')).toBe(1)
    })
  })

  describe('metadata round-trip', () => {
    it('preserves metadata and groupId', async () => {
      await adapter.move(makeEntry({
        metadata: { source: 'webhook', requestId: 'req-12345' },
        groupId: 'batch-001',
      }))

      const got = await adapter.get('dlq-001')
      expect(got?.metadata?.source).toBe('webhook')
      expect(got?.metadata?.requestId).toBe('req-12345')
      expect(got?.groupId).toBe('batch-001')
    })
  })

  describe('error history', () => {
    it('preserves error history array', async () => {
      await adapter.move(makeEntry({
        errorHistory: [
          'attempt 1: connection refused',
          'attempt 2: timeout after 30s',
          'attempt 3: context deadline exceeded',
        ],
      }))

      const got = await adapter.get('dlq-001')
      expect(got?.errorHistory).toHaveLength(3)
      expect(got?.errorHistory?.[0]).toBe('attempt 1: connection refused')
      expect(got?.errorHistory?.[2]).toBe('attempt 3: context deadline exceeded')
    })

    it('round-trips entries without error history', async () => {
      await adapter.move(makeEntry())

      const got = await adapter.get('dlq-001')
      expect(got?.errorHistory).toBeUndefined()
    })
  })
})
