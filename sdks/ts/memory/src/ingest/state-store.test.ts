// SPDX-License-Identifier: Apache-2.0

/**
 * Contract test suite for PipelineStateStore implementations.
 *
 * Both FilePipelineStateStore and PostgresPipelineStateStore must pass the
 * same behavioural contract. The file-based implementation is tested here
 * directly (no Docker). The Postgres implementation uses an in-memory fake
 * that honours the same interface.
 */

import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import type { Store } from '../store/index.js'
import type { PipelineStateEntry, PipelineStateStore } from './state-store.js'
import { FilePipelineStateStore } from './state-store.js'
import { PostgresPipelineStateStore } from './state-store-pg.js'
import type { PgSql } from './state-store-pg.js'

const BRAIN_ID = 'test-brain'

/** Valid 64-character lowercase hex hash for tests. */
const DEFAULT_HASH = 'abc123def4560000000000000000000000000000000000000000000000000000'

const makeEntry = (overrides: Partial<PipelineStateEntry> = {}): PipelineStateEntry => ({
  documentHash: DEFAULT_HASH,
  brainId: BRAIN_ID,
  stage: 'stored',
  retryCount: 0,
  createdAt: new Date('2026-01-01T00:00:00Z'),
  updatedAt: new Date('2026-01-01T00:00:00Z'),
  ...overrides,
})

/**
 * In-memory fake that mimics the PostgresPipelineStateStore's SQL behaviour
 * without requiring a real database connection. Routes based on the query
 * string passed to `unsafe()`.
 */
const createFakePgSql = (): PgSql => {
  const rows = new Map<string, Record<string, unknown>>()

  const taggedTemplate = (
    _strings: TemplateStringsArray,
    ..._values: unknown[]
  ): Promise<ReadonlyArray<unknown>> => Promise.resolve([])

  const unsafe = (query: string, params?: unknown[]): Promise<ReadonlyArray<unknown>> => {
    const safeParams = params ?? []

    if (query.includes('SELECT') && query.includes('LIMIT 1')) {
      const documentHash = safeParams[0] as string
      const row = rows.get(documentHash)
      if (row === undefined) return Promise.resolve([])
      return Promise.resolve([row])
    }

    if (query.includes('SELECT') && query.includes('NOT IN')) {
      const brainId = safeParams[0] as string
      const matching: Record<string, unknown>[] = []
      for (const row of rows.values()) {
        if (
          row['brain_id'] === brainId &&
          row['stage'] !== 'completed' &&
          row['stage'] !== 'failed'
        ) {
          matching.push(row)
        }
      }
      matching.sort((a, b) => {
        const aTime = (a['created_at'] as Date).getTime()
        const bTime = (b['created_at'] as Date).getTime()
        return aTime - bTime
      })
      return Promise.resolve(matching)
    }

    if (query.includes('INSERT')) {
      const documentHash = safeParams[0] as string
      const row: Record<string, unknown> = {
        document_hash: safeParams[0],
        brain_id: safeParams[1],
        stage: safeParams[2],
        retry_count: safeParams[3],
        last_error: safeParams[4],
        created_at: safeParams[5],
        updated_at: safeParams[6],
        completed_at: safeParams[7],
      }
      rows.set(documentHash, row)
      return Promise.resolve([])
    }

    if (query.includes('DELETE')) {
      const documentHash = safeParams[0] as string
      rows.delete(documentHash)
      return Promise.resolve([])
    }

    return Promise.resolve([])
  }

  const begin = async <T>(fn: (s: PgSql) => Promise<T>): Promise<T> => fn(sql)

  const sql = Object.assign(taggedTemplate, { begin, unsafe }) as unknown as PgSql

  return sql
}

type StoreFactory = {
  readonly name: string
  readonly create: () => { stateStore: PipelineStateStore; cleanup: () => Promise<void> }
}

const factories: readonly StoreFactory[] = [
  {
    name: 'FilePipelineStateStore',
    create: () => {
      const store: Store = createMemStore()
      const stateStore = new FilePipelineStateStore(store)
      return { stateStore, cleanup: async () => store.close() }
    },
  },
  {
    name: 'PostgresPipelineStateStore',
    create: () => {
      const sql = createFakePgSql()
      const stateStore = new PostgresPipelineStateStore({ sql })
      return { stateStore, cleanup: async () => {} }
    },
  },
]

for (const factory of factories) {
  describe(`PipelineStateStore contract: ${factory.name}`, () => {
    let stateStore: PipelineStateStore
    let cleanup: () => Promise<void>

    beforeEach(() => {
      const created = factory.create()
      stateStore = created.stateStore
      cleanup = created.cleanup
    })

    afterEach(async () => {
      await cleanup()
    })

    it('returns undefined when no state exists', async () => {
      const entry = await stateStore.get('0000000000000000000000000000000000000000000000000000000000000000')
      expect(entry).toBeUndefined()
    })

    it('round-trips state through set and get', async () => {
      const entry = makeEntry()
      await stateStore.set(entry)
      const retrieved = await stateStore.get(entry.documentHash)
      expect(retrieved).toBeDefined()
      expect(retrieved!.documentHash).toBe(entry.documentHash)
      expect(retrieved!.brainId).toBe(entry.brainId)
      expect(retrieved!.stage).toBe(entry.stage)
      expect(retrieved!.retryCount).toBe(entry.retryCount)
      expect(retrieved!.createdAt.toISOString()).toBe(entry.createdAt.toISOString())
      expect(retrieved!.updatedAt.toISOString()).toBe(entry.updatedAt.toISOString())
    })

    it('overwrites existing entry on set', async () => {
      const entry = makeEntry({ stage: 'stored' })
      await stateStore.set(entry)

      const updated = makeEntry({
        stage: 'chunked',
        updatedAt: new Date('2026-01-02T00:00:00Z'),
      })
      await stateStore.set(updated)

      const retrieved = await stateStore.get(entry.documentHash)
      expect(retrieved).toBeDefined()
      expect(retrieved!.stage).toBe('chunked')
      expect(retrieved!.updatedAt.toISOString()).toBe('2026-01-02T00:00:00.000Z')
    })

    it('lists only incomplete entries', async () => {
      const received = makeEntry({ documentHash: '1111111111111111111111111111111111111111111111111111111111111111', stage: 'received' })
      const stored = makeEntry({ documentHash: '2222222222222222222222222222222222222222222222222222222222222222', stage: 'stored' })
      const completed = makeEntry({ documentHash: '3333333333333333333333333333333333333333333333333333333333333333', stage: 'completed' })
      const failed = makeEntry({ documentHash: '4444444444444444444444444444444444444444444444444444444444444444', stage: 'failed' })
      const embedded = makeEntry({ documentHash: '5555555555555555555555555555555555555555555555555555555555555555', stage: 'embedded' })

      await stateStore.set(received)
      await stateStore.set(stored)
      await stateStore.set(completed)
      await stateStore.set(failed)
      await stateStore.set(embedded)

      const incomplete = await stateStore.listIncomplete(BRAIN_ID)
      const hashes = incomplete.map((e) => e.documentHash)

      expect(hashes).toContain('1111111111111111111111111111111111111111111111111111111111111111')
      expect(hashes).toContain('2222222222222222222222222222222222222222222222222222222222222222')
      expect(hashes).toContain('5555555555555555555555555555555555555555555555555555555555555555')
      expect(hashes).not.toContain('3333333333333333333333333333333333333333333333333333333333333333')
      expect(hashes).not.toContain('4444444444444444444444444444444444444444444444444444444444444444')
      expect(incomplete).toHaveLength(3)
    })

    it('listIncomplete filters by brainId', async () => {
      const ours = makeEntry({ documentHash: 'aaaa000000000000000000000000000000000000000000000000000000000000', brainId: BRAIN_ID, stage: 'stored' })
      const theirs = makeEntry({
        documentHash: 'bbbb000000000000000000000000000000000000000000000000000000000000',
        brainId: 'other-brain',
        stage: 'stored',
      })

      await stateStore.set(ours)
      await stateStore.set(theirs)

      const incomplete = await stateStore.listIncomplete(BRAIN_ID)
      expect(incomplete).toHaveLength(1)
      expect(incomplete[0]!.documentHash).toBe('aaaa000000000000000000000000000000000000000000000000000000000000')
    })

    it('deletes state by document hash', async () => {
      const entry = makeEntry()
      await stateStore.set(entry)

      await stateStore.delete(entry.documentHash)
      const retrieved = await stateStore.get(entry.documentHash)
      expect(retrieved).toBeUndefined()
    })

    it('delete on non-existent hash does not throw', async () => {
      await expect(stateStore.delete('0000000000000000000000000000000000000000000000000000000000000001')).resolves.toBeUndefined()
    })

    it('preserves lastError field', async () => {
      const entry = makeEntry({
        stage: 'failed',
        lastError: 'embedder timeout after 30s',
        retryCount: 3,
      })
      await stateStore.set(entry)

      const retrieved = await stateStore.get(entry.documentHash)
      expect(retrieved).toBeDefined()
      expect(retrieved!.lastError).toBe('embedder timeout after 30s')
      expect(retrieved!.retryCount).toBe(3)
    })

    it('preserves completedAt field', async () => {
      const completedAt = new Date('2026-01-05T12:00:00Z')
      const entry = makeEntry({ stage: 'completed', completedAt })
      await stateStore.set(entry)

      const retrieved = await stateStore.get(entry.documentHash)
      expect(retrieved).toBeDefined()
      expect(retrieved!.completedAt).toBeDefined()
      expect(retrieved!.completedAt!.toISOString()).toBe(completedAt.toISOString())
    })
  })
}

describe('FilePipelineStateStore prefix configuration', () => {
  it('uses a custom prefix when provided via options', async () => {
    const store: Store = createMemStore()
    const customStore = new FilePipelineStateStore({ store, prefix: 'raw/.custom-state-dir' })
    const defaultStore = new FilePipelineStateStore(store)

    const entry = makeEntry()
    await customStore.set(entry)

    const fromCustom = await customStore.get(entry.documentHash)
    expect(fromCustom).toBeDefined()
    expect(fromCustom!.documentHash).toBe(entry.documentHash)

    const fromDefault = await defaultStore.get(entry.documentHash)
    expect(fromDefault).toBeUndefined()

    await store.close()
  })

  it('uses the default prefix when prefix is omitted in options', async () => {
    const store: Store = createMemStore()
    const configStore = new FilePipelineStateStore({ store })
    const defaultStore = new FilePipelineStateStore(store)

    const entry = makeEntry()
    await configStore.set(entry)

    const fromDefault = await defaultStore.get(entry.documentHash)
    expect(fromDefault).toBeDefined()

    await store.close()
  })
})

describe('FilePipelineStateStore document hash validation', () => {
  let store: Store
  let stateStore: PipelineStateStore

  beforeEach(() => {
    store = createMemStore()
    stateStore = new FilePipelineStateStore(store)
  })

  afterEach(async () => {
    await store.close()
  })

  const invalidHashes = [
    '../../../etc/passwd',
    'short',
    'ABCD000000000000000000000000000000000000000000000000000000000000',
    'abc123def456',
    'abc/123/def/456',
  ]

  for (const hash of invalidHashes) {
    it(`rejects invalid hash "${hash}" on get`, async () => {
      await expect(stateStore.get(hash)).rejects.toThrow('Invalid document hash')
    })

    it(`rejects invalid hash "${hash}" on delete`, async () => {
      await expect(stateStore.delete(hash)).rejects.toThrow('Invalid document hash')
    })

    it(`rejects invalid hash "${hash}" on set`, async () => {
      const entry = makeEntry({ documentHash: hash })
      await expect(stateStore.set(entry)).rejects.toThrow('Invalid document hash')
    })
  }
})
