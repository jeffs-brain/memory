// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the pipeline state machine. Validates all stage transitions,
 * retry logic, dead letter behaviour, migration from V1, and listIncomplete.
 */

import { describe, expect, it } from 'vitest'
import type {
  PipelineStage,
  PipelineStateEntry,
  PipelineStateStore,
  TransitionCallback,
  V1PipelineStateEntry,
} from './state-machine.js'
import {
  createPipelineStateMachine,
  isValidTransition,
  migrateFromV1,
} from './state-machine.js'

/** In-memory test store for pipeline state entries. */
const createMemStore = (): PipelineStateStore => {
  const entries = new Map<string, PipelineStateEntry>()
  return {
    load: async (hash: string) => entries.get(hash),
    save: async (entry: PipelineStateEntry) => {
      entries.set(entry.documentHash, entry)
    },
    listIncomplete: async () => {
      const result: PipelineStateEntry[] = []
      for (const entry of entries.values()) {
        if (entry.stage !== 'indexed' && entry.stage !== 'dead_letter') {
          result.push(entry)
        }
      }
      return result
    },
  }
}

const seedEntry = async (
  store: PipelineStateStore,
  hash: string,
  stage: PipelineStage,
  retryCount = 0,
): Promise<void> => {
  const now = new Date().toISOString()
  await store.save({
    documentHash: hash,
    stage,
    retryCount,
    lastError: '',
    createdAt: now,
    updatedAt: now,
  })
}

describe('isValidTransition', () => {
  it('allows forward transitions', () => {
    expect(isValidTransition('received', 'stored')).toBe(true)
    expect(isValidTransition('stored', 'chunked')).toBe(true)
    expect(isValidTransition('chunked', 'embedded')).toBe(true)
    expect(isValidTransition('embedded', 'indexed')).toBe(true)
  })

  it('rejects backward transitions', () => {
    expect(isValidTransition('stored', 'received')).toBe(false)
    expect(isValidTransition('chunked', 'stored')).toBe(false)
    expect(isValidTransition('indexed', 'received')).toBe(false)
  })

  it('rejects skipped transitions', () => {
    expect(isValidTransition('received', 'chunked')).toBe(false)
    expect(isValidTransition('received', 'embedded')).toBe(false)
    expect(isValidTransition('stored', 'indexed')).toBe(false)
  })

  it('rejects transitions from terminal states', () => {
    expect(isValidTransition('dead_letter', 'received')).toBe(false)
    expect(isValidTransition('indexed', 'received')).toBe(false)
  })
})

describe('createPipelineStateMachine', () => {
  it('advances state forward and persists', async () => {
    const store = createMemStore()
    await seedEntry(store, 'doc-1', 'received')
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.advance('doc-1', 'stored')
    expect(entry.stage).toBe('stored')

    const loaded = await store.load('doc-1')
    expect(loaded?.stage).toBe('stored')
  })

  it('advances through the full pipeline', async () => {
    const store = createMemStore()
    await seedEntry(store, 'full-doc', 'received')
    const sm = createPipelineStateMachine({ stateStore: store })

    const stages: PipelineStage[] = ['stored', 'chunked', 'embedded', 'indexed']
    for (const target of stages) {
      const entry = await sm.advance('full-doc', target)
      expect(entry.stage).toBe(target)
    }
  })

  it('is idempotent for same-stage advance', async () => {
    const store = createMemStore()
    await seedEntry(store, 'idem-doc', 'chunked')
    const sm = createPipelineStateMachine({ stateStore: store })

    // Trying to advance to chunked when already at chunked is a no-op.
    const entry = await sm.advance('idem-doc', 'chunked')
    expect(entry.stage).toBe('chunked')
  })

  it('records failure and increments retry count', async () => {
    const store = createMemStore()
    await seedEntry(store, 'retry-doc', 'chunked')
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.recordFailure('retry-doc', 'timeout')
    expect(entry.stage).toBe('chunked')
    expect(entry.retryCount).toBe(1)
    expect(entry.lastError).toBe('timeout')
  })

  it('moves to dead_letter after max retries exhausted', async () => {
    const store = createMemStore()
    await seedEntry(store, 'exhaust-doc', 'embedded', 2)
    const sm = createPipelineStateMachine({ stateStore: store, maxRetries: 3 })

    const entry = await sm.recordFailure('exhaust-doc', 'permanent error')
    expect(entry.stage).toBe('dead_letter')
    expect(entry.retryCount).toBe(3)
  })

  it('markDeadLetter transitions immediately to dead_letter', async () => {
    const store = createMemStore()
    await seedEntry(store, 'mark-doc', 'stored')
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.markDeadLetter('mark-doc', 'fatal error')
    expect(entry.stage).toBe('dead_letter')
    expect(entry.lastError).toBe('fatal error')
  })

  it('shouldRetry returns true when under limit', () => {
    const sm = createPipelineStateMachine({ stateStore: createMemStore() })
    const entry: PipelineStateEntry = {
      documentHash: 'x',
      stage: 'chunked',
      retryCount: 0,
      lastError: '',
      createdAt: '',
      updatedAt: '',
    }
    expect(sm.shouldRetry(entry)).toBe(true)
    expect(sm.shouldRetry({ ...entry, retryCount: 2 })).toBe(true)
  })

  it('shouldRetry returns false at or above limit', () => {
    const sm = createPipelineStateMachine({ stateStore: createMemStore() })
    const entry: PipelineStateEntry = {
      documentHash: 'x',
      stage: 'chunked',
      retryCount: 3,
      lastError: '',
      createdAt: '',
      updatedAt: '',
    }
    expect(sm.shouldRetry(entry)).toBe(false)
    expect(sm.shouldRetry({ ...entry, retryCount: 5 })).toBe(false)
  })

  it('invokes transition callback on state change', async () => {
    const store = createMemStore()
    await seedEntry(store, 'cb-doc', 'received')

    const transitions: Array<{ from: PipelineStage; to: PipelineStage }> = []
    const onTransition: TransitionCallback = (_hash, from, to) => {
      transitions.push({ from, to })
    }

    const sm = createPipelineStateMachine({ stateStore: store, onTransition })
    await sm.advance('cb-doc', 'stored')
    await sm.advance('cb-doc', 'chunked')

    expect(transitions).toEqual([
      { from: 'received', to: 'stored' },
      { from: 'stored', to: 'chunked' },
    ])
  })

  it('does not invoke callback when state does not change', async () => {
    const store = createMemStore()
    await seedEntry(store, 'nocb-doc', 'received')

    let callCount = 0
    const sm = createPipelineStateMachine({
      stateStore: store,
      onTransition: () => { callCount++ },
    })

    // Trying to advance to 'chunked' from 'received' is invalid (skip), no transition.
    await sm.advance('nocb-doc', 'chunked')
    expect(callCount).toBe(0)
  })

  it('listIncomplete returns only non-terminal entries', async () => {
    const store = createMemStore()
    await seedEntry(store, 'active-1', 'chunked')
    await seedEntry(store, 'active-2', 'received')

    const now = new Date().toISOString()
    await store.save({
      documentHash: 'done-1',
      stage: 'indexed',
      retryCount: 0,
      lastError: '',
      createdAt: now,
      updatedAt: now,
    })
    await store.save({
      documentHash: 'dead-1',
      stage: 'dead_letter',
      retryCount: 3,
      lastError: 'permanent',
      createdAt: now,
      updatedAt: now,
    })

    const sm = createPipelineStateMachine({ stateStore: store })
    const incomplete = await sm.listIncomplete()
    const hashes = incomplete.map((e) => e.documentHash)
    expect(hashes).toHaveLength(2)
    expect(hashes).toContain('active-1')
    expect(hashes).toContain('active-2')
    expect(hashes).not.toContain('done-1')
    expect(hashes).not.toContain('dead-1')
  })

  it('creates entry for unknown document hash starting at received', async () => {
    const store = createMemStore()
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.advance('new-doc', 'stored')
    expect(entry.stage).toBe('stored')
    expect(entry.retryCount).toBe(0)
  })
})

describe('migrateFromV1', () => {
  it('converts a V1 entry to V2 format', () => {
    const v1: V1PipelineStateEntry = {
      documentId: 'brain-1:abc123',
      hash: 'abc123',
      stage: 'embedded',
      updatedAt: '2026-05-09T12:00:00.000Z',
      chunkCount: 5,
    }
    const v2 = migrateFromV1(v1)
    expect(v2.documentHash).toBe('abc123')
    expect(v2.stage).toBe('embedded')
    expect(v2.retryCount).toBe(0)
    expect(v2.lastError).toBe('')
    expect(v2.createdAt).toBe('2026-05-09T12:00:00.000Z')
  })

  it('maps all V1 stages correctly', () => {
    const stages: Array<V1PipelineStateEntry['stage']> = ['stored', 'chunked', 'embedded', 'indexed']
    for (const stage of stages) {
      const v1: V1PipelineStateEntry = {
        documentId: 'brain-1:test',
        hash: 'test-hash',
        stage,
        updatedAt: '2026-05-09T12:00:00.000Z',
      }
      const v2 = migrateFromV1(v1)
      expect(v2.stage).toBe(stage)
    }
  })
})
