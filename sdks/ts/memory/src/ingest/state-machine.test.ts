// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the pipeline state machine. Validates all stage transitions,
 * retry logic, dead letter behaviour, and persistence integration.
 */

import { describe, expect, it } from 'vitest'
import { createActor } from 'xstate'
import type { PipelineMachineEvent, PipelineStage } from './state-machine.js'
import {
  type PipelineMachineContext,
  type PipelineStateEntry,
  type PipelineStateStore,
  type TransitionCallback,
  createPipelineStateMachine,
  isValidTransition,
  pipelineMachine,
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
        if (entry.stage !== 'completed' && entry.stage !== 'failed') {
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
    expect(isValidTransition('indexed', 'completed')).toBe(true)
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
    expect(isValidTransition('chunked', 'completed')).toBe(false)
  })

  it('rejects transitions from terminal states', () => {
    expect(isValidTransition('completed', 'received')).toBe(false)
    expect(isValidTransition('failed', 'received')).toBe(false)
  })
})

describe('pipelineMachine (XState)', () => {
  const makeActor = (
    initialStage: PipelineStage = 'received',
    retryCount = 0,
  ) => {
    const context: PipelineMachineContext = {
      documentHash: 'test-hash',
      retryCount,
      maxRetries: 3,
      lastError: '',
    }

    if (initialStage === 'received') {
      return createActor(pipelineMachine, { input: context })
    }

    const resolved = pipelineMachine.resolveState({
      value: initialStage,
      context,
    })

    return createActor(pipelineMachine, {
      input: context,
      snapshot: resolved,
    })
  }

  it('transitions through all stages in order', () => {
    const actor = makeActor('received')
    actor.start()

    expect(actor.getSnapshot().value).toBe('received')

    actor.send({ type: 'STORE_COMPLETE' })
    expect(actor.getSnapshot().value).toBe('stored')

    actor.send({ type: 'CHUNK_COMPLETE' })
    expect(actor.getSnapshot().value).toBe('chunked')

    actor.send({ type: 'EMBED_COMPLETE' })
    expect(actor.getSnapshot().value).toBe('embedded')

    actor.send({ type: 'INDEX_COMPLETE' })
    expect(actor.getSnapshot().value).toBe('indexed')

    actor.send({ type: 'COMPLETE' })
    expect(actor.getSnapshot().value).toBe('completed')

    actor.stop()
  })

  it('rejects invalid events for the current state', () => {
    const actor = makeActor('received')
    actor.start()

    // CHUNK_COMPLETE is not valid in received state
    actor.send({ type: 'CHUNK_COMPLETE' })
    expect(actor.getSnapshot().value).toBe('received')

    actor.stop()
  })

  it('stays in current state on FAIL when retries remain', () => {
    const actor = makeActor('chunked', 0)
    actor.start()

    actor.send({ type: 'FAIL', error: 'transient error' })
    expect(actor.getSnapshot().value).toBe('chunked')
    expect(actor.getSnapshot().context.retryCount).toBe(1)
    expect(actor.getSnapshot().context.lastError).toBe('transient error')

    actor.stop()
  })

  it('transitions to failed when retries exhausted via FAIL', () => {
    const actor = makeActor('embedded', 2)
    actor.start()

    // retryCount=2, maxRetries=3, so canRetry guard fails (2 < 3 is true, one more)
    actor.send({ type: 'FAIL', error: 'third failure' })
    expect(actor.getSnapshot().value).toBe('embedded')
    expect(actor.getSnapshot().context.retryCount).toBe(3)

    // Now retryCount=3, guard fails
    actor.send({ type: 'FAIL', error: 'fourth failure' })
    expect(actor.getSnapshot().value).toBe('failed')

    actor.stop()
  })

  it('transitions to failed on RETRY_EXHAUSTED from any stage', () => {
    const actor = makeActor('stored')
    actor.start()

    actor.send({ type: 'RETRY_EXHAUSTED', error: 'gave up' })
    expect(actor.getSnapshot().value).toBe('failed')
    expect(actor.getSnapshot().context.lastError).toBe('gave up')

    actor.stop()
  })
})

describe('createPipelineStateMachine', () => {
  it('advances state forward and persists', async () => {
    const store = createMemStore()
    await seedEntry(store, 'doc-1', 'received')
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.advanceStage('doc-1', { type: 'STORE_COMPLETE' })
    expect(entry.stage).toBe('stored')

    const loaded = await store.load('doc-1')
    expect(loaded?.stage).toBe('stored')
  })

  it('advances through the full pipeline', async () => {
    const store = createMemStore()
    await seedEntry(store, 'full-doc', 'received')
    const sm = createPipelineStateMachine({ stateStore: store })

    const events: PipelineMachineEvent[] = [
      { type: 'STORE_COMPLETE' },
      { type: 'CHUNK_COMPLETE' },
      { type: 'EMBED_COMPLETE' },
      { type: 'INDEX_COMPLETE' },
      { type: 'COMPLETE' },
    ]
    const expectedStages: PipelineStage[] = ['stored', 'chunked', 'embedded', 'indexed', 'completed']

    for (let i = 0; i < events.length; i++) {
      const event = events[i]
      const expected = expectedStages[i]
      if (event === undefined || expected === undefined) break
      const entry = await sm.advanceStage('full-doc', event)
      expect(entry.stage).toBe(expected)
    }
  })

  it('records failure and increments retry count', async () => {
    const store = createMemStore()
    await seedEntry(store, 'retry-doc', 'chunked')
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.advanceStage('retry-doc', { type: 'FAIL', error: 'timeout' })
    expect(entry.stage).toBe('chunked')
    expect(entry.retryCount).toBe(1)
    expect(entry.lastError).toBe('timeout')
  })

  it('moves to failed after max retries exhausted', async () => {
    const store = createMemStore()
    await seedEntry(store, 'exhaust-doc', 'embedded', 2)
    const sm = createPipelineStateMachine({ stateStore: store, maxRetries: 3 })

    // retryCount=2, one more allowed
    const first = await sm.advanceStage('exhaust-doc', { type: 'FAIL', error: 'err-3' })
    expect(first.stage).toBe('embedded')
    expect(first.retryCount).toBe(3)

    // Now at max, next FAIL moves to failed
    const second = await sm.advanceStage('exhaust-doc', { type: 'FAIL', error: 'err-4' })
    expect(second.stage).toBe('failed')
  })

  it('markFailed transitions immediately to failed', async () => {
    const store = createMemStore()
    await seedEntry(store, 'mark-doc', 'stored')
    const sm = createPipelineStateMachine({ stateStore: store })

    const entry = await sm.markFailed('mark-doc', 'fatal error')
    expect(entry.stage).toBe('failed')
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
    await sm.advanceStage('cb-doc', { type: 'STORE_COMPLETE' })
    await sm.advanceStage('cb-doc', { type: 'CHUNK_COMPLETE' })

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

    // CHUNK_COMPLETE is invalid in received state, no transition
    await sm.advanceStage('nocb-doc', { type: 'CHUNK_COMPLETE' })
    expect(callCount).toBe(0)
  })

  it('failed documents queryable via listIncomplete exclusion', async () => {
    const store = createMemStore()
    await seedEntry(store, 'active-1', 'chunked')
    await seedEntry(store, 'active-2', 'received')

    const now = new Date().toISOString()
    await store.save({
      documentHash: 'done-1',
      stage: 'completed',
      retryCount: 0,
      lastError: '',
      createdAt: now,
      updatedAt: now,
    })
    await store.save({
      documentHash: 'dead-1',
      stage: 'failed',
      retryCount: 3,
      lastError: 'permanent',
      createdAt: now,
      updatedAt: now,
    })

    const incomplete = await store.listIncomplete()
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

    const entry = await sm.advanceStage('new-doc', { type: 'STORE_COMPLETE' })
    expect(entry.stage).toBe('stored')
    expect(entry.retryCount).toBe(0)
  })
})
