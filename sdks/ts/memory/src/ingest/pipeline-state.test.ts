// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for pipeline state persistence: read, write, delete, and
 * stage ordering logic.
 */

import { describe, expect, it } from 'vitest'
import { toPath } from '../store/index.js'
import {
  type PipelineState,
  deletePipelineState,
  isStageComplete,
  pipelineStatePath,
  readPipelineState,
  writePipelineState,
} from './pipeline-state.js'
import { createMockStore, testLogger } from './test-helpers.js'

const SAMPLE_HASH = 'abc123def456'

const sampleState: PipelineState = {
  documentId: 'brain-1:abc123def456ab',
  hash: SAMPLE_HASH,
  stage: 'embedded',
  updatedAt: '2026-05-09T12:00:00.000Z',
  chunkCount: 5,
}

describe('pipelineStatePath', () => {
  it('builds the correct store path for a given hash', () => {
    const path = pipelineStatePath('deadbeef1234')
    expect(path as string).toBe('raw/.pipeline-state/deadbeef1234.json')
  })
})

describe('isStageComplete', () => {
  const cases: ReadonlyArray<{
    readonly current: PipelineState['stage']
    readonly target: PipelineState['stage']
    readonly expected: boolean
  }> = [
    { current: 'stored', target: 'stored', expected: true },
    { current: 'stored', target: 'chunked', expected: false },
    { current: 'stored', target: 'embedded', expected: false },
    { current: 'stored', target: 'indexed', expected: false },
    { current: 'chunked', target: 'stored', expected: true },
    { current: 'chunked', target: 'chunked', expected: true },
    { current: 'chunked', target: 'embedded', expected: false },
    { current: 'embedded', target: 'stored', expected: true },
    { current: 'embedded', target: 'embedded', expected: true },
    { current: 'embedded', target: 'indexed', expected: false },
    { current: 'indexed', target: 'stored', expected: true },
    { current: 'indexed', target: 'indexed', expected: true },
  ]

  for (const { current, target, expected } of cases) {
    it(`${current} >= ${target} is ${expected}`, () => {
      expect(isStageComplete(current, target)).toBe(expected)
    })
  }
})

describe('readPipelineState', () => {
  it('returns undefined when no state file exists', async () => {
    const store = createMockStore()
    const state = await readPipelineState(store, SAMPLE_HASH, testLogger)
    expect(state).toBeUndefined()
  })

  it('returns parsed state when file exists and is valid', async () => {
    const store = createMockStore()
    const path = pipelineStatePath(SAMPLE_HASH)
    await store.write(toPath(path as string), Buffer.from(JSON.stringify(sampleState), 'utf8'))

    const state = await readPipelineState(store, SAMPLE_HASH, testLogger)
    expect(state).toEqual(sampleState)
  })

  it('returns undefined when state file contains invalid JSON', async () => {
    const store = createMockStore()
    const path = pipelineStatePath(SAMPLE_HASH)
    await store.write(toPath(path as string), Buffer.from('not json {{{', 'utf8'))

    const state = await readPipelineState(store, SAMPLE_HASH, testLogger)
    expect(state).toBeUndefined()
  })

  it('returns undefined when state file has missing required fields', async () => {
    const store = createMockStore()
    const path = pipelineStatePath(SAMPLE_HASH)
    const partial = { documentId: 'brain-1:abc', hash: SAMPLE_HASH }
    await store.write(toPath(path as string), Buffer.from(JSON.stringify(partial), 'utf8'))

    const state = await readPipelineState(store, SAMPLE_HASH, testLogger)
    expect(state).toBeUndefined()
  })

  it('returns undefined when stage field is not a valid pipeline stage', async () => {
    const store = createMockStore()
    const path = pipelineStatePath(SAMPLE_HASH)
    const invalid = { ...sampleState, stage: 'unknown-stage' }
    await store.write(toPath(path as string), Buffer.from(JSON.stringify(invalid), 'utf8'))

    const state = await readPipelineState(store, SAMPLE_HASH, testLogger)
    expect(state).toBeUndefined()
  })
})

describe('writePipelineState', () => {
  it('writes state as JSON to the correct path', async () => {
    const store = createMockStore()
    await writePipelineState(store, sampleState)

    const expectedPath = pipelineStatePath(SAMPLE_HASH)
    const buf = await store.read(toPath(expectedPath as string))
    const parsed = JSON.parse(buf.toString('utf8'))
    expect(parsed).toEqual(sampleState)
  })

  it('overwrites existing state at the same path', async () => {
    const store = createMockStore()
    await writePipelineState(store, sampleState)

    const updated: PipelineState = { ...sampleState, stage: 'indexed' }
    await writePipelineState(store, updated)

    const expectedPath = pipelineStatePath(SAMPLE_HASH)
    const buf = await store.read(toPath(expectedPath as string))
    const parsed = JSON.parse(buf.toString('utf8'))
    expect(parsed.stage).toBe('indexed')
  })
})

describe('deletePipelineState', () => {
  it('removes the state file from the store', async () => {
    const store = createMockStore()
    await writePipelineState(store, sampleState)

    await deletePipelineState(store, SAMPLE_HASH, testLogger)

    const exists = await store.exists(toPath(pipelineStatePath(SAMPLE_HASH) as string))
    expect(exists).toBe(false)
  })

  it('does not throw when the state file does not exist', async () => {
    const store = createMockStore()
    await expect(
      deletePipelineState(store, 'nonexistent-hash', testLogger),
    ).resolves.toBeUndefined()
  })
})
