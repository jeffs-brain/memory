// SPDX-License-Identifier: Apache-2.0

/**
 * End-to-end test for the ingest pipeline against the real SQLite search
 * index. Uses hashembed + memstore so it stays offline.
 *
 * Also includes crash recovery tests that simulate failures at each stage
 * boundary and verify the pipeline resumes correctly on re-entry.
 */

import { createHash } from 'node:crypto'
import { afterEach, describe, expect, it } from 'vitest'
import { createHashEmbedder } from '../llm/hashembed.js'
import { type SearchIndex, createSearchIndex } from '../search/index.js'
import { createMemStore } from '../store/memstore.js'
import { type IngestProgress, ingestDocument } from './pipeline.js'
import { pipelineStatePath, readPipelineState } from './pipeline-state.js'
import { toPath } from '../store/index.js'
import {
  createMockEmbedder,
  createMockSearchIndex,
  createMockStore,
  testLogger,
} from './test-helpers.js'

const sha256Hex = (s: string): string =>
  createHash('sha256').update(Buffer.from(s, 'utf8')).digest('hex')

const indices: SearchIndex[] = []

const fresh = async (dim = 32): Promise<SearchIndex> => {
  const idx = await createSearchIndex({ dbPath: ':memory:', vectorDim: dim })
  indices.push(idx)
  return idx
}

afterEach(async () => {
  while (indices.length > 0) {
    const idx = indices.pop()
    if (idx !== undefined) await idx.close()
  }
})

const MARKDOWN_DOC = [
  '# Alpha Section',
  '',
  'The quick brown fox jumps over the lazy dog.',
  '',
  '# Beta Section',
  '',
  'Polyphonic synthesisers create harmonious soundscapes.',
  '',
  '# Gamma Section',
  '',
  'Reticulating splines with alacrity and precision.',
].join('\n')

describe('ingestDocument', () => {
  it('chunks a markdown doc and indexes every chunk for BM25 + hybrid search', async () => {
    const store = createMemStore()
    const embedder = createHashEmbedder({ dim: 32 })
    const searchIndex = await fresh(32)
    const progress: IngestProgress[] = []
    const result = await ingestDocument({
      store,
      searchIndex,
      embedder,
      doc: {
        brainId: 'brain-1',
        content: MARKDOWN_DOC,
        mime: 'text/markdown',
        title: 'Sample',
      },
      onProgress: (p) => progress.push(p),
    })

    expect(result.chunkCount).toBe(3)
    expect(result.embeddedCount).toBe(3)
    expect(result.reused).toBe(false)
    expect(result.documentId.startsWith('brain-1:')).toBe(true)

    // BM25 find term from chunk 2.
    const bm25 = searchIndex.searchBM25('polyphonic', 5)
    expect(bm25.length).toBeGreaterThan(0)
    expect(bm25[0]?.chunk.content ?? '').toMatch(/Polyphonic/)

    // Hybrid-ish: query embedding via hashembed and vector search.
    const [q] = await embedder.embed(['polyphonic synthesisers'])
    if (q === undefined) throw new Error('embed returned empty array')
    const vec = searchIndex.searchVector(q, 5)
    expect(vec.length).toBeGreaterThan(0)

    // Progress events fired for all four stages.
    const stages = new Set(progress.map((p) => p.stage))
    expect(stages.has('store')).toBe(true)
    expect(stages.has('chunk')).toBe(true)
    expect(stages.has('embed')).toBe(true)
    expect(stages.has('index')).toBe(true)
  })

  it('is idempotent: repeat ingest returns the existing document id and skips re-embed', async () => {
    const store = createMemStore()
    const embedder = createHashEmbedder({ dim: 32 })
    const searchIndex = await fresh(32)
    const first = await ingestDocument({
      store,
      searchIndex,
      embedder,
      doc: { brainId: 'brain-x', content: MARKDOWN_DOC, mime: 'text/markdown' },
    })
    const second = await ingestDocument({
      store,
      searchIndex,
      embedder,
      doc: { brainId: 'brain-x', content: MARKDOWN_DOC, mime: 'text/markdown' },
    })
    expect(second.reused).toBe(true)
    expect(second.documentId).toBe(first.documentId)
    expect(second.embeddedCount).toBe(0)
  })

  it('produces deterministic embeddings across hashembed runs', async () => {
    const text = '# Title\n\nsome deterministic body'
    const e1 = createHashEmbedder({ dim: 32 })
    const e2 = createHashEmbedder({ dim: 32 })
    const [a] = await e1.embed([text])
    const [b] = await e2.embed([text])
    expect(a).toBeDefined()
    expect(b).toBeDefined()
    if (a !== undefined && b !== undefined) {
      expect(a).toEqual(b)
    }
  })
})

describe('ingestDocument crash recovery', () => {
  const DOC_CONTENT = '# Recovery Test\n\nSome content for crash recovery testing.'

  it('resumes from chunking when embedder fails on first attempt', async () => {
    const store = createMockStore()
    const mockSearchIndex = createMockSearchIndex()
    let embedCallCount = 0
    const failingEmbedder = createMockEmbedder({
      embedFn: async (texts) => {
        embedCallCount++
        if (embedCallCount === 1) {
          throw new Error('embedder crashed')
        }
        return texts.map((t) => new Array(32).fill(t.length / 100))
      },
    })

    // First attempt: should fail at embed stage.
    await expect(
      ingestDocument({
        store,
        searchIndex: mockSearchIndex as unknown as SearchIndex,
        embedder: failingEmbedder,
        doc: { brainId: 'brain-cr', content: DOC_CONTENT, mime: 'text/markdown' },
        logger: testLogger,
      }),
    ).rejects.toThrow('embedder crashed')

    // Verify state was persisted at 'chunked' stage.
    const hash = sha256Hex(DOC_CONTENT)
    const state = await readPipelineState(store, hash, testLogger)
    expect(state).toBeDefined()
    expect(state?.stage).toBe('chunked')

    // Second attempt: should succeed, resuming from stored stage.
    const result = await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder: failingEmbedder,
      doc: { brainId: 'brain-cr', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })

    expect(result.reused).toBe(false)
    expect(result.chunkCount).toBeGreaterThan(0)
    expect(result.embeddedCount).toBeGreaterThan(0)
    expect(mockSearchIndex.upsertCallCount).toBe(1)
  })

  it('resumes from indexing when search index fails on first attempt', async () => {
    const store = createMockStore()
    let upsertCallCount = 0
    const mockSearchIndex = createMockSearchIndex({
      upsertFn: (_chunks) => {
        upsertCallCount++
        if (upsertCallCount === 1) {
          throw new Error('index upsert crashed')
        }
      },
    })
    const embedder = createMockEmbedder()

    // First attempt: should fail at index stage.
    await expect(
      ingestDocument({
        store,
        searchIndex: mockSearchIndex as unknown as SearchIndex,
        embedder,
        doc: { brainId: 'brain-idx', content: DOC_CONTENT, mime: 'text/markdown' },
        logger: testLogger,
      }),
    ).rejects.toThrow('index upsert crashed')

    // Verify state was persisted at 'embedded' stage.
    const hash = sha256Hex(DOC_CONTENT)
    const state = await readPipelineState(store, hash, testLogger)
    expect(state).toBeDefined()
    expect(state?.stage).toBe('embedded')

    // Second attempt: should succeed.
    const result = await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder,
      doc: { brainId: 'brain-idx', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })

    expect(result.reused).toBe(false)
    expect(result.chunkCount).toBeGreaterThan(0)
  })

  it('returns reused true only when pipeline completed fully with matching hash', async () => {
    const store = createMemStore()
    const embedder = createHashEmbedder({ dim: 32 })
    const searchIndex = await fresh(32)

    const first = await ingestDocument({
      store,
      searchIndex,
      embedder,
      doc: { brainId: 'brain-dup', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })
    expect(first.reused).toBe(false)

    // Pipeline state should be marked as 'indexed' after successful completion.
    const hash = sha256Hex(DOC_CONTENT)
    const stateAfter = await readPipelineState(store, hash, testLogger)
    expect(stateAfter).toBeDefined()
    expect(stateAfter?.stage).toBe('indexed')

    // Second ingest with same content: should be reused.
    const second = await ingestDocument({
      store,
      searchIndex,
      embedder,
      doc: { brainId: 'brain-dup', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })
    expect(second.reused).toBe(true)
    expect(second.documentId).toBe(first.documentId)
  })

  it('re-ingests fully when content hash differs from stored state', async () => {
    const store = createMockStore()
    const mockSearchIndex = createMockSearchIndex()
    const embedder = createMockEmbedder()

    const contentV1 = '# Version 1\n\nOriginal content.'
    const contentV2 = '# Version 2\n\nUpdated content with different hash.'

    // Ingest V1 successfully.
    const firstResult = await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder,
      doc: { brainId: 'brain-v', content: contentV1, mime: 'text/markdown' },
      logger: testLogger,
    })
    expect(firstResult.reused).toBe(false)

    // Ingest V2 with different content at the same brain: full re-process.
    const secondResult = await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder,
      doc: { brainId: 'brain-v', content: contentV2, mime: 'text/markdown' },
      logger: testLogger,
    })
    expect(secondResult.reused).toBe(false)
    expect(secondResult.hash).not.toBe(firstResult.hash)
    expect(secondResult.chunkCount).toBeGreaterThan(0)
  })

  it('marks pipeline state as indexed on successful full completion', async () => {
    const store = createMockStore()
    const mockSearchIndex = createMockSearchIndex()
    const embedder = createMockEmbedder()

    await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder,
      doc: { brainId: 'brain-clean', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })

    const hash = sha256Hex(DOC_CONTENT)
    const finalState = await readPipelineState(store, hash, testLogger)
    expect(finalState).toBeDefined()
    expect(finalState?.stage).toBe('indexed')
    expect(finalState?.hash).toBe(hash)
  })

  it('document stored but not indexed is re-processed on next ingest call', async () => {
    const store = createMockStore()
    const mockSearchIndex = createMockSearchIndex()
    let embedCallCount = 0
    const trackingEmbedder = createMockEmbedder({
      embedFn: async (texts) => {
        embedCallCount++
        if (embedCallCount === 1) {
          throw new Error('first embed fails')
        }
        return texts.map(() => new Array(32).fill(0.5))
      },
    })

    // First attempt crashes at embed stage — document is stored.
    await expect(
      ingestDocument({
        store,
        searchIndex: mockSearchIndex as unknown as SearchIndex,
        embedder: trackingEmbedder,
        doc: { brainId: 'brain-retry', content: DOC_CONTENT, mime: 'text/markdown' },
        logger: testLogger,
      }),
    ).rejects.toThrow('first embed fails')

    // Document exists in store but has no search index entries.
    expect(mockSearchIndex.upsertCallCount).toBe(0)

    // Re-ingest: should NOT return reused=true, should complete pipeline.
    const result = await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder: trackingEmbedder,
      doc: { brainId: 'brain-retry', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })

    expect(result.reused).toBe(false)
    expect(result.chunkCount).toBeGreaterThan(0)
    expect(mockSearchIndex.upsertCallCount).toBe(1)
  })

  it('does not skip indexing when store batch fails mid-write', async () => {
    const store = createMockStore()
    const mockSearchIndex = createMockSearchIndex()
    const embedder = createMockEmbedder()

    // Simulate a store batch failure by making the batch throw.
    const originalBatch = store.batch.bind(store)
    let batchCallCount = 0
    store.batch = async (opts, fn) => {
      batchCallCount++
      if (batchCallCount === 1) {
        throw new Error('batch write failed')
      }
      return originalBatch(opts, fn)
    }

    // First attempt: fails at store stage.
    await expect(
      ingestDocument({
        store,
        searchIndex: mockSearchIndex as unknown as SearchIndex,
        embedder,
        doc: { brainId: 'brain-batch', content: DOC_CONTENT, mime: 'text/markdown' },
        logger: testLogger,
      }),
    ).rejects.toThrow('batch write failed')

    // No state should have been written since store failed.
    const hash = sha256Hex(DOC_CONTENT)
    const state = await readPipelineState(store, hash, testLogger)
    expect(state).toBeUndefined()

    // Retry: should succeed from scratch.
    const result = await ingestDocument({
      store,
      searchIndex: mockSearchIndex as unknown as SearchIndex,
      embedder,
      doc: { brainId: 'brain-batch', content: DOC_CONTENT, mime: 'text/markdown' },
      logger: testLogger,
    })
    expect(result.reused).toBe(false)
    expect(result.chunkCount).toBeGreaterThan(0)
  })
})
