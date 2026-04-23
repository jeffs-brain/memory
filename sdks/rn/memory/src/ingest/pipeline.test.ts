import { afterEach, describe, expect, it } from 'vitest'

import { createHashEmbedder } from '../llm/hashembed.js'
import { type SearchIndex, createSearchIndex } from '../search/index.js'
import { createMobileStore } from '../store/index.js'
import { createBetterSqliteOpenDb } from '../testing/better-sqlite-driver.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { type IngestProgress, ingestDocument } from './pipeline.js'

const searchIndices: SearchIndex[] = []

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

afterEach(async () => {
  while (searchIndices.length > 0) {
    const searchIndex = searchIndices.pop()
    if (searchIndex !== undefined) {
      await searchIndex.close()
    }
  }
})

const freshIndex = async (dim = 32): Promise<SearchIndex> => {
  const searchIndex = await createSearchIndex({
    dbPath: ':memory:',
    openDb: createBetterSqliteOpenDb(),
    vectorDim: dim,
  })
  searchIndices.push(searchIndex)
  return searchIndex
}

describe('ingest pipeline', () => {
  it('indexes markdown documents and emits progress across all stages', async () => {
    const store = await createMobileStore({
      root: '/brains/pipeline',
      adapter: createMemoryFileAdapter(),
    })
    const embedder = createHashEmbedder({ dim: 32 })
    const searchIndex = await freshIndex(32)
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
      onProgress: (event) => progress.push(event),
    })

    expect(result.chunkCount).toBe(3)
    expect(result.embeddedCount).toBe(3)
    expect(result.reused).toBe(false)
    expect(result.documentId.startsWith('brain-1:')).toBe(true)

    const bm25 = searchIndex.searchBm25('polyphonic', 5)
    expect(bm25.length).toBeGreaterThan(0)
    expect(bm25[0]?.chunk.content ?? '').toContain('Polyphonic')

    const [queryEmbedding] = await embedder.embed(['polyphonic synthesisers'])
    if (queryEmbedding === undefined) throw new Error('embed returned empty array')
    const vector = searchIndex.searchVector(queryEmbedding, 5)
    expect(vector.length).toBeGreaterThan(0)

    const stages = new Set(progress.map((event) => event.stage))
    expect(stages.has('store')).toBe(true)
    expect(stages.has('chunk')).toBe(true)
    expect(stages.has('embed')).toBe(true)
    expect(stages.has('index')).toBe(true)
  })

  it('reuses identical content on repeated ingest runs', async () => {
    const store = await createMobileStore({
      root: '/brains/pipeline-reuse',
      adapter: createMemoryFileAdapter(),
    })
    const embedder = createHashEmbedder({ dim: 32 })
    const searchIndex = await freshIndex(32)

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
})
