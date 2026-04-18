/**
 * End-to-end test for the ingest pipeline against the real SQLite search
 * index. Uses hashembed + memstore so it stays offline.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { createHashEmbedder } from '../llm/hashembed.js'
import { createMemStore } from '../store/memstore.js'
import { createSearchIndex, type SearchIndex } from '../search/index.js'
import { ingestDocument, type IngestProgress } from './pipeline.js'

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
