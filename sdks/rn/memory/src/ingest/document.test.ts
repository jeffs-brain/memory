import { afterEach, describe, expect, it } from 'vitest'

import { type SearchIndex, createSearchIndex } from '../search/index.js'
import { createMobileStore, toPath } from '../store/index.js'
import { createBetterSqliteOpenDb } from '../testing/better-sqlite-driver.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { ingestDocument } from './document.js'

const resources: Array<{ readonly searchIndex: SearchIndex }> = []

afterEach(async () => {
  while (resources.length > 0) {
    const resource = resources.pop()
    if (resource !== undefined) {
      await resource.searchIndex.close()
    }
  }
})

describe('ingestDocument', () => {
  it('stores the source document and indexes chunks with metadata and embeddings', async () => {
    const store = await createMobileStore({
      root: '/brains/docs',
      adapter: createMemoryFileAdapter(),
    })
    const searchIndex = await createSearchIndex({
      dbPath: ':memory:',
      openDb: createBetterSqliteOpenDb(),
      vectorDim: 2,
    })
    resources.push({ searchIndex })

    const embedder = {
      name: () => 'test-embedder',
      model: () => 'test-embedder-model',
      dimension: () => 2,
      embed: async (texts: readonly string[]) => texts.map(() => [1, 0]),
    }

    const result = await ingestDocument({
      path: 'docs/guide.txt',
      content: 'A concise guide to retrieval on mobile.',
      mime: 'text/plain',
      title: 'Guide',
      metadata: { source: 'test-suite' },
      searchIndex,
      embedder,
      store,
    })

    expect(result).toMatchObject({
      path: 'docs/guide.txt',
      title: 'Guide',
      mime: 'text/plain',
      chunkCount: 1,
      embeddedCount: 1,
    })
    await expect(store.read(toPath('docs/guide.txt'))).resolves.toBe(
      'A concise guide to retrieval on mobile.',
    )

    const chunks = searchIndex.indexedChunks()
    expect(chunks).toHaveLength(1)
    expect(chunks[0]).toMatchObject({
      path: 'docs/guide.txt',
      title: 'Guide',
      summary: 'Guide',
      metadata: {
        type: 'document',
        mime: 'text/plain',
        source: 'test-suite',
      },
    })
    expect(searchIndex.chunkIdsWithVectorForModel('test-embedder-model')).toEqual([
      'docs/guide.txt:0',
    ])
  })
})
