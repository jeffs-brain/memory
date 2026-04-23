import { afterEach, describe, expect, it } from 'vitest'

import { type SearchIndex, createSearchIndex } from '../search/index.js'
import { createBetterSqliteOpenDb } from '../testing/better-sqlite-driver.js'
import { createRetrieval } from './index.js'

const indices: SearchIndex[] = []

const fresh = async (vectorDim = 4): Promise<SearchIndex> => {
  const index = await createSearchIndex({
    dbPath: ':memory:',
    openDb: createBetterSqliteOpenDb(),
    vectorDim,
  })
  indices.push(index)
  return index
}

afterEach(async () => {
  while (indices.length > 0) {
    await indices.pop()?.close()
  }
})

describe('createRetrieval', () => {
  it('falls back to BM25 when semantic mode is requested without an embedder', async () => {
    const index = await fresh()
    index.upsertChunk({
      id: 'cycling',
      path: 'memory/global/cycling.md',
      title: 'Cycling',
      summary: 'Likes cycling',
      content: 'Cycling every weekend.',
      metadata: { scope: 'global' },
    })

    const retrieval = createRetrieval({ index })
    const response = await retrieval.searchRaw({
      query: 'cycling',
      mode: 'semantic',
      filters: { scope: 'global' },
    })

    expect(response.trace.mode).toBe('bm25')
    expect(response.results[0]?.id).toBe('cycling')
  })

  it('uses trigram fallback for slug typos and honours path filters', async () => {
    const index = await fresh()
    index.upsertChunks([
      {
        id: 'allowed',
        path: 'memory/global/photosynthesis.md',
        title: 'Allowed note',
        summary: 'summary',
        content: 'body with no matching terms',
        metadata: { scope: 'global' },
      },
      {
        id: 'blocked',
        path: 'memory/global/photosynthasia.md',
        title: 'Blocked note',
        summary: 'summary',
        content: 'body with no matching terms',
        metadata: { scope: 'global' },
      },
    ])

    const retrieval = createRetrieval({ index })
    const response = await retrieval.searchRaw({
      query: 'photosynthasis',
      filters: { paths: ['memory/global/photosynthesis.md'] },
    })

    expect(response.trace.attempts.map((attempt) => attempt.strategy)).toContain('trigram_fuzzy')
    expect(response.results.map((result) => result.id)).toEqual(['allowed'])
  })

  it('blends vector search with filters when an embedder is available', async () => {
    const index = await fresh(2)
    index.upsertChunks([
      {
        id: 'global-note',
        path: 'memory/global/cycling.md',
        title: 'Cycling',
        summary: 'Global cycling note',
        content: 'Road bike preference',
        embedding: [1, 0],
        metadata: { scope: 'global', tags: ['sport'] },
      },
      {
        id: 'project-note',
        path: 'memory/project/jb/cycling.md',
        title: 'Cycling project',
        summary: 'Project note',
        content: 'Sprint backlog',
        embedding: [1, 0],
        metadata: { scope: 'project', tags: ['work'] },
      },
    ])

    const retrieval = createRetrieval({
      index,
      embedder: {
        embed: async () => [[1, 0]],
      },
    })

    const results = await retrieval.search({
      query: 'exercise plan',
      mode: 'hybrid',
      filters: {
        pathPrefix: 'memory/global/',
        scope: 'global',
        tags: ['sport'],
      },
    })

    expect(results.map((result) => result.id)).toEqual(['global-note'])
  })

  it('reranks fused results when a reranker is configured', async () => {
    const index = await fresh(2)
    index.upsertChunks([
      {
        id: 'first',
        path: 'memory/global/first.md',
        title: 'First',
        summary: 'First note',
        content: 'alpha',
        embedding: [1, 0],
        metadata: { scope: 'global' },
      },
      {
        id: 'second',
        path: 'memory/global/second.md',
        title: 'Second',
        summary: 'Second note',
        content: 'beta',
        embedding: [1, 0],
        metadata: { scope: 'global' },
      },
    ])

    const retrieval = createRetrieval({
      index,
      embedder: {
        embed: async () => [[1, 0]],
      },
      reranker: {
        name: () => 'stub-reranker',
        rerank: async () => [
          { index: 1, id: 'second', score: 9 },
          { index: 0, id: 'first', score: 1 },
        ],
      },
    })

    const response = await retrieval.searchRaw({
      query: 'alpha beta',
      mode: 'hybrid',
      rerank: true,
      filters: { scope: 'global' },
    })

    expect(response.trace.reranked).toBe(true)
    expect(response.trace.rerankProvider).toBe('stub-reranker')
    expect(response.results[0]?.id).toBe('second')
    expect(response.results[0]?.rerankScore).toBe(9)
  })
})
