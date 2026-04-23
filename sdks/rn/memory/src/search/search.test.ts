import { afterEach, describe, expect, it } from 'vitest'

import { createBetterSqliteOpenDb } from '../testing/better-sqlite-driver.js'
import { type SearchIndex, createSearchIndex } from './index.js'

const DIM = 8
const indices: SearchIndex[] = []

const syntheticVector = (seed: number, dim = DIM): Float32Array => {
  const out = new Float32Array(dim)
  let state = (seed * 2654435761) >>> 0
  let norm = 0
  for (let index = 0; index < dim; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0
    const value = (state / 0xffffffff) * 2 - 1
    out[index] = value
    norm += value * value
  }
  norm = Math.sqrt(norm)
  for (let index = 0; index < dim; index += 1) {
    out[index] = (out[index] ?? 0) / norm
  }
  return out
}

const perturb = (vector: Float32Array, magnitude: number): Float32Array => {
  const out = new Float32Array(vector.length)
  let norm = 0
  for (let index = 0; index < vector.length; index += 1) {
    const value = (vector[index] ?? 0) + Math.sin(index + 1) * magnitude
    out[index] = value
    norm += value * value
  }
  norm = Math.sqrt(norm)
  for (let index = 0; index < vector.length; index += 1) {
    out[index] = (out[index] ?? 0) / norm
  }
  return out
}

const fresh = async (vectorDim = DIM): Promise<SearchIndex> => {
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

describe('createSearchIndex', () => {
  it('ranks title hits above content-only hits for the same token', async () => {
    const index = await fresh()

    index.upsertChunks([
      {
        id: 'title-hit',
        path: 'notes/title.md',
        title: 'Photosynthesis explained',
        summary: 'overview',
        content: 'Plants use light and water.',
      },
      {
        id: 'content-hit',
        path: 'notes/content.md',
        title: 'Gardening tips',
        summary: 'seasonal guide',
        content: 'Photosynthesis slows in autumn before dormancy.',
      },
    ])

    const results = index.searchBm25('photosynthesis', 10)
    expect(results[0]?.chunk.id).toBe('title-hit')
    expect(results[0]?.score).toBeLessThan(results[1]?.score ?? Number.POSITIVE_INFINITY)
  })

  it('returns the nearest vector result and exposes indexed chunk metadata', async () => {
    const index = await fresh(4)
    const alpha = syntheticVector(1, 4)
    const bravo = syntheticVector(2, 4)
    const charlie = syntheticVector(3, 4)

    index.upsertChunks([
      {
        id: 'alpha',
        path: 'memory/global/alpha.md',
        title: 'Alpha',
        content: 'alpha',
        embedding: alpha,
        embeddingModel: 'minilm',
      },
      {
        id: 'bravo',
        path: 'memory/global/bravo.md',
        title: 'Bravo',
        content: 'bravo',
        embedding: bravo,
        embeddingModel: 'minilm',
      },
      {
        id: 'charlie',
        path: 'memory/global/charlie.md',
        title: 'Charlie',
        content: 'charlie',
        embedding: charlie,
        embeddingModel: 'other-model',
      },
    ])

    const nearest = index.searchVector(perturb(bravo, 0.001), 3)
    expect(nearest[0]?.chunk.id).toBe('bravo')
    expect(index.chunkIdsWithVectorForModel('minilm')).toEqual(['alpha', 'bravo'])
    expect(index.indexedChunks().map((chunk) => chunk.id)).toEqual(['alpha', 'bravo', 'charlie'])
  })
})
