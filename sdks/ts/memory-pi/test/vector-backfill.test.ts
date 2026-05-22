// SPDX-License-Identifier: Apache-2.0

import type { Embedder } from '@jeffs-brain/memory'
import { createSearchIndex } from '@jeffs-brain/memory/search'
import { describe, expect, it } from 'vitest'
import { backfillSearchIndexVectors } from '../src/vector-backfill.js'

const fakeEmbedder = (dim: number): Embedder => ({
  name: () => 'fake',
  model: () => 'fake-embedder',
  dimension: () => dim,
  async embed(texts) {
    return texts.map((text, index) => {
      const seed = text.length + index + 1
      return Array.from({ length: dim }, (_, offset) => seed + offset)
    })
  },
})

describe('backfillSearchIndexVectors', () => {
  it('stores chunk-level vectors for indexed chunks missing the active model', async () => {
    const idx = await createSearchIndex({ dbPath: ':memory:', vectorDim: 3 })
    try {
      idx.upsertChunks([
        {
          id: 'memory/a.md#0',
          path: 'memory/a.md',
          ordinal: 0,
          title: 'A',
          content: 'alpha content',
        },
        {
          id: 'memory/a.md#1',
          path: 'memory/a.md',
          ordinal: 1,
          title: 'A',
          content: 'beta content',
        },
      ])

      const result = await backfillSearchIndexVectors({
        brainId: 'test',
        searchIndex: idx,
        embedder: fakeEmbedder(3),
        model: 'fake-embedder',
      })

      expect(result.embedded).toBe(2)
      expect(idx.chunkIdsWithVectorForModel('fake-embedder').sort()).toEqual([
        'memory/a.md#0',
        'memory/a.md#1',
      ])

      const second = await backfillSearchIndexVectors({
        brainId: 'test',
        searchIndex: idx,
        embedder: fakeEmbedder(3),
        model: 'fake-embedder',
      })
      expect(second.embedded).toBe(0)
    } finally {
      await idx.close()
    }
  })
})
