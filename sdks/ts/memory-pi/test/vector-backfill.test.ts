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

  it('keeps default backfill batches below TEI client batch limits', async () => {
    const idx = await createSearchIndex({ dbPath: ':memory:', vectorDim: 3 })
    const batchSizes: number[] = []
    const embedder: Embedder = {
      name: () => 'fake',
      model: () => 'fake-embedder',
      dimension: () => 3,
      async embed(texts) {
        batchSizes.push(texts.length)
        return texts.map((text, index) => {
          const seed = text.length + index + 1
          return Array.from({ length: 3 }, (_, offset) => seed + offset)
        })
      },
    }

    try {
      idx.upsertChunks(
        Array.from({ length: 40 }, (_, index) => ({
          id: `memory/b.md#${index}`,
          path: 'memory/b.md',
          ordinal: index,
          title: 'B',
          content: `content ${index}`,
        })),
      )

      const result = await backfillSearchIndexVectors({
        brainId: 'test',
        searchIndex: idx,
        embedder,
        model: 'fake-embedder',
      })

      expect(result.embedded).toBe(40)
      expect(batchSizes).toEqual([16, 16, 8])
    } finally {
      await idx.close()
    }
  })
})
