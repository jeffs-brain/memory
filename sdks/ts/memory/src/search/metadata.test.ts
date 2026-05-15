// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for per-chunk key-value metadata (knowledge_chunk_metadata).
 * Mirrors the Go test coverage in go/search/metadata_test.go.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { type SearchIndex, createSearchIndex } from './index.js'

const DIM = 8

const indices: SearchIndex[] = []

async function fresh(): Promise<SearchIndex> {
  const idx = await createSearchIndex({ dbPath: ':memory:', vectorDim: DIM })
  indices.push(idx)
  return idx
}

afterEach(async () => {
  while (indices.length > 0) {
    const idx = indices.pop()
    if (idx !== undefined) await idx.close()
  }
})

describe('setChunkMetadata / getChunkMetadata', () => {
  it('round-trips multiple key-value pairs', async () => {
    const idx = await fresh()

    const meta: Record<string, string> = {
      ontology_type: 'factual',
      confidence: '0.95',
      source_model: 'gpt-4o',
    }
    idx.setChunkMetadata('chunk_001', meta)

    const got = idx.getChunkMetadata('chunk_001')
    expect(Object.keys(got)).toHaveLength(3)
    expect(got['ontology_type']).toBe('factual')
    expect(got['confidence']).toBe('0.95')
    expect(got['source_model']).toBe('gpt-4o')
  })

  it('overwrites existing values for the same key', async () => {
    const idx = await fresh()

    idx.setChunkMetadata('chunk_002', { confidence: '0.5' })
    idx.setChunkMetadata('chunk_002', { confidence: '0.99' })

    const got = idx.getChunkMetadata('chunk_002')
    expect(got['confidence']).toBe('0.99')
  })

  it('returns an empty object for a non-existent chunk', async () => {
    const idx = await fresh()

    const got = idx.getChunkMetadata('nonexistent_chunk')
    expect(got).toBeDefined()
    expect(Object.keys(got)).toHaveLength(0)
  })
})

describe('queryByMetadata', () => {
  it('returns matching chunk IDs ordered by chunk_id', async () => {
    const idx = await fresh()

    idx.setChunkMetadata('chunk_a', {
      ontology_type: 'factual',
      topic: 'physics',
    })
    idx.setChunkMetadata('chunk_b', {
      ontology_type: 'opinion',
      topic: 'physics',
    })
    idx.setChunkMetadata('chunk_c', {
      ontology_type: 'factual',
      topic: 'chemistry',
    })

    const ids = idx.queryByMetadata('ontology_type', 'factual', 10)
    expect(ids).toHaveLength(2)
    expect(ids[0]).toBe('chunk_a')
    expect(ids[1]).toBe('chunk_c')

    const topicIds = idx.queryByMetadata('topic', 'physics', 10)
    expect(topicIds).toHaveLength(2)
  })

  it('respects the limit parameter', async () => {
    const idx = await fresh()

    for (const id of ['chunk_01', 'chunk_02', 'chunk_03', 'chunk_04', 'chunk_05']) {
      idx.setChunkMetadata(id, { type: 'fact' })
    }

    const ids = idx.queryByMetadata('type', 'fact', 3)
    expect(ids).toHaveLength(3)
  })

  it('returns empty array when no matches exist', async () => {
    const idx = await fresh()

    const ids = idx.queryByMetadata('nonexistent_key', 'value', 10)
    expect(ids).toHaveLength(0)
  })
})

describe('validation errors', () => {
  it('rejects empty chunk ID on set', async () => {
    const idx = await fresh()
    expect(() => idx.setChunkMetadata('', { key: 'val' })).toThrow()
  })

  it('rejects empty key on set', async () => {
    const idx = await fresh()
    expect(() => idx.setChunkMetadata('chunk_1', { '': 'val' })).toThrow()
  })

  it('rejects empty value on set', async () => {
    const idx = await fresh()
    expect(() => idx.setChunkMetadata('chunk_1', { key: '' })).toThrow()
  })

  it('rejects empty chunk ID on get', async () => {
    const idx = await fresh()
    expect(() => idx.getChunkMetadata('')).toThrow()
  })

  it('rejects empty key on query', async () => {
    const idx = await fresh()
    expect(() => idx.queryByMetadata('', 'value', 10)).toThrow()
  })
})

describe('empty map handling', () => {
  it('no-ops when meta is an empty object', async () => {
    const idx = await fresh()
    // Should not throw
    idx.setChunkMetadata('chunk_1', {})
  })
})

describe('delete cleanup', () => {
  it('removes metadata when the chunk is deleted via deleteChunk', async () => {
    const idx = await fresh()

    idx.upsertChunk({
      id: 'chunk_with_meta',
      path: 'raw/documents/doc.md',
      ordinal: 0,
      content: 'some content',
    })
    idx.setChunkMetadata('chunk_with_meta', {
      type: 'factual',
      confidence: '0.9',
    })

    // Verify metadata exists before delete
    const before = idx.getChunkMetadata('chunk_with_meta')
    expect(Object.keys(before)).toHaveLength(2)

    idx.deleteChunk('chunk_with_meta')

    const after = idx.getChunkMetadata('chunk_with_meta')
    expect(Object.keys(after)).toHaveLength(0)
  })

  it('removes metadata when chunks are deleted via deleteByPath', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      { id: 'path_chunk_1', path: 'raw/documents/target.md', ordinal: 0, content: 'first' },
      { id: 'path_chunk_2', path: 'raw/documents/target.md', ordinal: 1, content: 'second' },
      { id: 'other_chunk', path: 'raw/documents/other.md', ordinal: 0, content: 'keep' },
    ])
    idx.setChunkMetadata('path_chunk_1', { type: 'factual' })
    idx.setChunkMetadata('path_chunk_2', { type: 'opinion' })
    idx.setChunkMetadata('other_chunk', { type: 'factual' })

    idx.deleteByPath('raw/documents/target.md')

    expect(Object.keys(idx.getChunkMetadata('path_chunk_1'))).toHaveLength(0)
    expect(Object.keys(idx.getChunkMetadata('path_chunk_2'))).toHaveLength(0)
    // Other chunk's metadata should be untouched
    expect(idx.getChunkMetadata('other_chunk')['type']).toBe('factual')
  })
})
