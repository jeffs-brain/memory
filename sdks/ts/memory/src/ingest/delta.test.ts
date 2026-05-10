// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for chunk-level change detection (delta computation and
 * manifest persistence). Mirrors the test coverage in
 * go/knowledge/delta_test.go.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import type { Store } from '../store/index.js'
import type { Chunk } from './chunker.js'
import {
  buildChunkManifest,
  computeChunkDeltas,
  hashChunk,
  readChunkManifest,
  writeChunkManifest,
} from './delta.js'

const makeChunk = (content: string, ordinal: number): Chunk => ({
  content,
  ordinal,
  headingPath: [],
  startLine: 0,
  endLine: 0,
  tokens: Math.ceil(content.length / 4),
})

describe('hashChunk', () => {
  it('produces deterministic output', () => {
    const h1 = hashChunk('hello world')
    const h2 = hashChunk('hello world')
    expect(h1).toBe(h2)
  })

  it('produces different hashes for different content', () => {
    const h1 = hashChunk('hello world')
    const h2 = hashChunk('different content')
    expect(h1).not.toBe(h2)
  })

  it('returns a 64-character hex string', () => {
    const h = hashChunk('test')
    expect(h).toMatch(/^[a-f0-9]{64}$/)
  })
})

describe('computeChunkDeltas', () => {
  it('marks all chunks as new when no manifest exists', () => {
    const chunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
      makeChunk('gamma content', 2),
    ]

    const deltas = computeChunkDeltas(chunks, undefined)

    expect(deltas).toHaveLength(3)
    for (const d of deltas) {
      expect(d.category).toBe('new')
      expect(d.hash).toMatch(/^[a-f0-9]{64}$/)
    }
  })

  it('marks all chunks as unchanged when hashes match', () => {
    const chunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
    ]

    const manifest = buildChunkManifest('doc1hash', chunks)

    const deltas = computeChunkDeltas(chunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const added = deltas.filter((d) => d.category === 'new')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(2)
    expect(added).toHaveLength(0)
    expect(removed).toHaveLength(0)
  })

  it('detects added chunks by hash not in old manifest', () => {
    const oldChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
    ]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const newChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
      makeChunk('new paragraph', 2),
    ]

    const deltas = computeChunkDeltas(newChunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const added = deltas.filter((d) => d.category === 'new')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(2)
    expect(added).toHaveLength(1)
    expect(added[0]?.chunk.content).toBe('new paragraph')
    expect(removed).toHaveLength(0)
  })

  it('detects removed chunks by hash in old but not new', () => {
    const oldChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
    ]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const newChunks: Chunk[] = [makeChunk('alpha content', 0)]

    const deltas = computeChunkDeltas(newChunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(1)
    expect(removed).toHaveLength(1)
    expect(removed[0]?.hash).toBe(hashChunk('beta content'))
  })

  it('marks reordered chunks as unchanged (same hashes, different positions)', () => {
    const oldChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
      makeChunk('gamma content', 2),
    ]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const newChunks: Chunk[] = [
      makeChunk('gamma content', 0),
      makeChunk('alpha content', 1),
      makeChunk('beta content', 2),
    ]

    const deltas = computeChunkDeltas(newChunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const added = deltas.filter((d) => d.category === 'new')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(3)
    expect(added).toHaveLength(0)
    expect(removed).toHaveLength(0)
  })

  it('handles complete document rewrite (all new hashes)', () => {
    const oldChunks: Chunk[] = [
      makeChunk('old alpha', 0),
      makeChunk('old beta', 1),
    ]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const newChunks: Chunk[] = [
      makeChunk('new alpha', 0),
      makeChunk('new beta', 1),
      makeChunk('new gamma', 2),
    ]

    const deltas = computeChunkDeltas(newChunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const added = deltas.filter((d) => d.category === 'new')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(0)
    expect(added).toHaveLength(3)
    expect(removed).toHaveLength(2)
  })

  it('handles one chunk changed while others remain identical', () => {
    const oldChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
      makeChunk('gamma content', 2),
    ]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const newChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta MODIFIED content', 1),
      makeChunk('gamma content', 2),
    ]

    const deltas = computeChunkDeltas(newChunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const added = deltas.filter((d) => d.category === 'new')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(2)
    expect(added).toHaveLength(1)
    expect(added[0]?.chunk.content).toBe('beta MODIFIED content')
    expect(removed).toHaveLength(1)
    expect(removed[0]?.hash).toBe(hashChunk('beta content'))
  })

  it('handles duplicate content chunks correctly', () => {
    const oldChunks: Chunk[] = [makeChunk('repeated', 0)]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const newChunks: Chunk[] = [makeChunk('repeated', 0), makeChunk('repeated', 1)]

    const deltas = computeChunkDeltas(newChunks, manifest)

    const unchanged = deltas.filter((d) => d.category === 'unchanged')
    const added = deltas.filter((d) => d.category === 'new')
    const removed = deltas.filter((d) => d.category === 'removed')

    expect(unchanged).toHaveLength(1)
    expect(added).toHaveLength(1)
    expect(removed).toHaveLength(0)
  })

  it('returns empty when both inputs are empty', () => {
    const deltas = computeChunkDeltas([], undefined)
    expect(deltas).toHaveLength(0)
  })

  it('marks all old chunks as removed when new set is empty', () => {
    const oldChunks: Chunk[] = [
      makeChunk('alpha content', 0),
      makeChunk('beta content', 1),
    ]
    const manifest = buildChunkManifest('doc1hash', oldChunks)

    const deltas = computeChunkDeltas([], manifest)

    const removed = deltas.filter((d) => d.category === 'removed')
    expect(removed).toHaveLength(2)
  })
})

describe('buildChunkManifest', () => {
  it('builds entries with correct hash and chunkId', () => {
    const chunks: Chunk[] = [makeChunk('chunk A', 0), makeChunk('chunk B', 1)]

    const manifest = buildChunkManifest('hash123', chunks)

    expect(manifest.documentHash).toBe('hash123')
    expect(manifest.chunks).toHaveLength(2)
    expect(manifest.chunks[0]?.hash).toBe(hashChunk('chunk A'))
    expect(manifest.chunks[0]?.chunkId).toBe('hash123_0')
    expect(manifest.chunks[1]?.hash).toBe(hashChunk('chunk B'))
    expect(manifest.chunks[1]?.chunkId).toBe('hash123_1')
  })
})

describe('readChunkManifest/writeChunkManifest', () => {
  const stores: Store[] = []

  afterEach(async () => {
    while (stores.length > 0) {
      const s = stores.pop()
      if (s !== undefined) await s.close()
    }
  })

  const freshStore = (): Store => {
    const s = createMemStore()
    stores.push(s)
    return s
  }

  it('round-trips through store correctly', async () => {
    const store = freshStore()
    const chunks: Chunk[] = [
      makeChunk('first chunk', 0),
      makeChunk('second chunk', 1),
      makeChunk('third chunk', 2),
    ]

    const manifest = buildChunkManifest('abc123def456', chunks)
    await writeChunkManifest(store, manifest)

    const loaded = await readChunkManifest(store, 'abc123def456')

    expect(loaded).not.toBeUndefined()
    expect(loaded?.documentHash).toBe('abc123def456')
    expect(loaded?.generation).toBe(1)
    expect(loaded?.chunks).toHaveLength(3)
    expect(loaded?.updatedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/)

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]
      if (chunk === undefined) continue
      expect(loaded?.chunks[i]?.hash).toBe(hashChunk(chunk.content))
      expect(loaded?.chunks[i]?.chunkId).toBe(`abc123def456_${String(i)}`)
    }
  })

  it('returns undefined for missing manifest', async () => {
    const store = freshStore()
    const loaded = await readChunkManifest(store, 'nonexistent')
    expect(loaded).toBeUndefined()
  })

  it('increments generation on subsequent writes', async () => {
    const store = freshStore()
    const chunks1: Chunk[] = [makeChunk('content v1', 0)]
    const manifest1 = buildChunkManifest('docXYZ', chunks1)

    await writeChunkManifest(store, manifest1)
    const first = await readChunkManifest(store, 'docXYZ')
    expect(first?.generation).toBe(1)

    const chunks2: Chunk[] = [makeChunk('content v2', 0)]
    const manifest2 = buildChunkManifest('docXYZ', chunks2)

    await writeChunkManifest(store, manifest2)
    const second = await readChunkManifest(store, 'docXYZ')
    expect(second?.generation).toBe(2)
  })
})
