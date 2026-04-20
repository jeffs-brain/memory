// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the hybrid BM25 + vector search index.
 *
 * Scope is deliberately meaningful, not exhaustive:
 *   - DDL is idempotent (running createSearchIndex twice against the same
 *     database does not throw or duplicate state).
 *   - BM25 weights actually favour title hits over content hits.
 *   - Vector search returns the nearest of a small corpus.
 *   - ~100-chunk roundtrip works under a single transaction.
 *   - deleteByPath sweeps both indexes.
 *
 * No real embedder is invoked; we generate synthetic 1024-dim vectors
 * deterministically from a seed so tests stay reproducible.
 */

import { afterEach, describe, expect, it } from 'vitest'
import { type SearchIndex, createSearchIndex } from './index.js'

const DIM = 1024

function syntheticVector(seed: number, dim = DIM): Float32Array {
  // Simple LCG for deterministic pseudo-random floats. Not
  // cryptographically strong — we only need distinguishable vectors.
  const out = new Float32Array(dim)
  let s = (seed * 2654435761) >>> 0
  for (let i = 0; i < dim; i += 1) {
    s = (s * 1664525 + 1013904223) >>> 0
    out[i] = (s / 0xffffffff) * 2 - 1
  }
  // L2 normalise so cosine distance maths lines up with our docs.
  let norm = 0
  for (let i = 0; i < dim; i += 1) {
    const value = out[i] ?? 0
    norm += value * value
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < dim; i += 1) {
      out[i] = (out[i] ?? 0) / norm
    }
  }
  return out
}

function perturb(vec: Float32Array, magnitude: number): Float32Array {
  const out = new Float32Array(vec.length)
  let norm = 0
  for (let i = 0; i < vec.length; i += 1) {
    const noise = (Math.sin(i * 12.9898) + 1) * magnitude
    const value = (vec[i] ?? 0) + noise
    out[i] = value
    norm += value * value
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < vec.length; i += 1) {
      out[i] = (out[i] ?? 0) / norm
    }
  }
  return out
}

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

describe('createSearchIndex', () => {
  it('applies DDL idempotently when called twice against the same connection', async () => {
    const first = await createSearchIndex({ dbPath: ':memory:', vectorDim: DIM })
    indices.push(first)
    // Second call re-uses the already-opened connection and must not
    // explode on duplicate CREATE statements or rank re-configuration.
    const second = await createSearchIndex({ connection: first.db, vectorDim: DIM })
    indices.push(second)
    // Verify the FTS and vector tables exist exactly once.
    const tables = (
      first.db
        .prepare(
          "SELECT name FROM sqlite_master WHERE type IN ('table','index') AND name LIKE 'knowledge_%'",
        )
        .all() as Array<{ name: string }>
    ).map((r) => r.name)
    expect(tables).toContain('knowledge_chunks')
    expect(tables).toContain('knowledge_fts')
    expect(tables).toContain('knowledge_vectors')
    expect(tables).toContain('knowledge_meta')
  })
})

describe('BM25 weight ordering', () => {
  it('ranks a title match higher than a content-only match for the same term', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'title-hit',
        path: 'notes/alpha.md',
        title: 'Photosynthesis explained',
        summary: 'A short introduction',
        content: 'Plants use light and water in complicated biochemical pathways.',
      },
      {
        id: 'content-hit',
        path: 'notes/beta.md',
        title: 'Gardening tips',
        summary: 'Seasonal planting guide',
        content:
          'When you prune in autumn, photosynthesis in the remaining leaves slows slightly before dormancy begins.',
      },
      {
        id: 'noise-a',
        path: 'notes/gamma.md',
        title: 'Totally unrelated',
        summary: 'Filler',
        content: 'Lorem ipsum dolor sit amet.',
      },
    ])

    const results = idx.searchBM25('photosynthesis', 10)
    expect(results.length).toBeGreaterThanOrEqual(2)
    // FTS5 rank: lower (more negative) is a stronger match.
    const titleResult = results.find((r) => r.chunk.id === 'title-hit')
    const contentResult = results.find((r) => r.chunk.id === 'content-hit')
    expect(titleResult).toBeDefined()
    expect(contentResult).toBeDefined()
    expect(titleResult?.score).toBeLessThan(contentResult?.score)
    // And the ordering in the returned array matches: title first.
    const firstMatching = results.find(
      (r) => r.chunk.id === 'title-hit' || r.chunk.id === 'content-hit',
    )
    expect(firstMatching?.chunk.id).toBe('title-hit')
  })

  it('preserves compiled FTS expressions for column-scoped queries', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'path-hit',
        path: 'project/gift-note.md',
        title: 'Tracking note',
        summary: 'Project note',
        content: 'This note is about totals.',
      },
      {
        id: 'content-hit',
        path: 'project/other-note.md',
        title: 'Other note',
        summary: 'Gift details live in the body',
        content: 'gift content only',
      },
    ])

    const results = idx.searchBM25Compiled('path:gift', 10)
    expect(results.length).toBeGreaterThanOrEqual(1)
    expect(results[0]?.chunk.id).toBe('path-hit')
    expect(results.some((result) => result.chunk.id === 'path-hit')).toBe(true)
  })
})

describe('vector search', () => {
  it('returns the closest of three embeddings to the query', async () => {
    const idx = await fresh()

    const vA = syntheticVector(1)
    const vB = syntheticVector(2)
    const vC = syntheticVector(3)

    idx.upsertChunks([
      { id: 'a', path: 'vec/a.md', content: 'alpha', embedding: vA },
      { id: 'b', path: 'vec/b.md', content: 'bravo', embedding: vB },
      { id: 'c', path: 'vec/c.md', content: 'charlie', embedding: vC },
    ])

    const query = perturb(vB, 0.001)
    const results = idx.searchVector(query, 3)
    expect(results).toHaveLength(3)
    expect(results[0]?.chunk.id).toBe('b')
    // Distance to the nearest should be strictly less than to the others.
    expect(results[0]?.distance).toBeLessThan(results[1]?.distance)
  })
})

describe('roundtrip', () => {
  it('writes 100 chunks in one transaction and finds them via BM25 and vector search', async () => {
    const idx = await fresh()

    const chunks = Array.from({ length: 100 }, (_, i) => ({
      id: `chunk-${i}`,
      path: `roundtrip/${i}.md`,
      ordinal: 0,
      title: `Document number ${i}`,
      summary: i % 2 === 0 ? 'even entry' : 'odd entry',
      tags: ['batch', `parity-${i % 2}`],
      content: `Body text for entry ${i} discussing widgets and gadgets.`,
      embedding: syntheticVector(i + 1000),
    }))

    idx.upsertChunks(chunks)

    const bm25 = idx.searchBM25('widgets', 50)
    expect(bm25.length).toBeGreaterThanOrEqual(50)

    const nearest = idx.searchVector(chunks[42]?.embedding, 5)
    expect(nearest[0]?.chunk.id).toBe('chunk-42')

    // Spot-check hydration
    const c = idx.getChunk('chunk-7')
    expect(c).toBeDefined()
    expect(c?.title).toBe('Document number 7')
    expect(c?.tags).toEqual(['batch', 'parity-1'])
  })
})

describe('deleteByPath', () => {
  it('removes matching chunks from both BM25 and vector indexes', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'keep',
        path: 'keep.md',
        title: 'stays',
        content: 'persistent content about widgets',
        embedding: syntheticVector(111),
      },
      {
        id: 'goner-1',
        path: 'goner.md',
        title: 'disappears-one',
        content: 'ephemeral widgets info',
        embedding: syntheticVector(222),
      },
      {
        id: 'goner-2',
        path: 'goner.md',
        title: 'disappears-two',
        content: 'more ephemeral widgets',
        embedding: syntheticVector(223),
      },
    ])

    // Prove they are all present first.
    expect(idx.searchBM25('widgets', 10)).toHaveLength(3)
    expect(idx.searchVector(syntheticVector(222), 5).some((r) => r.chunk.id === 'goner-1')).toBe(
      true,
    )

    idx.deleteByPath('goner.md')

    const bm25 = idx.searchBM25('widgets', 10)
    expect(bm25).toHaveLength(1)
    expect(bm25[0]?.chunk.id).toBe('keep')

    const vec = idx.searchVector(syntheticVector(222), 5)
    expect(vec.some((r) => r.chunk.id.startsWith('goner'))).toBe(false)

    // chunks table also cleaned.
    expect(idx.getChunk('goner-1')).toBeUndefined()
    expect(idx.getChunk('goner-2')).toBeUndefined()
    expect(idx.getChunk('keep')).toBeDefined()
  })
})
