/**
 * Integration-level tests for the hybrid retrieval pipeline. Drives the
 * real SQLite + FTS5 + sqlite-vec index (via `createSearchIndex`) with
 * a small synthetic corpus so the pipeline wiring is exercised without
 * any network I/O or real LLM calls.
 *
 * Tests cover:
 *   - Happy path: BM25 + vector + RRF + no reranker returns the
 *     expected ordering.
 *   - Unanimity shortcut: when BM25 and vector top-3 agree on ≥ 2
 *     positions, the reranker is skipped even when supplied.
 *   - Reranker runs when the shortcut does not fire.
 *   - Retry ladder: zero initial hits trigger the strongest-term
 *     fallback and the trigram fuzzy fallback.
 */

import { afterEach, describe, expect, it } from 'vitest'
import type { Embedder } from '../llm/types.js'
import type { Reranker, RerankRequest, RerankResult } from '../rerank/index.js'
import { createSearchIndex, type SearchIndex } from '../search/index.js'
import { createRetrieval } from './hybrid.js'

const DIM = 16

/** Deterministic seeded vector so tests stay reproducible. */
function syntheticVector(seed: number, dim = DIM): Float32Array {
  const out = new Float32Array(dim)
  let s = (seed * 2654435761) >>> 0
  for (let i = 0; i < dim; i += 1) {
    s = (s * 1664525 + 1013904223) >>> 0
    out[i] = (s / 0xffffffff) * 2 - 1
  }
  let norm = 0
  for (let i = 0; i < dim; i += 1) norm += out[i]! * out[i]!
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < dim; i += 1) out[i] = out[i]! / norm
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

/** Embedder stub that routes seeded queries to known vectors. */
function makeStubEmbedder(byQuery: ReadonlyMap<string, Float32Array>): Embedder {
  return {
    name: () => 'stub-embedder',
    model: () => 'stub-v0',
    dimension: () => DIM,
    async embed(texts) {
      return texts.map((t) => {
        const v = byQuery.get(t)
        if (v !== undefined) return Array.from(v)
        // Default: return the seed-1 vector so the caller still gets
        // a populated ranking; tests that care about vector order
        // must route via `byQuery`.
        return Array.from(syntheticVector(1))
      })
    },
  }
}

describe('createRetrieval happy path', () => {
  it('returns a fused ranking over BM25 + vector with no reranker', async () => {
    const idx = await fresh()

    const vA = syntheticVector(1)
    const vB = syntheticVector(2)
    const vC = syntheticVector(3)

    idx.upsertChunks([
      {
        id: 'a',
        path: 'a.md',
        title: 'Photosynthesis basics',
        summary: 'Intro to plant biology',
        content: 'Plants use light and water for photosynthesis.',
        embedding: vA,
      },
      {
        id: 'b',
        path: 'b.md',
        title: 'Gardening tips',
        summary: 'Seasonal planting',
        content: 'Prune in autumn; photosynthesis slows before dormancy.',
        embedding: vB,
      },
      {
        id: 'c',
        path: 'c.md',
        title: 'Cooking recipes',
        summary: 'Soups and stews',
        content: 'Simmer stock for at least an hour.',
        embedding: vC,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['photosynthesis', vA]])),
    })

    const { results, trace } = await retrieval.searchRaw({
      query: 'photosynthesis',
      topK: 5,
    })

    expect(trace.mode).toBe('hybrid')
    expect(trace.embedderUsed).toBe(true)
    expect(trace.bm25Count).toBeGreaterThan(0)
    expect(trace.vectorCount).toBeGreaterThan(0)
    expect(trace.fusedCount).toBeGreaterThan(0)
    // "a" is the clear winner: title + summary + body hits on BM25,
    // and its vector is the query embedding.
    expect(results[0]!.id).toBe('a')
    // RRF produces a positive score.
    expect(results[0]!.score).toBeGreaterThan(0)
  })
})

describe('createRetrieval intent-aware reweighting', () => {
  it('prioritises durable preference notes for recommendation queries', async () => {
    const idx = await fresh()

    const vGuide = syntheticVector(501)
    const vPreference = perturb(vGuide, 0.01, 1)
    const vNoise = syntheticVector(503)

    idx.upsertChunks([
      {
        id: 'guide',
        path: 'memory/project/reference-netflix-guide.md',
        title: 'Netflix watch guide',
        summary: 'Movie suggestions and watchlist advice',
        content: 'Advice and recommendations for what to watch on Netflix tonight.',
        embedding: vGuide,
      },
      {
        id: 'preference',
        path: 'memory/global/user-preference-netflix-comedy.md',
        title: 'Entertainment preference',
        summary: 'Prefers Netflix stand-up comedy shows and movies',
        content:
          'The user prefers Netflix stand-up comedy shows and light comedy movies instead of serious dramas.',
        embedding: vPreference,
      },
      {
        id: 'noise',
        path: 'memory/project/reference-drama.md',
        title: 'Drama shortlist',
        summary: 'Heavy drama options',
        content: 'A list of intense period dramas.',
        embedding: vNoise,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['Can you recommend a funny show or movie to watch on Netflix?', vGuide]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'Can you recommend a funny show or movie to watch on Netflix?',
      topK: 5,
    })

    expect(results.length).toBeGreaterThan(1)
    expect(results[0]!.id).toBe('preference')
    expect(results[1]!.id).toBe('guide')
  })

  it('prioritises concrete pickup facts over generic tracking notes', async () => {
    const idx = await fresh()

    const vTracking = syntheticVector(601)
    const vBoots = perturb(vTracking, 0.01, 2)
    const vBlazer = perturb(vTracking, 0.015, 3)

    idx.upsertChunks([
      {
        id: 'tracking',
        path: 'memory/project/user-pickup-tracking.md',
        title: 'Clothing pickup tracking',
        summary: 'Tracking note for Zara clothing pickups',
        content: 'Tracking note for clothing pickups and collection reminders from Zara.',
        embedding: vTracking,
      },
      {
        id: 'boots',
        path: 'memory/global/user-fact-2026-03-01-zara-boots.md',
        title: 'Zara boots pickup',
        summary: 'Picked up black boots from Zara',
        content: 'I picked up black Zara boots for the party.',
        embedding: vBoots,
      },
      {
        id: 'blazer',
        path: 'memory/global/user-fact-2026-03-02-zara-blazer.md',
        title: 'Zara blazer pickup',
        summary: 'Picked up a navy blazer from Zara',
        content: 'I picked up a navy Zara blazer for dinner.',
        embedding: vBlazer,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['What clothes did I pick up from Zara?', vTracking]])),
    })

    const { results } = await retrieval.searchRaw({
      query: 'What clothes did I pick up from Zara?',
      topK: 5,
    })

    expect(results.length).toBeGreaterThan(2)
    expect(results.slice(0, 2).map((result) => result.id).sort()).toEqual([
      'blazer',
      'boots',
    ])
    expect(results[2]!.id).toBe('tracking')
  })

  it('prioritises atomic trip facts over roll-up notes for total queries', async () => {
    const idx = await fresh()

    const vRollUp = syntheticVector(701)
    const vYorkshire = perturb(vRollUp, 0.01, 4)
    const vScotland = perturb(vRollUp, 0.015, 5)

    idx.upsertChunks([
      {
        id: 'rollup',
        path: 'memory/global/road-trip-mileage-roll-up.md',
        title: 'Road trip mileage roll-up',
        summary: 'Total mileage across several road trips',
        content: 'Summary: in total the road trips covered 1200 miles overall.',
        embedding: vRollUp,
      },
      {
        id: 'yorkshire',
        path: 'memory/global/user-fact-2026-04-10-yorkshire-road-trip.md',
        title: 'Yorkshire road trip mileage',
        summary: 'Drove 450 miles on the Yorkshire trip',
        content: 'I drove 450 miles on our Yorkshire road trip.',
        embedding: vYorkshire,
      },
      {
        id: 'scotland',
        path: 'memory/global/user-fact-2026-04-18-scotland-road-trip.md',
        title: 'Scotland road trip mileage',
        summary: 'Drove 750 miles on the Scotland trip',
        content: 'I drove 750 miles on our Scotland road trip.',
        embedding: vScotland,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['What is the total mileage from my road trips?', vRollUp]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'What is the total mileage from my road trips?',
      topK: 5,
    })

    expect(results.length).toBeGreaterThan(2)
    expect(results.slice(0, 2).map((result) => result.id).sort()).toEqual([
      'scotland',
      'yorkshire',
    ])
    expect(results[2]!.id).toBe('rollup')
  })
})

describe('createRetrieval unanimity shortcut', () => {
  it('skips the reranker when BM25 and vector top-3 agree on two positions', async () => {
    const idx = await fresh()

    const vA = syntheticVector(10)
    const vB = syntheticVector(20)
    const vC = syntheticVector(30)
    const vD = syntheticVector(40)

    // Arrange the corpus so the BM25 ranking for "alpha beta gamma"
    // returns a, b, c, and the vector search (query = vA) returns
    // a, b, d. That gives two top-3 agreements -> unanimity fires.
    idx.upsertChunks([
      {
        id: 'a',
        path: 'a.md',
        title: 'alpha',
        summary: 'alpha beta gamma',
        content: 'alpha is primary here',
        embedding: vA,
      },
      {
        id: 'b',
        path: 'b.md',
        title: 'beta',
        summary: 'alpha beta',
        content: 'beta body',
        embedding: perturb(vA, 0.001, 1),
      },
      {
        id: 'c',
        path: 'c.md',
        title: 'gamma',
        summary: 'gamma only',
        content: 'gamma body',
        embedding: vC,
      },
      {
        id: 'd',
        path: 'd.md',
        title: 'delta',
        summary: '',
        content: 'delta body',
        embedding: perturb(vA, 0.01, 2),
      },
      { id: 'e', path: 'e.md', title: 'epsilon', content: 'noise', embedding: vD },
      { id: 'f', path: 'f.md', title: 'zeta', content: 'more noise', embedding: vB },
    ])

    let rerankCalls = 0
    const reranker: Reranker = {
      name: () => 'counting-reranker',
      async rerank(req: RerankRequest): Promise<readonly RerankResult[]> {
        rerankCalls++
        return req.documents.map((d, i) => ({ id: d.id, index: i, score: 0 }))
      },
    }

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['alpha beta gamma', vA]])),
      reranker,
    })

    const { trace } = await retrieval.searchRaw({
      query: 'alpha beta gamma',
      topK: 5,
      rerank: true,
    })

    expect(trace.rerankSkippedReason).toBe('unanimity')
    expect(trace.reranked).toBe(false)
    expect(trace.unanimity?.agreements).toBeGreaterThanOrEqual(2)
    expect(rerankCalls).toBe(0)
  })

  it('invokes the reranker when no shortcut fires', async () => {
    const idx = await fresh()

    // Disjoint top-3 between BM25 (keyword hit in title) and vector
    // (the embedder maps to an unrelated doc) so unanimity cannot
    // fire.
    const vA = syntheticVector(100)
    const vB = syntheticVector(200)
    const vC = syntheticVector(300)
    const vZ = syntheticVector(400)

    idx.upsertChunks([
      {
        id: 'x',
        path: 'x.md',
        title: 'quirkword',
        summary: 'matches keyword',
        content: 'quirkword body',
        embedding: vA,
      },
      {
        id: 'y',
        path: 'y.md',
        title: 'quirkword again',
        summary: 'another keyword match',
        content: 'quirkword again body',
        embedding: vB,
      },
      {
        id: 'z',
        path: 'z.md',
        title: 'unrelated',
        summary: 'no keyword',
        content: 'something else',
        embedding: vZ,
      },
      {
        id: 'w',
        path: 'w.md',
        title: 'also unrelated',
        summary: 'no keyword',
        content: 'more text',
        embedding: vC,
      },
    ])

    let rerankCalls = 0
    const reranker: Reranker = {
      name: () => 'fake-reranker',
      async rerank(req: RerankRequest): Promise<readonly RerankResult[]> {
        rerankCalls++
        // Invert the order so we can verify the reranker's output
        // actually shapes the final ranking.
        return req.documents.map((d, i) => ({
          id: d.id,
          index: i,
          score: req.documents.length - i,
        }))
      },
    }

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['quirkword', vZ]])),
      reranker,
    })

    const { trace } = await retrieval.searchRaw({
      query: 'quirkword',
      topK: 5,
      rerank: true,
    })

    expect(rerankCalls).toBe(1)
    expect(trace.reranked).toBe(true)
    expect(trace.rerankSkippedReason).toBeUndefined()
  })
})

describe('createRetrieval retry ladder', () => {
  it('falls through to the strongest-term rung when the initial query returns zero', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'a',
        path: 'a.md',
        title: 'uniqueword entry',
        summary: 'about uniqueword',
        content: 'uniqueword content',
      },
      {
        id: 'b',
        path: 'b.md',
        title: 'unrelated',
        summary: 'misc',
        content: 'misc body',
      },
    ])

    const retrieval = createRetrieval({ index: idx })

    // Initial pass searches for the verbatim phrase which does not
    // appear in any doc. Strongest-term rung then pulls "uniquewordxyz"
    // (the longest non-stop-word token) and retries, which misses too.
    // Rung 3 sanitises to "the uniqueword when" (punctuation stripped)
    // -> but parseQuery still drops stop-words; the retry ladder's
    // refreshed strongest rung picks "uniqueword" which matches doc a.
    const { results, trace } = await retrieval.searchRaw({
      query: '"missing phrase uniqueword"',
    })

    expect(results.length).toBeGreaterThan(0)
    expect(results[0]!.id).toBe('a')
    const strategies = trace.attempts.map((a) => a.strategy)
    expect(strategies[0]).toBe('initial')
    expect(strategies).toContain('strongest_term')
    const strongest = trace.attempts.find((a) => a.strategy === 'strongest_term')
    expect(strongest?.hits).toBeGreaterThan(0)
  })

  it('falls through to the trigram fuzzy rung on pure slug typos', async () => {
    const idx = await fresh()

    // A doc whose path slug is clearly related to the query token but
    // whose body / title share no searchable tokens with the query.
    idx.upsertChunks([
      {
        id: 'a',
        path: 'notes/photosynthesis.md',
        title: 'Body-only title',
        summary: 'no matching summary',
        content: 'content that does not contain the query',
      },
    ])

    const retrieval = createRetrieval({ index: idx })

    const { results, trace } = await retrieval.searchRaw({
      // Typo at the end so BM25 does not find it but trigrams will.
      query: 'photosynthasis',
    })

    const strategies = trace.attempts.map((a) => a.strategy)
    expect(strategies).toContain('trigram_fuzzy')
    const fuzzy = trace.attempts.find((a) => a.strategy === 'trigram_fuzzy')
    expect(fuzzy?.hits).toBeGreaterThan(0)
    expect(results[0]!.path).toBe('notes/photosynthesis.md')
  })

  it('leaves the ladder alone when skipRetryLadder is set', async () => {
    const idx = await fresh()
    idx.upsertChunks([
      { id: 'a', path: 'a.md', title: 'alpha', content: 'alpha body' },
    ])
    const retrieval = createRetrieval({ index: idx })

    const { results, trace } = await retrieval.searchRaw({
      query: 'missingterm',
      skipRetryLadder: true,
    })

    expect(results).toEqual([])
    expect(trace.attempts.map((a) => a.strategy)).toEqual(['initial'])
  })
})

function perturb(vec: Float32Array, magnitude: number, seed: number): Float32Array {
  const out = new Float32Array(vec.length)
  let s = (seed * 2654435761) >>> 0
  let norm = 0
  for (let i = 0; i < vec.length; i += 1) {
    s = (s * 1664525 + 1013904223) >>> 0
    const noise = ((s / 0xffffffff) * 2 - 1) * magnitude
    out[i] = vec[i]! + noise
    norm += out[i]! * out[i]!
  }
  norm = Math.sqrt(norm)
  if (norm > 0) {
    for (let i = 0; i < vec.length; i += 1) out[i] = out[i]! / norm
  }
  return out
}
