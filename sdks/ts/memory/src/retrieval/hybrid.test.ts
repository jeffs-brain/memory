// SPDX-License-Identifier: Apache-2.0

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

describe('createRetrieval reusable request surface', () => {
  it('applies questionDate-driven temporal augmentation inside retrieval', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'friday',
        path: 'memory/global/friday.md',
        title: 'Weekly note',
        summary: 'Met the supplier',
        tags: ['2024/03/08', 'Friday'],
        content: 'Met the supplier and agreed the new timeline.',
      },
      {
        id: 'noise',
        path: 'memory/global/noise.md',
        title: 'Weekly plan',
        summary: 'No matching date',
        content: 'Planned the next sprint.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results, trace } = await retrieval.searchRaw({
      query: 'What happened last Friday?',
      questionDate: '2024-03-15',
      topK: 5,
      rerank: false,
    })

    expect(results[0]?.id).toBe('friday')
    expect(trace.temporalAugmented).toBe(true)
    expect(trace.bm25Queries.length).toBeGreaterThan(1)
  })

  it('keeps semantic and rerank queries unaugmented when questionDate is supplied', async () => {
    const idx = await fresh()

    const vFriday = syntheticVector(77)
    idx.upsertChunks([
      {
        id: 'friday',
        path: 'memory/global/friday.md',
        title: 'Weekly note',
        summary: 'Met the supplier',
        tags: ['2024/03/08', 'Friday'],
        content: 'Met the supplier and agreed the new timeline.',
        embedding: vFriday,
      },
    ])

    const embedInputs: string[] = []
    let rerankQuery = ''
    const embedder: Embedder = {
      name: () => 'capturing-embedder',
      model: () => 'capturing-v0',
      dimension: () => DIM,
      async embed(texts) {
        embedInputs.push(...texts)
        return texts.map((text) =>
          text === 'What happened last Friday?'
            ? Array.from(vFriday)
            : Array.from(syntheticVector(78)),
        )
      },
    }
    const reranker: Reranker = {
      name: () => 'capturing-reranker',
      async rerank(req: RerankRequest): Promise<readonly RerankResult[]> {
        rerankQuery = req.query
        return req.documents.map((document, index) => ({
          id: document.id,
          index,
          score: 1,
        }))
      },
    }

    const retrieval = createRetrieval({ index: idx, embedder, reranker })
    const { trace } = await retrieval.searchRaw({
      query: 'What happened last Friday?',
      questionDate: '2024-03-15',
      topK: 5,
      mode: 'hybrid-rerank',
    })

    expect(trace.temporalAugmented).toBe(true)
    expect(embedInputs).toEqual(['What happened last Friday?'])
    expect(rerankQuery).toBe('What happened last Friday?')
  })

  it('hydrates candidate bodies from a body lookup when supplied', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'orchid',
        path: 'memory/project/demo/orchid.md',
        title: '',
        summary: '',
        content: 'orchid fertiliser placeholder snippet',
      },
    ])

    let hydratedLookupCalls = 0
    const reranker: Reranker = {
      name: () => 'capturing-reranker',
      async rerank(req: RerankRequest): Promise<readonly RerankResult[]> {
        return req.documents.map((document, index) => ({
          id: document.id,
          index,
          score: 1,
        }))
      },
    }

    const retrieval = createRetrieval({
      index: idx,
      reranker,
      bodyLookup: async (paths) => {
        hydratedLookupCalls += 1
        return [
          {
          path: paths[0] ?? '',
          title: 'Orchid fertiliser note',
          summary: 'Precise feeding guidance',
          content:
            '---\nname: Orchid fertiliser note\ndescription: Precise feeding guidance\n---\n\nUse the 12-6-8 orchid fertiliser every second Saturday.',
          metadata: {
            sessionId: 'orchid-session',
            sessionDate: '2024/03/02 (Sat) 10:00',
          },
          },
        ]
      },
    })

    const { results } = await retrieval.searchRaw({
      query: 'placeholder',
      topK: 5,
      mode: 'hybrid-rerank',
    })

    expect(hydratedLookupCalls).toBe(1)
    expect(results[0]?.content).toContain('Use the 12-6-8 orchid fertiliser every second Saturday.')
    expect(results[0]?.title).toBe('Orchid fertiliser note')
    expect(results[0]?.summary).toBe('Precise feeding guidance')
    expect(results[0]?.metadata?.['sessionId']).toBe('orchid-session')
  })

  it('boosts the most recent dated hit for recency questions', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'older',
        path: 'memory/global/a-older.md',
        title: 'Market visit',
        content: '[Observed on 2024/03/01 (Fri) 09:00]\nEarned $220 at the Downtown Farmers Market.',
      },
      {
        id: 'newer',
        path: 'memory/global/z-newer.md',
        title: 'Market visit',
        content: '[Observed on 2024/03/08 (Fri) 09:00]\nEarned $420 at the Downtown Farmers Market.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results } = await retrieval.searchRaw({
      query: 'How much did I earn at the Downtown Farmers Market on my most recent visit?',
      mode: 'bm25',
      rerank: false,
      topK: 5,
    })

    expect(results[0]?.id).toBe('newer')
  })

  it('boosts candidates closest to the resolved temporal hint date', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'far',
        path: 'memory/global/a-far.md',
        title: 'Weekly note',
        content: '[Observed on 2024/02/02 (Fri) 10:00]\nMet the supplier and agreed the timeline.',
      },
      {
        id: 'near',
        path: 'memory/global/z-near.md',
        title: 'Weekly note',
        content: '[Observed on 2024/03/08 (Fri) 10:00]\nMet the supplier and agreed the timeline.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results } = await retrieval.searchRaw({
      query: 'What happened last Friday?',
      questionDate: '2024-03-15',
      mode: 'bm25',
      rerank: false,
      topK: 5,
    })

    expect(results[0]?.id).toBe('near')
  })

  it('drops future-dated hits relative to the question date', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'past',
        path: 'memory/global/past.md',
        title: 'Supplier visit',
        content: '[Observed on 2024/03/10 (Sun) 09:00]\nMet the supplier and agreed the next steps.',
      },
      {
        id: 'future',
        path: 'memory/global/future.md',
        title: 'Supplier visit',
        content: '[Observed on 2024/03/20 (Wed) 09:00]\nMet the supplier and agreed the next steps.',
      },
      {
        id: 'undated',
        path: 'memory/global/undated.md',
        title: 'Supplier visit',
        content: 'Met the supplier and agreed the next steps.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results } = await retrieval.searchRaw({
      query: 'What is the most recent supplier visit?',
      questionDate: '2024/03/15 (Fri) 09:00',
      mode: 'bm25',
      rerank: false,
      topK: 5,
    })

    expect(results[0]?.id).toBe('past')
    expect(results.some((result) => result.id === 'future')).toBe(false)
  })

  it('drops drifted token probes from BM25 fanout', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'target',
        path: 'raw/lme/answer_sharegpt_hChsWOp_97.md',
        content:
          'We finally named the Radiation Amplified zombie Fissionator after trying several other names.',
      },
      {
        id: 'conversation',
        path: 'memory/project/conversation-note.md',
        title: 'Conversation archive',
        summary: 'Conversation recap and conversation metadata',
        content: 'Conversation recap with conversation follow-up notes.',
      },
      {
        id: 'remembered',
        path: 'memory/project/remembered-note.md',
        title: 'Remembered preferences',
        summary: 'Remembered recap and remembered preference note',
        content: 'Remembered choices and remembered follow-up details.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results } = await retrieval.searchRaw({
      query:
        'I was thinking back to our previous conversation about the Radiation Amplified zombie, and I was wondering if you remembered what we finally decided to name it?',
      mode: 'bm25',
      rerank: false,
      topK: 5,
    })

    expect(results[0]?.id).toBe('target')
  })

  it('adds phrase probes for compound total questions in the BM25 fanout plan', async () => {
    const idx = await fresh()

    const retrieval = createRetrieval({ index: idx })
    const { trace } = await retrieval.searchRaw({
      query:
        'What is the total amount I spent on the designer handbag and high-end skincare products?',
      mode: 'bm25',
      rerank: false,
      skipRetryLadder: true,
    })

    expect(
      trace.bm25Queries.some((query) => query.includes('handbag') && query.includes('cost')),
    ).toBe(true)
    expect(
      trace.bm25Queries.some(
        (query) => query.includes('products') && (query.includes('high-end') || query.includes('highend')),
      ),
    ).toBe(true)
    expect(
      trace.bm25Queries.some((query) => query.includes('designer') && query.includes('handbag')),
    ).toBe(true)
  })

  it('merges phrase-probe hits for compound total questions in BM25 mode', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'bag',
        path: 'raw/lme/designer-handbag.md',
        title: 'Designer handbag',
        summary: 'High-value purchase',
        content: 'I spent 1800 on the designer handbag.',
      },
      {
        id: 'skincare',
        path: 'raw/lme/skincare-products.md',
        title: 'Skincare products',
        summary: 'Beauty purchase',
        content: 'I spent 320 on the high-end skincare products.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results } = await retrieval.searchRaw({
      query:
        'What is the total amount I spent on the designer handbag and high-end skincare products?',
      mode: 'bm25',
      skipRetryLadder: true,
      rerank: false,
      topK: 5,
    })

    const topPaths = new Set(results.slice(0, 2).map((result) => result.path))
    expect(topPaths.has('raw/lme/designer-handbag.md')).toBe(true)
    expect(topPaths.has('raw/lme/skincare-products.md')).toBe(true)
  })

  it('adds a focused recommendation probe for exact back-end language recalls', async () => {
    const idx = await fresh()

    const retrieval = createRetrieval({ index: idx })
    const { trace } = await retrieval.searchRaw({
      query:
        'I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?',
      mode: 'bm25',
      rerank: false,
      skipRetryLadder: true,
    })

    expect(
      trace.bm25Queries.some(
        (query) =>
          (query.includes('programming') && query.includes('language')) ||
          query.includes('back-end development'),
      ),
    ).toBe(true)
  })

  it('adds an action-date probe for when-did-I submission questions', async () => {
    const idx = await fresh()

    const retrieval = createRetrieval({ index: idx })
    const { trace } = await retrieval.searchRaw({
      query: 'When did I submit my research paper on sentiment analysis?',
      mode: 'bm25',
      rerank: false,
      skipRetryLadder: true,
    })

    expect(
      trace.bm25Queries.some(
        (query) =>
          query.includes('submission') &&
          query.includes('date') &&
          query.includes('sentiment') &&
          query.includes('analysis'),
      ),
    ).toBe(true)
    expect(
      trace.bm25Queries.some(
        (query) =>
          query.includes('submission') &&
          query.includes('date') &&
          query.includes('research') &&
          query.includes('paper'),
      ),
    ).toBe(true)
  })

  it('adds an inspiration-source probe for painting inspiration questions', async () => {
    const idx = await fresh()

    const retrieval = createRetrieval({ index: idx })
    const { trace } = await retrieval.searchRaw({
      query: 'How can I find new inspiration for my paintings?',
      mode: 'bm25',
      rerank: false,
      skipRetryLadder: true,
    })

    expect(
      trace.bm25Queries.some(
        (query) =>
          query.includes('paintings') &&
          query.includes('social') &&
          query.includes('media') &&
          query.includes('tutorials'),
      ),
    ).toBe(true)
  })

  it('applies retrieval filters before fusion', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'billing',
        path: 'memory/project/billing/invoice.md',
        title: 'Invoice note',
        summary: 'Customer invoice and payment date',
        content: 'Invoice 42 has been paid.',
      },
      {
        id: 'global',
        path: 'memory/global/invoice.md',
        title: 'Global invoice note',
        summary: 'General invoice guidance',
        content: 'Invoice guidance and VAT notes.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results, trace } = await retrieval.searchRaw({
      query: 'invoice',
      topK: 5,
      rerank: false,
      filters: {
        pathPrefix: 'memory/project/billing/',
        scope: 'project',
        project: 'billing',
      },
    })

    expect(results.map((result) => result.id)).toEqual(['billing'])
    expect(trace.filtersApplied).toBe(true)
  })

  it('applies exact path-list filters before fusion', async () => {
    const idx = await fresh()

    idx.upsertChunks([
      {
        id: 'billing',
        path: 'memory/project/billing/invoice.md',
        title: 'Invoice note',
        summary: 'Customer invoice and payment date',
        content: 'Invoice 42 has been paid.',
      },
      {
        id: 'global',
        path: 'memory/global/invoice.md',
        title: 'Global invoice note',
        summary: 'General invoice guidance',
        content: 'Invoice guidance and VAT notes.',
      },
    ])

    const retrieval = createRetrieval({ index: idx })
    const { results, trace } = await retrieval.searchRaw({
      query: 'invoice',
      topK: 5,
      rerank: false,
      filters: {
        paths: ['memory/global/invoice.md'],
      },
    })

    expect(results.map((result) => result.id)).toEqual(['global'])
    expect(trace.filtersApplied).toBe(true)
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

  it('demotes advice-shaped user-fact notes for exact property lookups', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(801)
    const vDirect = perturb(vQuestion, 0.01, 6)

    idx.upsertChunks([
      {
        id: 'question-like',
        path: 'memory/global/user-fact-commute-question.md',
        title: 'Commute question',
        summary: 'Tips for staying awake during a 30-minute train commute',
        content: 'What are some tips for staying awake during morning commutes, especially when I am stuck on the train for 30 minutes?',
        embedding: vQuestion,
      },
      {
        id: 'direct',
        path: 'memory/global/user-fact-commute-duration.md',
        title: 'Commute duration',
        summary: 'Daily commute takes 45 minutes each way',
        content: 'My daily commute to work takes 45 minutes each way.',
        embedding: vDirect,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['How long is my daily commute to work?', vQuestion]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'How long is my daily commute to work?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('direct')
  })

  it('prefers routine user facts over project tips for first-person duration lookups', async () => {
    const idx = await fresh()

    const vTips = syntheticVector(841)
    const vMorning = perturb(vTips, 0.01, 2)
    const vRoutine = perturb(vTips, 0.02, 3)

    idx.upsertChunks([
      {
        id: 'tips',
        path: 'memory/project/eval-lme/morning-commute-tips.md',
        title: 'Morning commute tips',
        summary: 'Tips for staying awake during a 30-minute morning commute',
        content: 'Tips for staying awake during a 30-minute morning commute.',
        embedding: vTips,
      },
      {
        id: 'morning',
        path: 'memory/global/user-morning-commute-duration.md',
        title: 'Morning commute duration',
        summary: 'Often a 30-minute morning commute with shorter days',
        content:
          'User is often on a train for a 30-minute morning commute. Some days the commute is shorter, around 15-20 minutes.',
        embedding: vMorning,
      },
      {
        id: 'routine',
        path: 'memory/global/user-commute-time.md',
        title: 'Daily commute time',
        summary: 'Daily commute takes 45 minutes each way',
        content: 'I listen to audiobooks during my daily commute, which takes 45 minutes each way.',
        embedding: vRoutine,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['How long is my daily morning commute to work?', vTips]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'How long is my daily morning commute to work?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('routine')
  })

  it('treats first-person cadence lookups as concrete fact queries', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(844)
    const vReference = vQuestion
    const vCadence = perturb(vQuestion, 0.03, 2)

    idx.upsertChunks([
      {
        id: 'reference',
        path: 'memory/project/eval-lme/doctor-visit-reference.md',
        title: 'Doctor visit reference',
        summary: 'Reference note about how often to schedule doctor visits with Dr. Smith',
        content:
          'Reference: ask how often to schedule doctor visits with Dr. Smith and what to discuss in follow-up appointments.',
        metadata: {
          type: 'reference',
        },
        embedding: vReference,
      },
      {
        id: 'cadence',
        path: 'memory/global/user-fact-dr-smith-weekly.md',
        title: 'Weekly visit with Dr. Smith',
        summary: 'Sees Dr. Smith every week',
        content: 'I see Dr. Smith every week.',
        metadata: {
          type: 'user',
        },
        embedding: vCadence,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['How often do I see Dr. Smith?', vQuestion]])),
    })

    const { results } = await retrieval.searchRaw({
      query: 'How often do I see Dr. Smith?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('cadence')
  })

  it('prefers direct global transport facts over project reference notes', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(846)
    const vReference = vQuestion
    const vDirect = perturb(vQuestion, 0.03, 7)

    idx.upsertChunks([
      {
        id: 'reference',
        path: 'memory/project/eval-lme/office-transport-reference.md',
        title: 'Office transport reference',
        summary: 'Reference note listing tram, bus, and cycle commute options',
        content:
          'Reference: possible office commute transport options include the tram, the bus, and cycling.',
        metadata: {
          type: 'reference',
        },
        embedding: vReference,
      },
      {
        id: 'direct',
        path: 'memory/global/user-fact-office-commute-tram.md',
        title: 'Office commute transport',
        summary: 'Usually takes the tram to the office',
        content: 'I usually take the tram to the office for my commute.',
        metadata: {
          type: 'user',
        },
        embedding: vDirect,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['Which mode of transport did I use for my office commute?', vQuestion]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'Which mode of transport did I use for my office commute?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('direct')
  })

  it('diversifies first-person comparison lookups connected with or', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(847)
    const vGuide = vQuestion
    const vWalk = perturb(vQuestion, 0.02, 8)
    const vRun = perturb(vQuestion, 0.021, 9)

    idx.upsertChunks([
      {
        id: 'guide',
        path: 'memory/project/eval-lme/treadmill-mode-guide.md',
        title: 'Treadmill mode guide',
        summary: 'Comparison chart for walk and run mode speeds',
        content: 'Guide: walk mode is 4 km/h and run mode is 10 km/h on the treadmill.',
        metadata: {
          type: 'reference',
        },
        embedding: vGuide,
      },
      {
        id: 'walk',
        path: 'memory/global/user-fact-treadmill-walk-mode.md',
        title: 'Walk mode speed',
        summary: 'Treadmill walk mode is 4 km/h',
        content: 'My treadmill runs at 4 km/h in walk mode.',
        metadata: {
          type: 'user',
        },
        embedding: vWalk,
      },
      {
        id: 'run',
        path: 'memory/global/user-fact-treadmill-run-mode.md',
        title: 'Run mode speed',
        summary: 'Treadmill run mode is 10 km/h',
        content: 'My treadmill runs at 10 km/h in run mode.',
        metadata: {
          type: 'user',
        },
        embedding: vRun,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['What speed is my treadmill in walk or run mode?', vQuestion]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'What speed is my treadmill in walk or run mode?',
      topK: 5,
    })

    expect(results.slice(0, 2).map((result) => result.id).sort()).toEqual(['run', 'walk'])
  })

  it('demotes superseded notes when metadata marks them stale', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(848)
    const vStale = vQuestion
    const vCurrent = perturb(vQuestion, 0.02, 10)

    idx.upsertChunks([
      {
        id: 'stale',
        path: 'memory/global/user-fact-passports-hall-drawer.md',
        title: 'Old passport location',
        summary: 'Passports were in the hall drawer',
        content: 'I used to keep the passports in the hall drawer.',
        metadata: {
          type: 'user',
          superseded_by: 'user-fact-passports-bedroom-safe.md',
        },
        embedding: vStale,
      },
      {
        id: 'current',
        path: 'memory/global/user-fact-passports-bedroom-safe.md',
        title: 'Current passport location',
        summary: 'Passports are in the bedroom safe',
        content: 'I keep the passports in the bedroom safe.',
        metadata: {
          type: 'user',
        },
        embedding: vCurrent,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['Where do I keep the passports?', vQuestion]])),
    })

    const { results } = await retrieval.searchRaw({
      query: 'Where do I keep the passports?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('current')
  })

  it('diversifies composite total queries across the requested focuses', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(845)
    const vCoach = perturb(vQuestion, 0.005, 2)
    const vNordstrom = perturb(vQuestion, 0.01, 3)
    const vEbay = perturb(vQuestion, 0.012, 4)
    const vMoisturizer = perturb(vQuestion, 0.014, 5)

    idx.upsertChunks([
      {
        id: 'coach',
        path: 'memory/global/coach-handbag-800.md',
        title: 'Coach handbag purchase',
        summary: 'Coach handbag cost $800',
        content: 'User recently treated themself to a Coach handbag which cost $800 and they are really loving the quality.',
        embedding: vCoach,
      },
      {
        id: 'nordstrom',
        path: 'memory/global/user-fact-2023-05-28-recently-invested-some-high-end-products.md',
        title: 'High-end products purchase',
        summary: 'Invested $500 in high-end products',
        content: "I've recently invested $500 in some high-end products during the Nordstrom anniversary sale.",
        embedding: vNordstrom,
      },
      {
        id: 'ebay',
        path: 'memory/global/user_ebay_handbag_deal.md',
        title: 'Designer handbag eBay deal',
        summary: 'Designer handbag bought for $200',
        content: 'The user bought a designer handbag on eBay that originally retailed for $1,500 for $200.',
        embedding: vEbay,
      },
      {
        id: 'moisturizer',
        path: 'memory/global/user_high-end-moisturizer.md',
        title: 'High-end moisturizer purchase',
        summary: 'Splurged on a $150 moisturizer',
        content:
          'The user recently splurged on a $150 moisturizer and is asking for affordable alternatives to high-end skincare products.',
        embedding: vMoisturizer,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([
          [
            'What is the total amount I spent on the designer handbag and high-end skincare products?',
            vQuestion,
          ],
        ]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'What is the total amount I spent on the designer handbag and high-end skincare products?',
      topK: 5,
    })

    expect([results[0]!.id, results[1]!.id].sort()).toEqual(['coach', 'nordstrom'])
  })

  it('treats specific remind-me recalls as concrete fact lookups', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(851)
    const vBroad = perturb(vQuestion, 0.008, 4)
    const vFocused = perturb(vQuestion, 0.02, 4)

    idx.upsertChunks([
      {
        id: 'broad',
        path: 'memory/project/eval-lme/back-end-learning-resources.md',
        title: 'Back-end learning resources',
        summary: 'NodeSchool, Udacity, Coursera, Flask, Django, Spring, Hibernate, SQL',
        content:
          'Recommended back-end resources include NodeSchool, Udacity, Coursera, Flask, Django, Spring, Hibernate, SQL.',
        embedding: vBroad,
      },
      {
        id: 'focused',
        path: 'memory/project/eval-lme/study-tips-for-becoming-full-stack.md',
        title: 'Study tips for becoming full-stack',
        summary: 'Learn a back-end programming language, such as Ruby, Python, or PHP',
        content: 'Learn a back-end programming language, such as Ruby, Python, or PHP.',
        embedding: vFocused,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([
          [
            'I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?',
            vQuestion,
          ],
        ]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query:
        'I wanted to follow up on our previous conversation about front-end and back-end development. Can you remind me of the specific back-end programming languages you recommended I learn?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('focused')
  })

  it('boosts explicit dated facts for action-date questions', async () => {
    const idx = await fresh()

    const vQuestion = syntheticVector(911)
    const vPaper = perturb(vQuestion, 0.01, 6)
    const vDate = perturb(vQuestion, 0.011, 6)

    idx.upsertChunks([
      {
        id: 'paper',
        path: 'memory/global/user-fact-paper.md',
        title: 'Submitted paper',
        summary: 'User submitted a sentiment analysis paper to ACL',
        content: 'I submitted my research paper on sentiment analysis to ACL.',
        embedding: vPaper,
      },
      {
        id: 'dated',
        path: 'memory/global/user-fact-acl-date.md',
        title: 'ACL submission date',
        summary: 'ACL submission date note',
        content: "I'm reviewing for ACL, and their submission date was February 1st.",
        embedding: vDate,
      },
    ])

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(
        new Map([['When did I submit my research paper on sentiment analysis?', vQuestion]]),
      ),
    })

    const { results } = await retrieval.searchRaw({
      query: 'When did I submit my research paper on sentiment analysis?',
      topK: 5,
    })

    expect(results[0]!.id).toBe('dated')
  })
})

describe('createRetrieval unanimity shortcut', () => {
  it('skips the reranker when BM25 and vector top-3 agree on two positions', async () => {
    const idx = await fresh()

    const vA = syntheticVector(10)
    const vB = syntheticVector(20)
    const vC = syntheticVector(30)
    const vD = syntheticVector(40)

    // Arrange the corpus so the BM25 ranking for "alphabeta"
    // returns a, b, c, and the vector search (query = vA) returns
    // a, b, d. That gives two top-3 agreements -> unanimity fires.
    idx.upsertChunks([
      {
        id: 'a',
        path: 'a.md',
        title: 'alphabeta',
        summary: 'alphabeta primary',
        content: 'alphabeta is primary here',
        embedding: vA,
      },
      {
        id: 'b',
        path: 'b.md',
        title: 'second entry',
        summary: 'alphabeta secondary',
        content: 'alphabeta body',
        embedding: perturb(vA, 0.001, 1),
      },
      {
        id: 'c',
        path: 'c.md',
        title: 'third entry',
        summary: 'alphabeta tertiary',
        content: 'alphabeta third body',
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
      embedder: makeStubEmbedder(new Map([['alphabeta', vA]])),
      reranker,
    })

    const { trace } = await retrieval.searchRaw({
      query: 'alphabeta',
      topK: 5,
      mode: 'hybrid-rerank',
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

  it('breaks equal rerank scores using fused score and original order', async () => {
    const idx = await fresh()

    const vA = syntheticVector(100)
    const vB = syntheticVector(200)
    const vC = syntheticVector(300)

    idx.upsertChunks([
      {
        id: 'a',
        path: 'a.md',
        title: 'alphabeta one',
        summary: 'alphabeta primary',
        content: 'alphabeta first body',
        embedding: vA,
      },
      {
        id: 'b',
        path: 'b.md',
        title: 'alphabeta two',
        summary: 'alphabeta secondary',
        content: 'alphabeta second body',
        embedding: vB,
      },
      {
        id: 'c',
        path: 'c.md',
        title: 'alphabeta three',
        summary: 'alphabeta tertiary',
        content: 'alphabeta third body',
        embedding: vC,
      },
    ])

    const reranker: Reranker = {
      name: () => 'tie-reranker',
      async rerank(req: RerankRequest): Promise<readonly RerankResult[]> {
        return req.documents.map((document, index) => ({
          id: document.id,
          index,
          score: 5,
        }))
      },
    }

    const retrieval = createRetrieval({
      index: idx,
      embedder: makeStubEmbedder(new Map([['alphabeta', vA]])),
      reranker,
    })

    const results = await retrieval.search({
      query: 'alphabeta',
      topK: 3,
      rerank: true,
    })

    expect(results.map((result) => result.id)).toEqual(['a', 'b', 'c'])
  })
})

describe('createRetrieval retry ladder', () => {
  it('recovers on the initial fanout when a strongest token is enough', async () => {
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

    // The full phrase does not appear in any doc, but the initial
    // fanout now includes strongest-token probes so "uniqueword"
    // surfaces the relevant hit without entering the retry ladder.
    const { results, trace } = await retrieval.searchRaw({
      query: '"missing phrase uniqueword"',
    })

    expect(results.length).toBeGreaterThan(0)
    expect(results[0]!.id).toBe('a')
    const strategies = trace.attempts.map((a) => a.strategy)
    expect(strategies[0]).toBe('initial')
    expect(strategies).not.toContain('strongest_term')
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

  it('applies exact path-list filters to the trigram fuzzy rung', async () => {
    const idx = await fresh()

    const retrieval = createRetrieval({
      index: idx,
      trigramChunks: [
        {
          id: 'allowed',
          path: 'notes/photosynthesis.md',
          title: 'Allowed note',
          summary: 'allowed summary',
          content: 'allowed content',
        },
        {
          id: 'blocked',
          path: 'notes/photosynthasis.md',
          title: 'Blocked note',
          summary: 'blocked summary',
          content: 'blocked content',
        },
      ],
    })

    const { results, trace } = await retrieval.searchRaw({
      query: 'photosynthasis',
      filters: {
        paths: ['notes/photosynthesis.md'],
      },
    })

    expect(trace.attempts.map((attempt) => attempt.strategy)).toContain('trigram_fuzzy')
    expect(results.map((result) => result.id)).toEqual(['allowed'])
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
