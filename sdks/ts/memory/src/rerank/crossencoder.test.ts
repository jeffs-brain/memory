// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type {
  Reranker as TEIRerankerContract,
  RerankScore,
} from '../llm/types.js'
import { CrossEncoderReranker } from './crossencoder.js'

function stubClient(hits: readonly RerankScore[]): {
  client: TEIRerankerContract
  calls: { query: string; documents: readonly string[] }[]
} {
  const calls: { query: string; documents: readonly string[] }[] = []
  const client: TEIRerankerContract = {
    name: () => 'stub-tei',
    rerank: vi.fn(
      async (query: string, documents: readonly string[]) => {
        calls.push({ query, documents })
        return hits
      },
    ),
  }
  return { client, calls }
}

describe('CrossEncoderReranker', () => {
  it('passes the query + document payloads through to the TEI client', async () => {
    const { client, calls } = stubClient([
      { index: 0, score: 0.1 },
      { index: 1, score: 0.9 },
    ])
    const reranker = new CrossEncoderReranker({ client })
    const out = await reranker.rerank({
      query: 'hello',
      documents: [
        { id: 'a', text: 'alpha' },
        { id: 'b', text: 'bravo' },
      ],
    })
    expect(calls).toEqual([{ query: 'hello', documents: ['alpha', 'bravo'] }])
    expect(out.map((r) => r.id)).toEqual(['b', 'a'])
    expect(out[0]?.score).toBe(0.9)
  })

  it('sinks documents the backend did not score to the tail', async () => {
    const { client } = stubClient([{ index: 1, score: 5 }])
    const reranker = new CrossEncoderReranker({ client })
    const out = await reranker.rerank({
      query: 'q',
      documents: [
        { id: 'a', text: 'alpha' },
        { id: 'b', text: 'bravo' },
      ],
    })
    expect(out.map((r) => r.id)).toEqual(['b', 'a'])
    expect(out[1]?.score).toBe(Number.NEGATIVE_INFINITY)
  })

  it('short-circuits on empty documents without calling the client', async () => {
    const { client, calls } = stubClient([])
    const reranker = new CrossEncoderReranker({ client })
    const out = await reranker.rerank({ query: 'q', documents: [] })
    expect(out).toEqual([])
    expect(calls).toEqual([])
  })

  it('delegates availability checks to the TEI client when present', async () => {
    const { client } = stubClient([])
    client.isAvailable = vi.fn(async () => false)
    const reranker = new CrossEncoderReranker({ client })
    await expect(reranker.isAvailable?.()).resolves.toBe(false)
    expect(client.isAvailable).toHaveBeenCalledTimes(1)
  })

  it('shares the configured concurrency cap across reranker instances', async () => {
    let active = 0
    let maxActive = 0
    const client: TEIRerankerContract = {
      name: () => 'slow-tei',
      rerank: vi.fn(async () => {
        active += 1
        maxActive = Math.max(maxActive, active)
        await new Promise((resolve) => setTimeout(resolve, 10))
        active -= 1
        return [{ index: 0, score: 1 }]
      }),
    }

    const left = new CrossEncoderReranker({ client, concurrencyCap: 1 })
    const right = new CrossEncoderReranker({ client, concurrencyCap: 1 })

    await Promise.all([
      left.rerank({ query: 'q', documents: [{ id: 'a', text: 'alpha' }] }),
      right.rerank({ query: 'q', documents: [{ id: 'b', text: 'bravo' }] }),
    ])

    expect(maxActive).toBe(1)
  })
})
