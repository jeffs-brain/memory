// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/types.js'
import { LLMReranker, composeLLMRerankDocument, extractJSONArray } from './llm-rerank.js'

type StubProvider = Provider & {
  readonly calls: CompletionRequest[]
}

function stubProvider(
  respond: (req: CompletionRequest, callIndex: number) => string,
): StubProvider {
  const calls: CompletionRequest[] = []
  const p: StubProvider = {
    calls,
    name: () => 'stub',
    modelName: () => 'stub-model',
    supportsStructuredDecoding: () => false,
    async complete(req: CompletionRequest): Promise<CompletionResponse> {
      const idx = calls.length
      calls.push(req)
      return {
        content: respond(req, idx),
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    },
    // biome-ignore lint/correctness/useYield: deliberate empty generator stub
    async *stream(_req: CompletionRequest): AsyncIterable<StreamEvent> {
      return
    },
    async structured(_req: StructuredRequest): Promise<string> {
      throw new Error('not implemented')
    },
  }
  return p
}

function makeDocs(n: number) {
  return Array.from({ length: n }, (_, i) => ({ id: `doc-${i}`, text: `doc ${i} text` }))
}

describe('LLMReranker', () => {
  it('fans out 4 parallel batches of 5 documents by default', async () => {
    const provider = stubProvider((_req, idx) => {
      // Each batch has 5 local IDs; score them 9, 8, 7, 6, 5 so we can
      // map results back.
      return JSON.stringify([
        { id: 0, score: 9 + idx },
        { id: 1, score: 8 + idx },
        { id: 2, score: 7 + idx },
        { id: 3, score: 6 + idx },
        { id: 4, score: 5 + idx },
      ])
    })
    const reranker = new LLMReranker({ provider })
    const out = await reranker.rerank({ query: 'q', documents: makeDocs(20) })

    expect(provider.calls.length).toBe(4)
    // Highest score comes from the last batch (idx=3 offset 15), local 0.
    expect(out[0]?.id).toBe('doc-15')
    expect(out[0]?.score).toBe(12)
    // Each of the 20 documents must appear in the output exactly once.
    const ids = new Set(out.map((r) => r.id))
    expect(ids.size).toBe(20)
  })

  it('treats a malformed response as all-zero scores', async () => {
    const provider = stubProvider(() => 'not json at all')
    const reranker = new LLMReranker({ provider })
    const out = await reranker.rerank({ query: 'q', documents: makeDocs(3) })
    // Two calls: first attempt + strict retry. Both fail to parse.
    expect(provider.calls.length).toBe(2)
    expect(out.every((r) => r.score === 0)).toBe(true)
    // Stable ordering by original index when scores tie.
    expect(out.map((r) => r.id)).toEqual(['doc-0', 'doc-1', 'doc-2'])
  })

  it('orders results by descending score with a single batch', async () => {
    const provider = stubProvider(() =>
      JSON.stringify([
        { id: 0, score: 2 },
        { id: 1, score: 10 },
        { id: 2, score: 5 },
      ]),
    )
    const reranker = new LLMReranker({ provider, batchSize: 5, parallelism: 1 })
    const out = await reranker.rerank({ query: 'q', documents: makeDocs(3) })
    expect(out.map((r) => r.id)).toEqual(['doc-1', 'doc-2', 'doc-0'])
    expect(provider.calls.length).toBe(1)
  })

  it('accepts bare numeric score arrays', async () => {
    const provider = stubProvider(() => '[3, 1, 2]')
    const reranker = new LLMReranker({ provider, batchSize: 5, parallelism: 1 })
    const out = await reranker.rerank({ query: 'q', documents: makeDocs(3) })
    expect(out.map((r) => r.id)).toEqual(['doc-0', 'doc-2', 'doc-1'])
  })

  it('honours a custom batch size of 3 => two batches for six docs', async () => {
    const provider = stubProvider(() => '[1, 2, 3]')
    const reranker = new LLMReranker({ provider, batchSize: 3, parallelism: 2 })
    await reranker.rerank({ query: 'q', documents: makeDocs(6) })
    expect(provider.calls.length).toBe(2)
  })

  it('includes body content alongside summary in rerank documents', () => {
    const rendered = composeLLMRerankDocument({
      id: 0,
      path: 'memory/global/alpha.md',
      title: 'Alpha',
      summary: 'Summary line',
      content: 'Body line with 2024-02-01 and $250 raised.',
    })
    expect(rendered).toContain('summary: Summary line')
    expect(rendered).toContain('summary: Summary line\n\n    content:')
    expect(rendered).toContain('content: Body line with 2024-02-01 and $250 raised.')
  })

  it('parses fenced JSON responses from provider', async () => {
    const fenced =
      '```json\n[{"id": 0, "score": 9}, {"id": 1, "score": 3}, {"id": 2, "score": 6}]\n```'
    const provider = stubProvider(() => fenced)
    const reranker = new LLMReranker({ provider, batchSize: 5, parallelism: 1 })
    const out = await reranker.rerank({ query: 'q', documents: makeDocs(3) })
    expect(out.map((r) => r.id)).toEqual(['doc-0', 'doc-2', 'doc-1'])
    expect(provider.calls.length).toBe(1)
  })

  it('parses fenced JSON without language tag from provider', async () => {
    const fenced = '```\n[{"id": 0, "score": 2}, {"id": 1, "score": 8}, {"id": 2, "score": 5}]\n```'
    const provider = stubProvider(() => fenced)
    const reranker = new LLMReranker({ provider, batchSize: 5, parallelism: 1 })
    const out = await reranker.rerank({ query: 'q', documents: makeDocs(3) })
    expect(out.map((r) => r.id)).toEqual(['doc-1', 'doc-2', 'doc-0'])
    expect(provider.calls.length).toBe(1)
  })
})

describe('extractJSONArray', () => {
  it('extracts unfenced JSON array', () => {
    expect(extractJSONArray('[1, 3, 2]')).toBe('[1, 3, 2]')
  })

  it('extracts from fenced JSON with language tag', () => {
    expect(extractJSONArray('```json\n[1, 3, 2]\n```')).toBe('[1, 3, 2]')
  })

  it('extracts from fenced JSON without language tag', () => {
    expect(extractJSONArray('```\n[1, 3, 2]\n```')).toBe('[1, 3, 2]')
  })

  it('returns undefined for malformed input', () => {
    expect(extractJSONArray('not json at all')).toBeUndefined()
  })

  it('returns undefined for empty string', () => {
    expect(extractJSONArray('')).toBeUndefined()
  })

  it('handles fences with extra whitespace', () => {
    expect(extractJSONArray('  ```json  \n  [1, 3, 2]  \n  ```  ')).toBe('[1, 3, 2]')
  })

  it('handles nested fences without crashing', () => {
    const nested = '```json\n```inner\n[1, 2, 3]\n```\n```'
    const result = extractJSONArray(nested)
    // Should not throw; the exact extraction is best-effort.
    expect(result === undefined || result === '[1, 2, 3]' || typeof result === 'string').toBe(true)
  })

  it('extracts array surrounded by prose', () => {
    expect(extractJSONArray('Here is the result:\n[1, 2, 3]\nHope this helps!')).toBe('[1, 2, 3]')
  })

  it('extracts array from fenced block with trailing commentary', () => {
    const input = '```json\n[{"id": 0, "score": 8.5}]\n```\nSome trailing text.'
    expect(extractJSONArray(input)).toBe('[{"id": 0, "score": 8.5}]')
  })
})
