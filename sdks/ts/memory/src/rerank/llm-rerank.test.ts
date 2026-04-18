import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/types.js'
import { LLMReranker } from './llm-rerank.js'

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
})
