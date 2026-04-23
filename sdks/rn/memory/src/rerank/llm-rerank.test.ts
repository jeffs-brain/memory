import { describe, expect, it } from 'vitest'

import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/types.js'
import { LLMReranker, composeLLMRerankDocument } from './llm-rerank.js'

type StubProvider = Provider & {
  readonly calls: CompletionRequest[]
}

const stubProvider = (
  respond: (request: CompletionRequest, callIndex: number) => string,
): StubProvider => {
  const calls: CompletionRequest[] = []
  return {
    calls,
    name: () => 'stub',
    modelName: () => 'stub-model',
    supportsStructuredDecoding: () => false,
    complete: async (request): Promise<CompletionResponse> => {
      const index = calls.length
      calls.push(request)
      return {
        content: respond(request, index),
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    },
    async *stream(_request: CompletionRequest): AsyncIterable<StreamEvent> {
      yield* [] as readonly StreamEvent[]
    },
    structured: async (_request: StructuredRequest): Promise<string> => {
      throw new Error('not implemented')
    },
  }
}

const makeDocs = (count: number) =>
  Array.from({ length: count }, (_, index) => ({
    id: `doc-${index}`,
    text: `doc ${index} text`,
  }))

describe('LLMReranker', () => {
  it('batches documents and orders them by descending score', async () => {
    const provider = stubProvider(() =>
      JSON.stringify([
        { id: 0, score: 2 },
        { id: 1, score: 10 },
        { id: 2, score: 5 },
      ]),
    )
    const reranker = new LLMReranker({ provider, batchSize: 5, parallelism: 1 })
    const results = await reranker.rerank({ query: 'q', documents: makeDocs(3) })

    expect(results.map((result) => result.id)).toEqual(['doc-1', 'doc-2', 'doc-0'])
    expect(provider.calls.length).toBe(1)
  })

  it('falls back to zero scores on malformed output', async () => {
    const provider = stubProvider(() => 'not json at all')
    const reranker = new LLMReranker({ provider })
    const results = await reranker.rerank({ query: 'q', documents: makeDocs(3) })

    expect(provider.calls.length).toBe(2)
    expect(results.every((result) => result.score === 0)).toBe(true)
    expect(results.map((result) => result.id)).toEqual(['doc-0', 'doc-1', 'doc-2'])
  })

  it('renders rerank documents with summary and body content', () => {
    const rendered = composeLLMRerankDocument({
      id: 0,
      path: 'memory/global/alpha.md',
      title: 'Alpha',
      summary: 'Summary line',
      content: 'Body line with 2024-02-01 and 250 raised.',
    })
    expect(rendered).toContain('summary: Summary line')
    expect(rendered).toContain('content: Body line with 2024-02-01 and 250 raised.')
  })
})
