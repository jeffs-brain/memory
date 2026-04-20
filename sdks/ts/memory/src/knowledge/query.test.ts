// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  type CompletionRequest,
  type CompletionResponse,
  type Provider,
  noopLogger,
} from '../llm/index.js'
import { joinPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { createKnowledge } from './index.js'
import { WIKI_PREFIX } from './promote.js'
import type { KnowledgeQueryRetriever } from './types.js'

const makeProvider = (handler: (req: CompletionRequest) => CompletionResponse): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  supportsStructuredDecoding: () => false,
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  async complete(req) {
    return handler(req)
  },
  async structured() {
    return '{}'
  },
})

describe('query', () => {
  it('falls back to the local wiki scan when no retriever is configured', async () => {
    const store = createMemStore()
    const articlePath = joinPath(WIKI_PREFIX, 'transport.md')
    const article = serialiseFrontmatter(
      {
        title: 'Transport',
        summary: 'Notes about trains and trams.',
        tags: ['mobility'],
        sources: [],
        modified: '2026-04-17T00:00:00Z',
      },
      'Trains run on the main line and trams cover city routes.',
    )
    await store.write(articlePath, Buffer.from(article, 'utf8'))

    let captured: CompletionRequest | undefined
    const provider = makeProvider((req) => {
      captured = req
      return {
        content: 'Transport is covered in [[transport]].',
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    })

    const knowledge = createKnowledge({ store, provider, logger: noopLogger })
    const result = await knowledge.query('How does transport work?')

    expect(result.answer).toBe('Transport is covered in [[transport]].')
    expect(result.sourcePaths).toEqual([articlePath])
    expect(result.searchHits).toBe(1)
    expect(captured?.system).toContain('knowledge base query engine')
    expect(captured?.messages[0]?.content).toContain('How does transport work?')
    expect(captured?.messages[0]?.content).toContain('Trains run on the main line')
  })

  it('uses retrieval-backed article hydration when a retriever is configured', async () => {
    const store = createMemStore()
    const articlePath = joinPath(WIKI_PREFIX, 'wildlife.md')
    const article = serialiseFrontmatter(
      {
        title: 'Wetland Wildlife',
        summary: 'Species found near Dutch wetlands.',
        tags: ['wildlife'],
        sources: [],
        modified: '2026-04-18T00:00:00Z',
      },
      'Otters have returned to several wetland areas.',
    )
    await store.write(articlePath, Buffer.from(article, 'utf8'))

    let captured: CompletionRequest | undefined
    const provider = makeProvider((req) => {
      captured = req
      return {
        content: 'Wetland wildlife is described in [[wildlife]].',
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    })

    const retriever: KnowledgeQueryRetriever = {
      retrieve: async () => ({
        chunks: [
          {
            path: articlePath,
            score: 0.91,
            metadata: {
              title: 'Wetland Wildlife',
              modified: '2026-04-18T00:00:00Z',
            },
          },
          {
            path: articlePath,
            score: 0.55,
            metadata: {
              title: 'Wetland Wildlife',
              modified: '2026-04-18T00:00:00Z',
            },
          },
        ],
      }),
    }

    const knowledge = createKnowledge({ store, provider, retriever, logger: noopLogger })
    const result = await knowledge.query('What is the otter situation?')

    expect(result.answer).toBe('Wetland wildlife is described in [[wildlife]].')
    expect(result.sourcePaths).toEqual([articlePath])
    expect(result.searchHits).toBe(1)
    expect(captured?.messages[0]?.content).toContain('What is the otter situation?')
    expect(captured?.messages[0]?.content).toContain(
      'Otters have returned to several wetland areas.',
    )
  })

  it('returns the empty knowledge message when there are no wiki articles', async () => {
    const store = createMemStore()
    const provider = makeProvider(() => {
      throw new Error('provider should not be called')
    })

    const knowledge = createKnowledge({ store, provider, logger: noopLogger })
    const result = await knowledge.query('Anything at all')

    expect(result.answer).toBe('Knowledge base is empty.')
    expect(result.sourcePaths).toEqual([])
    expect(result.searchHits).toBe(0)
  })

  it('falls back to search results when synthesis fails', async () => {
    const store = createMemStore()
    const articlePath = joinPath(WIKI_PREFIX, 'transport.md')
    const article = serialiseFrontmatter(
      {
        title: 'Transport',
        summary: 'Notes about trains and trams.',
        tags: ['mobility'],
        sources: [],
      },
      'Trains run on the main line and trams cover city routes.',
    )
    await store.write(articlePath, Buffer.from(article, 'utf8'))

    const provider = makeProvider(() => {
      throw new Error('provider unavailable')
    })

    const knowledge = createKnowledge({ store, provider, logger: noopLogger })
    const result = await knowledge.query('How does transport work?')

    expect(result.answer).toContain(
      'LLM synthesis unavailable (stub/stub-model): provider unavailable',
    )
    expect(result.answer).toContain('[[transport]]')
    expect(result.sourcePaths).toEqual([articlePath])
    expect(result.searchHits).toBe(1)
  })
})
