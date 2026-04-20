// SPDX-License-Identifier: Apache-2.0

/**
 * Plugin hook ordering: Start events fire in registration order, End
 * events fire in reverse registration order. Payload contents are sane.
 */

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { createMemory } from './index.js'
import type { Plugin } from './types.js'

const stubProvider = (content: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  complete: async (_req: CompletionRequest): Promise<CompletionResponse> => ({
    content,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async (_req: StructuredRequest) => content,
})

describe('plugin hook ordering', () => {
  it('fires extraction Start in registration order, End in reverse order', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const events: string[] = []
    const mk = (name: string): Plugin => ({
      name,
      onExtractionStart: (ctx) => {
        events.push(`${name}:start:${ctx.scope}`)
      },
      onExtractionEnd: (ctx) => {
        events.push(`${name}:end:${ctx.extracted.length}`)
      },
    })
    const mem = createMemory({
      store,
      provider: stubProvider(JSON.stringify({ memories: [] })),
      cursorStore,
      scope: 'project',
      actorId: 't',
      plugins: [mk('a'), mk('b'), mk('c')],
    })
    await mem.extract({
      messages: Array.from({ length: 6 }, (_, i) => ({ role: 'user', content: `m${i}` })),
    })
    expect(events).toEqual([
      'a:start:project',
      'b:start:project',
      'c:start:project',
      'c:end:0',
      'b:end:0',
      'a:end:0',
    ])
  })

  it('fires reflection hooks with the parsed result in payload', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const payloads: unknown[] = []
    const plugin: Plugin = {
      name: 'probe',
      onReflectionStart: (ctx) => {
        payloads.push({ phase: 'start', messages: ctx.messages.length })
      },
      onReflectionEnd: (ctx) => {
        payloads.push({ phase: 'end', outcome: ctx.result?.outcome })
      },
    }
    const mem = createMemory({
      store,
      provider: stubProvider(JSON.stringify({ outcome: 'success', summary: 's', heuristics: [] })),
      cursorStore,
      scope: 'project',
      actorId: 't',
      plugins: [plugin],
    })
    await mem.reflect({ messages: [{ role: 'user', content: 'x' }], sessionId: 'sess' })
    expect(payloads).toEqual([
      { phase: 'start', messages: 1 },
      { phase: 'end', outcome: 'success' },
    ])
  })

  it('fires consolidation hooks with the report in payload', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const events: string[] = []
    const plugin: Plugin = {
      name: 'probe',
      onConsolidationStart: (ctx) => {
        events.push(`start:${ctx.scope}`)
      },
      onConsolidationEnd: (ctx) => {
        events.push(`end:${ctx.scope}:${ctx.report?.merged ?? 0}`)
      },
    }
    const mem = createMemory({
      store,
      provider: stubProvider(JSON.stringify({ verdict: 'distinct' })),
      cursorStore,
      scope: 'project',
      actorId: 't',
      plugins: [plugin],
    })
    await mem.consolidate()
    expect(events).toEqual(['start:project', 'end:project:0'])
  })
})
