// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/index.js'
import { createCache } from './cache.js'
import { createDistiller } from './distill.js'
import { DISTILL_SYSTEM_PROMPT } from './prompt.js'

type StubProvider = Provider & {
  calls: CompletionRequest[]
}

const expectDefined = <T>(value: T | undefined, message: string): T => {
  if (value === undefined) throw new Error(message)
  return value
}

/**
 * makeStubProvider spins up a minimal Provider that records every
 * completion request and returns the next canned response. Streaming
 * and structured helpers are unused by the distiller but required by
 * the Provider surface.
 */
function makeStubProvider(responses: readonly string[]): StubProvider {
  const queue = [...responses]
  const calls: CompletionRequest[] = []

  const stub: StubProvider = {
    calls,
    name: () => 'stub',
    modelName: () => 'stub-model',
    supportsStructuredDecoding: () => false,
    async *stream(): AsyncIterable<StreamEvent> {
      yield { type: 'done', stopReason: 'end_turn' }
    },
    async complete(req: CompletionRequest): Promise<CompletionResponse> {
      calls.push(req)
      const next = queue.shift() ?? ''
      return {
        content: next,
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    },
    async structured(_req: StructuredRequest): Promise<string> {
      return ''
    },
  }
  return stub
}

describe('createDistiller', () => {
  it('short-circuits empty input without calling the LLM', async () => {
    const provider = makeStubProvider(['unused'])
    const distiller = createDistiller({ provider })
    expect(await distiller.distill('')).toBe('')
    expect(await distiller.distill('   ')).toBe('')
    expect(provider.calls).toHaveLength(0)
  })

  it('returns the distilled query from the provider on a cache miss', async () => {
    const provider = makeStubProvider(['distilled output'])
    const distiller = createDistiller({ provider, model: 'gpt-test' })
    const out = await distiller.distill('raw query')
    expect(out).toBe('distilled output')
    expect(provider.calls).toHaveLength(1)
    const call = expectDefined(provider.calls[0], 'expected distiller call')
    expect(call.model).toBe('gpt-test')
    expect(call.temperature).toBe(0)
    expect(call.maxTokens).toBe(256)
    expect(call.messages[0]).toEqual({
      role: 'system',
      content: DISTILL_SYSTEM_PROMPT,
    })
    expect(call.messages[1]).toEqual({ role: 'user', content: 'raw query' })
  })

  it('returns the cached value without a second LLM call on repeat input', async () => {
    const provider = makeStubProvider(['first response', 'should never be used'])
    const distiller = createDistiller({ provider, model: 'gpt-test' })
    expect(await distiller.distill('raw query')).toBe('first response')
    expect(await distiller.distill('raw query')).toBe('first response')
    expect(provider.calls).toHaveLength(1)
  })

  it('treats casing and surrounding whitespace as cache-equivalent', async () => {
    const provider = makeStubProvider(['cached'])
    const distiller = createDistiller({ provider })
    await distiller.distill('Raw Query')
    await distiller.distill('  raw query  ')
    expect(provider.calls).toHaveLength(1)
  })

  it('re-runs the LLM when the model changes for the same query', async () => {
    const provider = makeStubProvider(['first', 'second'])
    const sharedCache = createCache({ capacity: 16 })
    const d1 = createDistiller({ provider, model: 'gpt-a', cache: sharedCache })
    const d2 = createDistiller({ provider, model: 'gpt-b', cache: sharedCache })
    expect(await d1.distill('hello')).toBe('first')
    expect(await d2.distill('hello')).toBe('second')
    expect(provider.calls).toHaveLength(2)
  })

  it('evicts the least-recently-used entry once the LRU is full', async () => {
    const provider = makeStubProvider(['r1', 'r2', 'r3', 'r1-again'])
    const distiller = createDistiller({ provider, cacheCapacity: 2 })

    expect(await distiller.distill('one')).toBe('r1')
    expect(await distiller.distill('two')).toBe('r2')
    expect(await distiller.distill('three')).toBe('r3') // evicts 'one'

    // 'two' and 'three' are cached; a fresh 'one' triggers another LLM call.
    expect(await distiller.distill('two')).toBe('r2')
    expect(await distiller.distill('three')).toBe('r3')
    expect(await distiller.distill('one')).toBe('r1-again')
    expect(provider.calls).toHaveLength(4)
  })

  it('trims trailing whitespace from the provider response', async () => {
    const provider = makeStubProvider(['  padded output\n'])
    const distiller = createDistiller({ provider })
    expect(await distiller.distill('x')).toBe('padded output')
  })

  it('propagates the abort signal to the provider', async () => {
    const provider = makeStubProvider(['ok'])
    const completeSpy = vi.spyOn(provider, 'complete')
    const distiller = createDistiller({ provider })
    const ctrl = new AbortController()
    await distiller.distill('abortable', ctrl.signal)
    expect(completeSpy).toHaveBeenCalledWith(expect.any(Object), ctrl.signal)
  })

  it('uses a caller-supplied cache so multiple distillers can share state', async () => {
    const provider = makeStubProvider(['from-llm', 'should-not-fire'])
    const shared = createCache({ capacity: 8 })
    const d1 = createDistiller({ provider, cache: shared })
    const d2 = createDistiller({ provider, cache: shared })
    expect(await d1.distill('overlapping')).toBe('from-llm')
    expect(await d2.distill('overlapping')).toBe('from-llm')
    expect(provider.calls).toHaveLength(1)
  })
})
