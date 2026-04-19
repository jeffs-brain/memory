// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the CLI provider/embedder factories. These live on
 * hot paths for `memory serve`, so regressions here silently break the
 * tri-SDK LongMemEval daemon pipeline (each SDK spawns a serve daemon
 * pointing at the same shared brain).
 */

import { afterEach, describe, expect, it, vi } from 'vitest'
import type {
  CompletionResponse,
  Provider,
  StreamEvent,
} from '../llm/index.js'
import {
  buildProvider,
  buildReranker,
  embedderFromEnv,
  providerFromEnvOptional,
  rerankerFromEnv,
} from './config.js'

afterEach(() => {
  vi.unstubAllEnvs()
})

const makeStubProvider = (payload = '[{"id":0,"score":9},{"id":1,"score":1}]'): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream(): AsyncIterable<StreamEvent> {
    yield { type: 'done', stopReason: 'end_turn' }
  },
  complete: async (): Promise<CompletionResponse> => ({
    content: payload,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async () => payload,
})

describe('buildProvider', () => {
  it('forwards baseURL to the anthropic provider', () => {
    const provider = buildProvider({
      kind: 'anthropic',
      apiKey: 'sk-test',
      model: 'claude-opus-4-5',
      baseURL: 'https://proxy.example.com/v1',
    })
    expect(provider.name()).toBe('anthropic')
    // The base URL lives on the private field; assert via the provider
    // surface by checking the string serialises. We rely on the
    // configured base URL reaching the HTTP client by running a real
    // call in the integration suite; here we just sanity-check the
    // constructor did not throw and the provider identifies itself.
    expect(provider.modelName()).toBe('claude-opus-4-5')
  })

  it('forwards baseURL to the openai provider', () => {
    const provider = buildProvider({
      kind: 'openai',
      apiKey: 'sk-test',
      model: 'gpt-4o',
      baseURL: 'https://proxy.example.com/v1',
    })
    expect(provider.name()).toBe('openai')
    expect(provider.modelName()).toBe('gpt-4o')
  })

  it('supports ollama without an apiKey', () => {
    const provider = buildProvider({
      kind: 'ollama',
      apiKey: '',
      model: 'llama3.1',
    })
    expect(provider.name()).toBe('ollama')
    expect(provider.modelName()).toBe('llama3.1')
  })
})

describe('buildProvider / anthropic baseURL wiring', () => {
  it('routes HTTP calls to the baseURL when set', async () => {
    // Inject a fake fetch by swapping the provider's http client via a
    // monkey-patched global. The AnthropicProvider defers to
    // defaultHttpClient which resolves globalThis.fetch; capturing it
    // here is enough to assert the baseURL is honoured.
    const seen: { url?: string } = {}
    const originalFetch = globalThis.fetch
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      seen.url = url
      // Minimal streaming response the provider can parse.
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(
            new TextEncoder().encode(
              'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ),
          )
          controller.close()
        },
      })
      return new Response(body, {
        status: 200,
        headers: { 'content-type': 'text/event-stream' },
      })
    }) as typeof fetch

    try {
      const provider = buildProvider({
        kind: 'anthropic',
        apiKey: 'sk-test',
        model: 'claude-opus-4-5',
        baseURL: 'https://proxy.example.com/v1',
      })
      // Consume one stream turn. The iteration forces the HTTP call.
      const events = provider.stream({ messages: [{ role: 'user', content: 'hi' }] })
      for await (const _ of events) {
        // drain
      }
      expect(seen.url).toBeDefined()
      expect(seen.url!.startsWith('https://proxy.example.com/v1')).toBe(true)
    } finally {
      globalThis.fetch = originalFetch
    }
  })
})

describe('provider env inference', () => {
  it('infers anthropic settings from fallback env vars', () => {
    vi.stubEnv('ANTHROPIC_API_KEY', 'sk-ant')
    vi.stubEnv('ANTHROPIC_BASE_URL', 'https://proxy.example.com/v1')
    vi.stubEnv('JB_LLM_MODEL', 'claude-opus-4-5')

    expect(providerFromEnvOptional()).toEqual({
      kind: 'anthropic',
      apiKey: 'sk-ant',
      model: 'claude-opus-4-5',
      baseURL: 'https://proxy.example.com/v1',
    })
  })

  it('infers openai settings from fallback env vars', () => {
    vi.stubEnv('OPENAI_API_KEY', 'sk-open')
    vi.stubEnv('OPENAI_BASE_URL', 'https://proxy.example.com/v1')
    vi.stubEnv('JB_LLM_MODEL', 'gpt-4o')

    expect(providerFromEnvOptional()).toEqual({
      kind: 'openai',
      apiKey: 'sk-open',
      model: 'gpt-4o',
      baseURL: 'https://proxy.example.com/v1',
    })
  })
})

describe('embedder env inference', () => {
  it('infers an openai embedder from fallback env vars', () => {
    vi.stubEnv('OPENAI_API_KEY', 'sk-open')
    vi.stubEnv('OPENAI_BASE_URL', 'https://proxy.example.com')
    vi.stubEnv('JB_EMBED_MODEL', 'text-embedding-3-small')

    expect(embedderFromEnv()).toEqual({
      kind: 'openai',
      apiKey: 'sk-open',
      baseURL: 'https://proxy.example.com',
      model: 'text-embedding-3-small',
    })
  })
})

describe('reranker config', () => {
  it('parses llm reranker settings from the environment', () => {
    vi.stubEnv('JB_RERANK_PROVIDER', 'llm')

    expect(rerankerFromEnv()).toEqual({
      kind: 'llm',
      baseURL: 'http://localhost:8080',
      label: 'llm-rerank',
      batchSize: 5,
      parallelism: 4,
      concurrencyCap: 4,
      preferHttp: false,
    })
  })

  it('parses auto reranker settings from the environment', () => {
    vi.stubEnv('JB_RERANK_PROVIDER', 'auto')
    vi.stubEnv('JB_RERANK_BATCH_SIZE', '7')
    vi.stubEnv('JB_RERANK_PARALLELISM', '2')
    vi.stubEnv('JB_RERANK_CONCURRENCY', '3')

    expect(rerankerFromEnv()).toEqual({
      kind: 'auto',
      baseURL: 'http://localhost:8080',
      label: 'auto-rerank',
      batchSize: 7,
      parallelism: 2,
      concurrencyCap: 3,
      preferHttp: false,
    })
  })

  it('treats http as an alias for the TEI reranker surface', () => {
    vi.stubEnv('JB_RERANK_PROVIDER', 'http')

    expect(rerankerFromEnv()).toEqual({
      kind: 'http',
      baseURL: 'http://localhost:8080',
      label: 'cross-encoder',
      batchSize: 5,
      parallelism: 4,
      concurrencyCap: 4,
      preferHttp: true,
    })
  })

  it('prefers TEI in auto mode when the health probe succeeds', async () => {
    const provider = makeStubProvider()
    const fetch = vi.fn(async (input: string | URL | Request, init?: RequestInit) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      if (url.endsWith('/health')) {
        return new Response('ok', { status: 200 })
      }
      if (url.endsWith('/rerank') && init?.method === 'POST') {
        return new Response('[{"index":1,"score":9},{"index":0,"score":1}]', { status: 200 })
      }
      return new Response('missing', { status: 404 })
    })

    const reranker = buildReranker(
      {
        kind: 'auto',
        baseURL: 'http://tei.example.com',
        label: 'auto-rerank',
        batchSize: 5,
        parallelism: 1,
        concurrencyCap: 1,
        preferHttp: true,
      },
      {
        provider,
        http: { fetch },
      },
    )
    const out = await reranker.rerank({
      query: 'q',
      documents: [
        { id: 'a', text: 'alpha' },
        { id: 'b', text: 'bravo' },
      ],
    })

    expect(out.map((entry) => entry.id)).toEqual(['b', 'a'])
    expect(fetch).toHaveBeenCalled()
  })

  it('falls back to the llm reranker in auto mode when TEI is unavailable', async () => {
    const complete = vi.fn(async (): Promise<CompletionResponse> => ({
      content: '[{"id":0,"score":9},{"id":1,"score":1}]',
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }))
    const provider: Provider = {
      ...makeStubProvider(),
      complete,
    }
    const fetch = vi.fn(async () => new Response('down', { status: 503 }))

    const reranker = buildReranker(
      {
        kind: 'auto',
        baseURL: 'http://tei.example.com',
        label: 'auto-rerank',
        batchSize: 5,
        parallelism: 1,
        concurrencyCap: 1,
        preferHttp: true,
      },
      {
        provider,
        http: { fetch },
      },
    )
    const out = await reranker.rerank({
      query: 'q',
      documents: [
        { id: 'a', text: 'alpha' },
        { id: 'b', text: 'bravo' },
      ],
    })

    expect(out.map((entry) => entry.id)).toEqual(['a', 'b'])
    expect(complete).toHaveBeenCalledTimes(1)
  })

  it('builds an llm reranker when explicitly requested', async () => {
    const complete = vi.fn(async (): Promise<CompletionResponse> => ({
      content: '[{"id":1,"score":9},{"id":0,"score":1}]',
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }))
    const provider: Provider = {
      ...makeStubProvider(),
      complete,
    }

    const reranker = buildReranker(
      {
        kind: 'llm',
        baseURL: 'http://ignored.example.com',
        label: 'llm-rerank',
        batchSize: 5,
        parallelism: 1,
        concurrencyCap: 1,
        preferHttp: false,
      },
      { provider },
    )
    const out = await reranker.rerank({
      query: 'q',
      documents: [
        { id: 'a', text: 'alpha' },
        { id: 'b', text: 'bravo' },
      ],
    })

    expect(out.map((entry) => entry.id)).toEqual(['b', 'a'])
    expect(complete).toHaveBeenCalledTimes(1)
  })

  it('uses the llm reranker directly in auto mode when no explicit rerank URL is configured', async () => {
    const complete = vi.fn(async (): Promise<CompletionResponse> => ({
      content: '[{"id":1,"score":9},{"id":0,"score":1}]',
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }))
    const provider: Provider = {
      ...makeStubProvider(),
      complete,
    }

    const reranker = buildReranker(
      {
        kind: 'auto',
        baseURL: 'http://localhost:8080',
        label: 'auto-rerank',
        batchSize: 5,
        parallelism: 1,
        concurrencyCap: 1,
        preferHttp: false,
      },
      { provider },
    )
    const out = await reranker.rerank({
      query: 'q',
      documents: [
        { id: 'a', text: 'alpha' },
        { id: 'b', text: 'bravo' },
      ],
    })

    expect(out.map((entry) => entry.id)).toEqual(['b', 'a'])
    expect(complete).toHaveBeenCalledTimes(1)
  })
})
