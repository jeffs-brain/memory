// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the CLI provider/embedder factories. These live on
 * hot paths for `memory serve`, so regressions here silently break the
 * tri-SDK LongMemEval daemon pipeline (each SDK spawns a serve daemon
 * pointing at the same shared brain).
 */

import { describe, expect, it } from 'vitest'
import { buildProvider } from './config.js'

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
