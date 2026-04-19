// SPDX-License-Identifier: Apache-2.0

/**
 * Regression tests for the OpenAI provider's streaming path. Exercise
 * the same code a caller would hit against cloud OpenAI-compat
 * endpoints (Cloud Run / vLLM / gemma4) so a plain curl-works-but-SDK-
 * does-not regression cannot slip through.
 */

import { describe, expect, it } from 'vitest'

import type { HttpClient } from './http.js'
import { OpenAIEmbedder, OpenAIProvider } from './openai.js'
import type { StreamEvent } from './types.js'

type RecordedCall = {
  url: string
  headers: Record<string, string>
  body: unknown
  accept: string
  contentType: string
}

type FakeServer = {
  body: string | (() => ReadableStream<Uint8Array>)
  status?: number
  headers?: Record<string, string>
}

/** Build an HttpClient whose `fetch` returns a canned response. */
const makeHttp = (
  impl: (url: string, call: RecordedCall) => FakeServer,
): { http: HttpClient; calls: RecordedCall[] } => {
  const calls: RecordedCall[] = []
  const http: HttpClient = {
    fetch: async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString()
      const headers = normaliseHeaders(init?.headers)
      const bodyStr = typeof init?.body === 'string' ? init.body : ''
      const body = bodyStr === '' ? null : JSON.parse(bodyStr)
      const call: RecordedCall = {
        url,
        headers,
        body,
        accept: headers.accept ?? '',
        contentType: headers['content-type'] ?? '',
      }
      calls.push(call)
      const fake = impl(url, call)
      const respBody = typeof fake.body === 'string' ? new Blob([fake.body]).stream() : fake.body()
      return new Response(respBody, {
        status: fake.status ?? 200,
        headers: {
          'content-type': 'text/event-stream',
          ...(fake.headers ?? {}),
        },
      })
    },
  }
  return { http, calls }
}

const normaliseHeaders = (input: unknown): Record<string, string> => {
  const out: Record<string, string> = {}
  if (input === undefined || input === null) return out
  if (Array.isArray(input)) {
    for (const [k, v] of input) out[String(k).toLowerCase()] = String(v)
    return out
  }
  if (typeof (input as Headers).forEach === 'function') {
    ;(input as Headers).forEach((v, k) => {
      out[k.toLowerCase()] = v
    })
    return out
  }
  for (const [k, v] of Object.entries(input as Record<string, string>)) {
    out[k.toLowerCase()] = String(v)
  }
  return out
}

/** Shape of a standard OpenAI-compat SSE chunk. */
const chatChunk = (delta: Record<string, unknown>, finishReason: string | null = null): string =>
  JSON.stringify({
    id: 'chatcmpl-test',
    object: 'chat.completion.chunk',
    created: 1714492800,
    model: 'gemma-4-31B-it',
    choices: [
      {
        index: 0,
        delta,
        finish_reason: finishReason,
      },
    ],
  })

/** Build the SSE body Cloud Run / vLLM typically returns: `data: {...}`
 *  frames separated by blank lines, terminated by `data: [DONE]`. */
const sseBodyFromChunks = (chunks: readonly string[]): string => {
  const frames = chunks.map((c) => `data: ${c}\n\n`).join('')
  return `${frames}data: [DONE]\n\n`
}

describe('OpenAIProvider streaming', () => {
  it('parses standard OpenAI-compat SSE and yields text_delta + done', async () => {
    const body = sseBodyFromChunks([
      chatChunk({ role: 'assistant', content: '' }),
      chatChunk({ content: 'Hello' }),
      chatChunk({ content: ' world' }),
      chatChunk({ content: '!' }),
      chatChunk({}, 'stop'),
    ])
    const { http } = makeHttp(() => ({ body }))

    const provider = new OpenAIProvider({
      apiKey: 'test-key',
      model: 'gemma-4-31B-it',
      baseURL: 'https://gemma4-31b-585310675348.europe-west4.run.app',
      http,
    })

    const events: StreamEvent[] = []
    for await (const evt of provider.stream({
      messages: [{ role: 'user', content: 'hi' }],
      maxTokens: 10,
    })) {
      events.push(evt)
    }

    const deltas = events.filter((e) => e.type === 'text_delta')
    expect(deltas.map((e) => (e as { text: string }).text).join('')).toBe('Hello world!')
    const last = events[events.length - 1]
    expect(last?.type).toBe('done')
    if (last?.type === 'done') {
      expect(last.stopReason).toBe('end_turn')
    }
  })

  it('still yields text_delta when the server chunks a single frame across reads', async () => {
    const lines = [
      'data: ',
      `${chatChunk({ content: 'A' })}\n\n`,
      `data: ${chatChunk({ content: 'B' })}\n\n`,
      `data: ${chatChunk({}, 'stop')}\n\n`,
      'data: [DONE]\n\n',
    ]
    const { http } = makeHttp(() => ({
      body: () =>
        new ReadableStream<Uint8Array>({
          start(controller) {
            const enc = new TextEncoder()
            for (const l of lines) controller.enqueue(enc.encode(l))
            controller.close()
          },
        }),
    }))

    const provider = new OpenAIProvider({
      apiKey: 'test-key',
      model: 'gemma-4-31B-it',
      baseURL: 'https://example.run.app',
      http,
    })

    const text: string[] = []
    for await (const evt of provider.stream({
      messages: [{ role: 'user', content: 'hi' }],
      maxTokens: 10,
    })) {
      if (evt.type === 'text_delta') text.push(evt.text)
    }
    expect(text.join('')).toBe('AB')
  })

  it('sends the bearer auth, content-type, and accept headers the gateway expects', async () => {
    const { http, calls } = makeHttp(() => ({
      body: sseBodyFromChunks([chatChunk({}, 'stop')]),
    }))
    const provider = new OpenAIProvider({
      apiKey: 'secret-token',
      model: 'gemma-4-31B-it',
      baseURL: 'https://example.run.app',
      http,
    })
    for await (const _ of provider.stream({
      messages: [{ role: 'user', content: 'hi' }],
      maxTokens: 5,
    })) {
      // drain
    }
    expect(calls).toHaveLength(1)
    const call = calls[0]
    expect(call?.url).toBe('https://example.run.app/v1/chat/completions')
    expect(call?.headers.authorization).toBe('Bearer secret-token')
    expect(call?.contentType).toBe('application/json')
    expect(call?.accept).toBe('text/event-stream')
    expect(call?.body).toMatchObject({
      model: 'gemma-4-31B-it',
      stream: true,
    })
  })

  it('strips a trailing slash from baseURL so the request path is correct', async () => {
    const { http, calls } = makeHttp(() => ({
      body: sseBodyFromChunks([chatChunk({}, 'stop')]),
    }))
    const provider = new OpenAIProvider({
      apiKey: 'k',
      model: 'm',
      baseURL: 'https://example.run.app/',
      http,
    })
    for await (const _ of provider.stream({
      messages: [{ role: 'user', content: 'hi' }],
    })) {
      // drain
    }
    expect(calls[0]?.url).toBe('https://example.run.app/v1/chat/completions')
  })

  it('surfaces an error frame when the gateway returns a non-2xx status', async () => {
    const { http } = makeHttp(() => ({
      status: 503,
      body: JSON.stringify({ error: { message: 'service unavailable' } }),
      headers: { 'content-type': 'application/json' },
    }))

    const provider = new OpenAIProvider({
      apiKey: 'k',
      model: 'm',
      baseURL: 'https://example.run.app',
      http,
    })

    let caught: unknown
    try {
      for await (const _ of provider.stream({
        messages: [{ role: 'user', content: 'hi' }],
      })) {
        // drain
      }
    } catch (err) {
      caught = err
    }
    expect(caught).toBeDefined()
    expect(String(caught)).toContain('503')
  })

  it('ignores blank data frames and the [DONE] sentinel, emits done with usage', async () => {
    const body = [
      'data: \n\n',
      `data: ${chatChunk({ content: 'hi' })}\n\n`,
      `data: ${JSON.stringify({
        id: 'x',
        object: 'chat.completion.chunk',
        created: 0,
        model: 'm',
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
        usage: { prompt_tokens: 3, completion_tokens: 1 },
      })}\n\n`,
      'data: [DONE]\n\n',
    ].join('')

    const { http } = makeHttp(() => ({ body }))
    const provider = new OpenAIProvider({
      apiKey: 'k',
      model: 'm',
      baseURL: 'https://example.run.app',
      http,
    })

    const events: StreamEvent[] = []
    for await (const evt of provider.stream({
      messages: [{ role: 'user', content: 'hi' }],
    })) {
      events.push(evt)
    }

    const done = events[events.length - 1]
    expect(done?.type).toBe('done')
    if (done?.type === 'done') {
      expect(done.stopReason).toBe('end_turn')
      expect(done.usage?.inputTokens).toBe(3)
      expect(done.usage?.outputTokens).toBe(1)
    }
  })
})

describe('OpenAIEmbedder', () => {
  it('posts to the embeddings endpoint and preserves response order', async () => {
    const seen: { url?: string; body?: unknown } = {}
    const http: HttpClient = {
      fetch: async (input, init) => {
        seen.url = typeof input === 'string' ? input : input.toString()
        seen.body = typeof init?.body === 'string' ? JSON.parse(init.body) : null
        return new Response(
          JSON.stringify({
            data: [
              { index: 1, embedding: [4, 5, 6] },
              { index: 0, embedding: [1, 2, 3] },
            ],
          }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' },
          },
        )
      },
    }

    const embedder = new OpenAIEmbedder({
      apiKey: 'sk-test',
      model: 'text-embedding-3-small',
      baseURL: 'https://proxy.example.com',
      http,
    })
    const out = await embedder.embed(['alpha', 'bravo'])

    expect(seen.url).toBe('https://proxy.example.com/v1/embeddings')
    expect(seen.body).toEqual({
      input: ['alpha', 'bravo'],
      model: 'text-embedding-3-small',
    })
    expect(out).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ])
    expect(embedder.dimension()).toBe(3)
  })
})
