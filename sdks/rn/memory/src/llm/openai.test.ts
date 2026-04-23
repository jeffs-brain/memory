import { describe, expect, it } from 'vitest'

import type { HttpClient } from './http.js'
import { OpenAIEmbedder, OpenAIProvider } from './openai.js'
import type { StreamEvent } from './types.js'

type RecordedCall = {
  readonly url: string
  readonly headers: Record<string, string>
  readonly body: unknown
  readonly accept: string
  readonly contentType: string
}

type FakeServer = {
  readonly body: string | (() => ReadableStream<Uint8Array>)
  readonly status?: number
  readonly headers?: Record<string, string>
}

const normaliseHeaders = (input: unknown): Record<string, string> => {
  const out: Record<string, string> = {}
  if (input === undefined || input === null) return out
  if (Array.isArray(input)) {
    for (const [key, value] of input) out[String(key).toLowerCase()] = String(value)
    return out
  }
  if (typeof (input as Headers).forEach === 'function') {
    ;(input as Headers).forEach((value, key) => {
      out[key.toLowerCase()] = value
    })
    return out
  }
  for (const [key, value] of Object.entries(input as Record<string, string>)) {
    out[key.toLowerCase()] = String(value)
  }
  return out
}

const makeHttp = (
  implementation: (url: string, call: RecordedCall) => FakeServer,
): {
  readonly http: HttpClient
  readonly calls: RecordedCall[]
} => {
  const calls: RecordedCall[] = []
  const http: HttpClient = {
    fetch: async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString()
      const headers = normaliseHeaders(init?.headers)
      const bodyText = typeof init?.body === 'string' ? init.body : ''
      const body = bodyText === '' ? null : JSON.parse(bodyText)
      const call: RecordedCall = {
        url,
        headers,
        body,
        accept: headers.accept ?? '',
        contentType: headers['content-type'] ?? '',
      }
      calls.push(call)
      const response = implementation(url, call)
      const responseBody =
        typeof response.body === 'string' ? new Blob([response.body]).stream() : response.body()
      return new Response(responseBody, {
        status: response.status ?? 200,
        headers: {
          'content-type': 'text/event-stream',
          ...(response.headers ?? {}),
        },
      })
    },
  }
  return { http, calls }
}

const chatChunk = (delta: Record<string, unknown>, finishReason: string | null = null): string =>
  JSON.stringify({
    id: 'chatcmpl-test',
    object: 'chat.completion.chunk',
    created: 1714492800,
    model: 'gemma-4-31B-it',
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })

const sseBodyFromChunks = (chunks: readonly string[]): string =>
  `${chunks.map((chunk) => `data: ${chunk}\n\n`).join('')}data: [DONE]\n\n`

const chatCompletionBody = (
  args: {
    readonly content?: string
    readonly finishReason?: string
    readonly toolCalls?: ReadonlyArray<{
      readonly id: string
      readonly name: string
      readonly arguments: string
    }>
  } = {},
): string =>
  JSON.stringify({
    id: 'chatcmpl-test',
    object: 'chat.completion',
    created: 1714492800,
    model: 'gemma-4-31B-it',
    choices: [
      {
        index: 0,
        message: {
          content: args.content ?? '',
          ...(args.toolCalls === undefined
            ? {}
            : {
                tool_calls: args.toolCalls.map((toolCall) => ({
                  id: toolCall.id,
                  type: 'function',
                  function: {
                    name: toolCall.name,
                    arguments: toolCall.arguments,
                  },
                })),
              }),
        },
        finish_reason: args.finishReason ?? 'stop',
      },
    ],
    usage: {
      prompt_tokens: 2,
      completion_tokens: 3,
    },
  })

describe('OpenAIProvider', () => {
  it('parses OpenAI-compatible SSE streams and sends bearer auth', async () => {
    const { http, calls } = makeHttp(() => ({
      body: sseBodyFromChunks([
        chatChunk({ role: 'assistant', content: '' }),
        chatChunk({ content: 'Hello' }),
        chatChunk({ content: ' world' }),
        chatChunk({}, 'stop'),
      ]),
    }))

    const provider = new OpenAIProvider({
      apiKey: 'secret-token',
      model: 'gemma-4-31B-it',
      baseURL: 'https://example.run.app/',
      http,
    })

    const events: StreamEvent[] = []
    for await (const event of provider.stream({
      messages: [{ role: 'user', content: 'hi' }],
      maxTokens: 10,
    })) {
      events.push(event)
    }

    expect(
      events
        .filter((event) => event.type === 'text_delta')
        .map((event) => event.text)
        .join(''),
    ).toBe('Hello world')
    expect(calls[0]?.url).toBe('https://example.run.app/v1/chat/completions')
    expect(calls[0]?.headers.authorization).toBe('Bearer secret-token')
    expect(calls[0]?.accept).toBe('text/event-stream')
    expect(calls[0]?.contentType).toBe('application/json')
  })

  it('posts embeddings to the OpenAI-compatible endpoint and updates dimension', async () => {
    const { http, calls } = makeHttp((url) => {
      if (url.endsWith('/v1/embeddings')) {
        return {
          body: JSON.stringify({
            data: [
              { index: 0, embedding: [0.1, 0.2, 0.3] },
              { index: 1, embedding: [0.4, 0.5, 0.6] },
            ],
          }),
          headers: { 'content-type': 'application/json' },
        }
      }
      throw new Error(`unexpected url: ${url}`)
    })

    const embedder = new OpenAIEmbedder({
      apiKey: 'secret-token',
      model: 'text-embedding-3-small',
      baseURL: 'https://example.run.app',
      http,
    })

    const vectors = await embedder.embed(['first', 'second'])
    expect(vectors).toEqual([
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ])
    expect(embedder.dimension()).toBe(3)
    expect(calls[0]?.url).toBe('https://example.run.app/v1/embeddings')
    expect(calls[0]?.headers.authorization).toBe('Bearer secret-token')
  })

  it('parses tool calls from complete responses', async () => {
    const { http } = makeHttp(() => ({
      body: chatCompletionBody({
        finishReason: 'tool_calls',
        toolCalls: [{ id: 'tool-1', name: 'lookup', arguments: '{"id":1}' }],
      }),
      headers: { 'content-type': 'application/json' },
    }))

    const provider = new OpenAIProvider({
      apiKey: 'secret-token',
      model: 'gemma-4-31B-it',
      baseURL: 'https://example.run.app',
      http,
    })

    const response = await provider.complete({
      messages: [{ role: 'user', content: 'look this up' }],
    })

    expect(response.toolCalls).toEqual([{ id: 'tool-1', name: 'lookup', arguments: '{"id":1}' }])
    expect(response.stopReason).toBe('tool_use')
    expect(response.usage).toEqual({ inputTokens: 2, outputTokens: 3 })
  })

  it('retries structured responses when the first payload fails schema validation', async () => {
    let callCount = 0
    const { http, calls } = makeHttp(() => {
      callCount += 1
      return {
        body:
          callCount === 1
            ? chatCompletionBody({ content: 'not valid json' })
            : chatCompletionBody({ content: '{"answer":"valid"}' }),
        headers: { 'content-type': 'application/json' },
      }
    })

    const provider = new OpenAIProvider({
      apiKey: 'secret-token',
      model: 'gemma-4-31B-it',
      baseURL: 'https://example.run.app',
      http,
    })

    const response = await provider.structured({
      messages: [{ role: 'user', content: 'answer in JSON' }],
      schema: JSON.stringify({
        type: 'object',
        properties: { answer: { type: 'string' } },
        required: ['answer'],
      }),
      schemaName: 'answer_payload',
      maxRetries: 2,
    })

    expect(response).toBe('{"answer":"valid"}')
    expect(calls).toHaveLength(2)
    expect(((calls[1]?.body as { messages?: unknown[] }).messages ?? []).length).toBeGreaterThan(
      ((calls[0]?.body as { messages?: unknown[] }).messages ?? []).length,
    )
  })

  it('preserves sparse embedding indexes and derives dimension from the first populated vector', async () => {
    const { http } = makeHttp((url) => {
      if (!url.endsWith('/v1/embeddings')) {
        throw new Error(`unexpected url: ${url}`)
      }
      return {
        body: JSON.stringify({
          data: [
            { index: 2, embedding: [0.7, 0.8] },
            { index: 0, embedding: [0.1, 0.2] },
          ],
        }),
        headers: { 'content-type': 'application/json' },
      }
    })

    const embedder = new OpenAIEmbedder({
      apiKey: 'secret-token',
      model: 'text-embedding-3-small',
      baseURL: 'https://example.run.app',
      http,
      dimensions: 8,
    })

    const vectors = await embedder.embed(['first', 'second', 'third'])

    expect(vectors).toEqual([[0.1, 0.2], [], [0.7, 0.8]])
    expect(embedder.dimension()).toBe(2)
  })
})
