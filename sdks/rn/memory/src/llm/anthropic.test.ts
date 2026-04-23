import { describe, expect, it } from 'vitest'

import { AnthropicProvider } from './anthropic.js'
import type { HttpClient } from './http.js'

type RecordedCall = {
  readonly url: string
  readonly headers: Record<string, string>
  readonly body: unknown
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
    for (const [key, value] of input) {
      out[String(key).toLowerCase()] = String(value)
    }
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
      const call: RecordedCall = { url, headers, body }
      calls.push(call)
      const response = implementation(url, call)
      const responseBody =
        typeof response.body === 'string' ? new Blob([response.body]).stream() : response.body()
      return new Response(responseBody, {
        status: response.status ?? 200,
        headers: {
          'content-type': 'application/json',
          ...(response.headers ?? {}),
        },
      })
    },
  }
  return { http, calls }
}

describe('AnthropicProvider', () => {
  it('parses content and tool use blocks from complete responses', async () => {
    const { http, calls } = makeHttp(() => ({
      body: JSON.stringify({
        content: [
          { type: 'text', text: 'Hello' },
          { type: 'tool_use', id: 'tool-1', name: 'lookup', input: { id: 1 } },
        ],
        stop_reason: 'tool_use',
        usage: {
          input_tokens: 2,
          output_tokens: 3,
          cache_read_input_tokens: 4,
          cache_creation_input_tokens: 5,
        },
      }),
    }))

    const provider = new AnthropicProvider({
      apiKey: 'secret-key',
      model: 'claude-3-7-sonnet',
      baseURL: 'https://example.run.app/',
      http,
    })

    const response = await provider.complete({
      messages: [{ role: 'user', content: 'hello' }],
    })

    expect(response.content).toBe('Hello')
    expect(response.toolCalls).toEqual([{ id: 'tool-1', name: 'lookup', arguments: '{"id":1}' }])
    expect(response.stopReason).toBe('tool_use')
    expect(response.usage).toEqual({
      inputTokens: 2,
      outputTokens: 3,
      cacheReadTokens: 4,
      cacheCreateTokens: 5,
    })
    expect(calls[0]?.url).toBe('https://example.run.app/v1/messages')
    expect(calls[0]?.headers['x-api-key']).toBe('secret-key')
    expect(calls[0]?.headers['anthropic-version']).toBe('2023-06-01')
  })

  it('retries structured responses until emit_structured validates', async () => {
    let callCount = 0
    const { http, calls } = makeHttp(() => {
      callCount += 1
      return {
        body: JSON.stringify(
          callCount === 1
            ? {
                content: [{ type: 'text', text: 'not valid json' }],
                stop_reason: 'end_turn',
                usage: { input_tokens: 1, output_tokens: 1 },
              }
            : {
                content: [
                  {
                    type: 'tool_use',
                    id: 'tool-1',
                    name: 'emit_structured',
                    input: { answer: 'valid' },
                  },
                ],
                stop_reason: 'tool_use',
                usage: { input_tokens: 1, output_tokens: 1 },
              },
        ),
      }
    })

    const provider = new AnthropicProvider({
      apiKey: 'secret-key',
      model: 'claude-3-7-sonnet',
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
    expect((calls[0]?.body as { tool_choice?: { name?: string } }).tool_choice?.name).toBe(
      'emit_structured',
    )
    expect(((calls[1]?.body as { messages?: unknown[] }).messages ?? []).length).toBeGreaterThan(
      ((calls[0]?.body as { messages?: unknown[] }).messages ?? []).length,
    )
  })
})
