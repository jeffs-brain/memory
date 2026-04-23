import { describe, expect, it, vi } from 'vitest'

import { createHttpStore } from './http.js'
import { ErrPayloadTooLarge, toPath } from './index.js'

type RecordedRequest = {
  readonly method: string
  readonly url: string
  readonly body: string
}

const readBody = async (body: BodyInit | null | undefined): Promise<string> => {
  if (body === undefined || body === null) return ''
  if (typeof body === 'string') return body
  if (body instanceof Blob) return await body.text()
  return String(body)
}

describe('HttpStore', () => {
  it('rejects oversized batches before sending them', async () => {
    const fetch = vi.fn(async () => new Response(null, { status: 200 }))
    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
      bodyLimits: {
        batchDecodedBytes: 4,
        batchOpCount: 2,
      },
    })

    await expect(
      store.batch({ reason: 'test' }, async (batch) => {
        await batch.write(toPath('docs/one.txt'), '12345')
      }),
    ).rejects.toBeInstanceOf(ErrPayloadTooLarge)
    expect(fetch).not.toHaveBeenCalled()
  })

  it('honours configured higher batch ceilings client-side', async () => {
    const fetch = vi.fn(async () => new Response(null, { status: 200 }))
    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
      bodyLimits: {
        batchDecodedBytes: 16,
      },
    })

    await store.batch({ reason: 'test' }, async (batch) => {
      await batch.write(toPath('docs/one.txt'), '1234567890')
    })

    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('maps 413 responses to ErrPayloadTooLarge', async () => {
    const fetch = vi.fn(
      async () =>
        new Response(
          JSON.stringify({
            status: 413,
            title: 'Payload Too Large',
            code: 'payload_too_large',
            detail: 'batch payload exceeds 8 bytes after decode',
          }),
          {
            status: 413,
            headers: { 'content-type': 'application/problem+json' },
          },
        ),
    )

    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
    })

    await expect(store.write(toPath('docs/one.txt'), 'ok')).rejects.toBeInstanceOf(
      ErrPayloadTooLarge,
    )
  })

  it('reads, lists, and replays batch changes before submitting them', async () => {
    const requests: RecordedRequest[] = []
    const fetch = vi.fn(async (input: string, init?: RequestInit) => {
      requests.push({
        method: init?.method ?? 'GET',
        url: input,
        body: await readBody(init?.body),
      })
      const url = new URL(input)
      if (url.pathname.endsWith('/documents/read')) {
        return new Response('hello', { status: 200 })
      }
      if (url.pathname.endsWith('/documents') && (init?.method ?? 'GET') === 'GET') {
        return new Response(
          JSON.stringify({
            items: [
              {
                path: 'docs/existing.txt',
                size: 5,
                mtime: '2026-04-23T00:00:00.000Z',
                is_dir: false,
              },
            ],
          }),
          {
            status: 200,
            headers: { 'content-type': 'application/json' },
          },
        )
      }
      if (url.pathname.endsWith('/documents/batch-ops')) {
        return new Response(null, { status: 200 })
      }
      throw new Error(`unexpected request: ${init?.method ?? 'GET'} ${input}`)
    })

    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
    })

    await expect(store.read(toPath('docs/existing.txt'))).resolves.toBe('hello')
    await expect(store.list(toPath('docs'), { recursive: true })).resolves.toMatchObject([
      { path: 'docs/existing.txt', size: 5 },
    ])

    await store.batch({ reason: 'test' }, async (batch) => {
      await batch.write(toPath('docs/new.txt'), 'hi')
      await expect(batch.read(toPath('docs/new.txt'))).resolves.toBe('hi')
      await expect(batch.list(toPath('docs'), { recursive: true })).resolves.toMatchObject([
        { path: 'docs/existing.txt' },
        { path: 'docs/new.txt' },
      ])
    })

    const batchRequest = requests.find((request) => request.url.includes('/documents/batch-ops'))
    expect(batchRequest).toBeDefined()
    expect(JSON.parse(batchRequest?.body ?? '')).toEqual({
      reason: 'test',
      ops: [{ type: 'write', path: 'docs/new.txt', content_base64: 'aGk=' }],
    })
  })

  it('subscribes to daemon change events over SSE', async () => {
    const fetch = vi.fn(async (input: string, init?: RequestInit) => {
      const url = new URL(input)
      if (url.pathname.endsWith('/events')) {
        return new Response(
          new Blob([
            'event: change\n',
            'data: {"kind":"updated","path":"docs/one.txt","when":"2026-04-23T00:00:00.000Z"}\n\n',
          ]).stream(),
          {
            status: 200,
            headers: { 'content-type': 'text/event-stream' },
          },
        )
      }
      throw new Error(`unexpected request: ${init?.method ?? 'GET'} ${input}`)
    })

    const store = createHttpStore({
      baseUrl: 'https://example.test',
      brainId: 'brain-a',
      fetch,
    })

    const events: Array<{ readonly kind: string; readonly path: string }> = []
    const unsubscribe = store.subscribe((event) => {
      events.push({ kind: event.kind, path: event.path })
    })

    await new Promise((resolve) => setTimeout(resolve, 10))
    unsubscribe()

    expect(events).toEqual([{ kind: 'updated', path: 'docs/one.txt' }])
  })
})
