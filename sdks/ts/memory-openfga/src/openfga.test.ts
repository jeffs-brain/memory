import { describe, expect, it, vi } from 'vitest'
import type { Resource, Subject } from '@jeffs-brain/memory/acl'
import { createOpenFgaProvider, OpenFgaHttpError, type FetchLike } from './openfga.js'

const user = (id: string): Subject => ({ kind: 'user', id })
const brain = (id: string): Resource => ({ type: 'brain', id })

const jsonResponse = (body: unknown, init: ResponseInit = {}): Response =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
    ...init,
  })

describe('openfga adapter - check', () => {
  it('sends a correctly shaped check request', async () => {
    const fetchMock = vi.fn<FetchLike>().mockResolvedValue(jsonResponse({ allowed: true }))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      modelId: 'model-42',
      token: 'secret',
      fetch: fetchMock,
    })

    const result = await acl.check(user('alice'), 'write', brain('notes'))
    expect(result.allowed).toBe(true)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(url).toBe('https://fga.example.com/stores/store-1/check')
    expect(init?.method).toBe('POST')
    const headers = init?.headers as Record<string, string> | undefined
    expect(headers?.['authorization']).toBe('Bearer secret')
    expect(headers?.['content-type']).toBe('application/json')
    const body = JSON.parse(String(init?.body ?? '{}')) as {
      tuple_key: { user: string; relation: string; object: string }
      authorization_model_id?: string
    }
    expect(body.tuple_key).toEqual({
      user: 'user:alice',
      relation: 'writer',
      object: 'brain:notes',
    })
    expect(body.authorization_model_id).toBe('model-42')
  })

  it('maps the allowed:false response to deny with a reason', async () => {
    const fetchMock = vi
      .fn<FetchLike>()
      .mockResolvedValue(jsonResponse({ allowed: false, resolution: 'no tuple' }))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com/',
      storeId: 'store-1',
      fetch: fetchMock,
    })
    const result = await acl.check(user('alice'), 'read', brain('notes'))
    expect(result.allowed).toBe(false)
    expect(result.reason).toBe('no tuple')
  })

  it('throws OpenFgaHttpError on non-2xx', async () => {
    const fetchMock = vi
      .fn<FetchLike>()
      .mockResolvedValue(new Response('boom', { status: 500 }))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      fetch: fetchMock,
    })
    await expect(acl.check(user('alice'), 'read', brain('notes'))).rejects.toBeInstanceOf(
      OpenFgaHttpError,
    )
  })

  it('omits the Authorization header when no token is configured', async () => {
    const fetchMock = vi.fn<FetchLike>().mockResolvedValue(jsonResponse({ allowed: true }))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      fetch: fetchMock,
    })
    await acl.check(user('alice'), 'read', brain('notes'))
    const headers = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string> | undefined
    expect(headers?.['authorization']).toBeUndefined()
  })
})

describe('openfga adapter - write', () => {
  it('sends the correct write body', async () => {
    const fetchMock = vi.fn<FetchLike>().mockResolvedValue(jsonResponse({}))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      modelId: 'model-42',
      fetch: fetchMock,
    })
    await acl.write?.({
      writes: [{ subject: user('alice'), relation: 'writer', resource: brain('notes') }],
      deletes: [{ subject: user('bob'), relation: 'reader', resource: brain('notes') }],
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0] ?? []
    expect(url).toBe('https://fga.example.com/stores/store-1/write')
    const body = JSON.parse(String(init?.body ?? '{}')) as {
      writes?: { tuple_keys: unknown[] }
      deletes?: { tuple_keys: unknown[] }
      authorization_model_id?: string
    }
    expect(body.writes?.tuple_keys).toEqual([
      { user: 'user:alice', relation: 'writer', object: 'brain:notes' },
    ])
    expect(body.deletes?.tuple_keys).toEqual([
      { user: 'user:bob', relation: 'reader', object: 'brain:notes' },
    ])
    expect(body.authorization_model_id).toBe('model-42')
  })

  it('does not call fetch when the request is empty', async () => {
    const fetchMock = vi.fn<FetchLike>().mockResolvedValue(jsonResponse({}))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      fetch: fetchMock,
    })
    await acl.write?.({})
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('wraps fetch rejections in OpenFgaRequestError', async () => {
    const fetchMock = vi.fn<FetchLike>().mockRejectedValue(new Error('network down'))
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      fetch: fetchMock,
    })
    await expect(
      acl.write?.({
        writes: [{ subject: user('a'), relation: 'reader', resource: brain('x') }],
      }),
    ).rejects.toThrow(/network down/)
  })
})

describe('openfga adapter - read', () => {
  it('decodes tuples from the response', async () => {
    const fetchMock = vi.fn<FetchLike>().mockResolvedValue(
      jsonResponse({
        tuples: [
          {
            key: { user: 'user:alice', relation: 'writer', object: 'brain:notes' },
          },
          { key: { user: 'junk', relation: 'writer', object: 'brain:notes' } },
        ],
      }),
    )
    const acl = createOpenFgaProvider({
      apiUrl: 'https://fga.example.com',
      storeId: 'store-1',
      fetch: fetchMock,
    })
    const tuples = await acl.read?.({ resource: brain('notes') })
    expect(tuples?.length).toBe(1)
    expect(tuples?.[0]?.subject).toEqual(user('alice'))
    expect(tuples?.[0]?.relation).toBe('writer')
    expect(tuples?.[0]?.resource).toEqual(brain('notes'))
  })
})
