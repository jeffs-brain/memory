import { describe, expect, it } from 'vitest'

import type { HttpClient } from './http.js'
import { OllamaEmbedder } from './ollama.js'

type RecordedCall = {
  readonly url: string
  readonly body: unknown
}

const makeHttp = (
  implementation: (url: string, body: unknown) => { status: number; body: string },
): {
  readonly http: HttpClient
  readonly calls: RecordedCall[]
} => {
  const calls: RecordedCall[] = []
  const http: HttpClient = {
    fetch: async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString()
      const bodyText = typeof init?.body === 'string' ? init.body : ''
      const body = bodyText === '' ? null : JSON.parse(bodyText)
      calls.push({ url, body })
      const response = implementation(url, body)
      return new Response(response.body, {
        status: response.status,
        headers: { 'content-type': 'application/json' },
      })
    },
  }
  return { http, calls }
}

describe('OllamaEmbedder', () => {
  it('falls back from /api/embed to /api/embeddings on 404', async () => {
    const { http, calls } = makeHttp((url) => {
      if (url.endsWith('/api/embed')) {
        return { status: 404, body: JSON.stringify({ error: 'not found' }) }
      }
      if (url.endsWith('/api/embeddings')) {
        return {
          status: 200,
          body: JSON.stringify({ embedding: [0.1, 0.2, 0.3] }),
        }
      }
      return { status: 500, body: 'unexpected' }
    })

    const embedder = new OllamaEmbedder({ model: 'bge-m3', cacheSize: 0, http })
    const vectors = await embedder.embed(['hello'])

    expect(vectors).toEqual([[0.1, 0.2, 0.3]])
    expect(embedder.dimension()).toBe(3)
    expect(calls.map((call) => new URL(call.url).pathname)).toEqual([
      '/api/embed',
      '/api/embeddings',
    ])
  })

  it('uses /api/embed when supported', async () => {
    const { http, calls } = makeHttp((url) => {
      if (url.endsWith('/api/embed')) {
        return {
          status: 200,
          body: JSON.stringify({ embeddings: [[1, 2, 3]] }),
        }
      }
      return { status: 500, body: 'should not be called' }
    })

    const embedder = new OllamaEmbedder({ model: 'bge-m3', cacheSize: 0, http })
    const vectors = await embedder.embed(['hello'])

    expect(vectors).toEqual([[1, 2, 3]])
    expect(calls).toHaveLength(1)
  })

  it('deduplicates concurrent requests for the same text', async () => {
    let hits = 0
    const { http } = makeHttp((url) => {
      hits += 1
      if (url.endsWith('/api/embed')) {
        return {
          status: 200,
          body: JSON.stringify({ embeddings: [[9, 8, 7]] }),
        }
      }
      return { status: 500, body: 'unexpected' }
    })

    const embedder = new OllamaEmbedder({ model: 'bge-m3', cacheSize: 0, http })
    const [left, middle, right] = await Promise.all([
      embedder.embed(['same']),
      embedder.embed(['same']),
      embedder.embed(['same']),
    ])

    expect(left).toEqual([[9, 8, 7]])
    expect(middle).toEqual([[9, 8, 7]])
    expect(right).toEqual([[9, 8, 7]])
    expect(hits).toBe(1)
  })

  it('serves repeated requests from the LRU cache', async () => {
    let hits = 0
    const { http } = makeHttp((url) => {
      hits += 1
      if (url.endsWith('/api/embed')) {
        return {
          status: 200,
          body: JSON.stringify({ embeddings: [[0.5, 0.5]] }),
        }
      }
      return { status: 500, body: 'unexpected' }
    })

    const embedder = new OllamaEmbedder({ model: 'bge-m3', cacheSize: 8, http })
    await embedder.embed(['cached'])
    await embedder.embed(['cached'])
    await embedder.embed(['cached'])

    expect(hits).toBe(1)
  })
})
