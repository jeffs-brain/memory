// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { OllamaEmbedder } from './ollama.js'
import type { HttpClient } from './http.js'

type RecordedCall = { url: string; body: unknown }

function makeHttp(
  impl: (url: string, body: unknown) => { status: number; body: string },
): { http: HttpClient; calls: RecordedCall[] } {
  const calls: RecordedCall[] = []
  const http: HttpClient = {
    fetch: async (input, init) => {
      const url = typeof input === 'string' ? input : input.toString()
      const bodyStr = typeof init?.body === 'string' ? init.body : ''
      const body = bodyStr === '' ? null : JSON.parse(bodyStr)
      calls.push({ url, body })
      const { status, body: respBody } = impl(url, body)
      return new Response(respBody, {
        status,
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
    const out = await embedder.embed(['hello'])
    expect(out).toEqual([[0.1, 0.2, 0.3]])
    expect(embedder.dimension()).toBe(3)
    const paths = calls.map((c) => new URL(c.url).pathname)
    expect(paths).toEqual(['/api/embed', '/api/embeddings'])
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
    const out = await embedder.embed(['hello'])
    expect(out).toEqual([[1, 2, 3]])
    expect(calls).toHaveLength(1)
  })

  it('deduplicates concurrent requests for the same text (single-flight)', async () => {
    let hits = 0
    const { http } = makeHttp((url) => {
      hits++
      if (url.endsWith('/api/embed')) {
        return {
          status: 200,
          body: JSON.stringify({ embeddings: [[9, 8, 7]] }),
        }
      }
      return { status: 500, body: 'unexpected' }
    })
    const embedder = new OllamaEmbedder({ model: 'bge-m3', cacheSize: 0, http })
    // Fire three concurrent embed calls for the same text.
    const [a, b, c] = await Promise.all([
      embedder.embed(['same']),
      embedder.embed(['same']),
      embedder.embed(['same']),
    ])
    expect(a).toEqual([[9, 8, 7]])
    expect(b).toEqual([[9, 8, 7]])
    expect(c).toEqual([[9, 8, 7]])
    expect(hits).toBe(1)
  })

  it('serves subsequent identical requests from the LRU cache', async () => {
    let hits = 0
    const { http } = makeHttp((url) => {
      hits++
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
