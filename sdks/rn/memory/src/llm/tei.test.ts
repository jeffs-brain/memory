import { describe, expect, it } from 'vitest'

import type { HttpClient } from './http.js'
import { TEIEmbedder, TEIReranker } from './tei.js'

describe('TEIEmbedder', () => {
  it('deduplicates repeated texts within a batch and updates dimension', async () => {
    let hits = 0
    const http: HttpClient = {
      fetch: async () => {
        hits += 1
        return new Response(JSON.stringify([[0.1, 0.2, 0.3]]), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        })
      },
    }

    const embedder = new TEIEmbedder({
      baseURL: 'http://tei.example.com',
      http,
      cacheSize: 8,
    })

    const vectors = await embedder.embed(['same', 'same'])

    expect(vectors).toEqual([
      [0.1, 0.2, 0.3],
      [0.1, 0.2, 0.3],
    ])
    expect(embedder.dimension()).toBe(3)
    expect(hits).toBe(1)
  })
})

describe('TEIReranker availability', () => {
  it('probes /health once and memoises the result inside the ttl window', async () => {
    const fetch = async () => new Response('ok', { status: 200 })
    const http: HttpClient = { fetch }
    const reranker = new TEIReranker({
      baseURL: 'http://tei.example.com',
      http,
      probeTtlMs: 60_000,
    })

    await expect(reranker.isAvailable?.()).resolves.toBe(true)
    await expect(reranker.isAvailable?.()).resolves.toBe(true)
  })

  it('falls back to /info when /health is unavailable', async () => {
    const calls: string[] = []
    const http: HttpClient = {
      fetch: async (input) => {
        const url =
          typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
        calls.push(url)
        if (url.endsWith('/health')) {
          return new Response('missing', { status: 404 })
        }
        return new Response('{"ok":true}', { status: 200 })
      },
    }

    const reranker = new TEIReranker({
      baseURL: 'http://tei.example.com',
      http,
      probeTtlMs: 0,
    })

    await expect(reranker.isAvailable?.()).resolves.toBe(true)
    expect(calls).toEqual(['http://tei.example.com/health', 'http://tei.example.com/info'])
  })
})
