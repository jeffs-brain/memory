// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type { HttpClient } from './http.js'
import { TEIReranker } from './tei.js'

describe('TEIReranker availability', () => {
  it('probes /health once and memoises the result inside the ttl window', async () => {
    const fetch = vi.fn(async () => new Response('ok', { status: 200 }))
    const http: HttpClient = { fetch }
    const reranker = new TEIReranker({
      baseURL: 'http://tei.example.com',
      http,
      probeTtlMs: 60_000,
    })

    await expect(reranker.isAvailable?.()).resolves.toBe(true)
    await expect(reranker.isAvailable?.()).resolves.toBe(true)

    expect(fetch).toHaveBeenCalledTimes(1)
    expect(fetch).toHaveBeenCalledWith(
      'http://tei.example.com/health',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('falls back to /info when /health is unavailable', async () => {
    const fetch = vi.fn(async (input: string | URL | Request) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.toString() : input.url
      if (url.endsWith('/health')) {
        return new Response('missing', { status: 404 })
      }
      return new Response('{"ok":true}', { status: 200 })
    })
    const http: HttpClient = { fetch }
    const reranker = new TEIReranker({
      baseURL: 'http://tei.example.com',
      http,
      probeTtlMs: 0,
    })

    await expect(reranker.isAvailable?.()).resolves.toBe(true)

    expect(fetch).toHaveBeenNthCalledWith(
      1,
      'http://tei.example.com/health',
      expect.objectContaining({ method: 'GET' }),
    )
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      'http://tei.example.com/info',
      expect.objectContaining({ method: 'GET' }),
    )
  })
})
