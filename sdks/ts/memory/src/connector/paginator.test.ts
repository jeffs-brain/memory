// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { collectPages, paginate, MaxPagesExceededError } from './paginator.js'
import { createRateLimiter } from './rate-limiter.js'
import type { PageResult } from './types.js'

describe('paginate', () => {
  it('iterates a single page', async () => {
    const items = await collectPages({
      fetchPage: async () => ({ items: ['a', 'b', 'c'] }),
    })
    expect(items).toEqual(['a', 'b', 'c'])
  })

  it('iterates multiple pages via cursor', async () => {
    let page = 0
    const items = await collectPages({
      fetchPage: async (cursor) => {
        const current = page++
        const pages: Record<number, PageResult<string>> = {
          0: { items: ['a', 'b'], nextCursor: 'p2' },
          1: { items: ['c', 'd'], nextCursor: 'p3' },
          2: { items: ['e'] },
        }
        return pages[current] ?? { items: [] }
      },
    })
    expect(items).toEqual(['a', 'b', 'c', 'd', 'e'])
  })

  it('stops at maxPages guard', async () => {
    await expect(
      collectPages({
        fetchPage: async () => ({ items: ['x'], nextCursor: 'always-more' }),
        maxPages: 3,
      }),
    ).rejects.toThrow(MaxPagesExceededError)
  })

  it('handles empty first page', async () => {
    const items = await collectPages({
      fetchPage: async () => ({ items: [] }),
    })
    expect(items).toEqual([])
  })

  it('propagates fetch errors', async () => {
    const sentinel = new Error('api error')
    await expect(
      collectPages({
        fetchPage: async () => {
          throw sentinel
        },
      }),
    ).rejects.toThrow(sentinel)
  })

  it('integrates with rate limiter', async () => {
    const rl = createRateLimiter({ maxTokens: 100, refillRate: 100 })
    let page = 0
    const items = await collectPages({
      fetchPage: async () => {
        const current = page++
        return current < 2
          ? { items: ['item'], nextCursor: 'next' }
          : { items: ['last'] }
      },
      rateLimiter: rl,
    })
    expect(items).toHaveLength(3)
    rl.close()
  })

  it('aborts via AbortSignal', async () => {
    const controller = new AbortController()
    let page = 0

    const gen = paginate({
      fetchPage: async () => {
        const current = page++
        if (current === 1) controller.abort()
        return { items: [`page-${current}`], nextCursor: 'more' }
      },
      signal: controller.signal,
    })

    const items: string[] = []
    await expect(
      (async () => {
        for await (const item of gen) {
          items.push(item)
        }
      })(),
    ).rejects.toThrow()

    // Should have collected at most the first two pages before abort kicks in.
    expect(items.length).toBeLessThanOrEqual(2)
  })
})
