// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the in-memory token bucket rate limiter.
 */

import { afterEach, describe, expect, it } from 'vitest'
import type { RateLimiter } from './types.js'
import { createTokenBucket } from './token-bucket.js'

describe('createTokenBucket', () => {
  const limitersTeardown: RateLimiter[] = []

  const makeBucket = (
    overrides: Partial<Parameters<typeof createTokenBucket>[0]> = {},
  ): RateLimiter => {
    const lim = createTokenBucket({
      maxTokens: 10,
      refillRatePerSecond: 100,
      tenantId: 'test',
      ...overrides,
    })
    limitersTeardown.push(lim)
    return lim
  }

  afterEach(async () => {
    for (const lim of limitersTeardown.splice(0)) {
      await lim.close()
    }
  })

  it('acquire succeeds when tokens are available', async () => {
    const lim = makeBucket()
    const tok = await lim.acquire(1)
    expect(tok).toBeDefined()
    tok.release()
  })

  it('acquire resolves after refill when tokens are exhausted', async () => {
    const lim = makeBucket({ maxTokens: 2, refillRatePerSecond: 200 })

    // Drain the bucket.
    const t1 = await lim.acquire(1)
    const t2 = await lim.acquire(1)
    t1.release()
    t2.release()

    // Next acquire should eventually resolve via refill.
    const start = Date.now()
    const t3 = await lim.acquire(1)
    t3.release()
    const elapsed = Date.now() - start
    // With 200 tokens/sec, refill should happen within ~100ms.
    expect(elapsed).toBeLessThan(500)
  })

  it('tryAcquire returns undefined when no tokens available', async () => {
    const lim = makeBucket({ maxTokens: 1, refillRatePerSecond: 0.001 })

    const tok = lim.tryAcquire(1)
    expect(tok).toBeDefined()
    tok!.release()

    // Bucket exhausted.
    const tok2 = lim.tryAcquire(1)
    expect(tok2).toBeUndefined()
  })

  it('release does not add tokens back (consumed model)', async () => {
    const lim = makeBucket({ maxTokens: 1, refillRatePerSecond: 0.001 })

    const tok = await lim.acquire(1)
    tok.release()

    // Even after release, tokens are consumed (not semaphore model).
    const tok2 = lim.tryAcquire(1)
    expect(tok2).toBeUndefined()
  })

  it('release is idempotent', async () => {
    const lim = makeBucket({ maxConcurrency: 1 })

    const tok = await lim.acquire(1)
    tok.release()
    tok.release() // Should not throw or double-free.
  })

  it('respects concurrency limit', async () => {
    const lim = makeBucket({
      maxTokens: 100,
      refillRatePerSecond: 1000,
      maxConcurrency: 2,
    })

    const tok1 = await lim.acquire(1)
    const tok2 = await lim.acquire(1)

    // Third acquire should be blocked by concurrency.
    const tok3 = lim.tryAcquire(1)
    expect(tok3).toBeUndefined()

    // Release one slot -> next should succeed.
    tok1.release()

    const tok4 = lim.tryAcquire(1)
    expect(tok4).toBeDefined()
    tok4!.release()

    tok2.release()
  })

  it('updateFromHeaders triggers back-off when remaining < burst/4', async () => {
    const lim = makeBucket({ maxTokens: 100, refillRatePerSecond: 50 })

    lim.updateFromHeaders({
      remaining: 10,
      limit: 100,
    })

    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBeLessThan(50)
  })

  it('updateFromHeaders pauses on retry-after', async () => {
    const lim = makeBucket({ maxTokens: 10, refillRatePerSecond: 50 })

    lim.updateFromHeaders({
      retryAfter: 0.05, // 50ms
    })

    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBe(0)

    // Wait for resume.
    await new Promise(r => setTimeout(r, 100))
    const m2 = lim.metrics()
    expect(m2.refillRatePerSecond).toBeGreaterThan(0)
  })

  it('updateFromHeaders updates burst from limit header', async () => {
    const lim = makeBucket({ maxTokens: 10 })

    lim.updateFromHeaders({
      remaining: 50,
      limit: 200,
    })

    const m = lim.metrics()
    expect(m.maxTokens).toBe(200)
  })

  it('metrics returns correct snapshot', async () => {
    const lim = makeBucket({ maxTokens: 20, refillRatePerSecond: 10 })

    const m = lim.metrics()
    expect(m.maxTokens).toBe(20)
    expect(m.refillRatePerSecond).toBe(10)
    expect(m.throttledTotal).toBe(0)
    expect(m.waitingRequests).toBe(0)
  })

  it('close rejects pending waiters', async () => {
    const lim = makeBucket({ maxTokens: 1, refillRatePerSecond: 0.001 })

    // Drain.
    const tok = await lim.acquire(1)
    tok.release()

    // Start a pending acquire.
    const pending = lim.acquire(1)

    // Close should reject it.
    await lim.close()

    await expect(pending).rejects.toThrow('rate limiter closed')

    // Remove from teardown since we already closed.
    const idx = limitersTeardown.indexOf(lim)
    if (idx >= 0) limitersTeardown.splice(idx, 1)
  })

  it('acquire rejects after close', async () => {
    const lim = makeBucket()
    await lim.close()

    await expect(lim.acquire(1)).rejects.toThrow('rate limiter is closed')

    const idx = limitersTeardown.indexOf(lim)
    if (idx >= 0) limitersTeardown.splice(idx, 1)
  })

  it('refill rate matches configured rate over time', async () => {
    // 100 tokens/sec, bucket of 5.
    const lim = makeBucket({ maxTokens: 5, refillRatePerSecond: 100 })

    // Drain.
    for (let i = 0; i < 5; i++) {
      const tok = await lim.acquire(1)
      tok.release()
    }

    // Wait 100ms -> should refill ~10 tokens, capped at 5.
    await new Promise(r => setTimeout(r, 100))

    const m = lim.metrics()
    // Should have refilled close to maxTokens.
    expect(m.availableTokens).toBeGreaterThanOrEqual(3)
    expect(m.availableTokens).toBeLessThanOrEqual(5)
  })
})
