// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the adaptive rate limiter wrapper.
 */

import { afterEach, describe, expect, it } from 'vitest'
import type { RateLimiter } from './types.js'
import { createAdaptiveRateLimiter } from './adaptive.js'
import { createTokenBucket } from './token-bucket.js'

describe('createAdaptiveRateLimiter', () => {
  const limitersTeardown: RateLimiter[] = []

  afterEach(async () => {
    for (const lim of limitersTeardown.splice(0)) {
      await lim.close()
    }
  })

  const makeAdaptive = (
    bucketOpts: Partial<Parameters<typeof createTokenBucket>[0]> = {},
    adaptiveOpts: Partial<Parameters<typeof createAdaptiveRateLimiter>[0]> = {},
  ): RateLimiter => {
    const bucket = createTokenBucket({
      maxTokens: 100,
      refillRatePerSecond: 50,
      tenantId: 'test',
      ...bucketOpts,
    })
    const lim = createAdaptiveRateLimiter({
      bucket,
      minRefillRate: 1,
      maxRefillRate: 100,
      recoveryFactor: 1.5,
      ...adaptiveOpts,
    })
    limitersTeardown.push(lim)
    return lim
  }

  it('reduces refill rate when remaining is low', () => {
    const lim = makeAdaptive()

    // remaining=5 < threshold (100/4=25) -> throttle.
    lim.updateFromHeaders({
      remaining: 5,
      limit: 100,
    })

    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBeLessThan(50)
  })

  it('recovers rate when remaining is above threshold', () => {
    const lim = makeAdaptive({ refillRatePerSecond: 10 })

    const before = lim.metrics().refillRatePerSecond

    lim.updateFromHeaders({
      remaining: 80,
      limit: 100,
    })

    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBeGreaterThan(before)
  })

  it('rate never drops below minRefillRate', () => {
    const lim = makeAdaptive(
      { refillRatePerSecond: 2 },
      { minRefillRate: 5 },
    )

    // Repeatedly throttle.
    for (let i = 0; i < 20; i++) {
      lim.updateFromHeaders({ remaining: 1, limit: 100 })
    }

    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBeGreaterThanOrEqual(5)
  })

  it('rate never exceeds maxRefillRate', () => {
    const lim = makeAdaptive(
      { refillRatePerSecond: 80 },
      { maxRefillRate: 100, recoveryFactor: 2.0 },
    )

    // Repeatedly recover.
    for (let i = 0; i < 20; i++) {
      lim.updateFromHeaders({ remaining: 90, limit: 100 })
    }

    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBeLessThanOrEqual(100)
  })

  it('delegates retry-after to the underlying bucket', () => {
    const bucket = createTokenBucket({
      maxTokens: 10,
      refillRatePerSecond: 50,
      tenantId: 'test',
    })
    const lim = createAdaptiveRateLimiter({
      bucket,
      minRefillRate: 1,
      maxRefillRate: 100,
    })
    limitersTeardown.push(lim)

    lim.updateFromHeaders({ retryAfter: 0.05 })

    const bm = bucket.metrics()
    expect(bm.refillRatePerSecond).toBe(0)
  })

  it('acquire delegates to underlying bucket', async () => {
    const lim = makeAdaptive()
    const tok = await lim.acquire(1)
    expect(tok).toBeDefined()
    tok.release()
  })

  it('tryAcquire delegates to underlying bucket', () => {
    const lim = makeAdaptive()
    const tok = lim.tryAcquire(1)
    expect(tok).toBeDefined()
    tok!.release()
  })

  it('uses default options when not specified', () => {
    const bucket = createTokenBucket({
      maxTokens: 10,
      refillRatePerSecond: 10,
      tenantId: 'test',
    })
    const lim = createAdaptiveRateLimiter({ bucket })
    limitersTeardown.push(lim)

    // Should not throw.
    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBe(10)
  })
})
