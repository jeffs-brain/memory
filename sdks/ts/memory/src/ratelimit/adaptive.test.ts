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
  ): { adaptive: RateLimiter; bucket: RateLimiter } => {
    const bucket = createTokenBucket({
      maxTokens: 100,
      refillRatePerSecond: 50,
      tenantId: 'test',
      ...bucketOpts,
    })
    const adaptive = createAdaptiveRateLimiter({
      bucket,
      minRefillRate: 1,
      maxRefillRate: 100,
      recoveryFactor: 1.5,
      ...adaptiveOpts,
    })
    limitersTeardown.push(adaptive)
    return { adaptive, bucket }
  }

  it('reduces refill rate when remaining is low', () => {
    const { adaptive } = makeAdaptive()

    // remaining=5 < threshold (100/4=25) -> throttle.
    adaptive.updateFromHeaders({
      remaining: 5,
      limit: 100,
    })

    const m = adaptive.metrics()
    expect(m.refillRatePerSecond).toBeLessThan(50)
  })

  it('recovers rate when remaining is above threshold', () => {
    const { adaptive } = makeAdaptive({ refillRatePerSecond: 10 })

    const before = adaptive.metrics().refillRatePerSecond

    adaptive.updateFromHeaders({
      remaining: 80,
      limit: 100,
    })

    const m = adaptive.metrics()
    expect(m.refillRatePerSecond).toBeGreaterThan(before)
  })

  it('rate never drops below minRefillRate', () => {
    const { adaptive } = makeAdaptive(
      { refillRatePerSecond: 2 },
      { minRefillRate: 5 },
    )

    // Repeatedly throttle.
    for (let i = 0; i < 20; i++) {
      adaptive.updateFromHeaders({ remaining: 1, limit: 100 })
    }

    const m = adaptive.metrics()
    expect(m.refillRatePerSecond).toBeGreaterThanOrEqual(5)
  })

  it('rate never exceeds maxRefillRate', () => {
    const { adaptive } = makeAdaptive(
      { refillRatePerSecond: 80 },
      { maxRefillRate: 100, recoveryFactor: 2.0 },
    )

    // Repeatedly recover.
    for (let i = 0; i < 20; i++) {
      adaptive.updateFromHeaders({ remaining: 90, limit: 100 })
    }

    const m = adaptive.metrics()
    expect(m.refillRatePerSecond).toBeLessThanOrEqual(100)
  })

  it('delegates retry-after to the underlying bucket', () => {
    const bucket = createTokenBucket({
      maxTokens: 10,
      refillRatePerSecond: 50,
      tenantId: 'test',
    })
    const adaptive = createAdaptiveRateLimiter({
      bucket,
      minRefillRate: 1,
      maxRefillRate: 100,
    })
    limitersTeardown.push(adaptive)

    adaptive.updateFromHeaders({ retryAfter: 0.05 })

    const bm = bucket.metrics()
    expect(bm.refillRatePerSecond).toBe(0)
  })

  it('acquire delegates to underlying bucket', async () => {
    const { adaptive } = makeAdaptive()
    const tok = await adaptive.acquire(1)
    expect(tok).toBeDefined()
    tok.release()
  })

  it('tryAcquire delegates to underlying bucket', () => {
    const { adaptive } = makeAdaptive()
    const tok = adaptive.tryAcquire(1)
    expect(tok).toBeDefined()
    tok!.release()
  })

  it('uses default options when not specified', () => {
    const bucket = createTokenBucket({
      maxTokens: 10,
      refillRatePerSecond: 10,
      tenantId: 'test',
    })
    const adaptive = createAdaptiveRateLimiter({ bucket })
    limitersTeardown.push(adaptive)

    // Should not throw.
    const m = adaptive.metrics()
    expect(m.refillRatePerSecond).toBe(10)
  })

  it('applies computed rate to the underlying bucket', () => {
    const { adaptive, bucket } = makeAdaptive()

    // Throttle via adaptive.
    adaptive.updateFromHeaders({
      remaining: 5,
      limit: 100,
    })

    const adaptiveRate = adaptive.metrics().refillRatePerSecond
    const bucketRate = bucket.metrics().refillRatePerSecond
    expect(adaptiveRate).toBe(bucketRate)
  })

  it('does not double-throttle the bucket', () => {
    const { adaptive, bucket } = makeAdaptive()

    // Throttle via adaptive.
    adaptive.updateFromHeaders({
      remaining: 5,
      limit: 100,
    })

    // The bucket should have exactly the rate computed by adaptive,
    // not an independently reduced rate from its own back-off logic.
    const adaptiveRate = adaptive.metrics().refillRatePerSecond
    const bucketRate = bucket.metrics().refillRatePerSecond
    expect(adaptiveRate).toBe(bucketRate)
  })

  it('setRefillRate applies to both adaptive and bucket', () => {
    const { adaptive, bucket } = makeAdaptive()

    adaptive.setRefillRate(25)

    expect(adaptive.metrics().refillRatePerSecond).toBe(25)
    expect(bucket.metrics().refillRatePerSecond).toBe(25)
  })
})
