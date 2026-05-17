// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { createRateLimiter } from './rate-limiter.js'

describe('createRateLimiter', () => {
  it('acquires within budget', async () => {
    const rl = createRateLimiter({ maxTokens: 10, refillRate: 1 })
    await rl.acquire(1)
    const remaining = rl.tokens()
    expect(remaining).toBeGreaterThanOrEqual(8)
    expect(remaining).toBeLessThanOrEqual(10)
    rl.close()
  })

  it('tryAcquire succeeds when tokens available', () => {
    const rl = createRateLimiter({ maxTokens: 5, refillRate: 1 })
    expect(rl.tryAcquire(1)).toBe(true)
    rl.close()
  })

  it('tryAcquire fails when bucket empty', () => {
    const rl = createRateLimiter({ maxTokens: 1, refillRate: 0.001 })
    rl.tryAcquire(1) // drain
    expect(rl.tryAcquire(1)).toBe(false)
    rl.close()
  })

  it('reset refills bucket to max', () => {
    const rl = createRateLimiter({ maxTokens: 10, refillRate: 0.001 })
    rl.tryAcquire(10) // drain
    expect(rl.tryAcquire(1)).toBe(false)

    rl.reset()
    expect(rl.tryAcquire(1)).toBe(true)
    rl.close()
  })

  it('adjustFromHeaders sets remaining tokens', () => {
    const rl = createRateLimiter({ maxTokens: 100, refillRate: 10 })
    rl.adjustFromHeaders({ remaining: '5', limit: '100' })
    expect(rl.tokens()).toBeLessThanOrEqual(6)
    rl.close()
  })

  it('backoff returns within reasonable time', async () => {
    const rl = createRateLimiter({
      maxTokens: 10,
      refillRate: 1,
      baseBackoff: 10,
      maxBackoff: 100,
    })

    const start = Date.now()
    await rl.backoff(0)
    const elapsed = Date.now() - start

    // Attempt 0: ~10ms base + up to 500ms jitter, capped at 100ms.
    expect(elapsed).toBeLessThan(700)
    rl.close()
  })

  it('backoff is capped at maxBackoff', async () => {
    const rl = createRateLimiter({
      maxTokens: 10,
      refillRate: 1,
      baseBackoff: 10,
      maxBackoff: 50,
    })

    const start = Date.now()
    await rl.backoff(10) // 2^10 * 10ms = way over 50ms
    const elapsed = Date.now() - start

    expect(elapsed).toBeLessThan(200)
    rl.close()
  })

  it('token refill over time', async () => {
    const rl = createRateLimiter({ maxTokens: 10, refillRate: 1000 })
    rl.tryAcquire(10) // drain

    await new Promise((r) => setTimeout(r, 20)) // 20ms -> ~20 tokens
    expect(rl.tryAcquire(1)).toBe(true)
    rl.close()
  })

  it('acquires after exhaust with fast refill', async () => {
    const rl = createRateLimiter({ maxTokens: 2, refillRate: 100 })
    await rl.acquire(2) // drain

    const start = Date.now()
    await rl.acquire(1)
    const elapsed = Date.now() - start

    expect(elapsed).toBeLessThan(500)
    rl.close()
  })
})
