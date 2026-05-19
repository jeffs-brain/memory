// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the per-tenant rate limiter factory.
 */

import { afterEach, describe, expect, it } from 'vitest'
import type { RateLimiterFactory } from './types.js'
import { createRateLimiterFactory } from './factory.js'

describe('createRateLimiterFactory', () => {
  const factoriesToTeardown: RateLimiterFactory[] = []

  afterEach(async () => {
    for (const f of factoriesToTeardown.splice(0)) {
      await f.close()
    }
  })

  const makeFactory = (
    overrides: Partial<Parameters<typeof createRateLimiterFactory>[0]> = {},
  ): RateLimiterFactory => {
    const f = createRateLimiterFactory({
      defaultMaxTokens: 10,
      defaultRefillRate: 100,
      ...overrides,
    })
    factoriesToTeardown.push(f)
    return f
  }

  it('returns isolated limiters for different tenants', () => {
    const f = makeFactory()
    const l1 = f.forTenant('tenant-a')
    const l2 = f.forTenant('tenant-b')
    expect(l1).not.toBe(l2)
  })

  it('returns the same limiter for the same tenant', () => {
    const f = makeFactory()
    const l1 = f.forTenant('tenant-a')
    const l2 = f.forTenant('tenant-a')
    expect(l1).toBe(l2)
  })

  it('tenants have independent token pools', async () => {
    const f = makeFactory({
      defaultMaxTokens: 2,
      defaultRefillRate: 0.001,
    })

    const l1 = f.forTenant('t1')
    const l2 = f.forTenant('t2')

    // Drain t1.
    const tok1 = await l1.acquire(1)
    tok1.release()
    const tok2 = await l1.acquire(1)
    tok2.release()

    // t2 should still have tokens.
    const tok3 = l2.tryAcquire(1)
    expect(tok3).toBeDefined()
    tok3!.release()
  })

  it('creates adaptive limiters when enabled', async () => {
    const f = makeFactory({
      adaptiveEnabled: true,
      minRefillRate: 1,
      maxRefillRate: 100,
      recoveryFactor: 1.5,
    })

    const lim = f.forTenant('t1')
    const tok = await lim.acquire(1)
    tok.release()

    // Update headers -> adaptive should adjust rate.
    lim.updateFromHeaders({ remaining: 1, limit: 10 })
    const m = lim.metrics()
    expect(m.refillRatePerSecond).toBeLessThan(100)
  })

  it('close releases all tenant limiters', async () => {
    const f = makeFactory()

    f.forTenant('a')
    f.forTenant('b')
    f.forTenant('c')

    await f.close()

    // Remove from teardown since we already closed.
    const idx = factoriesToTeardown.indexOf(f)
    if (idx >= 0) factoriesToTeardown.splice(idx, 1)
  })

  it('forTenant throws after close', async () => {
    const f = makeFactory()
    await f.close()

    expect(() => f.forTenant('should-fail')).toThrow('rate limiter factory is closed')

    // Remove from teardown since we already closed.
    const idx = factoriesToTeardown.indexOf(f)
    if (idx >= 0) factoriesToTeardown.splice(idx, 1)
  })

  it('constructor throws for zero defaultMaxTokens', () => {
    expect(() => createRateLimiterFactory({
      defaultMaxTokens: 0,
      defaultRefillRate: 10,
    })).toThrow('defaultMaxTokens must be a positive number')
  })

  it('constructor throws for zero defaultRefillRate', () => {
    expect(() => createRateLimiterFactory({
      defaultMaxTokens: 10,
      defaultRefillRate: 0,
    })).toThrow('defaultRefillRate must be a positive number')
  })
})
