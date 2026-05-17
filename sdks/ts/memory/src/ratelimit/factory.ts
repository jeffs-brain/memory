// SPDX-License-Identifier: Apache-2.0

/**
 * Per-tenant rate limiter factory. Each tenant receives an isolated
 * token bucket (optionally wrapped in an adaptive layer).
 */

import { noopLogger } from '../llm/types.js'
import { createAdaptiveRateLimiter } from './adaptive.js'
import { createTokenBucket } from './token-bucket.js'
import type { RateLimiter, RateLimiterFactory, RateLimiterFactoryOptions } from './types.js'

/**
 * Create a factory that produces per-tenant rate limiters. Calling
 * `forTenant` with the same ID returns the same limiter instance.
 */
export const createRateLimiterFactory = (opts: RateLimiterFactoryOptions): RateLimiterFactory => {
  const logger = opts.logger ?? noopLogger
  const tenants = new Map<string, RateLimiter>()

  const forTenant = (tenantId: string): RateLimiter => {
    const existing = tenants.get(tenantId)
    if (existing !== undefined) return existing

    const bucket = createTokenBucket({
      maxTokens: opts.defaultMaxTokens,
      refillRatePerSecond: opts.defaultRefillRate,
      ...(opts.defaultMaxConcurrency !== undefined ? { maxConcurrency: opts.defaultMaxConcurrency } : {}),
      tenantId,
      logger,
    })

    const limiter: RateLimiter = opts.adaptiveEnabled === true
      ? createAdaptiveRateLimiter({
          bucket,
          ...(opts.minRefillRate !== undefined ? { minRefillRate: opts.minRefillRate } : {}),
          ...(opts.maxRefillRate !== undefined ? { maxRefillRate: opts.maxRefillRate } : {}),
          ...(opts.recoveryFactor !== undefined ? { recoveryFactor: opts.recoveryFactor } : {}),
          logger,
        })
      : bucket

    tenants.set(tenantId, limiter)
    return limiter
  }

  const close = async (): Promise<void> => {
    const closers = [...tenants.values()].map(lim => lim.close())
    tenants.clear()
    await Promise.all(closers)
  }

  return { forTenant, close }
}
