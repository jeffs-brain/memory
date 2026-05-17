// SPDX-License-Identifier: Apache-2.0

/**
 * Per-tenant rate limiter factory. Each tenant receives an isolated
 * token bucket (optionally wrapped in an adaptive layer). Idle tenants
 * are evicted after a configurable TTL to prevent unbounded growth.
 */

import { noopLogger } from '../llm/types.js'
import { createAdaptiveRateLimiter } from './adaptive.js'
import { createTokenBucket } from './token-bucket.js'
import type { RateLimiter, RateLimiterFactory, RateLimiterFactoryOptions } from './types.js'

/** Default idle TTL: 5 minutes. */
const DEFAULT_TENANT_TTL_MS = 300_000

/** Eviction sweep interval: 60 seconds. */
const EVICTION_INTERVAL_MS = 60_000

type TenantEntry = {
  limiter: RateLimiter
  lastAccessedAt: number
}

/**
 * Create a factory that produces per-tenant rate limiters. Calling
 * `forTenant` with the same ID returns the same limiter instance.
 * Idle tenants are evicted after `tenantTtlMs` milliseconds.
 */
export const createRateLimiterFactory = (opts: RateLimiterFactoryOptions): RateLimiterFactory => {
  if (opts.defaultMaxTokens <= 0) {
    throw new Error('defaultMaxTokens must be a positive number')
  }
  if (opts.defaultRefillRate <= 0) {
    throw new Error('defaultRefillRate must be a positive number')
  }

  const logger = opts.logger ?? noopLogger
  const tenantTtlMs = opts.tenantTtlMs ?? DEFAULT_TENANT_TTL_MS
  const tenants = new Map<string, TenantEntry>()
  let closed = false

  const evictStale = (): void => {
    const now = Date.now()
    for (const [id, entry] of tenants) {
      if (now - entry.lastAccessedAt > tenantTtlMs) {
        tenants.delete(id)
        entry.limiter.close().catch(() => {
          // Swallow close errors during eviction.
        })
        logger.info('evicted idle tenant rate limiter', { tenantId: id })
      }
    }
  }

  // Periodic eviction sweep.
  const evictionTimer = setInterval(evictStale, EVICTION_INTERVAL_MS)
  if (typeof evictionTimer === 'object' && 'unref' in evictionTimer) {
    evictionTimer.unref()
  }

  const forTenant = (tenantId: string): RateLimiter => {
    if (closed) {
      throw new Error('rate limiter factory is closed')
    }

    const existing = tenants.get(tenantId)
    if (existing !== undefined) {
      existing.lastAccessedAt = Date.now()
      return existing.limiter
    }

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

    tenants.set(tenantId, { limiter, lastAccessedAt: Date.now() })
    return limiter
  }

  const close = async (): Promise<void> => {
    closed = true
    clearInterval(evictionTimer)
    const closers = [...tenants.values()].map(entry => entry.limiter.close())
    tenants.clear()
    await Promise.all(closers)
  }

  return { forTenant, close }
}
