// SPDX-License-Identifier: Apache-2.0

/**
 * Adaptive rate limiter wrapper that adjusts the underlying bucket's
 * refill rate based on provider response headers. Enforces floor and
 * ceiling bounds on the rate.
 */

import { noopLogger } from '../llm/types.js'
import type { AdaptiveOptions, RateLimitHeaders, RateLimitMetrics, RateLimitToken, RateLimiter } from './types.js'

const DEFAULT_MIN_REFILL_RATE = 1
const DEFAULT_MAX_REFILL_RATE = 100
const DEFAULT_RECOVERY_FACTOR = 1.5

/**
 * Wrap a base rate limiter with adaptive behaviour that reads
 * provider response headers and adjusts throughput accordingly.
 */
export const createAdaptiveRateLimiter = (opts: AdaptiveOptions): RateLimiter => {
  const logger = opts.logger ?? noopLogger
  const minRefillRate = opts.minRefillRate ?? DEFAULT_MIN_REFILL_RATE
  const maxRefillRate = opts.maxRefillRate ?? DEFAULT_MAX_REFILL_RATE
  const recoveryFactor = opts.recoveryFactor ?? DEFAULT_RECOVERY_FACTOR
  const bucket = opts.bucket

  let currentRate = bucket.metrics().refillRatePerSecond

  const acquire = (cost = 1): Promise<RateLimitToken> => bucket.acquire(cost)

  const tryAcquire = (cost = 1): RateLimitToken | undefined => bucket.tryAcquire(cost)

  const updateFromHeaders = (headers: RateLimitHeaders): void => {
    // Delegate retry-after directly to the underlying bucket.
    if (headers.retryAfter !== undefined && headers.retryAfter > 0) {
      bucket.updateFromHeaders(headers)
      return
    }

    const m = bucket.metrics()
    const maxTokens = Math.max(1, m.maxTokens)
    const threshold = Math.max(1, Math.floor(maxTokens / 4))

    const remaining = headers.remaining ?? 0

    if (remaining > 0 && remaining < threshold) {
      // Throttle: reduce rate proportionally.
      const ratio = remaining / maxTokens
      currentRate = Math.max(minRefillRate, currentRate * ratio)
      logger.info('adaptive limiter throttling', {
        remaining,
        threshold,
        newRate: currentRate,
      })
    } else if (remaining >= threshold) {
      // Recover: ramp up towards max.
      currentRate = Math.min(maxRefillRate, currentRate * recoveryFactor)
    }

    // Propagate to the bucket (without retry-after, to avoid re-pause).
    const propagated = {
      ...(headers.remaining !== undefined ? { remaining: headers.remaining } : {}),
      ...(headers.limit !== undefined ? { limit: headers.limit } : {}),
      ...(headers.resetAt !== undefined ? { resetAt: headers.resetAt } : {}),
    } satisfies RateLimitHeaders
    bucket.updateFromHeaders(propagated)
  }

  const metricsSnapshot = (): RateLimitMetrics => {
    const m = bucket.metrics()
    return {
      ...m,
      refillRatePerSecond: currentRate,
    }
  }

  const close = (): Promise<void> => bucket.close()

  return {
    acquire,
    tryAcquire,
    updateFromHeaders,
    metrics: metricsSnapshot,
    close,
  }
}
