// SPDX-License-Identifier: Apache-2.0

/**
 * In-memory token bucket rate limiter. Tokens are replenished at a
 * configurable rate and consumed on acquire. An optional concurrency
 * semaphore limits the number of in-flight operations.
 *
 * This is a native implementation (no external dependencies). The
 * bucket is suitable for single-process deployments; for multi-worker
 * scenarios a Redis-backed bucket can be layered on top.
 */

import { noopLogger } from '../llm/types.js'
import type { RateLimitHeaders, RateLimitMetrics, RateLimitToken, RateLimiter, TokenBucketOptions } from './types.js'

type Waiter = {
  readonly cost: number
  readonly resolve: (token: RateLimitToken) => void
  readonly reject: (err: Error) => void
}

/**
 * Create an in-memory token bucket rate limiter.
 *
 * Tokens are replenished continuously at `refillRatePerSecond`. When
 * tokens are unavailable, `acquire` queues the caller until enough
 * tokens accumulate. `tryAcquire` is non-blocking.
 */
export const createTokenBucket = (opts: TokenBucketOptions): RateLimiter => {
  const logger = opts.logger ?? noopLogger
  let tokens = opts.maxTokens
  let maxTokens = opts.maxTokens
  let refillRate = opts.refillRatePerSecond
  let lastRefill = Date.now()
  let throttledTotal = 0
  let concurrencyInFlight = 0
  const maxConcurrency = opts.maxConcurrency ?? 0

  const waitQueue: Waiter[] = []
  let refillTimer: ReturnType<typeof setInterval> | undefined
  let retryAfterTimer: ReturnType<typeof setTimeout> | undefined
  let closed = false

  const refill = (): void => {
    const now = Date.now()
    const elapsed = (now - lastRefill) / 1000
    lastRefill = now
    tokens = Math.min(maxTokens, tokens + elapsed * refillRate)
  }

  const makeToken = (): RateLimitToken => {
    let released = false
    return {
      release: () => {
        if (released) return
        released = true
        if (maxConcurrency > 0) {
          concurrencyInFlight--
        }
        drainQueue()
      },
    }
  }

  const canAcquire = (cost: number): boolean => {
    if (tokens < cost) return false
    if (maxConcurrency > 0 && concurrencyInFlight >= maxConcurrency) return false
    return true
  }

  const consumeTokens = (cost: number): void => {
    tokens -= cost
    if (maxConcurrency > 0) {
      concurrencyInFlight++
    }
  }

  const drainQueue = (): void => {
    refill()
    while (waitQueue.length > 0) {
      const head = waitQueue[0]
      if (head === undefined) break
      if (!canAcquire(head.cost)) break
      waitQueue.shift()
      consumeTokens(head.cost)
      head.resolve(makeToken())
    }
  }

  // Start periodic refill + queue drain.
  refillTimer = setInterval(() => {
    if (closed) return
    drainQueue()
  }, 50)

  // Prevent the timer from keeping the process alive.
  if (typeof refillTimer === 'object' && 'unref' in refillTimer) {
    refillTimer.unref()
  }

  const acquire = (cost = 1): Promise<RateLimitToken> => {
    if (closed) {
      return Promise.reject(new Error('rate limiter is closed'))
    }

    refill()
    if (canAcquire(cost)) {
      consumeTokens(cost)
      return Promise.resolve(makeToken())
    }

    throttledTotal++
    return new Promise<RateLimitToken>((resolve, reject) => {
      waitQueue.push({ cost, resolve, reject })
    })
  }

  const tryAcquire = (cost = 1): RateLimitToken | undefined => {
    if (closed) return undefined
    refill()
    if (!canAcquire(cost)) return undefined
    consumeTokens(cost)
    return makeToken()
  }

  const updateFromHeaders = (headers: RateLimitHeaders): void => {
    // Retry-After: pause the limiter.
    if (headers.retryAfter !== undefined && headers.retryAfter > 0) {
      logger.info('rate limiter pausing due to retry-after', {
        tenantId: opts.tenantId,
        retryAfter: headers.retryAfter,
      })
      throttledTotal++
      const savedRate = refillRate
      refillRate = 0
      if (retryAfterTimer !== undefined) {
        clearTimeout(retryAfterTimer)
      }
      retryAfterTimer = setTimeout(() => {
        refillRate = savedRate
        lastRefill = Date.now()
        logger.info('rate limiter resumed after retry-after', {
          tenantId: opts.tenantId,
          rate: savedRate,
        })
        drainQueue()
      }, headers.retryAfter * 1000)
      if (typeof retryAfterTimer === 'object' && 'unref' in retryAfterTimer) {
        retryAfterTimer.unref()
      }
      return
    }

    // Back off when remaining < burst/4.
    if (headers.remaining !== undefined && headers.remaining > 0 && headers.limit !== undefined && headers.limit > 0) {
      const threshold = Math.max(1, Math.floor(maxTokens / 4))
      if (headers.remaining < threshold) {
        const newRate = Math.max(0.1, refillRate / 2)
        logger.info('rate limiter backing off', {
          tenantId: opts.tenantId,
          remaining: headers.remaining,
          threshold,
          newRate,
        })
        throttledTotal++
        refillRate = newRate
      }
    }

    // Update burst from limit header.
    if (headers.limit !== undefined && headers.limit > 0 && headers.limit !== maxTokens) {
      maxTokens = headers.limit
    }
  }

  const metricsSnapshot = (): RateLimitMetrics => {
    refill()
    return {
      availableTokens: tokens,
      maxTokens,
      refillRatePerSecond: refillRate,
      waitingRequests: waitQueue.length,
      throttledTotal,
    }
  }

  const close = async (): Promise<void> => {
    closed = true
    if (refillTimer !== undefined) {
      clearInterval(refillTimer)
      refillTimer = undefined
    }
    if (retryAfterTimer !== undefined) {
      clearTimeout(retryAfterTimer)
      retryAfterTimer = undefined
    }
    // Reject any pending waiters.
    for (const waiter of waitQueue.splice(0)) {
      waiter.reject(new Error('rate limiter closed'))
    }
  }

  return {
    acquire,
    tryAcquire,
    updateFromHeaders,
    metrics: metricsSnapshot,
    close,
  }
}
