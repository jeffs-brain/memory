// SPDX-License-Identifier: Apache-2.0

/**
 * Token-bucket rate limiter with exponential backoff and adaptive
 * header-based adjustment. Suitable for throttling API calls to
 * external services.
 */

import type { RateLimiter, RateLimiterConfig, RateLimitHeaders } from './types.js'

const DEFAULT_REFILL_INTERVAL = 1000
const DEFAULT_MAX_RETRIES = 5
const DEFAULT_BASE_BACKOFF = 1000
const DEFAULT_MAX_BACKOFF = 60_000
const MAX_JITTER_MS = 500

/**
 * Create a new token-bucket rate limiter. The bucket starts at full
 * capacity.
 */
export const createRateLimiter = (config: RateLimiterConfig): RateLimiter => {
  const refillInterval = config.refillInterval ?? DEFAULT_REFILL_INTERVAL
  const baseBackoff = config.baseBackoff ?? DEFAULT_BASE_BACKOFF
  const maxBackoff = config.maxBackoff ?? DEFAULT_MAX_BACKOFF

  let currentTokens = config.maxTokens
  let lastTick = Date.now()
  let currentRefillRate = config.refillRate
  let closed = false

  const refill = (): void => {
    const now = Date.now()
    const elapsed = (now - lastTick) / 1000
    currentTokens = Math.min(config.maxTokens, currentTokens + elapsed * currentRefillRate)
    lastTick = now
  }

  const sleep = (ms: number, signal?: AbortSignal): Promise<void> =>
    new Promise((resolve, reject) => {
      if (signal?.aborted) {
        reject(signal.reason ?? new DOMException('The operation was aborted.', 'AbortError'))
        return
      }
      const timer = setTimeout(resolve, ms)
      const onAbort = (): void => {
        clearTimeout(timer)
        reject(signal!.reason ?? new DOMException('The operation was aborted.', 'AbortError'))
      }
      signal?.addEventListener('abort', onAbort, { once: true })
    })

  const limiter: RateLimiter = {
    async acquire(count = 1, signal?: AbortSignal): Promise<void> {
      const needed = Math.max(1, count)
      while (!closed) {
        if (signal?.aborted) {
          throw signal.reason ?? new DOMException('The operation was aborted.', 'AbortError')
        }
        refill()
        if (currentTokens >= needed) {
          currentTokens -= needed
          return
        }
        const deficit = needed - currentTokens
        const waitMs = (deficit / currentRefillRate) * 1000
        await sleep(Math.max(1, waitMs), signal)
      }
    },

    tryAcquire(count = 1): boolean {
      const needed = Math.max(1, count)
      refill()
      if (currentTokens >= needed) {
        currentTokens -= needed
        return true
      }
      return false
    },

    reset(): void {
      currentTokens = config.maxTokens
      lastTick = Date.now()
      currentRefillRate = config.refillRate
    },

    adjustFromHeaders(headers: RateLimitHeaders): void {
      if (headers.remaining !== undefined) {
        const rem = parseFloat(headers.remaining)
        if (!Number.isNaN(rem)) {
          currentTokens = Math.min(rem, config.maxTokens)
        }
      }

      if (headers.remaining !== undefined && headers.limit !== undefined) {
        const rem = parseFloat(headers.remaining)
        const lim = parseFloat(headers.limit)
        if (!Number.isNaN(rem) && !Number.isNaN(lim) && lim > 0) {
          const ratio = rem / lim
          currentRefillRate = ratio < 0.1
            ? config.refillRate / 2
            : config.refillRate
        }
      }
    },

    async backoff(attempt: number): Promise<void> {
      const multiplier = Math.pow(2, attempt)
      let delay = baseBackoff * multiplier
      const jitter = Math.random() * MAX_JITTER_MS
      delay = Math.min(delay + jitter, maxBackoff)
      await sleep(delay)
    },

    async retryAfter(headers: RateLimitHeaders, signal?: AbortSignal): Promise<void> {
      if (headers.retryAfter === undefined) return
      const seconds = parseFloat(headers.retryAfter)
      if (Number.isNaN(seconds)) return
      await sleep(seconds * 1000, signal)
    },

    tokens(): number {
      refill()
      return currentTokens
    },

    close(): void {
      closed = true
    },
  }

  return limiter
}
