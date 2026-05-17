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
 * Singly-linked list node for the waiter FIFO queue. Using a linked
 * list avoids O(n) Array.shift() on the hot path when draining waiters.
 */
type WaiterNode = {
  readonly value: Waiter
  next: WaiterNode | undefined
}

/**
 * O(1) enqueue/dequeue FIFO queue backed by a singly-linked list.
 * Replaces the naive array-based queue where Array.shift() is O(n).
 */
class WaiterQueue {
  private head: WaiterNode | undefined = undefined
  private tail: WaiterNode | undefined = undefined
  private _length = 0

  get length(): number {
    return this._length
  }

  enqueue(waiter: Waiter): void {
    const node: WaiterNode = { value: waiter, next: undefined }
    if (this.tail !== undefined) {
      this.tail.next = node
    } else {
      this.head = node
    }
    this.tail = node
    this._length++
  }

  dequeue(): Waiter | undefined {
    if (this.head === undefined) return undefined
    const value = this.head.value
    this.head = this.head.next
    if (this.head === undefined) {
      this.tail = undefined
    }
    this._length--
    return value
  }

  peek(): Waiter | undefined {
    return this.head?.value
  }

  /** Drain all entries, returning them as an array. Resets the queue. */
  drain(): Waiter[] {
    const items: Waiter[] = []
    let node = this.head
    while (node !== undefined) {
      items.push(node.value)
      node = node.next
    }
    this.head = undefined
    this.tail = undefined
    this._length = 0
    return items
  }
}

/** Maximum retry-after duration: 5 minutes in seconds. */
const MAX_RETRY_AFTER_SECS = 300

/**
 * Create an in-memory token bucket rate limiter.
 *
 * Tokens are replenished continuously at `refillRatePerSecond`. When
 * tokens are unavailable, `acquire` queues the caller until enough
 * tokens accumulate. `tryAcquire` is non-blocking.
 *
 * NOTE: This is a single-process implementation. For multi-worker
 * deployments requiring shared state, a Redis-backed bucket is planned
 * as a follow-up (see LLE-XXXX).
 */
export const createTokenBucket = (opts: TokenBucketOptions): RateLimiter => {
  if (opts.maxTokens <= 0) {
    throw new Error('maxTokens must be a positive number')
  }
  if (opts.refillRatePerSecond <= 0) {
    throw new Error('refillRatePerSecond must be a positive number')
  }

  const logger = opts.logger ?? noopLogger
  let tokens = opts.maxTokens
  let maxTokens = opts.maxTokens
  let refillRate = opts.refillRatePerSecond
  let lastRefill = Date.now()
  let throttledTotal = 0
  let concurrencyInFlight = 0
  const maxConcurrency = opts.maxConcurrency ?? 0

  const waitQueue = new WaiterQueue()
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
      const head = waitQueue.peek()
      if (head === undefined) break
      if (!canAcquire(head.cost)) break
      waitQueue.dequeue()
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

  const validateCost = (cost: number): void => {
    if (cost < 1) {
      throw new Error('cost must be a positive integer (>= 1)')
    }
  }

  const acquire = (cost = 1): Promise<RateLimitToken> => {
    validateCost(cost)
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
      waitQueue.enqueue({ cost, resolve, reject })
    })
  }

  const tryAcquire = (cost = 1): RateLimitToken | undefined => {
    validateCost(cost)
    if (closed) return undefined
    refill()
    if (!canAcquire(cost)) return undefined
    consumeTokens(cost)
    return makeToken()
  }

  const setRefillRate = (rate: number): void => {
    refillRate = rate
  }

  const updateFromHeaders = (headers: RateLimitHeaders): void => {
    // Retry-After: pause the limiter, capped at MAX_RETRY_AFTER_SECS.
    if (headers.retryAfter !== undefined && headers.retryAfter > 0) {
      const cappedRetryAfter = Math.min(headers.retryAfter, MAX_RETRY_AFTER_SECS)
      logger.info('rate limiter pausing due to retry-after', {
        tenantId: opts.tenantId,
        retryAfter: cappedRetryAfter,
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
      }, cappedRetryAfter * 1000)
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
    for (const waiter of waitQueue.drain()) {
      waiter.reject(new Error('rate limiter closed'))
    }
  }

  return {
    acquire,
    tryAcquire,
    updateFromHeaders,
    setRefillRate,
    metrics: metricsSnapshot,
    close,
  }
}
