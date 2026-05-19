// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for the per-tenant adaptive rate limiter. Consumed by
 * both the ingestion pipeline and connector layers.
 */

import type { Logger } from '../llm/types.js'

/** Token returned by a successful acquire. Callers must release it. */
export type RateLimitToken = {
  readonly release: () => void
}

/** Parsed rate-limit headers from a provider response. */
export type RateLimitHeaders = {
  readonly remaining?: number
  readonly limit?: number
  readonly resetAt?: Date
  readonly retryAfter?: number // seconds
}

/** Point-in-time snapshot of limiter state for monitoring. */
export type RateLimitMetrics = {
  readonly availableTokens: number
  readonly maxTokens: number
  readonly refillRatePerSecond: number
  readonly waitingRequests: number
  readonly throttledTotal: number
}

/** Core rate limiter interface. Thread-safe (single-threaded in JS). */
export type RateLimiter = {
  acquire(cost?: number): Promise<RateLimitToken>
  tryAcquire(cost?: number): RateLimitToken | undefined
  updateFromHeaders(headers: RateLimitHeaders): void
  /** Override the refill rate (tokens/sec). Used by the adaptive layer. */
  setRefillRate(rate: number): void
  metrics(): RateLimitMetrics
  close(): Promise<void>
}

/** Configuration for a token bucket limiter. */
export type TokenBucketOptions = {
  readonly maxTokens: number
  readonly refillRatePerSecond: number
  readonly maxConcurrency?: number
  readonly tenantId: string
  readonly logger?: Logger
}

/** Configuration for the adaptive wrapper. */
export type AdaptiveOptions = {
  readonly bucket: RateLimiter
  readonly minRefillRate?: number // default 1
  readonly maxRefillRate?: number // default 100
  readonly recoveryFactor?: number // default 1.5
  readonly logger?: Logger
}

/** Configuration for the limiter factory. */
export type RateLimiterFactoryOptions = {
  readonly defaultMaxTokens: number
  readonly defaultRefillRate: number
  readonly defaultMaxConcurrency?: number
  readonly adaptiveEnabled?: boolean
  readonly minRefillRate?: number
  readonly maxRefillRate?: number
  readonly recoveryFactor?: number
  /** TTL in milliseconds for idle tenants. Default 300_000 (5 min). */
  readonly tenantTtlMs?: number
  readonly logger?: Logger
}

/** Configurable header names for rate-limit parsing. */
export type HeaderNameOptions = {
  readonly remaining?: readonly string[]
  readonly limit?: readonly string[]
  readonly reset?: readonly string[]
  readonly retryAfter?: readonly string[]
}

/** Creates per-tenant limiter instances. */
export type RateLimiterFactory = {
  forTenant(tenantId: string): RateLimiter
  close(): Promise<void>
}
