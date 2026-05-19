// SPDX-License-Identifier: Apache-2.0

/**
 * Barrel export for the shared rate limiter package.
 */

export type {
  AdaptiveOptions,
  HeaderNameOptions,
  RateLimitHeaders,
  RateLimitMetrics,
  RateLimitToken,
  RateLimiter,
  RateLimiterFactory,
  RateLimiterFactoryOptions,
  TokenBucketOptions,
} from './types.js'

export { createTokenBucket } from './token-bucket.js'
export { createAdaptiveRateLimiter } from './adaptive.js'
export { createRateLimiterFactory } from './factory.js'
export { parseRateLimitHeaders, parseRateLimitHeaderRecord } from './headers.js'
export {
  createCircuitBreaker,
  CircuitOpenError,
  CircuitHalfOpenLimitError,
} from './circuit.js'
export type {
  CircuitBreaker,
  CircuitBreakerMetrics,
  CircuitBreakerOptions,
  CircuitState,
} from './circuit.js'
