// SPDX-License-Identifier: Apache-2.0

/**
 * Parse rate-limit headers from HTTP responses. Supports the common
 * header names used by OpenAI, Anthropic, and other LLM providers.
 */

import type { HeaderNameOptions, RateLimitHeaders } from './types.js'

/** Default candidate header names for remaining requests/tokens. */
const DEFAULT_REMAINING_HEADERS = [
  'x-ratelimit-remaining',
  'ratelimit-remaining',
  'x-ratelimit-remaining-requests',
  'x-ratelimit-remaining-tokens',
] as const

/** Default candidate header names for the rate limit ceiling. */
const DEFAULT_LIMIT_HEADERS = [
  'x-ratelimit-limit',
  'ratelimit-limit',
  'x-ratelimit-limit-requests',
  'x-ratelimit-limit-tokens',
] as const

/** Default candidate header names for the reset time. */
const DEFAULT_RESET_HEADERS = [
  'x-ratelimit-reset',
  'ratelimit-reset',
  'x-ratelimit-reset-requests',
  'x-ratelimit-reset-tokens',
] as const

/** Default candidate header names for retry-after. */
const DEFAULT_RETRY_AFTER_HEADERS = ['retry-after'] as const

/** Build a RateLimitHeaders object, omitting undefined properties. */
const buildHeaders = (
  remaining: number | undefined,
  limit: number | undefined,
  resetAt: Date | undefined,
  retryAfter: number | undefined,
): RateLimitHeaders => ({
  ...(remaining !== undefined ? { remaining } : {}),
  ...(limit !== undefined ? { limit } : {}),
  ...(resetAt !== undefined ? { resetAt } : {}),
  ...(retryAfter !== undefined ? { retryAfter } : {}),
})

/**
 * Resolve header name options, using defaults for any unspecified groups.
 */
const resolveNames = (custom?: HeaderNameOptions) => ({
  remaining: custom?.remaining ?? DEFAULT_REMAINING_HEADERS,
  limit: custom?.limit ?? DEFAULT_LIMIT_HEADERS,
  reset: custom?.reset ?? DEFAULT_RESET_HEADERS,
  retryAfter: custom?.retryAfter ?? DEFAULT_RETRY_AFTER_HEADERS,
})

/** Generic getter abstraction — unifies Headers (fetch API) and plain records. */
type HeaderGetter = (name: string) => string | undefined

const getterFromHeaders = (h: Headers): HeaderGetter =>
  (name) => h.get(name) ?? undefined

const getterFromRecord = (r: Readonly<Record<string, string | undefined>>): HeaderGetter =>
  (name) => r[name]

/** Parse rate-limit headers from a fetch API Headers object. */
export const parseRateLimitHeaders = (
  headers: Headers,
  headerNames?: HeaderNameOptions,
): RateLimitHeaders => {
  const names = resolveNames(headerNames)
  const get = getterFromHeaders(headers)
  return buildHeaders(
    firstInt(get, names.remaining),
    firstInt(get, names.limit),
    firstDate(get, names.reset),
    parseRetryAfterValue(get, names.retryAfter),
  )
}

/** Parse rate-limit headers from a plain record (e.g. from node http). */
export const parseRateLimitHeaderRecord = (
  headers: Readonly<Record<string, string | undefined>>,
  headerNames?: HeaderNameOptions,
): RateLimitHeaders => {
  const names = resolveNames(headerNames)
  const get = getterFromRecord(headers)
  return buildHeaders(
    firstInt(get, names.remaining),
    firstInt(get, names.limit),
    firstDate(get, names.reset),
    parseRetryAfterValue(get, names.retryAfter),
  )
}

const firstInt = (get: HeaderGetter, candidates: readonly string[]): number | undefined => {
  for (const name of candidates) {
    const v = get(name)?.trim()
    if (v === undefined || v === '') continue
    const n = Number.parseInt(v, 10)
    if (!Number.isNaN(n)) return n
  }
  return undefined
}

const firstDate = (get: HeaderGetter, candidates: readonly string[]): Date | undefined => {
  for (const name of candidates) {
    const v = get(name)?.trim()
    if (v === undefined || v === '') continue
    // Try Unix epoch seconds.
    const epoch = Number.parseInt(v, 10)
    if (!Number.isNaN(epoch) && String(epoch) === v) {
      return new Date(epoch * 1000)
    }
    // Try HTTP date.
    const d = new Date(v)
    if (!Number.isNaN(d.getTime())) return d
  }
  return undefined
}

const parseRetryAfterValue = (get: HeaderGetter, candidates: readonly string[]): number | undefined => {
  for (const name of candidates) {
    const v = get(name)?.trim()
    if (v === undefined || v === '') continue
    const secs = Number.parseInt(v, 10)
    if (!Number.isNaN(secs)) return secs
    // HTTP date.
    const d = new Date(v)
    if (!Number.isNaN(d.getTime())) {
      const delta = (d.getTime() - Date.now()) / 1000
      return delta > 0 ? Math.ceil(delta) : undefined
    }
  }
  return undefined
}
