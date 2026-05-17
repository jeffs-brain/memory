// SPDX-License-Identifier: Apache-2.0

/**
 * Parse rate-limit headers from HTTP responses. Supports the common
 * header names used by OpenAI, Anthropic, and other LLM providers.
 */

import type { RateLimitHeaders } from './types.js'

/** Candidate header names for remaining requests/tokens. */
const REMAINING_HEADERS = [
  'x-ratelimit-remaining',
  'ratelimit-remaining',
  'x-ratelimit-remaining-requests',
  'x-ratelimit-remaining-tokens',
] as const

/** Candidate header names for the rate limit ceiling. */
const LIMIT_HEADERS = [
  'x-ratelimit-limit',
  'ratelimit-limit',
  'x-ratelimit-limit-requests',
  'x-ratelimit-limit-tokens',
] as const

/** Candidate header names for the reset time. */
const RESET_HEADERS = [
  'x-ratelimit-reset',
  'ratelimit-reset',
  'x-ratelimit-reset-requests',
  'x-ratelimit-reset-tokens',
] as const

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

/** Parse rate-limit headers from a fetch API Headers object. */
export const parseRateLimitHeaders = (headers: Headers): RateLimitHeaders => {
  return buildHeaders(
    firstInt(headers, REMAINING_HEADERS),
    firstInt(headers, LIMIT_HEADERS),
    firstDate(headers, RESET_HEADERS),
    parseRetryAfter(headers),
  )
}

/** Parse rate-limit headers from a plain record (e.g. from node http). */
export const parseRateLimitHeaderRecord = (
  headers: Readonly<Record<string, string | undefined>>,
): RateLimitHeaders => {
  return buildHeaders(
    firstIntFromRecord(headers, REMAINING_HEADERS),
    firstIntFromRecord(headers, LIMIT_HEADERS),
    firstDateFromRecord(headers, RESET_HEADERS),
    parseRetryAfterFromRecord(headers),
  )
}

const firstInt = (headers: Headers, candidates: readonly string[]): number | undefined => {
  for (const name of candidates) {
    const v = headers.get(name)?.trim()
    if (v === undefined || v === '') continue
    const n = Number.parseInt(v, 10)
    if (!Number.isNaN(n)) return n
  }
  return undefined
}

const firstIntFromRecord = (
  headers: Readonly<Record<string, string | undefined>>,
  candidates: readonly string[],
): number | undefined => {
  for (const name of candidates) {
    const v = headers[name]?.trim()
    if (v === undefined || v === '') continue
    const n = Number.parseInt(v, 10)
    if (!Number.isNaN(n)) return n
  }
  return undefined
}

const firstDate = (headers: Headers, candidates: readonly string[]): Date | undefined => {
  for (const name of candidates) {
    const v = headers.get(name)?.trim()
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

const firstDateFromRecord = (
  headers: Readonly<Record<string, string | undefined>>,
  candidates: readonly string[],
): Date | undefined => {
  for (const name of candidates) {
    const v = headers[name]?.trim()
    if (v === undefined || v === '') continue
    const epoch = Number.parseInt(v, 10)
    if (!Number.isNaN(epoch) && String(epoch) === v) {
      return new Date(epoch * 1000)
    }
    const d = new Date(v)
    if (!Number.isNaN(d.getTime())) return d
  }
  return undefined
}

const parseRetryAfter = (headers: Headers): number | undefined => {
  const v = headers.get('retry-after')?.trim()
  if (v === undefined || v === '') return undefined
  const secs = Number.parseInt(v, 10)
  if (!Number.isNaN(secs)) return secs
  // HTTP date.
  const d = new Date(v)
  if (!Number.isNaN(d.getTime())) {
    const delta = (d.getTime() - Date.now()) / 1000
    return delta > 0 ? Math.ceil(delta) : undefined
  }
  return undefined
}

const parseRetryAfterFromRecord = (
  headers: Readonly<Record<string, string | undefined>>,
): number | undefined => {
  const v = headers['retry-after']?.trim()
  if (v === undefined || v === '') return undefined
  const secs = Number.parseInt(v, 10)
  if (!Number.isNaN(secs)) return secs
  const d = new Date(v)
  if (!Number.isNaN(d.getTime())) {
    const delta = (d.getTime() - Date.now()) / 1000
    return delta > 0 ? Math.ceil(delta) : undefined
  }
  return undefined
}
