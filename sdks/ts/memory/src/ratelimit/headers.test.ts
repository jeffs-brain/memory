// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for rate-limit header parsing.
 */

import { describe, expect, it } from 'vitest'
import { parseRateLimitHeaderRecord, parseRateLimitHeaders } from './headers.js'

describe('parseRateLimitHeaders', () => {
  it('parses x-ratelimit-remaining', () => {
    const headers = new Headers({ 'x-ratelimit-remaining': '42' })
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.remaining).toBe(42)
  })

  it('parses x-ratelimit-limit', () => {
    const headers = new Headers({ 'x-ratelimit-limit': '1000' })
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.limit).toBe(1000)
  })

  it('parses retry-after as seconds', () => {
    const headers = new Headers({ 'retry-after': '30' })
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.retryAfter).toBe(30)
  })

  it('parses x-ratelimit-reset as unix epoch', () => {
    const headers = new Headers({ 'x-ratelimit-reset': '1700000000' })
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.resetAt).toBeDefined()
    expect(parsed.resetAt!.getTime()).toBe(1700000000 * 1000)
  })

  it('returns undefined for missing headers', () => {
    const headers = new Headers()
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.remaining).toBeUndefined()
    expect(parsed.limit).toBeUndefined()
    expect(parsed.retryAfter).toBeUndefined()
    expect(parsed.resetAt).toBeUndefined()
  })

  it('handles non-numeric values gracefully', () => {
    const headers = new Headers({ 'x-ratelimit-remaining': 'abc' })
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.remaining).toBeUndefined()
  })

  it('trims whitespace from values', () => {
    const headers = new Headers({ 'x-ratelimit-remaining': '  25  ' })
    const parsed = parseRateLimitHeaders(headers)
    expect(parsed.remaining).toBe(25)
  })

  it('accepts custom header names', () => {
    const headers = new Headers({
      'my-custom-remaining': '77',
      'my-custom-limit': '200',
    })
    const parsed = parseRateLimitHeaders(headers, {
      remaining: ['my-custom-remaining'],
      limit: ['my-custom-limit'],
    })
    expect(parsed.remaining).toBe(77)
    expect(parsed.limit).toBe(200)
  })

  it('falls back to defaults for unspecified custom header groups', () => {
    const headers = new Headers({ 'x-ratelimit-remaining': '42' })
    const parsed = parseRateLimitHeaders(headers, {
      limit: ['my-custom-limit'],
    })
    // remaining should still use defaults.
    expect(parsed.remaining).toBe(42)
    // limit uses custom names — not present in headers.
    expect(parsed.limit).toBeUndefined()
  })
})

describe('parseRateLimitHeaderRecord', () => {
  it('parses from plain record', () => {
    const headers: Record<string, string | undefined> = {
      'x-ratelimit-remaining': '55',
      'retry-after': '10',
    }
    const parsed = parseRateLimitHeaderRecord(headers)
    expect(parsed.remaining).toBe(55)
    expect(parsed.retryAfter).toBe(10)
  })

  it('handles undefined values', () => {
    const headers: Record<string, string | undefined> = {
      'x-ratelimit-remaining': undefined,
    }
    const parsed = parseRateLimitHeaderRecord(headers)
    expect(parsed.remaining).toBeUndefined()
  })

  it('parses ratelimit-remaining (no x- prefix)', () => {
    const headers: Record<string, string | undefined> = {
      'ratelimit-remaining': '7',
    }
    const parsed = parseRateLimitHeaderRecord(headers)
    expect(parsed.remaining).toBe(7)
  })

  it('accepts custom header names from record', () => {
    const headers: Record<string, string | undefined> = {
      'provider-remaining': '33',
    }
    const parsed = parseRateLimitHeaderRecord(headers, {
      remaining: ['provider-remaining'],
    })
    expect(parsed.remaining).toBe(33)
  })
})
