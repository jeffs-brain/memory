// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import { cosineSimilarity, jaroWinklerDistance } from './similarity.js'

describe('jaroWinklerDistance', () => {
  it('returns 1.0 for identical strings', () => {
    expect(jaroWinklerDistance('Customer', 'Customer')).toBe(1.0)
  })

  it('is case insensitive', () => {
    expect(jaroWinklerDistance('ABC', 'abc')).toBe(1.0)
  })

  it('returns high similarity for singular vs plural', () => {
    const similarity = jaroWinklerDistance('Customer', 'Customers')
    expect(similarity).toBeGreaterThanOrEqual(0.9)
    expect(similarity).toBeLessThanOrEqual(1.0)
  })

  it('returns low similarity for different strings', () => {
    const similarity = jaroWinklerDistance('Customer', 'Product')
    expect(similarity).toBeLessThan(0.7)
  })

  it('returns 1.0 for both empty strings', () => {
    expect(jaroWinklerDistance('', '')).toBe(1.0)
  })

  it('returns 0.0 when first string is empty', () => {
    expect(jaroWinklerDistance('', 'hello')).toBe(0.0)
  })

  it('returns 0.0 when second string is empty', () => {
    expect(jaroWinklerDistance('hello', '')).toBe(0.0)
  })

  it('trims whitespace before comparison', () => {
    expect(jaroWinklerDistance('  hello  ', 'hello')).toBe(1.0)
  })

  it('returns high similarity for customer_record vs customer_records', () => {
    const similarity = jaroWinklerDistance('customer_record', 'customer_records')
    expect(similarity).toBeGreaterThanOrEqual(0.9)
  })
})

describe('cosineSimilarity', () => {
  it('returns 1.0 for identical unit vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1.0, 10)
  })

  it('returns 0.0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.0, 10)
  })

  it('returns -1.0 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0, 0], [-1, 0, 0])).toBeCloseTo(-1.0, 10)
  })

  it('returns 0.0 for zero first vector', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 0, 0])).toBe(0.0)
  })

  it('returns 0.0 for zero second vector', () => {
    expect(cosineSimilarity([1, 0, 0], [0, 0, 0])).toBe(0.0)
  })

  it('returns 0.0 for both zero vectors', () => {
    expect(cosineSimilarity([0, 0, 0], [0, 0, 0])).toBe(0.0)
  })

  it('returns cos(45) for 45-degree angle', () => {
    const expected = 1.0 / Math.sqrt(2)
    expect(cosineSimilarity([1, 1, 0], [1, 0, 0])).toBeCloseTo(expected, 10)
  })

  it('returns 0.0 for empty vectors', () => {
    expect(cosineSimilarity([], [])).toBe(0.0)
  })

  it('throws on mismatched lengths', () => {
    expect(() => cosineSimilarity([1, 0], [1, 0, 0])).toThrow()
  })
})
