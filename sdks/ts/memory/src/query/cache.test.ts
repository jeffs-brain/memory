// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { cacheKey, createCache } from './cache.js'

describe('createCache (LRU)', () => {
  it('returns undefined for a missing key', () => {
    const cache = createCache({ capacity: 4 })
    expect(cache.get('missing')).toBeUndefined()
    expect(cache.size()).toBe(0)
  })

  it('stores and retrieves values', () => {
    const cache = createCache({ capacity: 4 })
    cache.set('a', 'alpha')
    cache.set('b', 'bravo')
    expect(cache.get('a')).toBe('alpha')
    expect(cache.get('b')).toBe('bravo')
    expect(cache.size()).toBe(2)
  })

  it('reorders the recency queue on read (reverse-order retrieval)', () => {
    const cache = createCache({ capacity: 3 })
    cache.set('a', '1')
    cache.set('b', '2')
    cache.set('c', '3')

    // Reverse-order reads: c, b, a. After the loop a is the most recent.
    expect(cache.get('c')).toBe('3')
    expect(cache.get('b')).toBe('2')
    expect(cache.get('a')).toBe('1')

    // Inserting a new entry must evict c (now the least recently used).
    cache.set('d', '4')
    expect(cache.get('c')).toBeUndefined()
    expect(cache.get('a')).toBe('1')
    expect(cache.get('b')).toBe('2')
    expect(cache.get('d')).toBe('4')
    expect(cache.size()).toBe(3)
  })

  it('evicts the oldest entry when capacity is exceeded', () => {
    const cache = createCache({ capacity: 2 })
    cache.set('a', '1')
    cache.set('b', '2')
    cache.set('c', '3')
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toBe('2')
    expect(cache.get('c')).toBe('3')
    expect(cache.size()).toBe(2)
  })

  it('updating an existing key refreshes its recency', () => {
    const cache = createCache({ capacity: 2 })
    cache.set('a', '1')
    cache.set('b', '2')
    // Update a; a should be the most recent now.
    cache.set('a', 'updated')
    cache.set('c', '3')
    // b is the oldest and gets evicted; a stays.
    expect(cache.get('b')).toBeUndefined()
    expect(cache.get('a')).toBe('updated')
    expect(cache.get('c')).toBe('3')
  })

  it('clamps non-positive capacity to one', () => {
    const cache = createCache({ capacity: 0 })
    cache.set('a', '1')
    cache.set('b', '2')
    expect(cache.size()).toBe(1)
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toBe('2')
  })

  it('capacity-one eviction keeps only the most recent insert', () => {
    const cache = createCache({ capacity: 1 })
    cache.set('a', '1')
    cache.set('b', '2')
    cache.set('c', '3')
    expect(cache.get('a')).toBeUndefined()
    expect(cache.get('b')).toBeUndefined()
    expect(cache.get('c')).toBe('3')
    expect(cache.size()).toBe(1)
  })
})

describe('cacheKey', () => {
  it('includes the model name so different models do not collide', () => {
    expect(cacheKey('gpt-4', 'hello')).not.toBe(cacheKey('gpt-3', 'hello'))
  })

  it('is deterministic for identical inputs', () => {
    expect(cacheKey('m', 'q')).toBe(cacheKey('m', 'q'))
  })

  it('treats empty model as a distinct bucket', () => {
    expect(cacheKey('', 'q')).not.toBe(cacheKey('m', 'q'))
  })
})
