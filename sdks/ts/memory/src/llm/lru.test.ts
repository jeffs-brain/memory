// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { LRUCache, SingleFlight } from './lru.js'

describe('LRUCache', () => {
  it('rejects capacities of zero or less', () => {
    expect(() => new LRUCache<string, number>(0)).toThrow()
    expect(() => new LRUCache<string, number>(-1)).toThrow()
  })

  it('stores and retrieves values', () => {
    const c = new LRUCache<string, number>(3)
    c.set('a', 1)
    expect(c.get('a')).toBe(1)
    expect(c.size).toBe(1)
  })

  it('evicts the least recently used entry when at capacity', () => {
    const c = new LRUCache<string, number>(3)
    c.set('a', 1)
    c.set('b', 2)
    c.set('c', 3)
    c.set('d', 4)
    expect(c.has('a')).toBe(false)
    expect(Array.from(c.keys())).toEqual(['b', 'c', 'd'])
  })

  it('refreshes recency on a hit', () => {
    const c = new LRUCache<string, number>(3)
    c.set('a', 1)
    c.set('b', 2)
    c.set('c', 3)
    // touch 'a' so it becomes most recent
    expect(c.get('a')).toBe(1)
    c.set('d', 4)
    // 'b' is now LRU and should be evicted, not 'a'
    expect(c.has('b')).toBe(false)
    expect(c.has('a')).toBe(true)
    expect(Array.from(c.keys())).toEqual(['c', 'a', 'd'])
  })

  it('updates value and recency when setting an existing key', () => {
    const c = new LRUCache<string, number>(2)
    c.set('a', 1)
    c.set('b', 2)
    c.set('a', 10)
    c.set('c', 3)
    expect(c.has('b')).toBe(false)
    expect(c.get('a')).toBe(10)
  })
})

describe('SingleFlight', () => {
  it('collapses concurrent calls for the same key into one invocation', async () => {
    const sf = new SingleFlight<string, number>()
    let callCount = 0
    const fn = async (): Promise<number> => {
      callCount++
      await new Promise((resolve) => setTimeout(resolve, 5))
      return 42
    }
    const [a, b, c] = await Promise.all([sf.do('k', fn), sf.do('k', fn), sf.do('k', fn)])
    expect(a).toBe(42)
    expect(b).toBe(42)
    expect(c).toBe(42)
    expect(callCount).toBe(1)
  })

  it('treats different keys as independent', async () => {
    const sf = new SingleFlight<string, number>()
    const seen: string[] = []
    const make = (k: string) => async (): Promise<number> => {
      seen.push(k)
      return k.length
    }
    const [a, b] = await Promise.all([sf.do('one', make('one')), sf.do('two', make('two'))])
    expect(a).toBe(3)
    expect(b).toBe(3)
    expect(seen.sort()).toEqual(['one', 'two'])
  })

  it('clears inflight entries once the promise settles', async () => {
    const sf = new SingleFlight<string, number>()
    await sf.do('k', async () => 1)
    expect(sf.pending).toBe(0)
    let calls = 0
    await sf.do('k', async () => {
      calls++
      return 2
    })
    expect(calls).toBe(1)
  })
})
