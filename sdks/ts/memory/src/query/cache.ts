// SPDX-License-Identifier: Apache-2.0

/**
 * O(1) LRU cache keyed on strings with string values. Backed by a native
 * Map which preserves insertion order, so eviction of the least recently
 * used entry is the first key returned by `keys().next()`. Reads promote
 * the hit to the most recent position by delete + re-insert.
 *
 * Mirrors the bookkeeping semantics of go/query/cache.go without
 * the SHA-256 hashing step: TypeScript callers hash via `cacheKey`
 * before insertion when they want a deterministic, prompt-versioned
 * key.
 */

export type Cache = {
  get(key: string): string | undefined
  set(key: string, value: string): void
  size(): number
}

export type CacheOptions = {
  /** Maximum number of entries retained before LRU eviction kicks in. */
  capacity: number
}

/**
 * createCache builds a bounded LRU. A capacity of zero or below is
 * clamped to one so the cache always holds at least the most recent
 * insertion, matching the Go implementation.
 */
export function createCache(opts: CacheOptions): Cache {
  const capacity = opts.capacity > 0 ? opts.capacity : 1
  const store = new Map<string, string>()

  return {
    get(key) {
      const v = store.get(key)
      if (v === undefined) return undefined
      // Move to end (most recently used).
      store.delete(key)
      store.set(key, v)
      return v
    },
    set(key, value) {
      if (store.has(key)) store.delete(key)
      store.set(key, value)
      while (store.size > capacity) {
        const oldest = store.keys().next().value
        if (oldest === undefined) break
        store.delete(oldest)
      }
    },
    size: () => store.size,
  }
}

/**
 * cacheKey packs the model identifier and the normalised query into a
 * deterministic key. Callers typically pass the model name the
 * distiller is targeting so the same query against a different model
 * does not collide.
 */
export function cacheKey(model: string, query: string): string {
  return `${model}::${query}`
}
