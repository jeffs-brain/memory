// SPDX-License-Identifier: Apache-2.0

/**
 * Map-based LRU cache. Relies on the Map preserving insertion order:
 * deleting-then-re-setting a key bumps it to the tail. Eviction walks
 * the first key (least recently used) when the cache exceeds capacity.
 *
 * Not concurrency-safe across isolates, but every JavaScript runtime
 * targeted here is single-threaded per worker.
 */
export class LRUCache<K, V> {
  private readonly store = new Map<K, V>()

  constructor(readonly capacity: number) {
    if (capacity <= 0) {
      throw new Error(`LRUCache capacity must be > 0, got ${capacity}`)
    }
  }

  get size(): number {
    return this.store.size
  }

  get(key: K): V | undefined {
    const val = this.store.get(key)
    if (val === undefined) return undefined
    // Refresh recency: delete + re-set so this key becomes tail.
    this.store.delete(key)
    this.store.set(key, val)
    return val
  }

  has(key: K): boolean {
    return this.store.has(key)
  }

  set(key: K, value: V): void {
    if (this.store.has(key)) {
      this.store.delete(key)
    }
    this.store.set(key, value)
    while (this.store.size > this.capacity) {
      const first = this.store.keys().next()
      if (first.done) break
      this.store.delete(first.value)
    }
  }

  delete(key: K): boolean {
    return this.store.delete(key)
  }

  clear(): void {
    this.store.clear()
  }

  /** Iterate in LRU-to-MRU order. Used by tests to assert ordering. */
  *keys(): IterableIterator<K> {
    yield* this.store.keys()
  }
}

/**
 * Single-flight deduplication. Concurrent callers passing the same
 * key share a single invocation of {@link fn}; the second caller
 * receives the already-pending promise. Once the promise settles the
 * entry is cleared so future calls re-invoke.
 */
export class SingleFlight<K, V> {
  private readonly inflight = new Map<K, Promise<V>>()

  do(key: K, fn: () => Promise<V>): Promise<V> {
    const existing = this.inflight.get(key)
    if (existing !== undefined) return existing
    const promise = (async () => {
      try {
        return await fn()
      } finally {
        this.inflight.delete(key)
      }
    })()
    this.inflight.set(key, promise)
    return promise
  }

  get pending(): number {
    return this.inflight.size
  }
}
