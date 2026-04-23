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
    const value = this.store.get(key)
    if (value === undefined) return undefined
    this.store.delete(key)
    this.store.set(key, value)
    return value
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

  *keys(): IterableIterator<K> {
    yield* this.store.keys()
  }
}

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
