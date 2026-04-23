export const DEFAULT_SHARED_RERANK_CONCURRENCY = 4

class SharedConcurrencyGate {
  private active = 0
  private readonly queue: Array<() => void> = []

  constructor(private readonly limit: number) {}

  async withPermit<T>(task: () => Promise<T>): Promise<T> {
    if (this.active >= this.limit) {
      await new Promise<void>((resolve) => {
        this.queue.push(resolve)
      })
    }

    this.active += 1
    try {
      return await task()
    } finally {
      this.active -= 1
      const next = this.queue.shift()
      next?.()
    }
  }
}

const sharedGates = new Map<number, SharedConcurrencyGate>()

const normaliseConcurrencyCap = (value: number | undefined): number => {
  const parsed = Math.floor(value ?? DEFAULT_SHARED_RERANK_CONCURRENCY)
  return parsed > 0 ? parsed : DEFAULT_SHARED_RERANK_CONCURRENCY
}

export const runWithSharedRerankConcurrency = <T>(
  task: () => Promise<T>,
  concurrencyCap?: number,
): Promise<T> => {
  const limit = normaliseConcurrencyCap(concurrencyCap)
  const existing = sharedGates.get(limit)
  if (existing !== undefined) {
    return existing.withPermit(task)
  }

  const created = new SharedConcurrencyGate(limit)
  sharedGates.set(limit, created)
  return created.withPermit(task)
}
