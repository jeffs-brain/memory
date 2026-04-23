export type BatchRunnerOptions<TBatch, TResult> = {
  readonly batches: readonly TBatch[]
  readonly parallelism: number
  readonly worker: (batch: TBatch, index: number) => Promise<TResult>
  readonly onError?: (error: unknown, batch: TBatch, index: number) => void
}

export const runBatches = async <TBatch, TResult>(
  options: BatchRunnerOptions<TBatch, TResult>,
): Promise<readonly (TResult | undefined)[]> => {
  const { batches, parallelism, worker, onError } = options
  if (batches.length === 0) return []

  const limit = Math.max(1, Math.floor(parallelism))
  const results: Array<TResult | undefined> = new Array(batches.length)
  let cursor = 0

  const lanes = Array.from({ length: Math.min(limit, batches.length) }, async () => {
    for (;;) {
      const index = cursor
      cursor += 1
      if (index >= batches.length) return
      const batch = batches[index]
      if (batch === undefined) continue
      try {
        results[index] = await worker(batch, index)
      } catch (error) {
        onError?.(error, batch, index)
        results[index] = undefined
      }
    }
  })

  await Promise.all(lanes)
  return results
}
