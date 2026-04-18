/**
 * Bounded-parallelism batch runner used by the LLM reranker. Callers
 * supply the batched payloads and a per-batch worker; the runner
 * limits how many batches may be in flight at once and waits for all
 * of them to settle before returning. Errors are surfaced per batch
 * via the onError hook so the caller can count failures without
 * stopping the pipeline.
 */

export type BatchRunnerOptions<TBatch, TResult> = {
  /** Input batches. Each batch is passed to worker() unchanged. */
  batches: readonly TBatch[]
  /** Maximum number of worker() invocations in flight at once. */
  parallelism: number
  /** Called for each batch. May throw; errors are routed to onError. */
  worker: (batch: TBatch, index: number) => Promise<TResult>
  /**
   * Invoked when a batch's worker() rejects. Allows the caller to
   * track the error without failing the entire run.
   */
  onError?: (err: unknown, batch: TBatch, index: number) => void
}

/**
 * runBatches executes every batch through worker() with at most
 * `parallelism` invocations in flight at once. Results are returned in
 * input order; positions that rejected hold `undefined`. The runner
 * never throws: error handling is opt-in via onError.
 */
export async function runBatches<TBatch, TResult>(
  opts: BatchRunnerOptions<TBatch, TResult>,
): Promise<readonly (TResult | undefined)[]> {
  const { batches, parallelism, worker, onError } = opts
  if (batches.length === 0) return []
  const limit = Math.max(1, Math.floor(parallelism))
  const out: (TResult | undefined)[] = new Array(batches.length)
  let cursor = 0

  const lanes: Promise<void>[] = []
  const laneCount = Math.min(limit, batches.length)
  for (let lane = 0; lane < laneCount; lane++) {
    lanes.push(
      (async () => {
        while (true) {
          const idx = cursor++
          if (idx >= batches.length) return
          const batch = batches[idx]
          if (batch === undefined) continue
          try {
            out[idx] = await worker(batch, idx)
          } catch (err) {
            onError?.(err, batch, idx)
            out[idx] = undefined
          }
        }
      })(),
    )
  }
  await Promise.all(lanes)
  return out
}
