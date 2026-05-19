// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the ingestion worker pool. Uses a fake queue adapter so
 * all tests are deterministic with no network or database dependencies.
 */

import { afterEach, describe, expect, it } from 'vitest'

import type { ClaimOptions, QueueAdapter, QueueJob, QueueJobStatus } from './adapter.js'
import { createBackpressureChecker } from './backpressure.js'
import { type WorkerPool, createWorkerPool } from './worker-pool.js'

// -------------------------------------------------------------------
// Test helpers
// -------------------------------------------------------------------

type FakeAdapterState = {
  jobs: QueueJob[]
  claimed: string[]
  completed: string[]
  failed: string[]
  requeued: string[]
  failReasons: Map<string, string>
  claimedJobs: Map<string, QueueJob>
  pendingDepth: number
}

const createFakeAdapter = (initialJobs: ReadonlyArray<QueueJob> = []): QueueAdapter & {
  state: FakeAdapterState
} => {
  const state: FakeAdapterState = {
    jobs: [...initialJobs],
    claimed: [],
    completed: [],
    failed: [],
    requeued: [],
    failReasons: new Map(),
    claimedJobs: new Map(),
    pendingDepth: 0,
  }

  return {
    state,

    async enqueue(): Promise<QueueJob> {
      throw new Error('enqueue not implemented in fake adapter')
    },

    async claim(opts: ClaimOptions): Promise<readonly QueueJob[]> {
      const batchSize = opts.batchSize > 0 ? opts.batchSize : 1
      const result: QueueJob[] = []
      const remaining: QueueJob[] = []
      const now = new Date()

      for (const job of state.jobs) {
        if (job.status === 'pending' && result.length < batchSize) {
          const claimed: QueueJob = {
            ...job,
            status: 'processing',
            claimedBy: opts.workerId,
            claimedAt: now,
            updatedAt: now,
          }
          state.claimed.push(claimed.id)
          state.claimedJobs.set(claimed.id, claimed)
          result.push(claimed)
        } else {
          remaining.push(job)
        }
      }

      state.jobs = remaining
      return result
    },

    async complete(jobId: string): Promise<void> {
      state.completed.push(jobId)
    },

    async fail(jobId: string, reason: string): Promise<void> {
      state.failed.push(jobId)
      state.failReasons.set(jobId, reason)
    },

    async requeue(jobId: string): Promise<void> {
      state.requeued.push(jobId)
      // Return the job to the pending pool so another worker can claim it,
      // preserving the original brain ID and retry count.
      const original = state.claimedJobs.get(jobId)
      if (original !== undefined) {
        state.jobs.push({
          ...original,
          status: 'pending',
          claimedBy: undefined,
          claimedAt: undefined,
          updatedAt: new Date(),
        })
        state.claimedJobs.delete(jobId)
      }
    },

    async heartbeat(): Promise<void> {},

    async recoverStale(): Promise<number> {
      return 0
    },

    async countByStatus(): Promise<Readonly<Record<QueueJobStatus, number>>> {
      return {
        pending: state.pendingDepth,
        processing: 0,
        completed: 0,
        failed: 0,
        dead_letter: 0,
      }
    },

    async close(): Promise<void> {},
  }
}

const makeJobs = (count: number, brainId: string): ReadonlyArray<QueueJob> =>
  Array.from({ length: count }, (_, i): QueueJob => ({
    id: `job-${i}`,
    brainId,
    payload: { kind: 'raw', content: JSON.stringify({ doc: String(i) }) },
    status: 'pending',
    retryCount: 0,
    maxRetries: 3,
    createdAt: new Date(),
    updatedAt: new Date(),
  }))

const makeMultiBrainJobs = (
  perBrain: number,
  ...brainIds: ReadonlyArray<string>
): ReadonlyArray<QueueJob> => {
  let seq = 0
  return brainIds.flatMap((brainId) =>
    Array.from({ length: perBrain }, (): QueueJob => ({
      id: `job-${seq++}`,
      brainId,
      payload: { kind: 'raw' },
      status: 'pending',
      retryCount: 0,
      maxRetries: 3,
      createdAt: new Date(),
      updatedAt: new Date(),
    })),
  )
}

const waitFor = (
  predicate: () => boolean,
  timeoutMs: number = 5000,
  intervalMs: number = 10,
): Promise<void> =>
  new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs
    const tick = () => {
      if (predicate()) {
        resolve()
        return
      }
      if (Date.now() > deadline) {
        reject(new Error('waitFor timed out'))
        return
      }
      setTimeout(tick, intervalMs)
    }
    tick()
  })

// Pool instances to clean up after each test.
const pools: WorkerPool[] = []

afterEach(async () => {
  while (pools.length > 0) {
    const pool = pools.pop()
    if (pool !== undefined) await pool.stop()
  }
})

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

describe('createWorkerPool', () => {
  it('processes jobs concurrently across multiple workers', async () => {
    const adapter = createFakeAdapter(makeJobs(4, 'brain-a'))
    let maxConcurrent = 0
    let currentConcurrent = 0

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 4,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async () => {
        currentConcurrent++
        if (currentConcurrent > maxConcurrent) maxConcurrent = currentConcurrent
        await new Promise((r) => setTimeout(r, 50))
        currentConcurrent--
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(() => adapter.state.completed.length >= 4)
    await pool.stop()

    expect(maxConcurrent).toBeGreaterThanOrEqual(2)
    expect(adapter.state.completed.length).toBe(4)
  })

  it('enforces per-brain concurrency limit', async () => {
    const adapter = createFakeAdapter(makeJobs(6, 'brain-x'))
    let maxBrainConcurrent = 0
    let currentBrainConcurrent = 0

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 6,
      perBrainConcurrency: 2,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async () => {
        currentBrainConcurrent++
        if (currentBrainConcurrent > maxBrainConcurrent) {
          maxBrainConcurrent = currentBrainConcurrent
        }
        await new Promise((r) => setTimeout(r, 50))
        currentBrainConcurrent--
      },
    })
    pools.push(pool)
    pool.start()

    // Wait for all 6 jobs to complete. Jobs that hit the per-brain
    // concurrency limit are requeued (not failed), so they re-enter
    // the pending pool and are eventually processed.
    await waitFor(
      () => adapter.state.completed.length >= 6,
      30_000,
    )
    await pool.stop()

    expect(maxBrainConcurrent).toBeLessThanOrEqual(2)
    // Verify that requeue was used instead of fail for over-limit jobs.
    expect(adapter.state.requeued.length).toBeGreaterThan(0)
    expect(adapter.state.failed.length).toBe(0)
  })

  it('marks failed jobs when processor throws', async () => {
    const adapter = createFakeAdapter(makeJobs(1, 'brain-err'))

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async () => {
        throw new Error('extraction failed: unsupported mime type')
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(() => adapter.state.failed.length >= 1)
    await pool.stop()

    expect(adapter.state.failed.length).toBe(1)
    expect(adapter.state.failed[0]).toBe('job-0')
    expect(adapter.state.completed.length).toBe(0)
  })

  it('waits for inflight jobs on graceful shutdown', async () => {
    const adapter = createFakeAdapter(makeJobs(1, 'brain-slow'))
    let processingDone = false

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async () => {
        await new Promise((r) => setTimeout(r, 200))
        processingDone = true
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(() => adapter.state.claimed.length >= 1)
    await pool.stop()

    expect(processingDone).toBe(true)
  })

  it('reports accurate metrics during processing', async () => {
    const adapter = createFakeAdapter(makeJobs(3, 'brain-m'))
    let resolveProcessing: (() => void) | undefined
    const processingPromise = new Promise<void>((r) => {
      resolveProcessing = r
    })
    let blockCount = 0

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 2,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async () => {
        blockCount++
        if (blockCount <= 2) {
          await processingPromise
        }
      },
    })
    pools.push(pool)
    pool.start()

    // Wait until at least 2 jobs are claimed (and blocking).
    await waitFor(() => adapter.state.claimed.length >= 2)
    // Small delay to let the workers enter the processor.
    await new Promise((r) => setTimeout(r, 50))

    const snapshot = pool.metrics()
    expect(snapshot.activeWorkers).toBeGreaterThanOrEqual(1)

    // Unblock processing.
    resolveProcessing?.()

    await waitFor(() => adapter.state.completed.length >= 3)
    await pool.stop()

    const finalMetrics = pool.metrics()
    expect(finalMetrics.processedTotal).toBe(3)
  })

  it('processes multiple brains in parallel', async () => {
    const adapter = createFakeAdapter(makeMultiBrainJobs(2, 'brain-alpha', 'brain-beta'))
    let alphaActive = 0
    let betaActive = 0
    let bothObserved = false

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 4,
      perBrainConcurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async (job) => {
        if (job.brainId === 'brain-alpha') alphaActive++
        if (job.brainId === 'brain-beta') betaActive++
        if (alphaActive > 0 && betaActive > 0) bothObserved = true
        await new Promise((r) => setTimeout(r, 50))
        if (job.brainId === 'brain-alpha') alphaActive--
        if (job.brainId === 'brain-beta') betaActive--
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(
      () => adapter.state.completed.length >= 4,
    )
    await pool.stop()

    expect(bothObserved).toBe(true)
  })

  it('reports healthy when started and unhealthy when stopped', async () => {
    const adapter = createFakeAdapter()
    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 2,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 1000,
      processor: async () => {},
    })
    pools.push(pool)

    expect(pool.healthy()).toBe(false)

    pool.start()
    expect(pool.healthy()).toBe(true)

    await pool.stop()
    expect(pool.healthy()).toBe(false)
  })

  it('metrics track failed jobs separately', async () => {
    const succeedJobs = makeJobs(2, 'brain-ok')
    const failJobs: ReadonlyArray<QueueJob> = [
      {
        id: 'fail-1',
        brainId: 'brain-fail',
        payload: { kind: 'raw' },
        status: 'pending',
        retryCount: 0,
        maxRetries: 3,
        createdAt: new Date(),
        updatedAt: new Date(),
      },
    ]
    const adapter = createFakeAdapter([...succeedJobs, ...failJobs])

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 2,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async (job) => {
        if (job.brainId === 'brain-fail') {
          throw new Error('intentional failure')
        }
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(
      () => adapter.state.completed.length + adapter.state.failed.length >= 3,
    )
    await pool.stop()

    const finalMetrics = pool.metrics()
    expect(finalMetrics.processedTotal).toBe(2)
    expect(finalMetrics.failedTotal).toBe(1)
  })

  it('returns timed-out result when workers are stuck past shutdown timeout', async () => {
    const adapter = createFakeAdapter(makeJobs(1, 'brain-stuck'))

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 100,
      processor: async () => {
        // Simulate a stuck job that takes far longer than the shutdown timeout.
        await new Promise((r) => setTimeout(r, 5000))
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(() => adapter.state.claimed.length >= 1)

    const result = await pool.stop()
    expect(result.timedOut).toBe(true)
  })

  it('returns non-timed-out result on graceful shutdown', async () => {
    const adapter = createFakeAdapter(makeJobs(1, 'brain-fast'))

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async () => {
        await new Promise((r) => setTimeout(r, 10))
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(() => adapter.state.completed.length >= 1)

    const result = await pool.stop()
    expect(result.timedOut).toBe(false)
  })

  it('recovers from worker crash and continues processing', async () => {
    let crashCount = 0
    const jobIds: string[] = []

    // First job will cause a crash in the worker loop wrapper, second
    // should be processed after recovery.
    const jobs: QueueJob[] = [
      {
        id: 'crash-job',
        brainId: 'brain-crash',
        payload: { kind: 'raw' },
        status: 'pending',
        retryCount: 0,
        maxRetries: 3,
        createdAt: new Date(),
        updatedAt: new Date(),
      },
      {
        id: 'recovery-job',
        brainId: 'brain-crash',
        payload: { kind: 'raw' },
        status: 'pending',
        retryCount: 0,
        maxRetries: 3,
        createdAt: new Date(),
        updatedAt: new Date(),
      },
    ]
    const adapter = createFakeAdapter(jobs)

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      processor: async (job) => {
        if (job.id === 'crash-job' && crashCount === 0) {
          crashCount++
          // Processor errors are caught by claimAndProcess, not the
          // worker loop. This verifies the processor error path.
          throw new Error('simulated crash')
        }
        jobIds.push(job.id)
      },
    })
    pools.push(pool)
    pool.start()

    await waitFor(() => adapter.state.completed.length >= 1)
    await pool.stop()

    expect(jobIds).toContain('recovery-job')
    expect(adapter.state.failed.length).toBeGreaterThanOrEqual(1)
  })

  it('populates queueDepth in metrics from adapter', async () => {
    const adapter = createFakeAdapter()
    adapter.state.pendingDepth = 42

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 5000,
      maxQueueDepth: 1000,
      processor: async () => {},
    })
    pools.push(pool)
    pool.start()

    // Wait for the worker to idle and trigger a backpressure refresh.
    await waitFor(() => pool.metrics().queueDepth === 42, 5000)

    const snapshot = pool.metrics()
    expect(snapshot.queueDepth).toBe(42)

    await pool.stop()
  })
})

describe('resolveConcurrency and resolvePollInterval environment overrides', () => {
  const originalWorkerCount = process.env['MEMORY_WORKER_COUNT']
  const originalPollInterval = process.env['MEMORY_INGEST_WORKER_INTERVAL_MS']

  afterEach(() => {
    // Restore original environment.
    const restoreMap: Readonly<Record<string, string | undefined>> = {
      MEMORY_WORKER_COUNT: originalWorkerCount,
      MEMORY_INGEST_WORKER_INTERVAL_MS: originalPollInterval,
    }
    for (const [key, val] of Object.entries(restoreMap)) {
      if (val === undefined) {
        delete process.env[key]
      } else {
        process.env[key] = val
      }
    }
  })

  it('uses MEMORY_WORKER_COUNT env var when concurrency config is omitted', async () => {
    process.env['MEMORY_WORKER_COUNT'] = '7'
    const adapter = createFakeAdapter()
    let observedConcurrency = 0

    // Track how many workers start simultaneously.
    let concurrent = 0
    let maxConcurrent = 0
    const processingPromise = new Promise<void>(() => {})

    const pool = createWorkerPool({
      queue: adapter,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 1000,
      processor: async () => {
        concurrent++
        maxConcurrent = Math.max(maxConcurrent, concurrent)
        await processingPromise
      },
    })
    pools.push(pool)

    // The pool should have 7 idle workers based on env var.
    pool.start()
    await new Promise((r) => setTimeout(r, 50))

    const m = pool.metrics()
    observedConcurrency = m.activeWorkers + m.idleWorkers
    expect(observedConcurrency).toBe(7)

    await pool.stop()
  })

  it('config concurrency takes precedence over env var', async () => {
    process.env['MEMORY_WORKER_COUNT'] = '20'
    const adapter = createFakeAdapter()

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 3,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 1000,
      processor: async () => {},
    })
    pools.push(pool)
    pool.start()

    await new Promise((r) => setTimeout(r, 50))

    const m = pool.metrics()
    expect(m.activeWorkers + m.idleWorkers).toBe(3)

    await pool.stop()
  })

  it('uses MEMORY_INGEST_WORKER_INTERVAL_MS env var for poll interval', async () => {
    process.env['MEMORY_INGEST_WORKER_INTERVAL_MS'] = '50'
    const adapter = createFakeAdapter()
    let pollCount = 0

    // Count how often the worker polls (each poll cycle increments).
    const originalClaim = adapter.claim.bind(adapter)
    adapter.claim = async (opts: ClaimOptions) => {
      pollCount++
      return originalClaim(opts)
    }

    const pool = createWorkerPool({
      queue: adapter,
      concurrency: 1,
      shutdownTimeoutMs: 1000,
      processor: async () => {},
    })
    pools.push(pool)
    pool.start()

    // Wait 250ms -- with 50ms poll interval, expect ~5 polls.
    await new Promise((r) => setTimeout(r, 250))

    await pool.stop()

    // With 50ms poll interval over 250ms, expect at least 3 polls
    // (accounting for backpressure refresh overhead).
    expect(pollCount).toBeGreaterThanOrEqual(3)
  })

  it('ignores invalid MEMORY_WORKER_COUNT and uses default', async () => {
    process.env['MEMORY_WORKER_COUNT'] = 'not-a-number'
    const adapter = createFakeAdapter()

    const pool = createWorkerPool({
      queue: adapter,
      pollIntervalMs: 10,
      shutdownTimeoutMs: 1000,
      processor: async () => {},
    })
    pools.push(pool)
    pool.start()

    await new Promise((r) => setTimeout(r, 50))

    const m = pool.metrics()
    // Should fall back to default (4 * availableParallelism).
    expect(m.activeWorkers + m.idleWorkers).toBeGreaterThan(0)

    await pool.stop()
  })
})

describe('createBackpressureChecker', () => {
  it('detects backpressure when depth exceeds threshold', async () => {
    const adapter = createFakeAdapter()
    adapter.state.pendingDepth = 1500
    const checker = createBackpressureChecker(adapter, 1000)

    const pressured = await checker.check('')
    expect(pressured).toBe(true)
    expect(checker.isBackpressured()).toBe(true)
  })

  it('clears backpressure when depth drops below threshold', async () => {
    const adapter = createFakeAdapter()
    adapter.state.pendingDepth = 1500
    const checker = createBackpressureChecker(adapter, 1000)

    await checker.check('')
    expect(checker.isBackpressured()).toBe(true)

    adapter.state.pendingDepth = 500
    await checker.check('')
    expect(checker.isBackpressured()).toBe(false)
  })

  it('falls back to default threshold when given zero', () => {
    const adapter = createFakeAdapter()
    const checker = createBackpressureChecker(adapter, 0)
    expect(checker.maxDepth()).toBe(1000)
  })

  it('uses custom threshold when provided', () => {
    const adapter = createFakeAdapter()
    const checker = createBackpressureChecker(adapter, 500)
    expect(checker.maxDepth()).toBe(500)
  })
})
