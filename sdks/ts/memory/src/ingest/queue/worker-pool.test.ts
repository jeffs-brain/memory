// SPDX-License-Identifier: Apache-2.0

/**
 * Tests for the ingestion worker pool. Uses a fake queue adapter so
 * all tests are deterministic with no network or database dependencies.
 */

import { afterEach, describe, expect, it } from 'vitest'

import type { QueueAdapter, QueueJob } from './adapter.js'
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
  failReasons: Map<string, string>
  depth: number
}

const createFakeAdapter = (initialJobs: ReadonlyArray<QueueJob> = []): QueueAdapter & {
  state: FakeAdapterState
} => {
  const state: FakeAdapterState = {
    jobs: [...initialJobs],
    claimed: [],
    completed: [],
    failed: [],
    failReasons: new Map(),
    depth: 0,
  }

  return {
    state,

    async claim(workerId: string): Promise<QueueJob | undefined> {
      if (state.jobs.length === 0) return undefined
      const job = state.jobs.shift()
      if (job === undefined) return undefined
      const claimed: QueueJob = { ...job, status: 'running', claimedAt: new Date() }
      state.claimed.push(claimed.id)
      return claimed
    },

    async complete(jobId: string): Promise<void> {
      state.completed.push(jobId)
    },

    async fail(jobId: string, reason: string): Promise<void> {
      state.failed.push(jobId)
      state.failReasons.set(jobId, reason)
    },

    async heartbeat(): Promise<void> {},

    async depth(): Promise<number> {
      return state.depth
    },
  }
}

const makeJobs = (count: number, brainId: string): ReadonlyArray<QueueJob> =>
  Array.from({ length: count }, (_, i): QueueJob => ({
    id: `job-${i}`,
    brainId,
    payload: { doc: String(i) },
    status: 'pending',
    attempts: 0,
    createdAt: new Date(),
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
      payload: {},
      status: 'pending',
      attempts: 0,
      createdAt: new Date(),
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

    await waitFor(
      () => adapter.state.completed.length + adapter.state.failed.length >= 6,
    )
    await pool.stop()

    expect(maxBrainConcurrent).toBeLessThanOrEqual(2)
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
      () => adapter.state.completed.length + adapter.state.failed.length >= 4,
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
        payload: {},
        status: 'pending',
        attempts: 0,
        createdAt: new Date(),
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
})

describe('createBackpressureChecker', () => {
  it('detects backpressure when depth exceeds threshold', async () => {
    const adapter = createFakeAdapter()
    adapter.state.depth = 1500
    const checker = createBackpressureChecker(adapter, 1000)

    const pressured = await checker.check('')
    expect(pressured).toBe(true)
    expect(checker.isBackpressured()).toBe(true)
  })

  it('clears backpressure when depth drops below threshold', async () => {
    const adapter = createFakeAdapter()
    adapter.state.depth = 1500
    const checker = createBackpressureChecker(adapter, 1000)

    await checker.check('')
    expect(checker.isBackpressured()).toBe(true)

    adapter.state.depth = 500
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
