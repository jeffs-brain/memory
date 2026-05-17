// SPDX-License-Identifier: Apache-2.0

/**
 * Concurrent worker pool for processing ingestion pipeline jobs. Spawns
 * configurable parallel async loops that claim jobs from a queue adapter,
 * enforce per-brain concurrency limits, and support graceful shutdown.
 *
 * The pool reads MEMORY_WORKER_COUNT and MEMORY_INGEST_WORKER_INTERVAL_MS
 * from the environment when the corresponding config fields are not set.
 */

import { randomUUID } from 'node:crypto'
import { availableParallelism } from 'node:os'

import type { PoolLogger, QueueAdapter, QueueJob } from './adapter.js'
import { noopPoolLogger } from './adapter.js'
import { type BackpressureChecker, createBackpressureChecker } from './backpressure.js'

/**
 * Default concurrency is 4x the CPU count, following Celery/Airflow
 * convention for I/O-bound workloads where workers spend most time
 * waiting on network calls (embedding, LLM).
 */
const DEFAULT_CONCURRENCY = 4 * availableParallelism()

/**
 * Per-brain concurrency limit following the AWS fair scheduling
 * pattern for multi-tenant systems. Prevents a single large brain
 * from monopolising all workers.
 */
const DEFAULT_PER_BRAIN_CONCURRENCY = 5

/**
 * How often idle workers poll the queue for new jobs. 15 seconds
 * balances responsiveness against database load.
 */
const DEFAULT_POLL_INTERVAL_MS = 15_000

/**
 * Maximum time stop() waits for in-flight jobs before aborting.
 * 2 minutes follows Google Cloud K8s best practice for ingestion
 * pipeline drain windows.
 */
const DEFAULT_SHUTDOWN_TIMEOUT_MS = 120_000

/**
 * Environment variable that overrides worker count.
 */
const ENV_WORKER_COUNT = 'MEMORY_WORKER_COUNT'

/**
 * Environment variable that overrides poll interval in milliseconds.
 */
const ENV_POLL_INTERVAL = 'MEMORY_INGEST_WORKER_INTERVAL_MS'

/**
 * Callback invoked by the pool for each claimed job.
 */
export type JobProcessor = (job: QueueJob) => Promise<void>

/**
 * Configuration for the worker pool. All fields except queue and
 * processor are optional with documented defaults.
 */
export type WorkerPoolOptions = {
  readonly queue: QueueAdapter
  readonly processor: JobProcessor
  readonly concurrency?: number
  readonly perBrainConcurrency?: number
  readonly pollIntervalMs?: number
  readonly shutdownTimeoutMs?: number
  readonly maxQueueDepth?: number
  readonly logger?: PoolLogger
  readonly workerId?: string
}

/**
 * Point-in-time snapshot of pool health and throughput counters.
 */
export type WorkerPoolMetrics = {
  readonly activeWorkers: number
  readonly idleWorkers: number
  readonly queueDepth: number
  readonly processedTotal: number
  readonly failedTotal: number
  readonly perBrainActive: Readonly<Record<string, number>>
}

/**
 * Public surface of a running worker pool. Call start() to launch
 * workers, stop() for graceful shutdown.
 */
export type WorkerPool = {
  start(): void
  stop(): Promise<void>
  metrics(): WorkerPoolMetrics
  isBackpressured(): boolean
  healthy(): boolean
}

type ResolvedConfig = {
  readonly queue: QueueAdapter
  readonly processor: JobProcessor
  readonly concurrency: number
  readonly perBrainConcurrency: number
  readonly pollIntervalMs: number
  readonly shutdownTimeoutMs: number
  readonly maxQueueDepth: number
  readonly logger: PoolLogger
  readonly workerId: string
}

/**
 * Creates a worker pool with the given options. Workers are not started
 * until start() is called.
 *
 * Time: O(1) construction.
 * Space: O(C) where C = concurrency for worker state tracking.
 */
export const createWorkerPool = (opts: WorkerPoolOptions): WorkerPool => {
  const cfg = resolveConfig(opts)
  const brainActive = new Map<string, number>()
  const backpressure: BackpressureChecker = createBackpressureChecker(cfg.queue, cfg.maxQueueDepth)

  let processedTotal = 0
  let failedTotal = 0
  let activeWorkers = 0
  let abortController: AbortController | undefined
  let workerPromises: Promise<void>[] = []
  let stopped = false

  const acquireBrainSlot = (brainId: string): boolean => {
    const current = brainActive.get(brainId) ?? 0
    if (current >= cfg.perBrainConcurrency) {
      return false
    }
    brainActive.set(brainId, current + 1)
    return true
  }

  const releaseBrainSlot = (brainId: string): void => {
    const current = brainActive.get(brainId) ?? 0
    if (current <= 1) {
      brainActive.delete(brainId)
      return
    }
    brainActive.set(brainId, current - 1)
  }

  const pollWait = (signal: AbortSignal): Promise<void> =>
    new Promise((resolve) => {
      if (signal.aborted) {
        resolve()
        return
      }
      const timer = setTimeout(resolve, cfg.pollIntervalMs)
      const onAbort = () => {
        clearTimeout(timer)
        resolve()
      }
      signal.addEventListener('abort', onAbort, { once: true })
    })

  const claimAndProcess = async (qualifiedId: string, signal: AbortSignal): Promise<boolean> => {
    let claimed: QueueJob | undefined
    try {
      claimed = await cfg.queue.claim(qualifiedId)
    } catch (err: unknown) {
      if (signal.aborted) return false
      cfg.logger.warn('claim failed', { worker: qualifiedId, error: String(err) })
      return false
    }

    if (claimed === undefined) return false

    if (!acquireBrainSlot(claimed.brainId)) {
      cfg.logger.debug('per-brain concurrency limit reached, releasing job', {
        worker: qualifiedId,
        brainId: claimed.brainId,
      })
      try {
        await cfg.queue.fail(claimed.id, 'per-brain concurrency limit reached')
      } catch (failErr: unknown) {
        cfg.logger.error('failed to release over-limit job', {
          worker: qualifiedId,
          jobId: claimed.id,
          error: String(failErr),
        })
      }
      return true
    }

    activeWorkers++
    try {
      await cfg.processor(claimed)
      processedTotal++
      try {
        await cfg.queue.complete(claimed.id)
      } catch (completeErr: unknown) {
        cfg.logger.error('failed to mark job as completed', {
          worker: qualifiedId,
          jobId: claimed.id,
          error: String(completeErr),
        })
      }
    } catch (processErr: unknown) {
      failedTotal++
      try {
        await cfg.queue.fail(claimed.id, String(processErr))
      } catch (failErr: unknown) {
        cfg.logger.error('failed to mark job as failed', {
          worker: qualifiedId,
          jobId: claimed.id,
          error: String(failErr),
        })
      }
      cfg.logger.warn('job processing failed', {
        worker: qualifiedId,
        jobId: claimed.id,
        brainId: claimed.brainId,
        error: String(processErr),
      })
    } finally {
      activeWorkers--
      releaseBrainSlot(claimed.brainId)
    }

    return true
  }

  const runWorker = async (workerIdx: number, signal: AbortSignal): Promise<void> => {
    const qualifiedId = `${cfg.workerId}-${workerIdx}`
    cfg.logger.debug('worker started', { worker: qualifiedId })

    while (!signal.aborted) {
      const didWork = await claimAndProcess(qualifiedId, signal)
      if (!didWork && !signal.aborted) {
        await pollWait(signal)
      }
    }

    cfg.logger.debug('worker stopped', { worker: qualifiedId })
  }

  const refreshBackpressure = async (): Promise<void> => {
    try {
      await backpressure.check('')
    } catch (err: unknown) {
      cfg.logger.warn('backpressure check failed', { error: String(err) })
    }
  }

  return {
    start(): void {
      if (abortController !== undefined) return

      abortController = new AbortController()
      const { signal } = abortController

      cfg.logger.info('pool starting', {
        concurrency: cfg.concurrency,
        perBrainConcurrency: cfg.perBrainConcurrency,
        pollIntervalMs: cfg.pollIntervalMs,
        workerId: cfg.workerId,
      })

      workerPromises = Array.from({ length: cfg.concurrency }, (_, i) => runWorker(i, signal))
    },

    async stop(): Promise<void> {
      if (stopped) return
      stopped = true

      cfg.logger.info('pool stopping', { shutdownTimeoutMs: cfg.shutdownTimeoutMs })

      if (abortController === undefined) return
      abortController.abort()

      const allDone = Promise.all(workerPromises)
      const timeout = new Promise<'timeout'>((resolve) =>
        setTimeout(() => resolve('timeout'), cfg.shutdownTimeoutMs),
      )

      const outcome = await Promise.race([allDone.then(() => 'done' as const), timeout])

      if (outcome === 'timeout') {
        cfg.logger.warn('pool shutdown timed out, some workers may still be running')
      } else {
        cfg.logger.info('pool stopped gracefully')
      }
    },

    metrics(): WorkerPoolMetrics {
      const perBrainSnapshot: Record<string, number> = {}
      for (const [brainId, count] of brainActive) {
        perBrainSnapshot[brainId] = count
      }
      return {
        activeWorkers,
        idleWorkers: cfg.concurrency - activeWorkers,
        queueDepth: 0,
        processedTotal,
        failedTotal,
        perBrainActive: perBrainSnapshot,
      }
    },

    isBackpressured(): boolean {
      return backpressure.isBackpressured()
    },

    healthy(): boolean {
      return abortController !== undefined && !stopped
    },
  }
}

const resolveConfig = (opts: WorkerPoolOptions): ResolvedConfig => ({
  queue: opts.queue,
  processor: opts.processor,
  concurrency: resolveConcurrency(opts.concurrency),
  perBrainConcurrency: opts.perBrainConcurrency ?? DEFAULT_PER_BRAIN_CONCURRENCY,
  pollIntervalMs: resolvePollInterval(opts.pollIntervalMs),
  shutdownTimeoutMs: opts.shutdownTimeoutMs ?? DEFAULT_SHUTDOWN_TIMEOUT_MS,
  maxQueueDepth: opts.maxQueueDepth ?? 0,
  logger: opts.logger ?? noopPoolLogger,
  workerId: opts.workerId ?? randomUUID(),
})

const resolveConcurrency = (configured: number | undefined): number => {
  if (configured !== undefined && configured > 0) return configured

  const envVal = process.env[ENV_WORKER_COUNT]
  if (envVal !== undefined) {
    const parsed = Number.parseInt(envVal, 10)
    if (!Number.isNaN(parsed) && parsed > 0) return parsed
  }

  return DEFAULT_CONCURRENCY
}

const resolvePollInterval = (configured: number | undefined): number => {
  if (configured !== undefined && configured > 0) return configured

  const envVal = process.env[ENV_POLL_INTERVAL]
  if (envVal !== undefined) {
    const parsed = Number.parseInt(envVal, 10)
    if (!Number.isNaN(parsed) && parsed > 0) return parsed
  }

  return DEFAULT_POLL_INTERVAL_MS
}
