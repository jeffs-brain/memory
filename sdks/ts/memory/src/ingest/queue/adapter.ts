// SPDX-License-Identifier: Apache-2.0

/**
 * Queue adapter contract for the ingestion worker pool. Defines the
 * interface that P3-1 (PostgreSQL queue) will implement. The worker
 * pool is developed against this contract so both can land
 * independently.
 */

/**
 * Processing state of a queued ingestion job.
 */
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed'

/**
 * A single ingestion pipeline job in the queue. The brainId field
 * drives per-brain concurrency limiting.
 */
export type QueueJob = {
  readonly id: string
  readonly brainId: string
  readonly payload: Readonly<Record<string, unknown>>
  readonly status: JobStatus
  readonly attempts: number
  readonly createdAt: Date
  readonly claimedAt?: Date
}

/**
 * Abstracts the queue storage backend. P3-1 will provide a
 * PostgreSQL-backed implementation using FOR UPDATE SKIP LOCKED.
 */
export type QueueAdapter = {
  /**
   * Atomically selects and locks the next available job for
   * processing. Returns undefined when no jobs are available.
   */
  claim(workerId: string): Promise<QueueJob | undefined>

  /**
   * Marks a job as successfully processed.
   */
  complete(jobId: string): Promise<void>

  /**
   * Marks a job as failed with the given reason.
   */
  fail(jobId: string, reason: string): Promise<void>

  /**
   * Extends the claim lease for an in-progress job so stale
   * detection does not reclaim it prematurely.
   */
  heartbeat(jobId: string): Promise<void>

  /**
   * Returns the number of pending jobs in the queue, optionally
   * scoped to a specific brain. Pass an empty string for global depth.
   */
  depth(brainId: string): Promise<number>
}

/**
 * Logging contract for the pool. Callers inject a concrete
 * implementation; the pool never writes to stdout/stderr directly.
 */
export type PoolLogger = {
  debug(msg: string, meta?: Readonly<Record<string, unknown>>): void
  info(msg: string, meta?: Readonly<Record<string, unknown>>): void
  warn(msg: string, meta?: Readonly<Record<string, unknown>>): void
  error(msg: string, meta?: Readonly<Record<string, unknown>>): void
}

/**
 * Logger that discards all messages. Used when the caller does not
 * provide one.
 */
export const noopPoolLogger: PoolLogger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
}
