// SPDX-License-Identifier: Apache-2.0

/**
 * Queue adapter types for the ingestion worker pool. These types mirror
 * P3-1's canonical definitions in types.ts. When P3-1 and P3-2 merge,
 * this file is replaced by P3-1's types.ts with zero interface changes.
 *
 * Re-exports the shared Logger from llm/types.js so the worker pool
 * uses the same logging contract as the rest of the SDK.
 */

import type { Logger } from '../../llm/index.js'
import { noopLogger } from '../../llm/index.js'

/** Lifecycle state of an ingest queue job. */
export type QueueJobStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'dead_letter'

/** Set of valid job statuses for O(1) membership checks. */
export const VALID_STATUSES: ReadonlySet<QueueJobStatus> = new Set([
  'pending',
  'processing',
  'completed',
  'failed',
  'dead_letter',
])

/** Describes what the queue job should process. */
export type QueueJobPayload = {
  readonly kind: 'file' | 'url' | 'raw'
  readonly path?: string
  readonly url?: string
  readonly content?: string
  readonly title?: string
  readonly mime?: string
}

/** A single ingest queue entry with its full metadata. */
export type QueueJob = {
  readonly id: string
  readonly brainId: string
  readonly status: QueueJobStatus
  readonly payload: QueueJobPayload
  readonly retryCount: number
  readonly maxRetries: number
  readonly error?: string
  readonly claimedBy?: string
  readonly claimedAt?: Date
  readonly lastHeartbeat?: Date
  readonly nextRetryAt?: Date
  readonly createdAt: Date
  readonly updatedAt: Date
  readonly completedAt?: Date
  readonly metadata?: Readonly<Record<string, string>>
  readonly groupId?: string
  readonly idempotencyKey?: string
}

/** Parameters for creating a new queue job. */
export type EnqueueInput = {
  readonly brainId: string
  readonly payload: QueueJobPayload
  readonly maxRetries?: number
  readonly idempotencyKey?: string
  readonly groupId?: string
  readonly metadata?: Readonly<Record<string, string>>
}

/** Configures how a worker claims jobs from the queue. */
export type ClaimOptions = {
  readonly batchSize: number
  readonly workerId: string
}

/**
 * QueueAdapter defines the contract for an ingest queue backend.
 * Both PostgreSQL and SQLite implementations satisfy this type so
 * the pipeline code is storage-agnostic.
 */
export type QueueAdapter = {
  /** Enqueue a new job. Returns the created (or existing idempotent) job. */
  enqueue(input: EnqueueInput): Promise<QueueJob>

  /** Claim up to batchSize pending jobs for the given worker. */
  claim(opts: ClaimOptions): Promise<readonly QueueJob[]>

  /** Refresh the liveness timestamp for a processing job. */
  heartbeat(jobId: string): Promise<void>

  /** Mark a job as successfully completed. */
  complete(jobId: string, result?: Readonly<Record<string, string>>): Promise<void>

  /**
   * Record a failure against a job. When retryable is true and retries
   * remain, the job returns to pending with exponential backoff. Otherwise
   * it moves to dead_letter.
   */
  fail(jobId: string, error: string, retryable: boolean): Promise<void>

  /**
   * Return a claimed job to pending status WITHOUT incrementing the
   * retry count. Use when a job cannot be processed due to transient
   * conditions (e.g. per-brain concurrency limit reached) rather than
   * an actual processing failure.
   */
  requeue(jobId: string): Promise<void>

  /** Reset stale processing jobs to pending. Returns the count recovered. */
  recoverStale(staleThresholdMs: number): Promise<number>

  /** Count jobs grouped by status, optionally filtered by brain. */
  countByStatus(brainId?: string): Promise<Readonly<Record<QueueJobStatus, number>>>

  /** Release resources held by the adapter (connections, timers, listeners). */
  close(): Promise<void>
}

/** Default maximum retry count when the caller does not specify one. */
export const DEFAULT_MAX_RETRIES = 3

/** Default claim batch size when the caller does not specify one. */
export const DEFAULT_BATCH_SIZE = 1

/** Re-export Logger and noopLogger for convenience. */
export type { Logger }
export { noopLogger }
