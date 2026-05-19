// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for the ingest queue abstraction. Both the PostgreSQL
 * adapter and any future SQLite adapter implement the same QueueAdapter
 * contract so the pipeline code is storage-agnostic.
 */

import type { Logger } from '../../llm/index.js'

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

/**
 * Validate a status string at runtime and return the typed value.
 * Throws if the status is not a recognised QueueJobStatus.
 */
export const parseJobStatus = (raw: string): QueueJobStatus => {
  if (!VALID_STATUSES.has(raw as QueueJobStatus)) {
    throw new Error(`ingest: unrecognised job status: ${raw}`)
  }
  return raw as QueueJobStatus
}

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

/** Default heartbeat interval in milliseconds (30 seconds). */
export const DEFAULT_HEARTBEAT_INTERVAL_MS = 30_000

/** Default stale threshold in milliseconds (5 minutes). */
export const DEFAULT_STALE_THRESHOLD_MS = 300_000

/** Default LISTEN/NOTIFY channel name. */
export const DEFAULT_NOTIFY_CHANNEL = 'ingest_queue_new_job'

/** Base delay for exponential retry backoff (1 second). */
export const BACKOFF_BASE_DELAY_MS = 1_000

/** Lower jitter multiplier bound (inclusive). */
export const BACKOFF_JITTER_MIN = 0.5

/** Upper jitter multiplier bound (exclusive). */
export const BACKOFF_JITTER_MAX = 1.5

/**
 * Compute the next retry time using exponential backoff with jitter:
 * baseDelay * 2^retryCount * random(0.5, 1.5).
 */
export const computeBackoff = (retryCount: number): Date => {
  const multiplier = Math.pow(2, retryCount)
  const jitter = BACKOFF_JITTER_MIN + Math.random() * (BACKOFF_JITTER_MAX - BACKOFF_JITTER_MIN)
  const delayMs = BACKOFF_BASE_DELAY_MS * multiplier * jitter
  return new Date(Date.now() + delayMs)
}

/**
 * Validate a SQL identifier against injection. Only [a-zA-Z0-9_] are
 * permitted. This is a path traversal defence for schema and table names.
 */
export const validateIdentifier = (s: string): void => {
  if (s === '') {
    throw new Error('ingest: identifier must not be empty')
  }
  if (!/^[a-zA-Z0-9_]+$/.test(s)) {
    throw new Error(`ingest: identifier contains invalid characters: ${s}`)
  }
}

/**
 * Compute a stable advisory lock key from a brain ID using FNV-1a.
 * Returns a signed BigInt (int64) matching Go's int64(fnv.Sum64()) so
 * Go and TS workers mutually exclude each other on the same brain.
 *
 * PostgreSQL pg_try_advisory_lock accepts a signed bigint. Go's
 * hash/fnv returns uint64 then casts to int64, which wraps values
 * above 2^63-1 to negative. We replicate that wrapping here.
 */
export const advisoryLockKey = (brainId: string): bigint => {
  // FNV-1a 64-bit (unsigned computation)
  let hash = 0xcbf29ce484222325n
  const prime = 0x100000001b3n
  for (let i = 0; i < brainId.length; i++) {
    hash ^= BigInt(brainId.charCodeAt(i))
    hash = (hash * prime) & 0xffffffffffffffffn
  }
  // Convert to signed int64 to match Go's int64(uint64) cast.
  const signBit = 1n << 63n
  return hash >= signBit ? hash - (1n << 64n) : hash
}

/**
 * Environment variable name for the PostgreSQL connection URL.
 * Read by createPostgresQueueFromEnv to configure the adapter.
 */
export const ENV_POSTGRES_URL = 'MEMORY_POSTGRES_URL'

/**
 * Environment variable name for the worker poll interval in milliseconds.
 * Defaults to the heartbeat interval when not set.
 */
export const ENV_INGEST_WORKER_INTERVAL_MS = 'MEMORY_INGEST_WORKER_INTERVAL_MS'

/** Re-export Logger for convenience so consumers do not need llm imports. */
export type { Logger }
