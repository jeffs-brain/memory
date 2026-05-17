// SPDX-License-Identifier: Apache-2.0

/**
 * Barrel export for the ingest queue module.
 */

export {
  type QueueJobStatus,
  type QueueJobPayload,
  type QueueJob,
  type EnqueueInput,
  type ClaimOptions,
  type QueueAdapter,
  VALID_STATUSES,
  DEFAULT_MAX_RETRIES,
  DEFAULT_BATCH_SIZE,
  DEFAULT_HEARTBEAT_INTERVAL_MS,
  DEFAULT_STALE_THRESHOLD_MS,
  DEFAULT_NOTIFY_CHANNEL,
  BACKOFF_BASE_DELAY_MS,
  BACKOFF_JITTER_MIN,
  BACKOFF_JITTER_MAX,
  computeBackoff,
  validateIdentifier,
  advisoryLockKey,
} from './types.js'

export { createPostgresQueue, type PostgresQueueOptions, type PgClient, type PgListenClient } from './postgres.js'
