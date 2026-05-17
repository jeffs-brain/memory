// SPDX-License-Identifier: Apache-2.0

/**
 * PostgreSQL-backed ingest queue adapter using FOR UPDATE SKIP LOCKED
 * for safe multi-worker job claiming. Supports LISTEN/NOTIFY for
 * immediate dispatch and heartbeat-based stale job recovery.
 */

import { noopLogger } from '../../llm/index.js'
import {
  BACKOFF_BASE_DELAY_MS,
  BACKOFF_JITTER_MAX,
  BACKOFF_JITTER_MIN,
  DEFAULT_BATCH_SIZE,
  DEFAULT_HEARTBEAT_INTERVAL_MS,
  DEFAULT_MAX_RETRIES,
  DEFAULT_NOTIFY_CHANNEL,
  DEFAULT_STALE_THRESHOLD_MS,
  type ClaimOptions,
  type EnqueueInput,
  type Logger,
  type QueueAdapter,
  type QueueJob,
  type QueueJobPayload,
  type QueueJobStatus,
  advisoryLockKey,
  validateIdentifier,
} from './types.js'

/**
 * Minimal SQL client interface. Structurally typed so any PostgreSQL
 * driver that supports parameterised queries can be used (e.g. `pg`,
 * `postgres`, `@neondatabase/serverless`).
 */
export type PgClient = {
  query<R = Record<string, unknown>>(
    text: string,
    values?: ReadonlyArray<unknown>,
  ): Promise<{ readonly rows: readonly R[]; readonly rowCount: number }>
}

/**
 * Extended PgClient that supports LISTEN/NOTIFY. Optional -- when
 * provided, the adapter subscribes for immediate wake on new jobs.
 */
export type PgListenClient = PgClient & {
  on(event: 'notification', listener: (msg: { readonly channel: string; readonly payload?: string }) => void): void
  query<R = Record<string, unknown>>(
    text: string,
    values?: ReadonlyArray<unknown>,
  ): Promise<{ readonly rows: readonly R[]; readonly rowCount: number }>
}

/** Configuration for the PostgreSQL queue adapter. */
export type PostgresQueueOptions = {
  /** A pg-compatible client for queries. Required. */
  readonly client: PgClient
  /**
   * An optional dedicated client for LISTEN/NOTIFY. When provided,
   * the adapter subscribes for immediate wake on new jobs. This should
   * be a separate, long-lived connection (not from a pool).
   */
  readonly listenClient?: PgListenClient
  /** PostgreSQL schema name. Defaults to "public". */
  readonly schema?: string
  /** Queue table name. Defaults to "ingest_queue". */
  readonly tableName?: string
  /** Heartbeat refresh interval in milliseconds. Defaults to 30000. */
  readonly heartbeatIntervalMs?: number
  /** Stale job threshold in milliseconds. Defaults to 300000 (5 min). */
  readonly staleThresholdMs?: number
  /** Structured logger. Defaults to silent. */
  readonly logger?: Logger
  /** LISTEN/NOTIFY channel name. Defaults to "ingest_queue_new_job". */
  readonly notifyChannel?: string
}

/** Row shape returned by PostgreSQL RETURNING clauses. */
type QueueRow = {
  readonly id: string
  readonly brain_id: string
  readonly status: string
  readonly payload: QueueJobPayload | string
  readonly retry_count: number
  readonly max_retries: number
  readonly error: string | null
  readonly claimed_by: string | null
  readonly claimed_at: string | Date | null
  readonly last_heartbeat: string | Date | null
  readonly next_retry_at: string | Date | null
  readonly created_at: string | Date
  readonly updated_at: string | Date
  readonly completed_at: string | Date | null
  readonly metadata: Record<string, string> | null
  readonly group_id: string | null
  readonly idempotency_key: string | null
}

/**
 * Create a PostgreSQL-backed queue adapter.
 *
 * The adapter starts a heartbeat timer immediately. Call `close()` to
 * stop the timer and release resources.
 */
export const createPostgresQueue = (opts: PostgresQueueOptions): QueueAdapter => {
  const client = opts.client
  const schema = opts.schema ?? 'public'
  const tableName = opts.tableName ?? 'ingest_queue'
  validateIdentifier(schema)
  validateIdentifier(tableName)

  const heartbeatIntervalMs = opts.heartbeatIntervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS
  const staleThresholdMs = opts.staleThresholdMs ?? DEFAULT_STALE_THRESHOLD_MS
  const log: Logger = opts.logger ?? noopLogger
  const notifyChannel = opts.notifyChannel ?? DEFAULT_NOTIFY_CHANNEL
  const qualifiedTable = `${schema}.${tableName}`

  // Notification callbacks registered via onNotify.
  const notifyCallbacks: Array<(jobId: string) => void> = []

  // Track claimed job IDs for heartbeat refresh.
  const claimedJobs = new Set<string>()
  let closed = false

  // Set up LISTEN/NOTIFY if a dedicated listen client is provided.
  if (opts.listenClient !== undefined) {
    const listenClient = opts.listenClient
    listenClient.on('notification', (msg) => {
      if (msg.channel === notifyChannel && msg.payload !== undefined) {
        for (const cb of notifyCallbacks) {
          cb(msg.payload)
        }
      }
    })
    listenClient.query(`LISTEN ${notifyChannel}`).catch((err: unknown) => {
      log.warn('ingest: LISTEN setup failed', { error: String(err) })
    })
  }

  // Heartbeat timer refreshes liveness for all claimed jobs.
  const heartbeatTimer = setInterval(() => {
    if (closed) return
    const ids = [...claimedJobs]
    for (const id of ids) {
      heartbeat(id).catch((err: unknown) => {
        log.warn('ingest: heartbeat refresh failed', { jobId: id, error: String(err) })
      })
    }
  }, heartbeatIntervalMs)

  // Prevent the timer from keeping the process alive.
  if (typeof heartbeatTimer === 'object' && 'unref' in heartbeatTimer) {
    heartbeatTimer.unref()
  }

  const ensureOpen = (): void => {
    if (closed) throw new Error('ingest: queue adapter is closed')
  }

  const rowToJob = (row: QueueRow): QueueJob => {
    const payload: QueueJobPayload =
      typeof row.payload === 'string' ? JSON.parse(row.payload) : row.payload

    const parsedMeta: Record<string, string> | undefined =
      row.metadata !== null && row.metadata !== undefined
        ? (typeof row.metadata === 'string' ? JSON.parse(row.metadata) : row.metadata)
        : undefined

    const base = {
      id: row.id,
      brainId: row.brain_id,
      status: row.status as QueueJobStatus,
      payload,
      retryCount: row.retry_count,
      maxRetries: row.max_retries,
      createdAt: new Date(row.created_at),
      updatedAt: new Date(row.updated_at),
    }

    // Build optional fields conditionally so exactOptionalPropertyTypes
    // is satisfied (properties are absent rather than set to undefined).
    const optional: Record<string, unknown> = {}
    if (row.error !== null) optional['error'] = row.error
    if (row.claimed_by !== null) optional['claimedBy'] = row.claimed_by
    if (row.claimed_at !== null) optional['claimedAt'] = new Date(row.claimed_at)
    if (row.last_heartbeat !== null) optional['lastHeartbeat'] = new Date(row.last_heartbeat)
    if (row.next_retry_at !== null) optional['nextRetryAt'] = new Date(row.next_retry_at)
    if (row.completed_at !== null) optional['completedAt'] = new Date(row.completed_at)
    if (parsedMeta !== undefined) optional['metadata'] = parsedMeta
    if (row.group_id !== null) optional['groupId'] = row.group_id
    if (row.idempotency_key !== null) optional['idempotencyKey'] = row.idempotency_key

    return { ...base, ...optional } as QueueJob
  }

  const findByIdempotencyKey = async (key: string): Promise<QueueJob | undefined> => {
    const result = await client.query<QueueRow>(
      `SELECT id, brain_id, status, payload, retry_count, max_retries, error,
              claimed_by, claimed_at, last_heartbeat, next_retry_at,
              created_at, updated_at, completed_at, metadata, group_id, idempotency_key
       FROM ${qualifiedTable}
       WHERE idempotency_key = $1
         AND status NOT IN ('dead_letter', 'completed', 'failed')
       LIMIT 1`,
      [key],
    )
    if (result.rows.length === 0) return undefined
    return rowToJob(result.rows[0]!)
  }

  const enqueue = async (input: EnqueueInput): Promise<QueueJob> => {
    ensureOpen()
    const maxRetries = input.maxRetries ?? DEFAULT_MAX_RETRIES

    // Check idempotency key first.
    if (input.idempotencyKey !== undefined) {
      const existing = await findByIdempotencyKey(input.idempotencyKey)
      if (existing !== undefined) {
        log.debug('ingest: idempotent enqueue returned existing job', {
          jobId: existing.id,
          idempotencyKey: input.idempotencyKey,
        })
        return existing
      }
    }

    const payloadJson = JSON.stringify(input.payload)
    const metadataJson = input.metadata !== undefined ? JSON.stringify(input.metadata) : null

    const result = await client.query<QueueRow>(
      `INSERT INTO ${qualifiedTable}
         (brain_id, status, payload, max_retries, metadata, group_id, idempotency_key)
       VALUES ($1, $2, $3, $4, $5, $6, $7)
       RETURNING id, brain_id, status, payload, retry_count, max_retries, error,
                 claimed_by, claimed_at, last_heartbeat, next_retry_at,
                 created_at, updated_at, completed_at, metadata, group_id, idempotency_key`,
      [
        input.brainId,
        'pending',
        payloadJson,
        maxRetries,
        metadataJson,
        input.groupId ?? null,
        input.idempotencyKey ?? null,
      ],
    )

    const job = rowToJob(result.rows[0]!)
    log.info('ingest: job enqueued', { jobId: job.id, brainId: job.brainId })
    return job
  }

  const claim = async (opts: ClaimOptions): Promise<readonly QueueJob[]> => {
    ensureOpen()
    const batchSize = opts.batchSize > 0 ? opts.batchSize : DEFAULT_BATCH_SIZE
    if (opts.workerId === '') {
      throw new Error('ingest: claim requires a non-empty worker ID')
    }

    const result = await client.query<QueueRow>(
      `UPDATE ${qualifiedTable}
       SET status = $1,
           claimed_by = $2,
           claimed_at = NOW(),
           last_heartbeat = NOW(),
           updated_at = NOW()
       WHERE id IN (
         SELECT id FROM ${qualifiedTable}
         WHERE status = 'pending'
           AND (next_retry_at IS NULL OR next_retry_at <= NOW())
         ORDER BY created_at ASC
         LIMIT $3
         FOR UPDATE SKIP LOCKED
       )
       RETURNING id, brain_id, status, payload, retry_count, max_retries, error,
                 claimed_by, claimed_at, last_heartbeat, next_retry_at,
                 created_at, updated_at, completed_at, metadata, group_id, idempotency_key`,
      ['processing', opts.workerId, batchSize],
    )

    const jobs = result.rows.map(rowToJob)

    // Track claimed jobs for heartbeat and attempt advisory locks.
    for (const job of jobs) {
      claimedJobs.add(job.id)
      const lockKey = advisoryLockKey(job.brainId)
      client.query('SELECT pg_try_advisory_lock($1)', [lockKey.toString()]).catch((err: unknown) => {
        log.debug('ingest: advisory lock acquisition failed', {
          key: lockKey.toString(),
          error: String(err),
        })
      })
    }

    if (jobs.length > 0) {
      log.info('ingest: claimed jobs', { count: jobs.length, workerId: opts.workerId })
    }

    return jobs
  }

  const heartbeat = async (jobId: string): Promise<void> => {
    ensureOpen()
    const result = await client.query(
      `UPDATE ${qualifiedTable}
       SET last_heartbeat = NOW(), updated_at = NOW()
       WHERE id = $1 AND status = $2`,
      [jobId, 'processing'],
    )
    if (result.rowCount === 0) {
      throw new Error(`ingest: heartbeat found no processing job with id ${jobId}`)
    }
  }

  const complete = async (jobId: string, result?: Readonly<Record<string, string>>): Promise<void> => {
    ensureOpen()
    const resultJson = result !== undefined ? JSON.stringify(result) : null

    const res = await client.query<{ readonly brain_id: string }>(
      `UPDATE ${qualifiedTable}
       SET status = $1,
           completed_at = NOW(),
           updated_at = NOW(),
           metadata = COALESCE($3::jsonb, metadata)
       WHERE id = $2 AND status = $4
       RETURNING brain_id`,
      ['completed', jobId, resultJson, 'processing'],
    )
    if (res.rows.length === 0) {
      throw new Error(`ingest: complete found no processing job with id ${jobId}`)
    }

    claimedJobs.delete(jobId)
    const brainId = res.rows[0]!.brain_id
    const lockKey = advisoryLockKey(brainId)
    client.query('SELECT pg_advisory_unlock($1)', [lockKey.toString()]).catch((err: unknown) => {
      log.debug('ingest: advisory lock release failed', { key: lockKey.toString(), error: String(err) })
    })

    log.info('ingest: job completed', { jobId, brainId })
  }

  const fail = async (jobId: string, error: string, retryable: boolean): Promise<void> => {
    ensureOpen()

    // Fetch current state in a transactional manner.
    const fetchResult = await client.query<{
      readonly retry_count: number
      readonly max_retries: number
      readonly brain_id: string
    }>(
      `SELECT retry_count, max_retries, brain_id
       FROM ${qualifiedTable}
       WHERE id = $1 AND status = $2`,
      [jobId, 'processing'],
    )

    if (fetchResult.rows.length === 0) {
      throw new Error(`ingest: fail found no processing job with id ${jobId}`)
    }

    const row = fetchResult.rows[0]!
    const newRetryCount = row.retry_count + 1
    const canRetry = retryable && newRetryCount < row.max_retries

    let nextStatus: QueueJobStatus
    let nextRetryAt: Date | null = null

    if (canRetry) {
      nextStatus = 'pending'
      const multiplier = Math.pow(2, newRetryCount)
      const jitter = BACKOFF_JITTER_MIN + Math.random() * (BACKOFF_JITTER_MAX - BACKOFF_JITTER_MIN)
      const delayMs = BACKOFF_BASE_DELAY_MS * multiplier * jitter
      nextRetryAt = new Date(Date.now() + delayMs)
    } else {
      nextStatus = 'dead_letter'
    }

    await client.query(
      `UPDATE ${qualifiedTable}
       SET status = $1,
           retry_count = $2,
           error = $3,
           next_retry_at = $4,
           claimed_by = NULL,
           claimed_at = NULL,
           last_heartbeat = NULL,
           updated_at = NOW()
       WHERE id = $5`,
      [nextStatus, newRetryCount, error, nextRetryAt, jobId],
    )

    claimedJobs.delete(jobId)
    const lockKey = advisoryLockKey(row.brain_id)
    client.query('SELECT pg_advisory_unlock($1)', [lockKey.toString()]).catch((err: unknown) => {
      log.debug('ingest: advisory lock release failed', { key: lockKey.toString(), error: String(err) })
    })

    log.info('ingest: job failed', {
      jobId,
      status: nextStatus,
      retryCount: newRetryCount,
      retryable: canRetry,
    })
  }

  const recoverStale = async (thresholdMs: number): Promise<number> => {
    ensureOpen()
    const threshold = thresholdMs > 0 ? thresholdMs : staleThresholdMs
    const intervalStr = `${Math.floor(threshold / 1000)} seconds`

    const result = await client.query(
      `UPDATE ${qualifiedTable}
       SET status = $1,
           claimed_by = NULL,
           claimed_at = NULL,
           last_heartbeat = NULL,
           updated_at = NOW()
       WHERE status = $2
         AND last_heartbeat < NOW() - $3::interval`,
      ['pending', 'processing', intervalStr],
    )

    if (result.rowCount > 0) {
      log.warn('ingest: recovered stale jobs', { count: result.rowCount })
    }

    return result.rowCount
  }

  const countByStatus = async (brainId?: string): Promise<Readonly<Record<QueueJobStatus, number>>> => {
    ensureOpen()

    const counts: Record<QueueJobStatus, number> = {
      pending: 0,
      processing: 0,
      completed: 0,
      failed: 0,
      dead_letter: 0,
    }

    const result =
      brainId !== undefined
        ? await client.query<{ readonly status: string; readonly count: string }>(
            `SELECT status, COUNT(*)::int as count FROM ${qualifiedTable}
             WHERE brain_id = $1 GROUP BY status`,
            [brainId],
          )
        : await client.query<{ readonly status: string; readonly count: string }>(
            `SELECT status, COUNT(*)::int as count FROM ${qualifiedTable}
             GROUP BY status`,
          )

    for (const row of result.rows) {
      const status = row.status as QueueJobStatus
      counts[status] = Number(row.count)
    }

    return counts
  }

  const close = async (): Promise<void> => {
    if (closed) return
    closed = true
    clearInterval(heartbeatTimer)
    claimedJobs.clear()
    notifyCallbacks.length = 0
    log.info('ingest: queue adapter closed')
  }

  return {
    enqueue,
    claim,
    heartbeat,
    complete,
    fail,
    recoverStale,
    countByStatus,
    close,
  }
}
