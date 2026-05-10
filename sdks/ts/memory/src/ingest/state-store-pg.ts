// SPDX-License-Identifier: Apache-2.0

/**
 * PostgreSQL-backed pipeline state store.
 *
 * Stores pipeline state in a `memory.pipeline_state` table. Designed for
 * hosted deployments where multiple ingestion workers share state via a
 * central database rather than the brain's file-based Store.
 *
 * Uses a structural `PgSql` interface to avoid a hard dependency on the
 * `postgres` driver package.
 */

import type { PipelineStage, PipelineStateEntry, PipelineStateStore } from './state-store.js'

/**
 * Minimal SQL execution interface. Accepts a tagged template and returns
 * an array of row objects. Structurally compatible with the `postgres`
 * package's Sql type.
 */
export type PgSql = {
  <T = unknown>(strings: TemplateStringsArray, ...values: unknown[]): Promise<ReadonlyArray<T>>
  begin<T>(fn: (sql: PgSql) => Promise<T>): Promise<T>
  unsafe<T = unknown>(query: string, params?: unknown[]): Promise<ReadonlyArray<T>>
}

export type PostgresPipelineStateStoreOptions = {
  readonly sql: PgSql
  readonly schema?: string
}

type StateRow = {
  document_hash: string
  brain_id: string
  stage: string
  retry_count: number
  last_error: string | null
  created_at: Date
  updated_at: Date
  completed_at: Date | null
}

const SCHEMA_NAME_PATTERN = /^[a-z_][a-z0-9_]*$/

const rowToEntry = (row: StateRow): PipelineStateEntry => ({
  documentHash: row.document_hash,
  brainId: row.brain_id,
  stage: row.stage as PipelineStage,
  retryCount: row.retry_count,
  ...(row.last_error !== null ? { lastError: row.last_error } : {}),
  createdAt: row.created_at,
  updatedAt: row.updated_at,
  ...(row.completed_at !== null ? { completedAt: row.completed_at } : {}),
})

/**
 * PostgreSQL implementation of PipelineStateStore.
 *
 * Expects the `memory.pipeline_state` table to exist (see migration
 * `0005_pipeline_state.sql`). Uses upsert semantics for Set so that
 * concurrent calls for the same document hash are safe.
 */
export class PostgresPipelineStateStore implements PipelineStateStore {
  private readonly sql: PgSql
  private readonly table: string

  constructor(opts: PostgresPipelineStateStoreOptions) {
    this.sql = opts.sql
    const schema = opts.schema ?? 'memory'
    if (!SCHEMA_NAME_PATTERN.test(schema)) {
      throw new Error(
        `Invalid schema name "${schema}": must match ^[a-z_][a-z0-9_]*$`,
      )
    }
    this.table = `${schema}.pipeline_state`
  }

  async get(documentHash: string): Promise<PipelineStateEntry | undefined> {
    const rows = await this.sql.unsafe<StateRow>(
      `SELECT document_hash, brain_id, stage, retry_count, last_error,
              created_at, updated_at, completed_at
         FROM ${this.table}
        WHERE document_hash = $1
        LIMIT 1`,
      [documentHash],
    )
    const row = rows[0]
    if (row === undefined) return undefined
    return rowToEntry(row)
  }

  async set(entry: PipelineStateEntry): Promise<void> {
    await this.sql.unsafe(
      `INSERT INTO ${this.table}
         (document_hash, brain_id, stage, retry_count, last_error, created_at, updated_at, completed_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       ON CONFLICT (document_hash) DO UPDATE SET
         brain_id = EXCLUDED.brain_id,
         stage = EXCLUDED.stage,
         retry_count = EXCLUDED.retry_count,
         last_error = EXCLUDED.last_error,
         updated_at = EXCLUDED.updated_at,
         completed_at = EXCLUDED.completed_at`,
      [
        entry.documentHash,
        entry.brainId,
        entry.stage,
        entry.retryCount,
        entry.lastError ?? null,
        entry.createdAt,
        entry.updatedAt,
        entry.completedAt ?? null,
      ],
    )
  }

  async listIncomplete(brainId: string): Promise<readonly PipelineStateEntry[]> {
    const rows = await this.sql.unsafe<StateRow>(
      `SELECT document_hash, brain_id, stage, retry_count, last_error,
              created_at, updated_at, completed_at
         FROM ${this.table}
        WHERE brain_id = $1
          AND stage NOT IN ('completed', 'failed')
        ORDER BY created_at ASC`,
      [brainId],
    )
    return rows.map(rowToEntry)
  }

  async delete(documentHash: string): Promise<void> {
    await this.sql.unsafe(
      `DELETE FROM ${this.table} WHERE document_hash = $1`,
      [documentHash],
    )
  }
}
