// SPDX-License-Identifier: Apache-2.0

/**
 * Postgres-backed migration state backend for the BLAKE3 hash migrator.
 * Uses the `memory.blake3_migration_state` singleton table created by
 * migration 0005_blake3_migration_state.sql.
 *
 * This backend replaces the default file-based JSON sidecar approach
 * when running against a PostgresStore, providing atomic state updates
 * and compatibility with concurrent migration workers.
 */

import type { PgSql } from './store.js'
import type { MigrationState, MigrationStateBackend } from '@jeffs-brain/memory/ingest'

type Blake3MigrationRow = {
  cursor: string
  migrated: number
  total: number
}

/**
 * Creates a MigrationStateBackend backed by the
 * `memory.blake3_migration_state` Postgres table. Requires migration
 * 0005 to have been applied.
 */
export const createPostgresMigrationStateBackend = (sql: PgSql): MigrationStateBackend => ({
  load: async (): Promise<MigrationState> => {
    const rows = await sql<Blake3MigrationRow>`
      SELECT cursor, migrated, total
      FROM memory.blake3_migration_state
      WHERE id = 1
    `
    const row = rows[0]
    if (row === undefined) {
      return { cursor: '', migrated: 0, total: 0 }
    }
    return {
      cursor: row.cursor,
      migrated: row.migrated,
      total: row.total,
    }
  },

  save: async (state: MigrationState): Promise<void> => {
    await sql`
      UPDATE memory.blake3_migration_state
      SET cursor = ${state.cursor},
          migrated = ${state.migrated},
          total = ${state.total},
          updated_at = now()
      WHERE id = 1
    `
  },
})
