// SPDX-License-Identifier: Apache-2.0

/**
 * Extensible per-chunk key-value metadata for the search index.
 *
 * Stores arbitrary string metadata (ontology type, confidence scores,
 * embedding model, etc.) against chunk IDs. The composite primary key
 * (chunk_id, key) ensures at most one value per pair.
 *
 * Port of go/search/metadata.go.
 */

import type { SqlDb } from './driver.js'

/**
 * DDL for the knowledge_chunk_metadata table. Idempotent via
 * CREATE TABLE IF NOT EXISTS.
 */
export const CHUNK_METADATA_SCHEMA = `
CREATE TABLE IF NOT EXISTS knowledge_chunk_metadata (
    chunk_id TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (chunk_id, key)
);
`

/**
 * Idempotently create the knowledge_chunk_metadata table. Called
 * during index construction alongside the other DDL statements.
 */
export function createChunkMetadataTable(db: SqlDb): void {
  db.exec(CHUNK_METADATA_SCHEMA)
}

/**
 * Upsert key-value metadata for a chunk. Empty keys or values are
 * rejected. Existing entries for the same (chunk_id, key) pair are
 * overwritten via ON CONFLICT. The full map is written in a single
 * transaction for atomicity.
 *
 * Time: O(n) where n = Object.keys(meta).length.
 */
export function setChunkMetadata(db: SqlDb, chunkId: string, meta: Record<string, string>): void {
  if (chunkId === '') {
    throw new Error('search: setChunkMetadata requires non-empty chunk ID')
  }
  const keys = Object.keys(meta)
  if (keys.length === 0) return

  db.transaction(() => {
    const stmt = db.prepare(
      `INSERT INTO knowledge_chunk_metadata (chunk_id, key, value)
       VALUES (?, ?, ?)
       ON CONFLICT(chunk_id, key) DO UPDATE SET value = excluded.value`,
    )
    for (const k of keys) {
      if (k === '') {
        throw new Error(`search: metadata key must not be empty for chunk ${chunkId}`)
      }
      const v = meta[k]
      if (v === undefined || v === '') {
        throw new Error(`search: metadata value must not be empty for chunk ${chunkId} key ${k}`)
      }
      stmt.run(chunkId, k, v)
    }
  })
}

/**
 * Retrieve all key-value metadata for a chunk. Returns an empty object
 * (not undefined) when the chunk has no metadata.
 *
 * Time: O(n) where n = number of metadata rows for the chunk.
 */
export function getChunkMetadata(db: SqlDb, chunkId: string): Record<string, string> {
  if (chunkId === '') {
    throw new Error('search: getChunkMetadata requires non-empty chunk ID')
  }

  const rows = db
    .prepare('SELECT key, value FROM knowledge_chunk_metadata WHERE chunk_id = ?')
    .all(chunkId) as Array<{ key: string; value: string }>

  const out: Record<string, string> = {}
  for (const row of rows) {
    out[row.key] = row.value
  }
  return out
}

/**
 * Return chunk IDs where the given key matches the given value.
 * Results are ordered by chunk_id for determinism. A limit of zero
 * or negative defaults to 100.
 *
 * Time: O(n) where n = matching rows (bounded by limit).
 */
export function queryByMetadata(
  db: SqlDb,
  key: string,
  value: string,
  limit: number,
): string[] {
  if (key === '') {
    throw new Error('search: queryByMetadata requires non-empty key')
  }
  const effectiveLimit = limit <= 0 ? 100 : limit

  const rows = db
    .prepare(
      `SELECT chunk_id FROM knowledge_chunk_metadata
       WHERE key = ? AND value = ?
       ORDER BY chunk_id
       LIMIT ?`,
    )
    .all(key, value, effectiveLimit) as Array<{ chunk_id: string }>

  return rows.map((r) => r.chunk_id)
}

/**
 * Delete metadata rows for a batch of chunk IDs. Used during chunk
 * deletion to clean up associated metadata. No-op for an empty list.
 *
 * Time: O(n) where n = chunkIds.length.
 */
export function deleteChunkMetadataBatch(db: SqlDb, chunkIds: readonly string[]): void {
  if (chunkIds.length === 0) return

  db.transaction(() => {
    const stmt = db.prepare('DELETE FROM knowledge_chunk_metadata WHERE chunk_id = ?')
    for (const id of chunkIds) {
      stmt.run(id)
    }
  })
}
