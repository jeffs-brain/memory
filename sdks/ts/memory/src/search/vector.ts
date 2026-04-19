// SPDX-License-Identifier: Apache-2.0

/**
 * sqlite-vec integration. Manages the `knowledge_vectors` virtual table
 * and a text-id to integer-rowid mapping so callers can continue to refer
 * to chunks by their string id.
 *
 * Distance metric: sqlite-vec's default for vec0 virtual tables is L2.
 * We convert distance to a cosine-style similarity score by assuming the
 * inputs are L2-normalised (which is the standard output of bge-m3,
 * OpenAI text-embedding-3-*, and Voyage). With unit-length vectors,
 *
 *   cosine_similarity = 1 - (L2_distance^2 / 2)
 *
 * so the ordering produced by ORDER BY distance ASC is identical to
 * ordering by cosine similarity DESC, and no extra maths is required on
 * the SQLite side. Callers that want a similarity score get it from the
 * conversion below.
 */

import type { SqlDb } from './driver.js'

export type VectorRow = {
  readonly chunk_id: string
  readonly distance: number
  readonly similarity: number
}

/**
 * Convert a Float32 array into a Buffer suitable for sqlite-vec's BLOB
 * input. sqlite-vec accepts JSON strings too but the binary path is
 * strictly faster and avoids any precision loss on the JSON round-trip.
 */
export function encodeVector(vec: Float32Array | number[]): Buffer {
  const arr = vec instanceof Float32Array ? vec : Float32Array.from(vec)
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)
}

/**
 * Allocate (or re-use) the integer rowid that backs a chunk_id. We keep a
 * small side table (knowledge_vec_map) because sqlite-vec requires
 * integer rowids but our chunks use text ids for portability.
 *
 * Re-uses an existing mapping when one is present so vector upserts keep
 * the same rowid and overwrite the previous embedding cleanly. When
 * `model` is supplied, the mapping row's model column is updated to
 * pin the embedding model alongside the rowid; mirrors the `model`
 * column on the Go VectorIndex so switching embedders is detectable.
 */
export function ensureVectorRowid(db: SqlDb, chunkId: string, model?: string): number {
  const existing = db
    .prepare('SELECT vec_rowid FROM knowledge_vec_map WHERE chunk_id = ?')
    .get(chunkId) as { vec_rowid: number } | undefined | null
  if (existing != null) {
    const value = existing.vec_rowid
    const rowid = typeof value === 'bigint' ? Number(value) : value
    if (model !== undefined) {
      db.prepare('UPDATE knowledge_vec_map SET model = ? WHERE chunk_id = ?').run(model, chunkId)
    }
    return rowid
  }

  const nextRow = db
    .prepare('SELECT COALESCE(MAX(vec_rowid), 0) + 1 AS next FROM knowledge_vec_map')
    .get() as { next: number | bigint }
  // better-sqlite3 may return a bigint for integer aggregates; coerce so
  // sqlite-vec's rowid typecheck passes cleanly on both drivers.
  const next = typeof nextRow.next === 'bigint' ? Number(nextRow.next) : nextRow.next
  db.prepare('INSERT INTO knowledge_vec_map(chunk_id, vec_rowid, model) VALUES (?, ?, ?)').run(
    chunkId,
    next,
    model ?? null,
  )
  return next
}

/**
 * Upsert a single embedding into the vector table. The caller is
 * responsible for matching `embedding.length` to the table's configured
 * dimension; sqlite-vec otherwise raises a dimension mismatch error.
 *
 * When `model` is supplied, the embedding model name is pinned on the
 * mapping row so backfill callers can probe existing coverage per model.
 */
export function upsertVector(
  db: SqlDb,
  chunkId: string,
  embedding: Float32Array | number[],
  model?: string,
): void {
  const rowid = ensureVectorRowid(db, chunkId, model)
  const blob = encodeVector(embedding)
  // sqlite-vec's vec0 virtual table rejects a JS `number` as the primary
  // key on better-sqlite3 ("Only integers are allowed..." — the native
  // binding reports the value as a double when it arrives as a JS
  // number). Passing a BigInt satisfies the check on both drivers;
  // bun:sqlite happily coerces BigInt down for small values.
  const rowidBig = BigInt(rowid)
  db.prepare('DELETE FROM knowledge_vectors WHERE rowid = ?').run(rowidBig)
  db.prepare('INSERT INTO knowledge_vectors(rowid, embedding) VALUES (?, ?)').run(rowidBig, blob)
}

/**
 * Return the chunk ids that already carry a vector for the given
 * embedding model. Used by the daemon's vector backfill to skip work
 * on a restart. Missing (NULL) models are excluded because they come
 * from a pre-migration era and cannot be proven to match the current
 * embedder; we let the backfill overwrite them.
 */
export function chunkIdsWithVectorForModel(db: SqlDb, model: string): string[] {
  const rows = db
    .prepare('SELECT chunk_id FROM knowledge_vec_map WHERE model = ?')
    .all(model) as Array<{ chunk_id: string }>
  return rows.map((r) => r.chunk_id)
}

/**
 * Delete a vector by chunk id. Removes both the vector row and the
 * rowid mapping so future embeddings for the same chunk get a fresh rowid.
 */
export function deleteVector(db: SqlDb, chunkId: string): void {
  const row = db.prepare('SELECT vec_rowid FROM knowledge_vec_map WHERE chunk_id = ?').get(chunkId) as
    | { vec_rowid: number | bigint }
    | undefined
  if (row === undefined) return
  const vecRowidBig =
    typeof row.vec_rowid === 'bigint' ? row.vec_rowid : BigInt(row.vec_rowid)
  db.prepare('DELETE FROM knowledge_vectors WHERE rowid = ?').run(vecRowidBig)
  db.prepare('DELETE FROM knowledge_vec_map WHERE chunk_id = ?').run(chunkId)
}

/**
 * Delete every vector whose chunk belongs to the given path. Walks the
 * chunks table to resolve ids, then removes mapping + vector rows in one
 * transaction-adjacent sweep.
 */
export function deleteVectorsByPath(db: SqlDb, path: string): void {
  const rows = db.prepare('SELECT id FROM knowledge_chunks WHERE path = ?').all(path) as Array<{ id: string }>
  for (const { id } of rows) {
    deleteVector(db, id)
  }
}

/**
 * k-nearest-neighbour query. Returns chunk ids ordered by ascending L2
 * distance (which for unit-length inputs is equivalent to descending
 * cosine similarity). The `similarity` field is derived from distance
 * under the unit-norm assumption.
 *
 * The `MATCH ? AND k = ?` syntax is how vec0 exposes its KNN API — see
 * https://alexgarcia.xyz/sqlite-vec/python/knn.html for the canonical
 * reference.
 */
export function knnSearch(db: SqlDb, query: Float32Array | number[], k: number): VectorRow[] {
  if (k <= 0) return []
  const blob = encodeVector(query)
  const rows = db
    .prepare(
      `SELECT m.chunk_id AS chunk_id, v.distance AS distance
         FROM knowledge_vectors v
         JOIN knowledge_vec_map m ON m.vec_rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance`,
    )
    .all(blob, k) as Array<{ chunk_id: string; distance: number }>

  return rows.map((r) => ({
    chunk_id: r.chunk_id,
    distance: r.distance,
    similarity: 1 - (r.distance * r.distance) / 2,
  }))
}
