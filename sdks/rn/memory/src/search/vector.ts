import type { SqlDb } from './sqlite-types.js'

export type VectorRow = {
  readonly chunk_id: string
  readonly distance: number
  readonly similarity: number
}

export const encodeVector = (vector: Float32Array | readonly number[]): Uint8Array => {
  const arr = vector instanceof Float32Array ? vector : Float32Array.from(vector)
  const out = new Uint8Array(arr.byteLength)
  out.set(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength))
  return out
}

export const ensureVectorRowid = (db: SqlDb, chunkId: string, model?: string): number => {
  const existing = db
    .prepare('SELECT vec_rowid FROM knowledge_vec_map WHERE chunk_id = ?')
    .get(chunkId) as { vec_rowid: number | bigint } | undefined | null

  if (existing != null) {
    const value =
      typeof existing.vec_rowid === 'bigint' ? Number(existing.vec_rowid) : existing.vec_rowid
    if (model !== undefined) {
      db.prepare('UPDATE knowledge_vec_map SET model = ? WHERE chunk_id = ?').run(model, chunkId)
    }
    return value
  }

  const nextRow = db
    .prepare('SELECT COALESCE(MAX(vec_rowid), 0) + 1 AS next FROM knowledge_vec_map')
    .get() as { next: number | bigint }
  const next = typeof nextRow.next === 'bigint' ? Number(nextRow.next) : nextRow.next
  db.prepare('INSERT INTO knowledge_vec_map(chunk_id, vec_rowid, model) VALUES (?, ?, ?)').run(
    chunkId,
    next,
    model ?? null,
  )
  return next
}

export const upsertVector = (
  db: SqlDb,
  chunkId: string,
  embedding: Float32Array | readonly number[],
  model?: string,
): void => {
  const rowid = ensureVectorRowid(db, chunkId, model)
  const blob = encodeVector(embedding)
  const rowidBig = BigInt(rowid)
  db.prepare('DELETE FROM knowledge_vectors WHERE rowid = ?').run(rowidBig)
  db.prepare('INSERT INTO knowledge_vectors(rowid, embedding) VALUES (?, ?)').run(rowidBig, blob)
}

export const deleteVector = (db: SqlDb, chunkId: string): void => {
  const row = db
    .prepare('SELECT vec_rowid FROM knowledge_vec_map WHERE chunk_id = ?')
    .get(chunkId) as { vec_rowid: number | bigint } | undefined
  if (row === undefined) return
  const rowidBig = typeof row.vec_rowid === 'bigint' ? row.vec_rowid : BigInt(row.vec_rowid)
  db.prepare('DELETE FROM knowledge_vectors WHERE rowid = ?').run(rowidBig)
  db.prepare('DELETE FROM knowledge_vec_map WHERE chunk_id = ?').run(chunkId)
}

export const deleteVectorsByPath = (db: SqlDb, path: string): void => {
  const rows = db.prepare('SELECT id FROM knowledge_chunks WHERE path = ?').all(path) as Array<{
    id: string
  }>
  for (const row of rows) {
    deleteVector(db, row.id)
  }
}

export const chunkIdsWithVectorForModel = (db: SqlDb, model: string): string[] => {
  const rows = db
    .prepare('SELECT chunk_id FROM knowledge_vec_map WHERE model = ?')
    .all(model) as Array<{ chunk_id: string }>
  return rows.map((row) => row.chunk_id)
}

export const knnSearch = (
  db: SqlDb,
  query: Float32Array | readonly number[],
  k: number,
): VectorRow[] => {
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

  return rows.map((row) => ({
    chunk_id: row.chunk_id,
    distance: row.distance,
    similarity: 1 - (row.distance * row.distance) / 2,
  }))
}
