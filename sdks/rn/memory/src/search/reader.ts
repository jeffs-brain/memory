import { compileFts5Query, runBm25 } from './bm25.js'
import type { SqlDb } from './sqlite-types.js'
import { knnSearch } from './vector.js'
import type { Chunk } from './writer.js'

export type BM25Result = {
  readonly chunk: Chunk
  readonly score: number
}

export type VectorResult = {
  readonly chunk: Chunk
  readonly distance: number
  readonly similarity: number
}

type StoredChunkRow = {
  id: string
  path: string
  ordinal: number
  title: string
  summary: string
  tags: string
  content: string
  metadata_json: string | null
  embedding_dim: number | null
}

const hydrateRow = (row: StoredChunkRow): Chunk => {
  const tags = row.tags === '' ? [] : row.tags.split(/\s+/)
  const metadata =
    row.metadata_json === null
      ? undefined
      : (JSON.parse(row.metadata_json) as Record<string, unknown>)
  return {
    id: row.id,
    path: row.path,
    ordinal: row.ordinal,
    title: row.title,
    summary: row.summary,
    tags,
    content: row.content,
    ...(metadata === undefined ? {} : { metadata }),
  }
}

export const getChunk = (db: SqlDb, id: string): Chunk | undefined => {
  const row = db
    .prepare(
      `SELECT id, path, ordinal, title, summary, tags, content, metadata_json, embedding_dim
         FROM knowledge_chunks
        WHERE id = ?`,
    )
    .get(id) as StoredChunkRow | undefined
  return row === undefined ? undefined : hydrateRow(row)
}

export const listChunks = (db: SqlDb): Chunk[] => {
  const rows = db
    .prepare(
      `SELECT id, path, ordinal, title, summary, tags, content, metadata_json, embedding_dim
         FROM knowledge_chunks
        ORDER BY path, ordinal, id`,
    )
    .all() as StoredChunkRow[]
  return rows.map((row) => hydrateRow(row))
}

const hydrateMany = (db: SqlDb, ids: readonly string[]): Map<string, Chunk> => {
  if (ids.length === 0) return new Map()
  const placeholders = ids.map(() => '?').join(',')
  const rows = db
    .prepare(
      `SELECT id, path, ordinal, title, summary, tags, content, metadata_json, embedding_dim
         FROM knowledge_chunks
        WHERE id IN (${placeholders})`,
    )
    .all(...ids) as StoredChunkRow[]
  const out = new Map<string, Chunk>()
  for (const row of rows) out.set(row.id, hydrateRow(row))
  return out
}

export const searchBm25Compiled = (db: SqlDb, expr: string, limit: number): BM25Result[] => {
  if (expr === '' || limit <= 0) return []
  const rows = runBm25(db, expr, limit)
  if (rows.length === 0) return []
  const chunks = hydrateMany(
    db,
    rows.map((row) => row.chunk_id),
  )
  const out: BM25Result[] = []
  for (const row of rows) {
    const chunk = chunks.get(row.chunk_id)
    if (chunk === undefined) continue
    out.push({ chunk, score: row.rank })
  }
  return out
}

export const searchBm25 = (db: SqlDb, query: string, limit: number): BM25Result[] => {
  return searchBm25Compiled(db, compileFts5Query(query), limit)
}

export const searchVector = (
  db: SqlDb,
  embedding: Float32Array | readonly number[],
  limit: number,
): VectorResult[] => {
  if (limit <= 0) return []
  const rows = knnSearch(db, embedding, limit)
  if (rows.length === 0) return []
  const chunks = hydrateMany(
    db,
    rows.map((row) => row.chunk_id),
  )
  const out: VectorResult[] = []
  for (const row of rows) {
    const chunk = chunks.get(row.chunk_id)
    if (chunk === undefined) continue
    out.push({
      chunk,
      distance: row.distance,
      similarity: row.similarity,
    })
  }
  return out
}
