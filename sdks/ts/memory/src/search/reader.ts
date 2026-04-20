// SPDX-License-Identifier: Apache-2.0

/**
 * Read-side primitives: BM25 search, vector search, chunk hydration.
 *
 * Results are hydrated from knowledge_chunks so callers get a consistent
 * {@link Chunk}-shaped payload whether the match came via FTS5 or via
 * sqlite-vec. Missing chunk rows (possible if someone DELETEs from
 * knowledge_chunks directly) are filtered out — we do not surface ghost
 * ids to callers.
 */

import { compileFts5Query, runBm25 } from './bm25.js'
import type { SqlDb } from './driver.js'
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

function hydrateRow(row: StoredChunkRow): Chunk {
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
    ...(metadata !== undefined ? { metadata } : {}),
  }
}

/**
 * Load a single chunk by id. Returns undefined when the chunk has been
 * deleted or never existed. Does not touch FTS5 or the vector table.
 */
export function getChunk(db: SqlDb, id: string): Chunk | undefined {
  const row = db
    .prepare(
      `SELECT id, path, ordinal, title, summary, tags, content, metadata_json, embedding_dim
         FROM knowledge_chunks
        WHERE id = ?`,
    )
    .get(id) as StoredChunkRow | undefined
  return row === undefined ? undefined : hydrateRow(row)
}

function hydrateMany(db: SqlDb, ids: readonly string[]): Map<string, Chunk> {
  if (ids.length === 0) return new Map()
  const placeholders = ids.map(() => '?').join(',')
  const rows = db
    .prepare(
      `SELECT id, path, ordinal, title, summary, tags, content, metadata_json, embedding_dim
         FROM knowledge_chunks
        WHERE id IN (${placeholders})`,
    )
    .all(...ids) as StoredChunkRow[]
  const map = new Map<string, Chunk>()
  for (const r of rows) {
    map.set(r.id, hydrateRow(r))
  }
  return map
}

/**
 * BM25-ranked full-text search. Query is compiled via
 * {@link compileFts5Query} before hitting FTS5 so callers can pass raw
 * natural-language input.
 *
 * Returns results ordered by ascending FTS5 rank (lower is better). The
 * raw rank is exposed as `score` so downstream rerankers / fusion can
 * consume it without re-querying.
 */
export function searchBM25Compiled(db: SqlDb, expr: string, limit: number): BM25Result[] {
  if (expr === '' || limit <= 0) return []
  const rows = runBm25(db, expr, limit)
  if (rows.length === 0) return []
  const chunks = hydrateMany(
    db,
    rows.map((r) => r.chunk_id),
  )
  const out: BM25Result[] = []
  for (const r of rows) {
    const chunk = chunks.get(r.chunk_id)
    if (chunk === undefined) continue
    out.push({ chunk, score: r.rank })
  }
  return out
}

export function searchBM25(db: SqlDb, query: string, limit: number): BM25Result[] {
  const expr = compileFts5Query(query)
  return searchBM25Compiled(db, expr, limit)
}

/**
 * k-nearest-neighbour vector search over stored embeddings. Caller is
 * responsible for matching the embedding dimension; a mismatch raises a
 * sqlite-vec error.
 */
export function searchVector(
  db: SqlDb,
  embedding: Float32Array | number[],
  limit: number,
): VectorResult[] {
  if (limit <= 0) return []
  const rows = knnSearch(db, embedding, limit)
  if (rows.length === 0) return []
  const chunks = hydrateMany(
    db,
    rows.map((r) => r.chunk_id),
  )
  const out: VectorResult[] = []
  for (const r of rows) {
    const chunk = chunks.get(r.chunk_id)
    if (chunk === undefined) continue
    out.push({ chunk, distance: r.distance, similarity: r.similarity })
  }
  return out
}
