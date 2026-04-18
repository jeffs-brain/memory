// SPDX-License-Identifier: Apache-2.0

/**
 * Write-side primitives: upsert chunks, delete by path, batch write.
 *
 * All writes go through the driver's transaction wrapper. SQLite allows
 * one writer at a time; wrapping in BEGIN IMMEDIATE serialises writers
 * and keeps WAL snapshot reads available to concurrent searchers.
 */

import type { SqlDb } from './driver.js'
import { deleteVector, deleteVectorsByPath, upsertVector } from './vector.js'

export type Chunk = {
  readonly id: string
  readonly path: string
  readonly ordinal?: number
  readonly title?: string
  readonly summary?: string
  readonly tags?: readonly string[] | string
  readonly content: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly embedding?: Float32Array | number[]
}

function normaliseTags(tags: readonly string[] | string | undefined): string {
  if (tags === undefined) return ''
  if (typeof tags === 'string') return tags
  return tags.join(' ')
}

/**
 * Upsert a single chunk's BM25 and (optionally) vector rows. Exists
 * mainly for readability — batch callers should use
 * {@link upsertChunksBatch} so every write lands in one transaction.
 */
export function upsertChunk(db: SqlDb, chunk: Chunk): void {
  db.transaction(() => {
    upsertChunkInner(db, chunk)
  })
}

/**
 * Upsert many chunks in a single transaction. Matches the original
 * Go indexer's 100-per-batch shape but accepts any length — callers
 * that need streaming should chunk the input themselves.
 */
export function upsertChunksBatch(db: SqlDb, chunks: readonly Chunk[]): void {
  if (chunks.length === 0) return
  db.transaction(() => {
    for (const c of chunks) {
      upsertChunkInner(db, c)
    }
  })
}

function upsertChunkInner(db: SqlDb, chunk: Chunk): void {
  const tags = normaliseTags(chunk.tags)
  const metadataJson = chunk.metadata === undefined ? null : JSON.stringify(chunk.metadata)
  const dim = chunk.embedding === undefined ? null : chunk.embedding.length
  const ordinal = chunk.ordinal ?? 0
  const title = chunk.title ?? ''
  const summary = chunk.summary ?? ''

  db.prepare(
    `INSERT INTO knowledge_chunks(id, path, ordinal, title, summary, tags, content, metadata_json, embedding_dim, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
     ON CONFLICT(id) DO UPDATE SET
        path          = excluded.path,
        ordinal       = excluded.ordinal,
        title         = excluded.title,
        summary       = excluded.summary,
        tags          = excluded.tags,
        content       = excluded.content,
        metadata_json = excluded.metadata_json,
        embedding_dim = excluded.embedding_dim,
        updated_at    = CURRENT_TIMESTAMP`,
  ).run(chunk.id, chunk.path, ordinal, title, summary, tags, chunk.content, metadataJson, dim)

  // FTS5 has no ON CONFLICT — delete-then-insert keeps the index in sync.
  db.prepare('DELETE FROM knowledge_fts WHERE chunk_id = ?').run(chunk.id)
  db.prepare(
    `INSERT INTO knowledge_fts(path, title, summary, tags, content, chunk_id)
     VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(chunk.path, title, summary, tags, chunk.content, chunk.id)

  if (chunk.embedding !== undefined) {
    upsertVector(db, chunk.id, chunk.embedding)
  }
}

/**
 * Delete a single chunk by id from chunks, FTS, and vector tables.
 */
export function deleteChunk(db: SqlDb, id: string): void {
  db.transaction(() => {
    db.prepare('DELETE FROM knowledge_fts WHERE chunk_id = ?').run(id)
    deleteVector(db, id)
    db.prepare('DELETE FROM knowledge_chunks WHERE id = ?').run(id)
  })
}

/**
 * Delete every chunk belonging to a source path. Used when a document
 * is removed from the brain store so stale chunks do not linger in
 * either the BM25 or vector index.
 */
export function deleteByPath(db: SqlDb, path: string): void {
  db.transaction(() => {
    db.prepare('DELETE FROM knowledge_fts WHERE path = ?').run(path)
    deleteVectorsByPath(db, path)
    db.prepare('DELETE FROM knowledge_chunks WHERE path = ?').run(path)
  })
}
