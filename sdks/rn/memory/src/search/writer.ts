import type { SqlDb } from './sqlite-types.js'
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
  readonly embedding?: Float32Array | readonly number[]
  readonly embeddingModel?: string
}

export type ChunkWriteOptions = {
  readonly vectorEnabled?: boolean
}

const normaliseTags = (tags: readonly string[] | string | undefined): string => {
  if (tags === undefined) return ''
  if (typeof tags === 'string') return tags
  return tags.join(' ')
}

const upsertChunkInner = (db: SqlDb, chunk: Chunk, opts: ChunkWriteOptions = {}): void => {
  const vectorEnabled = opts.vectorEnabled !== false
  const tags = normaliseTags(chunk.tags)
  const metadataJson = chunk.metadata === undefined ? null : JSON.stringify(chunk.metadata)
  const dimension = chunk.embedding === undefined || !vectorEnabled ? null : chunk.embedding.length
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
  ).run(chunk.id, chunk.path, ordinal, title, summary, tags, chunk.content, metadataJson, dimension)

  db.prepare('DELETE FROM knowledge_fts WHERE chunk_id = ?').run(chunk.id)
  db.prepare(
    `INSERT INTO knowledge_fts(path, title, summary, tags, content, chunk_id)
     VALUES (?, ?, ?, ?, ?, ?)`,
  ).run(chunk.path, title, summary, tags, chunk.content, chunk.id)

  if (vectorEnabled && chunk.embedding !== undefined) {
    upsertVector(db, chunk.id, chunk.embedding, chunk.embeddingModel)
  }
}

export const upsertChunk = (db: SqlDb, chunk: Chunk, opts: ChunkWriteOptions = {}): void => {
  db.transaction(() => {
    upsertChunkInner(db, chunk, opts)
  })
}

export const upsertChunksBatch = (
  db: SqlDb,
  chunks: readonly Chunk[],
  opts: ChunkWriteOptions = {},
): void => {
  if (chunks.length === 0) return
  db.transaction(() => {
    for (const chunk of chunks) {
      upsertChunkInner(db, chunk, opts)
    }
  })
}

export const deleteChunk = (db: SqlDb, id: string, opts: ChunkWriteOptions = {}): void => {
  const vectorEnabled = opts.vectorEnabled !== false
  db.transaction(() => {
    db.prepare('DELETE FROM knowledge_fts WHERE chunk_id = ?').run(id)
    if (vectorEnabled) {
      deleteVector(db, id)
    }
    db.prepare('DELETE FROM knowledge_chunks WHERE id = ?').run(id)
  })
}

export const deleteByPath = (db: SqlDb, path: string, opts: ChunkWriteOptions = {}): void => {
  const vectorEnabled = opts.vectorEnabled !== false
  db.transaction(() => {
    db.prepare('DELETE FROM knowledge_fts WHERE path = ?').run(path)
    if (vectorEnabled) {
      deleteVectorsByPath(db, path)
    }
    db.prepare('DELETE FROM knowledge_chunks WHERE path = ?').run(path)
  })
}
