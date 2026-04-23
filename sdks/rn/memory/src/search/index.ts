import { getChunk, listChunks, searchBm25, searchBm25Compiled, searchVector } from './reader.js'
import type { BM25Result, VectorResult } from './reader.js'
import { applyDdl } from './schema.js'
import type { OpenSqliteDb, SqlDb } from './sqlite-types.js'
import { chunkIdsWithVectorForModel } from './vector.js'
import { type Chunk, deleteByPath, deleteChunk, upsertChunk, upsertChunksBatch } from './writer.js'

export type { Chunk } from './writer.js'
export type { BM25Result, VectorResult } from './reader.js'
export type { OpenSqliteDb, SqlDb } from './sqlite-types.js'
export { BM25_WEIGHTS, SCHEMA_VERSION } from './schema.js'

export type CreateSearchIndexOptions = {
  readonly dbPath: string
  readonly openDb: OpenSqliteDb
  readonly vectorDim?: number
}

export type SearchIndex = {
  readonly db: SqlDb
  readonly vectorDim: number
  upsertChunk(chunk: Chunk): void
  upsertChunks(chunks: readonly Chunk[]): void
  deleteChunk(id: string): void
  deleteByPath(path: string): void
  searchBm25(query: string, limit: number): BM25Result[]
  searchBm25Compiled(expr: string, limit: number): BM25Result[]
  searchVector(embedding: Float32Array | readonly number[], limit: number): VectorResult[]
  getChunk(id: string): Chunk | undefined
  indexedChunks(): Chunk[]
  indexedPaths(): string[]
  chunkIdsWithVectorForModel(model: string): string[]
  close(): Promise<void>
}

export const createSearchIndex = async (opts: CreateSearchIndexOptions): Promise<SearchIndex> => {
  const vectorDim = opts.vectorDim ?? 384
  const db = await opts.openDb(opts.dbPath)
  try {
    await db.loadVectorSupport?.()
    applyDdl((sql) => db.exec(sql), vectorDim)
  } catch (error) {
    try {
      db.close()
    } catch {
      // Ignore secondary close failures.
    }
    throw error
  }

  let closed = false
  return {
    db,
    vectorDim,
    upsertChunk: (chunk) => upsertChunk(db, chunk),
    upsertChunks: (chunks) => upsertChunksBatch(db, chunks),
    deleteChunk: (id) => deleteChunk(db, id),
    deleteByPath: (path) => deleteByPath(db, path),
    searchBm25: (query, limit) => searchBm25(db, query, limit),
    searchBm25Compiled: (expr, limit) => searchBm25Compiled(db, expr, limit),
    searchVector: (embedding, limit) => searchVector(db, embedding, limit),
    getChunk: (id) => getChunk(db, id),
    indexedChunks: () => listChunks(db),
    indexedPaths: () => {
      const rows = db
        .prepare('SELECT DISTINCT path FROM knowledge_chunks ORDER BY path')
        .all() as Array<{ path: string }>
      return rows.map((row) => row.path)
    },
    chunkIdsWithVectorForModel: (model) => chunkIdsWithVectorForModel(db, model),
    close: async () => {
      if (closed) return
      closed = true
      db.close()
    },
  }
}
