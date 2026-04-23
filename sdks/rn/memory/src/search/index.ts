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
  readonly vectorEnabled: boolean
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
  let vectorEnabled = true
  try {
    try {
      await db.loadVectorSupport?.()
    } catch {
      vectorEnabled = false
    }
    try {
      applyDdl((sql) => db.exec(sql), vectorDim, { vectorEnabled })
    } catch (error) {
      if (!vectorEnabled) {
        throw error
      }
      vectorEnabled = false
      applyDdl((sql) => db.exec(sql), vectorDim, { vectorEnabled: false })
    }
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
    vectorEnabled,
    upsertChunk: (chunk) => upsertChunk(db, chunk, { vectorEnabled }),
    upsertChunks: (chunks) => upsertChunksBatch(db, chunks, { vectorEnabled }),
    deleteChunk: (id) => deleteChunk(db, id, { vectorEnabled }),
    deleteByPath: (path) => deleteByPath(db, path, { vectorEnabled }),
    searchBm25: (query, limit) => searchBm25(db, query, limit),
    searchBm25Compiled: (expr, limit) => searchBm25Compiled(db, expr, limit),
    searchVector: (embedding, limit) => {
      if (!vectorEnabled) return []
      return searchVector(db, embedding, limit)
    },
    getChunk: (id) => getChunk(db, id),
    indexedChunks: () => listChunks(db),
    indexedPaths: () => {
      const rows = db
        .prepare('SELECT DISTINCT path FROM knowledge_chunks ORDER BY path')
        .all() as Array<{ path: string }>
      return rows.map((row) => row.path)
    },
    chunkIdsWithVectorForModel: (model) => {
      if (!vectorEnabled) return []
      return chunkIdsWithVectorForModel(db, model)
    },
    close: async () => {
      if (closed) return
      closed = true
      db.close()
    },
  }
}
