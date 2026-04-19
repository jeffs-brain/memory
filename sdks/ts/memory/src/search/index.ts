// SPDX-License-Identifier: Apache-2.0

/**
 * Public surface for the hybrid SQLite search index (BM25 + sqlite-vec).
 *
 * Tenant model: each brain / tenant gets its own SQLite file and its own
 * `SearchIndex` instance. Callers can either pass a `dbPath` and have the
 * module open + own the connection, or pass an already-opened `SqlDb`
 * when reusing a connection across modules in the same tenant.
 *
 * There is NO process-wide connection pool keyed by path — that was the
 * multi-tenant anti-pattern flagged in PORTING-SPEC §13. Callers are
 * responsible for connection lifecycle; `close()` is idempotent and only
 * closes the connection when this module opened it.
 */

import { openDatabase } from './driver.js'
import type { DriverKind, SqlDb } from './driver.js'
import { getChunk, searchBM25, searchBM25Compiled, searchVector } from './reader.js'
import type { BM25Result, VectorResult } from './reader.js'
import { applyDDL } from './schema.js'
import { chunkIdsWithVectorForModel } from './vector.js'
import { deleteByPath, deleteChunk, upsertChunk, upsertChunksBatch } from './writer.js'
import type { Chunk } from './writer.js'

export type { Chunk } from './writer.js'
export type { BM25Result, VectorResult } from './reader.js'
export type { SqlDb, DriverKind } from './driver.js'
export { BM25_WEIGHTS, SCHEMA_VERSION } from './schema.js'

export type CreateSearchIndexOptions = {
  /**
   * Filesystem path to the tenant's SQLite database. `:memory:` is
   * accepted for ephemeral (test) indices. Mutually exclusive with
   * `connection`.
   */
  readonly dbPath?: string
  /**
   * An already-opened connection. Supplied by callers that share a
   * single DB across modules (e.g. the SQLite-backed memory store).
   * The module will NOT close a caller-owned connection.
   */
  readonly connection?: SqlDb
  /**
   * Vector dimension. Baked into the sqlite-vec virtual table at
   * creation time; changing it later requires a manual rebuild.
   * Defaults to 1024 (bge-m3, voyage-large-2).
   */
  readonly vectorDim?: number
  /**
   * Force a driver (bun or better-sqlite3). Default: auto-detect.
   * Ignored when `connection` is supplied.
   */
  readonly driver?: DriverKind
  /**
   * Override the sqlite-vec extension path. Useful for tests running
   * under a custom bundler that cannot resolve the platform binary.
   * When undefined we delegate to sqlite-vec/getLoadablePath().
   */
  readonly vectorExtensionPath?: string
}

export type SearchIndex = {
  /** The underlying connection. Exposed so callers can run ad-hoc SQL. */
  readonly db: SqlDb
  /** Indexed vector dimension. */
  readonly vectorDim: number

  upsertChunk(chunk: Chunk): void
  upsertChunks(chunks: readonly Chunk[]): void
  deleteChunk(id: string): void
  deleteByPath(path: string): void

  searchBM25(query: string, limit: number): BM25Result[]
  searchBM25Compiled(expr: string, limit: number): BM25Result[]
  searchVector(embedding: Float32Array | number[], limit: number): VectorResult[]
  getChunk(id: string): Chunk | undefined

  /**
   * Return every distinct path currently indexed in knowledge_chunks.
   * Parity with the Go `Index.IndexedPaths()` used by the daemon's
   * vector backfill to enumerate candidates without re-scanning the
   * store.
   */
  indexedPaths(): string[]
  /**
   * Return the chunk ids that already have a vector for the given
   * embedding model. Used by backfill to skip work on restart; mirrors
   * Go's `VectorIndex.LoadAll(model)` probe.
   */
  chunkIdsWithVectorForModel(model: string): string[]

  /**
   * Close the connection iff this instance opened it. No-op when the
   * caller supplied their own `connection`. Safe to call multiple times.
   */
  close(): Promise<void>
}

async function loadSqliteVec(db: SqlDb, override: string | undefined): Promise<void> {
  if (override !== undefined) {
    db.loadExtension(override)
    return
  }
  // sqlite-vec ships an ESM + CJS entry that only exposes getLoadablePath
  // and load(). Dynamic import so a missing extension does not crash at
  // module-parse time.
  const sqliteVec = (await import('sqlite-vec')) as typeof import('sqlite-vec')
  const path = sqliteVec.getLoadablePath()
  db.loadExtension(path)
}

/**
 * Create (or attach to) a search index. DDL is applied idempotently;
 * repeated calls against the same database are safe.
 */
export async function createSearchIndex(opts: CreateSearchIndexOptions = {}): Promise<SearchIndex> {
  if (opts.dbPath !== undefined && opts.connection !== undefined) {
    throw new Error('createSearchIndex: pass either dbPath or connection, not both')
  }

  const vectorDim = opts.vectorDim ?? 1024
  const owned = opts.connection === undefined

  const db: SqlDb =
    opts.connection ??
    (await openDatabase({
      path: opts.dbPath ?? ':memory:',
      ...(opts.driver !== undefined ? { driver: opts.driver } : {}),
    }))

  try {
    await loadSqliteVec(db, opts.vectorExtensionPath)
    applyDDL((sql) => db.exec(sql), vectorDim)
  } catch (err) {
    if (owned) {
      try {
        db.close()
      } catch {
        /* ignore secondary failure */
      }
    }
    throw err
  }

  let closed = false
  return {
    db,
    vectorDim,
    upsertChunk: (chunk) => upsertChunk(db, chunk),
    upsertChunks: (chunks) => upsertChunksBatch(db, chunks),
    deleteChunk: (id) => deleteChunk(db, id),
    deleteByPath: (path) => deleteByPath(db, path),
    searchBM25: (query, limit) => searchBM25(db, query, limit),
    searchBM25Compiled: (expr, limit) => searchBM25Compiled(db, expr, limit),
    searchVector: (embedding, limit) => searchVector(db, embedding, limit),
    getChunk: (id) => getChunk(db, id),
    indexedPaths: () => {
      const rows = db
        .prepare('SELECT DISTINCT path FROM knowledge_chunks ORDER BY path')
        .all() as Array<{ path: string }>
      return rows.map((r) => r.path)
    },
    chunkIdsWithVectorForModel: (model) => chunkIdsWithVectorForModel(db, model),
    close: async () => {
      if (closed) return
      closed = true
      if (owned) db.close()
    },
  }
}
