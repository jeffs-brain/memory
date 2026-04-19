// SPDX-License-Identifier: Apache-2.0

/**
 * SQLite schema for the hybrid BM25 + vector search index.
 *
 * Three tables:
 *
 *   - knowledge_chunks      — addressable chunk metadata keyed by id, used to
 *                             hydrate results after BM25 / vector lookups.
 *   - knowledge_fts         — FTS5 virtual table over the chunk fields.
 *                             Column weights are configured via the
 *                             `bm25(...)` rank expression so FTS5 can still
 *                             short-circuit `ORDER BY rank LIMIT N`.
 *   - knowledge_vectors     — sqlite-vec virtual table holding the embedding
 *                             blob, with rowid matching knowledge_chunks.id.
 *
 * Column weights (bm25 takes one weight per indexed FTS5 column, in declaration
 * order). Title dominates, content is the baseline:
 *
 *   path     3x   — file location signals
 *   title    10x  — strongest identifier
 *   summary  5x   — high-level overview
 *   tags     4x   — explicit markers
 *   content  1x   — body text
 *
 * These are baked into knowledge_fts at creation via an INSERT into the
 * config row. Tweaking weights requires a schema_version bump and a rebuild.
 */

export const SCHEMA_VERSION = 1

/**
 * BM25 column weights in FTS5 declaration order (path, title, summary, tags,
 * content). Exported so callers / tests can assert the ordering.
 */
export const BM25_WEIGHTS = Object.freeze({
  path: 3.0,
  title: 10.0,
  summary: 5.0,
  tags: 4.0,
  content: 1.0,
})

/**
 * Build the FTS5 rank configuration expression. Must match the FTS5
 * declaration order in {@link FTS_SCHEMA}. Quoted because SQLite FTS5
 * parses the rank config value as a string literal.
 */
function buildRankExpr(): string {
  const { path, title, summary, tags, content } = BM25_WEIGHTS
  return `'bm25(${path}, ${title}, ${summary}, ${tags}, ${content})'`
}

/**
 * Metadata table: stores the schema version and any future migration markers.
 */
export const META_SCHEMA = `
CREATE TABLE IF NOT EXISTS knowledge_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
`

/**
 * Chunks table. `id` is the stable identifier callers supply (typically
 * `${path}#${ordinal}` or a UUID); `path` groups chunks by source document so
 * deleteByPath can sweep them in one statement. `embedding_dim` is NULL
 * for chunks that have no vector stored yet.
 */
export const CHUNK_SCHEMA = `
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id            TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    ordinal       INTEGER NOT NULL DEFAULT 0,
    title         TEXT NOT NULL DEFAULT '',
    summary       TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL DEFAULT '',
    metadata_json TEXT,
    embedding_dim INTEGER,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_path ON knowledge_chunks(path);
`

/**
 * FTS5 schema. knowledge_chunks.id is stored in the FTS `chunk_id` column;
 * we use FTS5's "external content" via a plain UNINDEXED column rather
 * than the contentless / content= mechanism because we want fully
 * self-contained BM25 over the normalised fields.
 */
export const FTS_SCHEMA = `
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    path,
    title,
    summary,
    tags,
    content,
    chunk_id UNINDEXED,
    tokenize='porter unicode61'
);
`

/**
 * Vector virtual table via sqlite-vec. The `rowid` is populated from a
 * monotonically increasing integer keyed off knowledge_chunks; we store
 * that integer in a mapping table (knowledge_vec_map) so callers using
 * text ids can round-trip.
 *
 * The `model` column on `knowledge_vec_map` pins the embedding model
 * name alongside each vector so switching models cleanly invalidates
 * stale vectors. Parity with the Go VectorIndex `Model` column in
 * `sdks/go/search/vectors.go`. Added via ALTER TABLE so existing
 * deployments migrate in-place; rows without a model stay NULL and are
 * treated as belonging to no model for backfill probing purposes.
 *
 * Dimension is 1024 to suit bge-m3 / voyage-large-2; callers supplying
 * a different dim should pass `vectorDim` to {@link createSearchIndex}.
 */
export function buildVectorSchema(vectorDim: number): string {
  if (!Number.isInteger(vectorDim) || vectorDim <= 0) {
    throw new Error(`vectorDim must be a positive integer, got ${vectorDim}`)
  }
  return `
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vectors USING vec0(
    embedding float[${vectorDim}]
);

CREATE TABLE IF NOT EXISTS knowledge_vec_map (
    chunk_id  TEXT PRIMARY KEY,
    vec_rowid INTEGER NOT NULL UNIQUE,
    model     TEXT
);

CREATE INDEX IF NOT EXISTS idx_knowledge_vec_map_model ON knowledge_vec_map(model);
`
}

/**
 * Apply all DDL idempotently, configure BM25 weights, and record the
 * schema version. Safe to run on an empty or already-initialised DB.
 *
 * The rank INSERT is guarded: FTS5 stores it in its config shadow table so
 * repeated applications simply overwrite the same row.
 */
export const applyDDL = (
  exec: (sql: string) => void,
  vectorDim: number,
): void => {
  exec(META_SCHEMA)
  exec(CHUNK_SCHEMA)
  exec(FTS_SCHEMA)
  exec(buildVectorSchema(vectorDim))
  // Idempotent migration for databases created before the `model`
  // column existed on knowledge_vec_map. SQLite raises "duplicate
  // column name" when the column already exists, which we swallow so
  // subsequent boots remain a no-op.
  try {
    exec('ALTER TABLE knowledge_vec_map ADD COLUMN model TEXT')
  } catch (err) {
    const msg = err instanceof Error ? err.message.toLowerCase() : String(err).toLowerCase()
    if (!msg.includes('duplicate column name')) throw err
  }
  exec('CREATE INDEX IF NOT EXISTS idx_knowledge_vec_map_model ON knowledge_vec_map(model)')
  exec(
    `INSERT INTO knowledge_fts(knowledge_fts, rank) VALUES('rank', ${buildRankExpr()})`,
  )
  exec(
    `INSERT INTO knowledge_meta(key, value) VALUES ('schema_version', '${SCHEMA_VERSION}') ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  )
  exec(
    `INSERT INTO knowledge_meta(key, value) VALUES ('vector_dim', '${vectorDim}') ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  )
}
