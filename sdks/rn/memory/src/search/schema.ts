export const SCHEMA_VERSION = 1

export const BM25_WEIGHTS = Object.freeze({
  path: 3.0,
  title: 10.0,
  summary: 5.0,
  tags: 4.0,
  content: 1.0,
})

const buildRankExpr = (): string => {
  const { path, title, summary, tags, content } = BM25_WEIGHTS
  return `'bm25(${path}, ${title}, ${summary}, ${tags}, ${content})'`
}

export const META_SCHEMA = `
CREATE TABLE IF NOT EXISTS knowledge_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
`

export const CHUNK_SCHEMA = `
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  ordinal INTEGER NOT NULL DEFAULT 0,
  title TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  tags TEXT NOT NULL DEFAULT '',
  content TEXT NOT NULL DEFAULT '',
  metadata_json TEXT,
  embedding_dim INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_path ON knowledge_chunks(path);
`

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

export const buildVectorSchema = (vectorDim: number): string => {
  if (!Number.isInteger(vectorDim) || vectorDim <= 0) {
    throw new Error(`vectorDim must be a positive integer, got ${vectorDim}`)
  }
  return `
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vectors USING vec0(
  embedding float[${vectorDim}]
);

CREATE TABLE IF NOT EXISTS knowledge_vec_map (
  chunk_id TEXT PRIMARY KEY,
  vec_rowid INTEGER NOT NULL UNIQUE,
  model TEXT
);

CREATE INDEX IF NOT EXISTS idx_knowledge_vec_map_model ON knowledge_vec_map(model);
`
}

export const applyCoreDdl = (exec: (sql: string) => void): void => {
  exec(META_SCHEMA)
  exec(CHUNK_SCHEMA)
  exec(FTS_SCHEMA)
}

export const applyVectorDdl = (exec: (sql: string) => void, vectorDim: number): void => {
  exec(buildVectorSchema(vectorDim))
  try {
    exec('ALTER TABLE knowledge_vec_map ADD COLUMN model TEXT')
  } catch (error) {
    const message = error instanceof Error ? error.message.toLocaleLowerCase('en') : String(error)
    if (!message.includes('duplicate column name')) throw error
  }
  exec('CREATE INDEX IF NOT EXISTS idx_knowledge_vec_map_model ON knowledge_vec_map(model)')
}

export const writeSchemaMeta = (
  exec: (sql: string) => void,
  vectorDim: number,
  vectorEnabled: boolean,
): void => {
  exec(`INSERT INTO knowledge_fts(knowledge_fts, rank) VALUES('rank', ${buildRankExpr()})`)
  exec(
    `INSERT INTO knowledge_meta(key, value) VALUES ('schema_version', '${SCHEMA_VERSION}') ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  )
  exec(
    `INSERT INTO knowledge_meta(key, value) VALUES ('vector_dim', '${vectorDim}') ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  )
  exec(
    `INSERT INTO knowledge_meta(key, value) VALUES ('vector_enabled', '${vectorEnabled ? 'true' : 'false'}') ON CONFLICT(key) DO UPDATE SET value = excluded.value`,
  )
}

export const applyDdl = (
  exec: (sql: string) => void,
  vectorDim: number,
  opts: { readonly vectorEnabled?: boolean } = {},
): void => {
  const vectorEnabled = opts.vectorEnabled !== false
  applyCoreDdl(exec)
  if (vectorEnabled) {
    applyVectorDdl(exec, vectorDim)
  }
  writeSchemaMeta(exec, vectorDim, vectorEnabled)
}
