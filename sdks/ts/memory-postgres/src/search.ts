/**
 * Postgres-backed hybrid search index.
 *
 * Mirrors the surface of the SQLite {@link SearchIndex} from
 * `./index.ts`, adapted to async I/O and the partitioned `memory.chunks`
 * table. ParadeDB's `pg_search` is optional in our deployments so BM25 is
 * expressed via the built-in tsvector + `@@` operator with `ts_rank_cd`;
 * swapping in pg_search later is a one-function change.
 *
 * See plan.md §4.3 for the hybrid CTE. We keep the same RRF formulation
 * (constant k = 60, full-outer-join on chunk_id) so SQLite and Postgres
 * paths return structurally compatible results.
 *
 * Tenancy: every query runs inside `sql.begin()` with `app.tenant_id` set
 * via `set_config(...)`. RLS enforces the isolation invariant — we do not
 * rely on the application filtering by tenant_id.
 */

import { expandAliases, parseQuery, type AliasTable, type Token } from '@jeffs-brain/memory/query'
import type { TrigramSourceChunk } from '@jeffs-brain/memory/retrieval'
import type { PgSql } from './store.js'

export type PostgresChunk = {
  readonly chunkId: bigint
  readonly documentId: string
  readonly path?: string
  readonly ordinal: number
  readonly content: string
  readonly tokens: number
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
  readonly embedding?: Float32Array | number[]
}

export type PostgresBM25Result = {
  readonly chunkId: bigint
  readonly score: number
}

export type PostgresVectorResult = {
  readonly chunkId: bigint
  readonly score: number
  readonly distance: number
}

export type PostgresHybridResult = {
  readonly chunkId: bigint
  readonly rrfScore: number
  readonly documentId: string
  readonly path?: string
  readonly ordinal: number
  readonly content: string
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
}

export type PostgresSearchIndexOptions = {
  readonly sql: PgSql
  readonly tenantId: string
  readonly brainId: string
  readonly aliases?: AliasTable
  /**
   * FTS dictionary for `to_tsquery` / `to_tsvector`. The migration hard-codes
   * `english` on the generated `fts_vector`, so queries must use the same
   * dictionary or the index will miss.
   */
  readonly tsConfig?: string
  /** Vector dimension. Matches `halfvec(N)` in the schema (default 1024). */
  readonly vectorDim?: number
  /** RRF constant k (defaults to 60 per Jeff's pipeline). */
  readonly rrfK?: number
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

const assertUuid = (value: string, label: string): void => {
  if (!UUID_RE.test(value)) throw new Error(`postgres-search: ${label} must be a uuid, got ${value}`)
}

const sanitiseLexeme = (value: string): string =>
  value.replace(/[^\p{L}\p{N}_]+/gu, '').trim()

const renderTsQueryToken = (token: Token): string => {
  switch (token.kind) {
    case 'phrase': {
      const parts = token.text
        .split(/\s+/)
        .map((part) => sanitiseLexeme(part))
        .filter((part) => part !== '')
      if (parts.length === 0) return ''
      if (parts.length === 1) return parts[0] ?? ''
      return `(${parts.join(' <-> ')})`
    }
    case 'prefix': {
      const lexeme = sanitiseLexeme(token.text)
      return lexeme === '' ? '' : `${lexeme}:*`
    }
    default: {
      return sanitiseLexeme(token.text)
    }
  }
}

/**
 * Compile a natural-language query into a `to_tsquery`-safe expression.
 * Mirrors the shared parser semantics used by the SQLite path so
 * boolean operators, quoted phrases, and prefix tokens survive on the
 * Postgres retrieval path as well.
 */
const compileTsQuery = (query: string, aliases?: AliasTable): string => {
  const ast = parseQuery(query)
  const expanded = aliases === undefined ? ast : expandAliases(ast, aliases)
  if (expanded.tokens.length === 0) return ''

  const pieces: string[] = []
  for (const token of expanded.tokens) {
    const rendered = renderTsQueryToken(token)
    if (rendered === '') continue
    if (pieces.length === 0) {
      pieces.push(rendered)
      continue
    }
    switch (token.operator ?? 'OR') {
      case 'AND':
        pieces.push('&', rendered)
        break
      case 'NOT':
        pieces.push('&', '!', rendered)
        break
      default:
        pieces.push('|', rendered)
        break
    }
  }

  return pieces.join(' ').trim()
}

const serialiseHalfvec = (embedding: Float32Array | number[]): string => {
  const values: number[] = embedding instanceof Float32Array ? Array.from(embedding) : embedding
  return `[${values.join(',')}]`
}

/**
 * Create a Postgres search index bound to (tenantId, brainId).
 */
export const createPostgresSearchIndex = (opts: PostgresSearchIndexOptions): PostgresSearchIndex => {
  assertUuid(opts.tenantId, 'tenantId')
  assertUuid(opts.brainId, 'brainId')
  return new PostgresSearchIndex(opts)
}

export class PostgresSearchIndex {
  readonly sql: PgSql
  readonly tenantId: string
  readonly brainId: string
  readonly aliases: AliasTable | undefined
  readonly tsConfig: string
  readonly vectorDim: number
  readonly rrfK: number
  private closed = false

  constructor(opts: PostgresSearchIndexOptions) {
    this.sql = opts.sql
    this.tenantId = opts.tenantId
    this.brainId = opts.brainId
    this.aliases = opts.aliases
    this.tsConfig = opts.tsConfig ?? 'english'
    this.vectorDim = opts.vectorDim ?? 1024
    this.rrfK = opts.rrfK ?? 60
  }

  async upsert(chunk: PostgresChunk): Promise<void> {
    this.ensureOpen()
    const embeddingLiteral =
      chunk.embedding === undefined ? null : serialiseHalfvec(chunk.embedding)
    await this.withTenant(async (tx) => {
      if (embeddingLiteral === null) {
        await tx`
          insert into memory.chunks
            (chunk_id, tenant_id, document_id, ordinal, content, tokens, embedding)
          values
            (${chunk.chunkId}, ${this.tenantId}::uuid, ${chunk.documentId}::uuid,
             ${chunk.ordinal}, ${chunk.content}, ${chunk.tokens}, null)
          on conflict (chunk_id, tenant_id) do update set
            document_id = excluded.document_id,
            ordinal     = excluded.ordinal,
            content     = excluded.content,
            tokens      = excluded.tokens,
            embedding   = excluded.embedding
        `
      } else {
        await tx`
          insert into memory.chunks
            (chunk_id, tenant_id, document_id, ordinal, content, tokens, embedding)
          values
            (${chunk.chunkId}, ${this.tenantId}::uuid, ${chunk.documentId}::uuid,
             ${chunk.ordinal}, ${chunk.content}, ${chunk.tokens},
             ${embeddingLiteral}::halfvec)
          on conflict (chunk_id, tenant_id) do update set
            document_id = excluded.document_id,
            ordinal     = excluded.ordinal,
            content     = excluded.content,
            tokens      = excluded.tokens,
            embedding   = excluded.embedding
        `
      }
    })
  }

  /**
   * Batch-upsert many chunks inside a single transaction. Implementation is
   * a row-at-a-time loop for now — partitioned batched INSERTs would need
   * the `postgres` driver's `.values` helper which we do not model on the
   * structural `PgSql` type.
   */
  async upsertMany(chunks: readonly PostgresChunk[]): Promise<void> {
    if (chunks.length === 0) return
    this.ensureOpen()
    await this.withTenant(async (tx) => {
      for (const chunk of chunks) {
        const embeddingLiteral =
          chunk.embedding === undefined ? null : serialiseHalfvec(chunk.embedding)
        if (embeddingLiteral === null) {
          await tx`
            insert into memory.chunks
              (chunk_id, tenant_id, document_id, ordinal, content, tokens, embedding)
            values
              (${chunk.chunkId}, ${this.tenantId}::uuid, ${chunk.documentId}::uuid,
               ${chunk.ordinal}, ${chunk.content}, ${chunk.tokens}, null)
            on conflict (chunk_id, tenant_id) do update set
              document_id = excluded.document_id,
              ordinal     = excluded.ordinal,
              content     = excluded.content,
              tokens      = excluded.tokens,
              embedding   = excluded.embedding
          `
        } else {
          await tx`
            insert into memory.chunks
              (chunk_id, tenant_id, document_id, ordinal, content, tokens, embedding)
            values
              (${chunk.chunkId}, ${this.tenantId}::uuid, ${chunk.documentId}::uuid,
               ${chunk.ordinal}, ${chunk.content}, ${chunk.tokens},
               ${embeddingLiteral}::halfvec)
            on conflict (chunk_id, tenant_id) do update set
              document_id = excluded.document_id,
              ordinal     = excluded.ordinal,
              content     = excluded.content,
              tokens      = excluded.tokens,
              embedding   = excluded.embedding
          `
        }
      }
    })
  }

  async deleteChunk(chunkId: bigint): Promise<void> {
    this.ensureOpen()
    await this.withTenant(async (tx) => {
      await tx`
        delete from memory.chunks
        where tenant_id = ${this.tenantId}::uuid and chunk_id = ${chunkId}
      `
    })
  }

  async deleteByDocument(documentId: string): Promise<void> {
    this.ensureOpen()
    assertUuid(documentId, 'documentId')
    await this.withTenant(async (tx) => {
      await tx`
        delete from memory.chunks
        where tenant_id = ${this.tenantId}::uuid and document_id = ${documentId}::uuid
      `
    })
  }

  /**
   * Delete every chunk whose source document matches the given `memory.documents.path`.
   * Joins through `memory.documents` so callers can key on the logical path.
   */
  async deleteByPath(path: string): Promise<void> {
    this.ensureOpen()
    await this.withTenant(async (tx) => {
      await tx`
        delete from memory.chunks c
        using memory.documents d
        where c.tenant_id = ${this.tenantId}::uuid
          and d.tenant_id = ${this.tenantId}::uuid
          and d.brain_id  = ${this.brainId}::uuid
          and d.path      = ${path}
          and c.document_id = d.document_id
      `
    })
  }

  async searchBM25(query: string, limit: number): Promise<PostgresBM25Result[]> {
    this.ensureOpen()
    if (limit <= 0) return []
    const expr = compileTsQuery(query, this.aliases)
    if (expr === '') return []
    return this.withTenant(async (tx) => {
      const rows = (await tx<{ chunk_id: string; score: number }>`
        select c.chunk_id::text as chunk_id,
               ts_rank_cd(c.fts_vector, to_tsquery(${this.tsConfig}, ${expr})) as score
        from memory.chunks c
        join memory.documents d on d.document_id = c.document_id
        where c.tenant_id = ${this.tenantId}::uuid
          and d.tenant_id = ${this.tenantId}::uuid
          and d.brain_id  = ${this.brainId}::uuid
          and c.fts_vector @@ to_tsquery(${this.tsConfig}, ${expr})
        order by score desc, c.chunk_id asc
        limit ${limit}
      `) as ReadonlyArray<{ chunk_id: string; score: number }>
      return rows.map((r) => ({ chunkId: BigInt(r.chunk_id), score: Number(r.score) }))
    })
  }

  async searchVector(
    embedding: Float32Array | number[],
    limit: number,
  ): Promise<PostgresVectorResult[]> {
    this.ensureOpen()
    if (limit <= 0) return []
    const literal = serialiseHalfvec(embedding)
    return this.withTenant(async (tx) => {
      const rows = (await tx<{ chunk_id: string; distance: number }>`
        select c.chunk_id::text as chunk_id,
               (c.embedding <=> ${literal}::halfvec) as distance
        from memory.chunks c
        join memory.documents d on d.document_id = c.document_id
        where c.tenant_id = ${this.tenantId}::uuid
          and d.tenant_id = ${this.tenantId}::uuid
          and d.brain_id  = ${this.brainId}::uuid
          and c.embedding is not null
        order by c.embedding <=> ${literal}::halfvec, c.chunk_id asc
        limit ${limit}
      `) as ReadonlyArray<{ chunk_id: string; distance: number }>
      return rows.map((r) => ({
        chunkId: BigInt(r.chunk_id),
        distance: Number(r.distance),
        score: 1 - Number(r.distance),
      }))
    })
  }

  async hasEmbeddings(): Promise<boolean> {
    this.ensureOpen()
    return this.withTenant(async (tx) => {
      const rows = (await tx<{ present: boolean }>`
        select exists(
          select 1
          from memory.chunks c
          join memory.documents d on d.document_id = c.document_id
          where c.tenant_id = ${this.tenantId}::uuid
            and d.tenant_id = ${this.tenantId}::uuid
            and d.brain_id  = ${this.brainId}::uuid
            and c.embedding is not null
        ) as present
      `) as ReadonlyArray<{ present: boolean }>
      return rows[0]?.present ?? false
    })
  }

  /**
   * Hybrid BM25 + vector + RRF. Matches plan.md §4.3 except that BM25 runs
   * via `ts_rank_cd` against the tsvector column rather than ParadeDB's
   * `paradedb.rank_bm25` (the migration uses GIN tsvector).
   *
   * `bm25Limit` / `vectorLimit` control how many rows each side contributes
   * before fusion (default 60 each). `limit` caps the fused output.
   */
  async searchHybrid(opts: {
    readonly query: string
    readonly embedding: Float32Array | number[]
    readonly limit: number
    readonly bm25Limit?: number
    readonly vectorLimit?: number
  }): Promise<PostgresHybridResult[]> {
    this.ensureOpen()
    if (opts.limit <= 0) return []
    const expr = compileTsQuery(opts.query, this.aliases)
    const bm25Limit = opts.bm25Limit ?? 60
    const vectorLimit = opts.vectorLimit ?? 60
    const literal = serialiseHalfvec(opts.embedding)
    const k = this.rrfK

    // Empty BM25 query: fall back to pure vector search.
    if (expr === '') {
      const vec = await this.searchVector(opts.embedding, opts.limit)
      if (vec.length === 0) return []
      return this.withTenant(async (tx) => {
        const ids = vec.map((r) => r.chunkId.toString())
        const rows = (await tx<{
          chunk_id: string
          document_id: string
          path: string
          ordinal: number
          content: string
          metadata: Record<string, string | number | boolean | null> | null
        }>`
          select c.chunk_id::text as chunk_id,
                 c.document_id::text as document_id,
                 d.path,
                 c.ordinal,
                 c.content,
                 d.metadata
          from memory.chunks c
          join memory.documents d
            on d.document_id = c.document_id
           and d.tenant_id = ${this.tenantId}::uuid
           and d.brain_id = ${this.brainId}::uuid
          where c.tenant_id = ${this.tenantId}::uuid
            and c.chunk_id = any (${ids}::bigint[])
        `) as ReadonlyArray<{
          chunk_id: string
          document_id: string
          path: string
          ordinal: number
          content: string
          metadata: Record<string, string | number | boolean | null> | null
        }>
        const lookup = new Map(rows.map((r) => [r.chunk_id, r]))
        const out: PostgresHybridResult[] = []
        for (const v of vec) {
          const r = lookup.get(v.chunkId.toString())
          if (r === undefined) continue
          out.push({
            chunkId: v.chunkId,
            rrfScore: v.score,
            documentId: r.document_id,
            path: r.path,
            ordinal: r.ordinal,
            content: r.content,
            ...(r.metadata !== null ? { metadata: r.metadata } : {}),
          })
        }
        return out
      })
    }

    return this.withTenant(async (tx) => {
      const rows = (await tx<{
        chunk_id: string
        rrf_score: number
        document_id: string
        path: string
        ordinal: number
        content: string
        metadata: Record<string, string | number | boolean | null> | null
      }>`
        with bm25 as (
          select c.chunk_id,
                 ts_rank_cd(c.fts_vector, to_tsquery(${this.tsConfig}, ${expr})) as score,
                 row_number() over (
                   order by ts_rank_cd(c.fts_vector, to_tsquery(${this.tsConfig}, ${expr})) desc, c.chunk_id asc
                 ) as rn
          from memory.chunks c
          join memory.documents d on d.document_id = c.document_id
          where c.tenant_id = ${this.tenantId}::uuid
            and d.tenant_id = ${this.tenantId}::uuid
            and d.brain_id  = ${this.brainId}::uuid
            and c.fts_vector @@ to_tsquery(${this.tsConfig}, ${expr})
          order by score desc
          limit ${bm25Limit}
        ),
        vec as (
          select c.chunk_id,
                 1 - (c.embedding <=> ${literal}::halfvec) as score,
                 row_number() over (
                   order by c.embedding <=> ${literal}::halfvec, c.chunk_id asc
                 ) as rn
          from memory.chunks c
          join memory.documents d on d.document_id = c.document_id
          where c.tenant_id = ${this.tenantId}::uuid
            and d.tenant_id = ${this.tenantId}::uuid
            and d.brain_id  = ${this.brainId}::uuid
            and c.embedding is not null
          order by c.embedding <=> ${literal}::halfvec
          limit ${vectorLimit}
        ),
        fused as (
          select coalesce(b.chunk_id, v.chunk_id) as chunk_id,
                 coalesce(1.0 / (${k} + b.rn), 0) + coalesce(1.0 / (${k} + v.rn), 0) as rrf_score
          from bm25 b
          full outer join vec v using (chunk_id)
        )
        select c.chunk_id::text as chunk_id,
               f.rrf_score,
               c.document_id::text as document_id,
               d.path,
               c.ordinal,
               c.content,
               d.metadata
        from fused f
        join memory.chunks c
          on c.chunk_id = f.chunk_id
         and c.tenant_id = ${this.tenantId}::uuid
        join memory.documents d
          on d.document_id = c.document_id
         and d.tenant_id = ${this.tenantId}::uuid
         and d.brain_id = ${this.brainId}::uuid
        order by f.rrf_score desc, c.chunk_id asc
        limit ${opts.limit}
      `) as ReadonlyArray<{
        chunk_id: string
        rrf_score: number
        document_id: string
        path: string
        ordinal: number
        content: string
        metadata: Record<string, string | number | boolean | null> | null
      }>
      return rows.map((r) => ({
        chunkId: BigInt(r.chunk_id),
        rrfScore: Number(r.rrf_score),
        documentId: r.document_id,
        path: r.path,
        ordinal: Number(r.ordinal),
        content: r.content,
        ...(r.metadata !== null ? { metadata: r.metadata } : {}),
      }))
    })
  }

  async getTrigramChunks(): Promise<readonly TrigramSourceChunk[]> {
    this.ensureOpen()
    return this.withTenant(async (tx) => {
      const rows = (await tx<{
        chunk_id: string
        path: string
        content: string
      }>`
        select c.chunk_id::text as chunk_id,
               d.path,
               c.content
        from memory.chunks c
        join memory.documents d
          on d.document_id = c.document_id
         and d.tenant_id = ${this.tenantId}::uuid
         and d.brain_id = ${this.brainId}::uuid
        where c.tenant_id = ${this.tenantId}::uuid
        order by c.chunk_id asc
      `) as ReadonlyArray<{
        chunk_id: string
        path: string
        content: string
      }>
      return rows.map((row) => ({
        id: row.chunk_id,
        path: row.path,
        content: row.content,
      }))
    })
  }

  async getChunk(chunkId: bigint): Promise<PostgresChunk | undefined> {
    this.ensureOpen()
    return this.withTenant(async (tx) => {
      const rows = (await tx<{
        chunk_id: string
        document_id: string
        path: string
        ordinal: number
        content: string
        tokens: number
        metadata: Record<string, string | number | boolean | null> | null
      }>`
        select c.chunk_id::text as chunk_id,
               c.document_id::text as document_id,
               d.path,
               c.ordinal, c.content, c.tokens,
               d.metadata
        from memory.chunks c
        join memory.documents d
          on d.document_id = c.document_id
         and d.tenant_id = ${this.tenantId}::uuid
         and d.brain_id = ${this.brainId}::uuid
        where c.tenant_id = ${this.tenantId}::uuid
          and c.chunk_id = ${chunkId}
        limit 1
      `) as ReadonlyArray<{
        chunk_id: string
        document_id: string
        path: string
        ordinal: number
        content: string
        tokens: number
        metadata: Record<string, string | number | boolean | null> | null
      }>
      const row = rows[0]
      if (row === undefined) return undefined
      return {
        chunkId: BigInt(row.chunk_id),
        documentId: row.document_id,
        path: row.path,
        ordinal: Number(row.ordinal),
        content: row.content,
        tokens: Number(row.tokens),
        ...(row.metadata !== null ? { metadata: row.metadata } : {}),
      }
    })
  }

  async close(): Promise<void> {
    // The caller owns the connection pool; we simply flag this index as
    // closed so further operations throw.
    this.closed = true
  }

  private async withTenant<T>(fn: (tx: PgSql) => Promise<T>): Promise<T> {
    return this.sql.begin(async (tx) => {
      await tx`select set_config('app.tenant_id', ${this.tenantId}, true)`
      return fn(tx)
    })
  }

  private ensureOpen(): void {
    if (this.closed) throw new Error('postgres-search: index is closed')
  }
}
