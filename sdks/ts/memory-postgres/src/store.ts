// SPDX-License-Identifier: Apache-2.0

/**
 * Postgres-backed Store adapter.
 *
 * The core memory package is OSS-standalone. This package is an optional
 * adapter that adds a PostgreSQL-backed Store + SearchIndex. It ships its
 * own SQL migrations under `./migrations/` and depends only on the
 * `postgres` driver; no Drizzle dep.
 *
 * Contract:
 *   - `memory.documents` stores metadata (path, content_hash, size, source).
 *   - Git remains the source of truth for content history — we do not try to
 *     implement git semantics on top of Postgres.
 *   - We add a `content bytea` column to `memory.documents` on init via
 *     `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so the standalone OSS path
 *     (no git remote) still has something to read back. The column is
 *     additive and idempotent; production deployments can leave the default
 *     `''` if they only use Postgres as the index.
 *   - RLS is honoured by setting `app.tenant_id` via `set_config(...,true)`
 *     inside every transaction we open. The adapter never spans a user
 *     connection across tenants.
 */

import { createHash } from 'node:crypto'
import {
  ErrNotFound,
  ErrReadOnly,
  StoreError,
  isGenerated as pathIsGenerated,
  lastSegment,
  matchGlob,
  validatePath,
  type Batch,
  type BatchOptions,
  type ChangeEvent,
  type EventSink,
  type FileInfo,
  type ListOpts,
  type Path,
  type Store,
  type Unsubscribe,
} from '@jeffs-brain/memory/store'

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

const assertUuid = (value: string, label: string): void => {
  if (!UUID_RE.test(value)) throw new StoreError(`postgres-store: ${label} must be a uuid, got ${value}`)
}

/**
 * Minimal `postgres.Sql`-shaped interface. We type this structurally so the
 * memory package stays free of a hard dependency on the `postgres` driver.
 */
export type PgSql = {
  <T = unknown>(strings: TemplateStringsArray, ...values: unknown[]): PgPendingQuery<T>
  begin<T>(fn: (sql: PgSql) => Promise<T>): Promise<T>
  unsafe<T = unknown>(sql: string, params?: unknown[]): PgPendingQuery<T>
}

export type PgPendingQuery<T> = Promise<ReadonlyArray<T>> & {
  simple(): Promise<unknown>
}

export type PostgresStoreOptions = {
  /**
   * A raw `postgres.Sql` client (from `postgres` / `postgres.js`). We take
   * the tagged-template client directly rather than any ORM wrapper so that
   * SET LOCAL and RLS paths stay explicit.
   */
  readonly sql: PgSql
  readonly tenantId: string
  readonly brainId: string
  /**
   * Optional label stored in `memory.documents.source`. Defaults to
   * `'postgres-store'`.
   */
  readonly source?: string
  /**
   * If true (default) the adapter runs `ALTER TABLE memory.documents ADD
   * COLUMN IF NOT EXISTS content bytea NOT NULL DEFAULT ''` on first use.
   * Set to false in managed deployments that provision the column up front.
   */
  readonly initContentColumn?: boolean
}

/**
 * Convert a glob pattern into a Postgres LIKE pattern. Unlike {@link matchGlob}
 * this lowers only `*` → `%` and `?` → `_`; charset classes are not
 * supported in LIKE. Unknown specials collapse to their literal character.
 */
const globToLike = (glob: string): string => {
  let out = ''
  for (const ch of glob) {
    if (ch === '*') out += '%'
    else if (ch === '?') out += '_'
    else if (ch === '\\' || ch === '%' || ch === '_') out += `\\${ch}`
    else out += ch
  }
  return out
}

const sha256 = (content: Buffer): Buffer => {
  const hash = createHash('sha256')
  hash.update(content)
  return hash.digest()
}

type DocumentRow = {
  path: string
  content: Buffer
  size: number
  content_hash: Buffer
  updated_at: Date
}

const rowToFileInfo = (row: Pick<DocumentRow, 'path' | 'size' | 'updated_at'>): FileInfo => ({
  path: row.path as Path,
  size: row.size,
  modTime: row.updated_at,
  isDir: false,
})

/**
 * Create a Postgres-backed {@link Store}. Idempotently runs the additive
 * `content` column migration on the first call and returns a `Store` that
 * scopes every write to `(tenant_id, brain_id)` with RLS.
 */
export const createPostgresStore = async (opts: PostgresStoreOptions): Promise<PostgresStore> => {
  assertUuid(opts.tenantId, 'tenantId')
  assertUuid(opts.brainId, 'brainId')
  const store = new PostgresStore(opts)
  if (opts.initContentColumn !== false) {
    await store.init()
  }
  return store
}

export class PostgresStore implements Store {
  readonly sql: PgSql
  readonly tenantId: string
  readonly brainId: string
  private readonly source: string
  private readonly sinks = new Map<number, EventSink>()
  private nextSinkId = 0
  private closed = false

  constructor(opts: PostgresStoreOptions) {
    this.sql = opts.sql
    this.tenantId = opts.tenantId
    this.brainId = opts.brainId
    this.source = opts.source ?? 'postgres-store'
  }

  /**
   * Apply the additive `content` column so `read()` has something to return.
   * Idempotent — safe to call on every process boot.
   */
  async init(): Promise<void> {
    const { sql } = this
    await sql.unsafe(
      "ALTER TABLE memory.documents ADD COLUMN IF NOT EXISTS content bytea NOT NULL DEFAULT ''::bytea",
    )
  }

  async read(p: Path): Promise<Buffer> {
    this.ensureOpen()
    validatePath(p)
    return this.withTenant(async (tx) => {
      const rows = (await tx<DocumentRow>`
        select path, content, size, content_hash, updated_at
        from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${p as string}
        limit 1
      `) as ReadonlyArray<DocumentRow>
      const row = rows[0]
      if (row === undefined) throw new ErrNotFound(p)
      return Buffer.from(row.content)
    })
  }

  async exists(p: Path): Promise<boolean> {
    this.ensureOpen()
    validatePath(p)
    return this.withTenant(async (tx) => {
      const rows = (await tx<{ one: number }>`
        select 1 as one from memory.documents
        where brain_id = ${this.brainId}::uuid
          and (path = ${p as string} or path like ${`${p}/%`})
        limit 1
      `) as ReadonlyArray<{ one: number }>
      return rows.length > 0
    })
  }

  async stat(p: Path): Promise<FileInfo> {
    this.ensureOpen()
    validatePath(p)
    return this.withTenant(async (tx) => {
      const rows = (await tx<DocumentRow>`
        select path, content, size, content_hash, updated_at
        from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${p as string}
        limit 1
      `) as ReadonlyArray<DocumentRow>
      const row = rows[0]
      if (row !== undefined) return rowToFileInfo(row)
      // Treat any path with matching children as a directory, mirroring
      // MemStore semantics.
      const childRows = (await tx<{ one: number }>`
        select 1 as one from memory.documents
        where brain_id = ${this.brainId}::uuid and path like ${`${p}/%`}
        limit 1
      `) as ReadonlyArray<{ one: number }>
      if (childRows.length > 0) {
        return { path: p, size: 0, modTime: new Date(0), isDir: true }
      }
      throw new ErrNotFound(p)
    })
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    const prefix = dir === '' ? '' : dir.endsWith('/') ? dir : `${dir}/`
    const glob = opts.glob ?? ''
    const includeGenerated = opts.includeGenerated === true
    const recursive = opts.recursive === true

    // Approach: load all document rows under the prefix (LIKE `${prefix}%`)
    // and filter in code. For the typical per-brain scale this is cheap and
    // matches the semantics MemStore/FsStore expose (directory collapsing,
    // glob matching on last segment, generated filtering).
    const pattern = prefix === '' ? '%' : `${globToLike(prefix)}%`
    const rows = await this.withTenant(async (tx) =>
      (await tx<DocumentRow>`
        select path, content, size, content_hash, updated_at
        from memory.documents
        where brain_id = ${this.brainId}::uuid
          and path like ${pattern}
        order by path asc
      `) as ReadonlyArray<DocumentRow>,
    )

    const seenDirs = new Set<string>()
    const out: FileInfo[] = []
    for (const row of rows) {
      const full = row.path
      if (prefix !== '' && !full.startsWith(prefix)) continue
      const rest = full.slice(prefix.length)
      if (rest === '') continue
      if (recursive) {
        const p = full as Path
        if (!includeGenerated && pathIsGenerated(p)) continue
        if (glob !== '' && !matchGlob(glob, lastSegment(rest))) continue
        out.push(rowToFileInfo(row))
        continue
      }
      const slash = rest.indexOf('/')
      if (slash === -1) {
        const p = full as Path
        if (!includeGenerated && pathIsGenerated(p)) continue
        if (glob !== '' && !matchGlob(glob, rest)) continue
        out.push(rowToFileInfo(row))
      } else {
        const childDir = `${prefix}${rest.slice(0, slash)}`
        if (seenDirs.has(childDir)) continue
        seenDirs.add(childDir)
        if (glob !== '' && !matchGlob(glob, rest.slice(0, slash))) continue
        out.push({
          path: childDir as Path,
          size: 0,
          modTime: new Date(0),
          isDir: true,
        })
      }
    }
    out.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return out
  }

  async write(p: Path, content: Buffer): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const existed = await this.putInner(p, content)
    this.emit({
      kind: existed ? 'updated' : 'created',
      path: p,
      when: new Date(),
    })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    // Postgres has no cheap "append bytea" that preserves sha256 of the
    // concatenation, so we read-modify-write in one transaction.
    let existed = false
    await this.withTenant(async (tx) => {
      const rows = (await tx<DocumentRow>`
        select path, content, size, content_hash, updated_at
        from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${p as string}
        for update
      `) as ReadonlyArray<DocumentRow>
      const base = rows[0]?.content ?? Buffer.alloc(0)
      existed = rows[0] !== undefined
      const merged = Buffer.concat([base, content])
      await this.upsertInTx(tx, p, merged)
    })
    this.emit({
      kind: existed ? 'updated' : 'created',
      path: p,
      when: new Date(),
    })
  }

  async delete(p: Path): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const deleted = await this.withTenant(async (tx) => {
      const rows = (await tx<{ path: string }>`
        delete from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${p as string}
        returning path
      `) as ReadonlyArray<{ path: string }>
      return rows.length > 0
    })
    if (!deleted) throw new ErrNotFound(p)
    this.emit({ kind: 'deleted', path: p, when: new Date() })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    this.ensureOpen()
    validatePath(src)
    validatePath(dst)
    await this.withTenant(async (tx) => {
      const rows = (await tx<DocumentRow>`
        select path, content, size, content_hash, updated_at
        from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${src as string}
        for update
      `) as ReadonlyArray<DocumentRow>
      const row = rows[0]
      if (row === undefined) throw new ErrNotFound(src)
      await this.upsertInTx(tx, dst, row.content)
      await tx`
        delete from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${src as string}
      `
    })
    this.emit({ kind: 'renamed', path: dst, oldPath: src, when: new Date() })
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    this.ensureOpen()
    const events: ChangeEvent[] = []
    await this.withTenant(async (tx) => {
      const batch = new PostgresBatch(this, tx, events, opts.reason)
      await fn(batch)
    })
    for (const e of events) this.emit(e)
  }

  subscribe(sink: EventSink): Unsubscribe {
    const id = this.nextSinkId++
    this.sinks.set(id, sink)
    return () => {
      this.sinks.delete(id)
    }
  }

  /** Postgres rows have no filesystem path to surface to callers. */
  localPath(_p: Path): string | undefined {
    return undefined
  }

  async close(): Promise<void> {
    // We do not own the connection pool — the caller supplied the `sql`
    // client and is responsible for ending it. Flag as closed so any further
    // operations throw ErrReadOnly.
    this.closed = true
    this.sinks.clear()
  }

  // ------------------------------------------------------------------
  // helpers

  /**
   * Run `fn` inside `sql.begin` with `app.tenant_id` set. RLS policies on
   * `memory.*` tables scope to that session variable.
   */
  async withTenant<T>(fn: (tx: PgSql) => Promise<T>): Promise<T> {
    return this.sql.begin(async (tx) => {
      await tx`select set_config('app.tenant_id', ${this.tenantId}, true)`
      return fn(tx)
    })
  }

  private async putInner(p: Path, content: Buffer): Promise<boolean> {
    let existed = false
    await this.withTenant(async (tx) => {
      const rows = (await tx<{ one: number }>`
        select 1 as one from memory.documents
        where brain_id = ${this.brainId}::uuid and path = ${p as string}
        limit 1
      `) as ReadonlyArray<{ one: number }>
      existed = rows.length > 0
      await this.upsertInTx(tx, p, content)
    })
    return existed
  }

  async upsertInTx(tx: PgSql, p: Path, content: Buffer): Promise<void> {
    const hash = sha256(content)
    await tx`
      insert into memory.documents
        (brain_id, tenant_id, path, content_hash, size, source, content, updated_at)
      values
        (${this.brainId}::uuid, ${this.tenantId}::uuid, ${p as string},
         ${hash}, ${content.length}, ${this.source}, ${content}, now())
      on conflict (brain_id, path) do update set
        content_hash = excluded.content_hash,
        size         = excluded.size,
        content      = excluded.content,
        source       = excluded.source,
        updated_at   = now()
    `
  }

  emit(event: ChangeEvent): void {
    for (const sink of this.sinks.values()) sink(event)
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }
}

/**
 * Batch implementation. Every operation runs on the shared transaction the
 * parent PostgresStore opened, so a thrown error rolls every write back.
 */
class PostgresBatch implements Batch {
  constructor(
    private readonly store: PostgresStore,
    private readonly tx: PgSql,
    private readonly events: ChangeEvent[],
    private readonly reason: string,
  ) {}

  async read(p: Path): Promise<Buffer> {
    validatePath(p)
    const rows = (await this.tx<DocumentRow>`
      select path, content, size, content_hash, updated_at
      from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${p as string}
      limit 1
    `) as ReadonlyArray<DocumentRow>
    const row = rows[0]
    if (row === undefined) throw new ErrNotFound(p)
    return Buffer.from(row.content)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    const existed = await this.existsInTx(p)
    await this.store.upsertInTx(this.tx, p, content)
    this.events.push({
      kind: existed ? 'updated' : 'created',
      path: p,
      reason: this.reason,
      when: new Date(),
    })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    const rows = (await this.tx<DocumentRow>`
      select content from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${p as string}
      for update
    `) as ReadonlyArray<DocumentRow>
    const existed = rows[0] !== undefined
    const base = rows[0]?.content ?? Buffer.alloc(0)
    const merged = Buffer.concat([base, content])
    await this.store.upsertInTx(this.tx, p, merged)
    this.events.push({
      kind: existed ? 'updated' : 'created',
      path: p,
      reason: this.reason,
      when: new Date(),
    })
  }

  async delete(p: Path): Promise<void> {
    validatePath(p)
    const rows = (await this.tx<{ path: string }>`
      delete from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${p as string}
      returning path
    `) as ReadonlyArray<{ path: string }>
    if (rows.length === 0) throw new ErrNotFound(p)
    this.events.push({
      kind: 'deleted',
      path: p,
      reason: this.reason,
      when: new Date(),
    })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    validatePath(src)
    validatePath(dst)
    const rows = (await this.tx<DocumentRow>`
      select content from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${src as string}
      for update
    `) as ReadonlyArray<DocumentRow>
    const row = rows[0]
    if (row === undefined) throw new ErrNotFound(src)
    await this.store.upsertInTx(this.tx, dst, row.content)
    await this.tx`
      delete from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${src as string}
    `
    this.events.push({
      kind: 'renamed',
      path: dst,
      oldPath: src,
      reason: this.reason,
      when: new Date(),
    })
  }

  async exists(p: Path): Promise<boolean> {
    validatePath(p)
    return this.existsInTx(p)
  }

  async stat(p: Path): Promise<FileInfo> {
    validatePath(p)
    const rows = (await this.tx<DocumentRow>`
      select path, content, size, content_hash, updated_at
      from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${p as string}
      limit 1
    `) as ReadonlyArray<DocumentRow>
    const row = rows[0]
    if (row === undefined) throw new ErrNotFound(p)
    return rowToFileInfo(row)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    // Batch list shares the store's list semantics but must run inside the
    // transaction so reads see prior writes. Delegate by executing the
    // same query on `this.tx`.
    const prefix = dir === '' ? '' : dir.endsWith('/') ? dir : `${dir}/`
    const pattern = prefix === '' ? '%' : `${globToLike(prefix)}%`
    const rows = (await this.tx<DocumentRow>`
      select path, content, size, content_hash, updated_at
      from memory.documents
      where brain_id = ${this.store.brainId}::uuid
        and path like ${pattern}
      order by path asc
    `) as ReadonlyArray<DocumentRow>

    const glob = opts.glob ?? ''
    const includeGenerated = opts.includeGenerated === true
    const recursive = opts.recursive === true
    const seenDirs = new Set<string>()
    const out: FileInfo[] = []
    for (const row of rows) {
      const full = row.path
      if (prefix !== '' && !full.startsWith(prefix)) continue
      const rest = full.slice(prefix.length)
      if (rest === '') continue
      if (recursive) {
        const p = full as Path
        if (!includeGenerated && pathIsGenerated(p)) continue
        if (glob !== '' && !matchGlob(glob, lastSegment(rest))) continue
        out.push(rowToFileInfo(row))
        continue
      }
      const slash = rest.indexOf('/')
      if (slash === -1) {
        const p = full as Path
        if (!includeGenerated && pathIsGenerated(p)) continue
        if (glob !== '' && !matchGlob(glob, rest)) continue
        out.push(rowToFileInfo(row))
      } else {
        const childDir = `${prefix}${rest.slice(0, slash)}`
        if (seenDirs.has(childDir)) continue
        seenDirs.add(childDir)
        if (glob !== '' && !matchGlob(glob, rest.slice(0, slash))) continue
        out.push({
          path: childDir as Path,
          size: 0,
          modTime: new Date(0),
          isDir: true,
        })
      }
    }
    out.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return out
  }

  private async existsInTx(p: Path): Promise<boolean> {
    const rows = (await this.tx<{ one: number }>`
      select 1 as one from memory.documents
      where brain_id = ${this.store.brainId}::uuid and path = ${p as string}
      limit 1
    `) as ReadonlyArray<{ one: number }>
    return rows.length > 0
  }
}

