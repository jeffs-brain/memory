// SPDX-License-Identifier: Apache-2.0

/**
 * Minimal SQLite driver abstraction.
 *
 * We support two backends:
 *
 *   - bun:sqlite (preferred). Built into the Bun runtime. Supports
 *     loadExtension, which sqlite-vec requires.
 *   - better-sqlite3 (fallback for Node / vitest-in-node runners).
 *
 * Both expose a superset of what this module needs — we trim down to a
 * tiny surface here so the rest of the code stays driver-agnostic.
 *
 * Drivers are loaded lazily via dynamic import so the package installs
 * cleanly on platforms missing better-sqlite3's native bindings.
 */

export type SqlStatement = {
  all(...params: unknown[]): unknown[]
  get(...params: unknown[]): unknown
  run(...params: unknown[]): unknown
}

export type SqlDb = {
  /** Execute one or more SQL statements with no bound parameters. */
  exec(sql: string): void
  /** Prepare a parameterised statement. */
  prepare(sql: string): SqlStatement
  /** Load a native extension from an absolute path. */
  loadExtension(path: string): void
  /** Run a callback inside a single transaction. */
  transaction<T>(fn: () => T): T
  /** Close the underlying database handle. */
  close(): void
  /**
   * Some drivers (better-sqlite3) require an unsafe flag to be flipped
   * before loadExtension works. Drivers without this concern no-op.
   */
  readonly native: unknown
}

export type DriverKind = 'bun' | 'better-sqlite3'

export type OpenOptions = {
  readonly path: string
  /**
   * Force a specific driver. When undefined we auto-detect: prefer
   * bun:sqlite when running under Bun, fall back to better-sqlite3.
   */
  readonly driver?: DriverKind
}

/**
 * Detect whether the Bun runtime is available. Running under Node (e.g.
 * vitest via node) will see globalThis.Bun === undefined.
 */
function isBunRuntime(): boolean {
  return typeof (globalThis as { Bun?: unknown }).Bun !== 'undefined'
}

/**
 * Open a fresh SQLite connection using the preferred driver. The returned
 * wrapper exposes the minimum surface this package needs.
 *
 * WAL + busy_timeout + synchronous=NORMAL pragmas are applied up front so
 * every tenant's connection behaves consistently. `_txlock=immediate` is
 * not expressible via pragma so writers go through {@link SqlDb.transaction}
 * which both drivers implement as BEGIN IMMEDIATE under the hood.
 */
export async function openDatabase(opts: OpenOptions): Promise<SqlDb> {
  const preferred: DriverKind = opts.driver ?? (isBunRuntime() ? 'bun' : 'better-sqlite3')

  if (preferred === 'bun') {
    try {
      return await openBun(opts.path)
    } catch (err) {
      // Bun was selected but failed (e.g. running under Node). Fall
      // through to better-sqlite3 so the caller's tests still run.
      if (opts.driver === 'bun') throw err
      return await openBetter(opts.path)
    }
  }
  return await openBetter(opts.path)
}

async function openBun(path: string): Promise<SqlDb> {
  // Dynamic import so Node-based runners don't blow up resolving bun:sqlite.
  const mod = (await import('bun:sqlite')) as typeof import('bun:sqlite')
  const Database = mod.Database
  const db = new Database(path)
  db.exec('PRAGMA journal_mode=WAL')
  db.exec('PRAGMA busy_timeout=10000')
  db.exec('PRAGMA synchronous=NORMAL')

  return {
    exec: (sql) => db.exec(sql),
    prepare: (sql) => {
      const stmt = db.prepare(sql)
      return {
        all: (...params) => stmt.all(...(params as never[])) as unknown[],
        get: (...params) => stmt.get(...(params as never[])) as unknown,
        run: (...params) => stmt.run(...(params as never[])) as unknown,
      }
    },
    loadExtension: (extPath) => db.loadExtension(extPath),
    transaction: <T>(fn: () => T): T => {
      const tx = db.transaction(fn)
      return tx() as T
    },
    close: () => db.close(),
    native: db,
  }
}

async function openBetter(path: string): Promise<SqlDb> {
  let Better: typeof import('better-sqlite3')
  try {
    Better = (await import('better-sqlite3')).default as unknown as typeof import('better-sqlite3')
  } catch (err) {
    throw new Error(
      `better-sqlite3 not installed. Install the optional dependency or run on Bun. Cause: ${
        (err as Error).message
      }`,
    )
  }
  // The constructor accepts an options bag; `allowExtension` must be true
  // for loadExtension to work in better-sqlite3 >= 11.
  const db = new Better(path, { allowExtension: true } as unknown as ConstructorParameters<typeof Better>[1])
  db.pragma('journal_mode = WAL')
  db.pragma('busy_timeout = 10000')
  db.pragma('synchronous = NORMAL')

  return {
    exec: (sql) => db.exec(sql),
    prepare: (sql) => {
      const stmt = db.prepare(sql)
      return {
        all: (...params) => stmt.all(...(params as never[])) as unknown[],
        get: (...params) => stmt.get(...(params as never[])) as unknown,
        run: (...params) => stmt.run(...(params as never[])) as unknown,
      }
    },
    loadExtension: (extPath) => db.loadExtension(extPath),
    transaction: <T>(fn: () => T): T => {
      const tx = db.transaction(fn)
      return tx() as T
    },
    close: () => db.close(),
    native: db,
  }
}
