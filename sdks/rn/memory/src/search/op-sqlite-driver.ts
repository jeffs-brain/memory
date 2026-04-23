import type { OpenSqliteDb, SqlDb } from './sqlite-types.js'

type OpSqliteScalar = string | number | boolean | null | ArrayBuffer | ArrayBufferView

const trimFilePrefix = (path: string): string => {
  return path.startsWith('file://') ? path.slice(7) : path
}

const splitPath = (path: string): { readonly location?: string; readonly name: string } => {
  if (path === ':memory:') return { name: path }
  const trimmed = trimFilePrefix(path).replace(/\/+$/, '')
  const index = trimmed.lastIndexOf('/')
  if (index === -1) return { name: trimmed }
  return {
    location: trimmed.slice(0, index),
    name: trimmed.slice(index + 1),
  }
}

const rowsFromResult = (result: { readonly rows: Array<Record<string, unknown>> }): unknown[] => {
  return result.rows as unknown[]
}

export const createOpSqliteOpenDb = (): OpenSqliteDb => {
  return async (dbPath: string): Promise<SqlDb> => {
    const sqlite = await import('@op-engineering/op-sqlite')
    const { location, name } = splitPath(dbPath)
    const db = sqlite.open(location === undefined ? { name } : { name, location })

    return {
      exec: (sql) => {
        db.executeSync(sql)
      },
      prepare: (sql) => ({
        all: (...params) =>
          rowsFromResult(db.executeSync(sql, params as readonly OpSqliteScalar[])),
        get: (...params) => {
          const rows = rowsFromResult(db.executeSync(sql, params as readonly OpSqliteScalar[]))
          return rows[0]
        },
        run: (...params) => db.executeSync(sql, params as readonly OpSqliteScalar[]),
      }),
      transaction: <T>(fn: () => T): T => {
        db.executeSync('BEGIN TRANSACTION;')
        try {
          const result = fn()
          db.executeSync('COMMIT;')
          return result
        } catch (error) {
          db.executeSync('ROLLBACK;')
          throw error
        }
      },
      close: () => {
        db.close()
      },
      loadVectorSupport: () => {
        // sqlite-vec is expected to be compiled into op-sqlite via
        // the package-level sqliteVec flag in the consumer app.
      },
    }
  }
}
