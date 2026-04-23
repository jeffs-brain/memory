import type { OpenSqliteDb, SqlDb } from '../search/sqlite-types.js'

export const createBetterSqliteOpenDb = (): OpenSqliteDb => {
  return async (dbPath: string): Promise<SqlDb> => {
    const BetterSqlite = (await import('better-sqlite3')).default
    const db = new BetterSqlite(dbPath, { allowExtension: true })
    db.pragma('journal_mode = WAL')
    db.pragma('busy_timeout = 10000')
    db.pragma('synchronous = NORMAL')

    return {
      exec: (sql) => db.exec(sql),
      prepare: (sql) => {
        const statement = db.prepare(sql)
        return {
          all: (...params) => statement.all(...params),
          get: (...params) => statement.get(...params),
          run: (...params) => statement.run(...params),
        }
      },
      transaction: <T>(fn: () => T): T => {
        const transaction = db.transaction(fn)
        return transaction()
      },
      close: () => {
        db.close()
      },
      loadVectorSupport: async () => {
        const sqliteVec = await import('sqlite-vec')
        db.loadExtension(sqliteVec.getLoadablePath())
      },
    }
  }
}
