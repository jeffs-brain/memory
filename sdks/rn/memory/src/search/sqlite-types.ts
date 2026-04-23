export type SqlStatement = {
  all(...params: readonly unknown[]): unknown[]
  get(...params: readonly unknown[]): unknown
  run(...params: readonly unknown[]): unknown
}

export type SqlDb = {
  exec(sql: string): void
  prepare(sql: string): SqlStatement
  transaction<T>(fn: () => T): T
  close(): void
  loadVectorSupport?(): Promise<void> | void
}

export type OpenSqliteDb = (path: string) => Promise<SqlDb>
