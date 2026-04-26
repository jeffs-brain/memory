import { describe, expect, it } from 'vitest'

import type { SqlDb } from './sqlite-types.js'
import { deleteVector, upsertVector } from './vector.js'

type RecordedRun = {
  readonly sql: string
  readonly params: readonly unknown[]
}

const createRecordingDb = (rowid: number | bigint | undefined): {
  readonly db: SqlDb
  readonly runs: readonly RecordedRun[]
} => {
  const runs: RecordedRun[] = []

  return {
    runs,
    db: {
      exec: () => undefined,
      close: () => undefined,
      transaction: (fn) => fn(),
      prepare: (sql) => ({
        all: () => [],
        get: () => {
          if (sql.includes('SELECT vec_rowid')) {
            return rowid === undefined ? undefined : { vec_rowid: rowid }
          }
          if (sql.includes('COALESCE')) {
            return { next: 1 }
          }
          return undefined
        },
        run: (...params) => {
          runs.push({ sql, params })
          return undefined
        },
      }),
    },
  }
}

const runParams = (runs: readonly RecordedRun[]): readonly unknown[] =>
  runs.flatMap((run) => run.params)

describe('vector rowid writes', () => {
  it('uses number rowids so React Native op-sqlite can bridge vector writes', () => {
    const { db, runs } = createRecordingDb(undefined)

    upsertVector(db, 'memory/global/example.md', [0.25, 0.75], 'hash')

    expect(runParams(runs).some((param) => typeof param === 'bigint')).toBe(false)
    expect(runs.some((run) => run.sql.includes('knowledge_vectors'))).toBe(true)
  })

  it('normalises bigint rowids read from SQLite before deleting vectors', () => {
    const { db, runs } = createRecordingDb(7n)

    deleteVector(db, 'memory/global/example.md')

    expect(runParams(runs)).toContain(7)
    expect(runParams(runs).some((param) => typeof param === 'bigint')).toBe(false)
  })
})
