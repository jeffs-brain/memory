// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  DEFAULT_INIT_LOCK_TIMEOUT_MS,
  PostgresStore,
  type PgPendingQuery,
  type PgSql,
} from './store.js'

const TENANT_ID = '11111111-1111-1111-1111-111111111111'
const BRAIN_ID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'

type Recorder = {
  readonly ddl: string[]
  readonly setConfig: Array<{ key: unknown; value: unknown }>
  beginCalls: number
}

/**
 * Build a structural {@link PgSql} fake that records every `unsafe` DDL
 * statement and every parameterised `set_config(...)` tagged-template call.
 * `begin` invokes the callback with a transaction-scoped fake whose calls are
 * recorded against the same recorder, mirroring postgres.js semantics closely
 * enough to assert on the ensure-schema path without a live database.
 */
const makeFakeSql = (
  recorder: Recorder,
  opts: { failBegin?: boolean } = {},
): PgSql => {
  const resolved = <T>(rows: ReadonlyArray<T>): PgPendingQuery<T> => {
    const p = Promise.resolve(rows) as PgPendingQuery<T>
    p.simple = async () => undefined
    return p
  }

  const setConfigKey = /set_config\(\s*'([^']+)'/
  const tagged = (<T>(strings: TemplateStringsArray, ...values: unknown[]): PgPendingQuery<T> => {
    const text = strings.join('?')
    const match = setConfigKey.exec(text)
    if (match !== null) {
      recorder.setConfig.push({ key: match[1], value: values[0] })
    }
    return resolved<T>([])
  }) as PgSql

  tagged.unsafe = <T>(sql: string): PgPendingQuery<T> => {
    recorder.ddl.push(sql)
    return resolved<T>([])
  }

  tagged.begin = async <T>(fn: (sql: PgSql) => Promise<T>): Promise<T> => {
    recorder.beginCalls += 1
    if (opts.failBegin === true) throw new Error('lock_timeout: canceling statement (55P03)')
    return fn(tagged)
  }

  return tagged
}

const newRecorder = (): Recorder => ({ ddl: [], setConfig: [], beginCalls: 0 })

const makeStore = (sql: PgSql, initLockTimeoutMs?: number): PostgresStore =>
  new PostgresStore({ sql, tenantId: TENANT_ID, brainId: BRAIN_ID, initLockTimeoutMs })

describe('PostgresStore.init ensure-schema guard', () => {
  it('runs the additive content + metadata DDL exactly once across many init() calls', async () => {
    const recorder = newRecorder()
    const store = makeStore(makeFakeSql(recorder))

    await store.init()
    await store.init()
    await store.init()

    const alters = recorder.ddl.filter((s) => s.startsWith('ALTER TABLE memory.documents'))
    expect(alters).toHaveLength(2)
    expect(alters[0]).toContain('ADD COLUMN IF NOT EXISTS content bytea')
    expect(alters[1]).toContain('ADD COLUMN IF NOT EXISTS metadata jsonb')
    expect(recorder.beginCalls).toBe(1)
  })

  it('is single-flight under concurrent init() calls', async () => {
    const recorder = newRecorder()
    const store = makeStore(makeFakeSql(recorder))

    await Promise.all([store.init(), store.init(), store.init()])

    expect(recorder.beginCalls).toBe(1)
    expect(recorder.ddl.filter((s) => s.startsWith('ALTER TABLE'))).toHaveLength(2)
  })

  it('sets a transaction-local lock_timeout on the ensure-schema path', async () => {
    const recorder = newRecorder()
    const store = makeStore(makeFakeSql(recorder))

    await store.init()

    expect(recorder.setConfig).toContainEqual({
      key: 'lock_timeout',
      value: `${DEFAULT_INIT_LOCK_TIMEOUT_MS}`,
    })
  })

  it('honours a caller-supplied initLockTimeoutMs', async () => {
    const recorder = newRecorder()
    const store = makeStore(makeFakeSql(recorder), 1500)

    await store.init()

    expect(recorder.setConfig).toContainEqual({ key: 'lock_timeout', value: '1500' })
  })

  it('does not cache a failed ensure-schema so a later init() can retry', async () => {
    const recorder = newRecorder()
    const failing = makeFakeSql(recorder, { failBegin: true })
    const failingStore = makeStore(failing)

    await expect(failingStore.init()).rejects.toThrow(/lock_timeout/)
    await expect(failingStore.init()).rejects.toThrow(/lock_timeout/)
    expect(recorder.beginCalls).toBe(2)
  })
})
