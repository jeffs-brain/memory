// SPDX-License-Identifier: Apache-2.0

/**
 * Persistent cursor store. Cursors are stored as JSON blobs in the Store
 * at `memory/_cursors/<actorId>.json`, with optional per-session entries
 * kept inside the same actor record. No in-memory cache. Every call
 * round-trips through the Store so multiple memory instances can never
 * disagree on cursor state.
 */

import { ErrNotFound } from '../store/errors.js'
import type { Store } from '../store/index.js'
import { joinPath } from '../store/path.js'
import type { CursorScope, CursorStore } from './types.js'

const CURSOR_DIR = 'memory/_cursors'

type CursorRecord = {
  readonly cursor?: number
  readonly written?: string
  readonly sessions?: readonly SessionCursorRecord[]
}

type SessionCursorRecord = {
  readonly sessionId: string
  readonly cursor: number
  readonly written: string
}

const sanitiseActorId = (actorId: string): string => {
  if (!actorId) throw new Error('cursor store: actorId must not be empty')
  // Keep POSIX-friendly characters; everything else collapses to `_`. Good
  // enough for tenant/user IDs which are already UUIDs or slugs in practice.
  const cleaned = actorId.replace(/[^A-Za-z0-9._-]/g, '_')
  if (cleaned === '' || cleaned.startsWith('.')) {
    throw new Error(`cursor store: actorId '${actorId}' sanitises to an invalid filename`)
  }
  return cleaned
}

export class StoreBackedCursorStore implements CursorStore {
  constructor(private readonly store: Store) {}

  async get(actorId: string, scope?: CursorScope): Promise<number> {
    const sessionId = normaliseSessionId(scope?.sessionId)
    const record = await this.readRecord(actorId)
    if (record === undefined) return 0
    if (sessionId !== undefined) {
      return readCursor(findSessionCursor(record.sessions, sessionId))
    }
    return readCursor(record)
  }

  async set(actorId: string, cursor: number, scope?: CursorScope): Promise<void> {
    const path = joinPath(CURSOR_DIR, `${sanitiseActorId(actorId)}.json`)
    const nextCursor = sanitiseCursor(cursor)
    const written = new Date().toISOString()
    const sessionId = normaliseSessionId(scope?.sessionId)
    const current = (await this.readRecord(actorId)) ?? {}

    const record: CursorRecord =
      sessionId === undefined
        ? {
            cursor: nextCursor,
            written,
            ...(current.sessions !== undefined ? { sessions: current.sessions } : {}),
          }
        : {
            ...(hasCursor(current) ? { cursor: readCursor(current) } : {}),
            written,
            sessions: upsertSessionCursor(current.sessions, {
              sessionId,
              cursor: nextCursor,
              written,
            }),
          }

    await this.store.write(path, Buffer.from(JSON.stringify(record), 'utf8'))
  }

  private async readRecord(actorId: string): Promise<CursorRecord | undefined> {
    const path = joinPath(CURSOR_DIR, `${sanitiseActorId(actorId)}.json`)
    try {
      const raw = await this.store.read(path)
      const parsed = JSON.parse(raw.toString('utf8')) as CursorRecord
      return parsed !== null && typeof parsed === 'object' ? parsed : undefined
    } catch (err) {
      if (err instanceof ErrNotFound) return undefined
      if (err instanceof SyntaxError) return undefined
      throw err
    }
  }
}

export const createStoreBackedCursorStore = (store: Store): CursorStore =>
  new StoreBackedCursorStore(store)

const normaliseSessionId = (sessionId: string | undefined): string | undefined =>
  typeof sessionId === 'string' && sessionId !== '' ? sessionId : undefined

const sanitiseCursor = (cursor: number): number => Math.max(0, Math.trunc(cursor))

const readCursor = (record: Pick<CursorRecord, 'cursor'> | undefined): number => {
  if (record === undefined) return 0
  if (typeof record.cursor !== 'number' || !Number.isFinite(record.cursor)) {
    return 0
  }
  return sanitiseCursor(record.cursor)
}

const hasCursor = (record: CursorRecord): boolean =>
  typeof record.cursor === 'number' && Number.isFinite(record.cursor)

const findSessionCursor = (
  sessions: unknown,
  sessionId: string,
): SessionCursorRecord | undefined => {
  if (!Array.isArray(sessions)) return undefined
  for (let i = sessions.length - 1; i >= 0; i--) {
    const session = sessions[i]
    if (
      session !== null &&
      typeof session === 'object' &&
      (session as SessionCursorRecord).sessionId === sessionId &&
      typeof (session as SessionCursorRecord).written === 'string' &&
      typeof (session as SessionCursorRecord).cursor === 'number' &&
      Number.isFinite((session as SessionCursorRecord).cursor)
    ) {
      return {
        sessionId,
        written: (session as SessionCursorRecord).written,
        cursor: sanitiseCursor((session as SessionCursorRecord).cursor),
      }
    }
  }
  return undefined
}

const upsertSessionCursor = (
  sessions: unknown,
  next: SessionCursorRecord,
): readonly SessionCursorRecord[] => {
  const kept = Array.isArray(sessions)
    ? sessions.filter((session): session is SessionCursorRecord => {
        if (session === null || typeof session !== 'object') return false
        return (
          typeof (session as SessionCursorRecord).sessionId === 'string' &&
          (session as SessionCursorRecord).sessionId !== '' &&
          (session as SessionCursorRecord).sessionId !== next.sessionId &&
          typeof (session as SessionCursorRecord).written === 'string' &&
          typeof (session as SessionCursorRecord).cursor === 'number' &&
          Number.isFinite((session as SessionCursorRecord).cursor)
        )
      })
    : []

  return [...kept, next]
}
