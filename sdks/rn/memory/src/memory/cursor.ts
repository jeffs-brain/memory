import { ErrNotFound, joinPath } from '../store/index.js'
import type { Store } from '../store/index.js'
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
  if (actorId === '') throw new Error('memory cursor: actorId must not be empty')
  const cleaned = actorId.replace(/[^A-Za-z0-9._-]/g, '_')
  if (cleaned === '' || cleaned.startsWith('.')) {
    throw new Error(`memory cursor: actorId '${actorId}' sanitises to an invalid filename`)
  }
  return cleaned
}

const normaliseSessionId = (sessionId: string | undefined): string | undefined =>
  typeof sessionId === 'string' && sessionId !== '' ? sessionId : undefined

const sanitiseCursor = (cursor: number): number => Math.max(0, Math.trunc(cursor))

const readCursor = (record: Pick<CursorRecord, 'cursor'> | undefined): number => {
  if (record === undefined) return 0
  if (typeof record.cursor !== 'number' || !Number.isFinite(record.cursor)) return 0
  return sanitiseCursor(record.cursor)
}

const hasCursor = (record: CursorRecord): boolean =>
  typeof record.cursor === 'number' && Number.isFinite(record.cursor)

const findSessionCursor = (
  sessions: unknown,
  sessionId: string,
): SessionCursorRecord | undefined => {
  if (!Array.isArray(sessions)) return undefined
  for (let index = sessions.length - 1; index >= 0; index -= 1) {
    const session = sessions[index]
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
    const written = new Date().toISOString()
    const nextCursor = sanitiseCursor(cursor)
    const sessionId = normaliseSessionId(scope?.sessionId)
    const current = (await this.readRecord(actorId)) ?? {}

    const record: CursorRecord =
      sessionId === undefined
        ? {
            cursor: nextCursor,
            written,
            ...(current.sessions === undefined ? {} : { sessions: current.sessions }),
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

    await this.store.write(path, JSON.stringify(record))
  }

  private async readRecord(actorId: string): Promise<CursorRecord | undefined> {
    const path = joinPath(CURSOR_DIR, `${sanitiseActorId(actorId)}.json`)
    try {
      const raw = await this.store.read(path)
      const parsed = JSON.parse(raw) as CursorRecord
      return parsed !== null && typeof parsed === 'object' ? parsed : undefined
    } catch (error) {
      if (error instanceof ErrNotFound) return undefined
      if (error instanceof SyntaxError) return undefined
      throw error
    }
  }
}

export const createStoreBackedCursorStore = (store: Store): CursorStore =>
  new StoreBackedCursorStore(store)
