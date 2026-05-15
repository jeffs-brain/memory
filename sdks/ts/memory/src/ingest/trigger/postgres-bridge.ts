// SPDX-License-Identifier: Apache-2.0

/**
 * Optional PostgreSQL LISTEN/NOTIFY bridge. Listens on a configurable
 * channel and forwards parsed events to the local trigger bus. The
 * PostgreSQL client is injected via the PgListener interface — no hard
 * dependency on any particular driver.
 */

import type { Logger } from '../../llm/types.js'
import type { IngestTriggerEvent, TriggerBus } from './types.js'

/**
 * Abstraction over a PostgreSQL LISTEN/NOTIFY client. Callers wrap
 * their pg driver (pgx, pg, etc.) to implement this interface.
 */
export type PgListener = {
  listen(channel: string, onNotify: (payload: string) => void): Promise<void>
  unlisten(channel: string): Promise<void>
}

export type PostgresBridgeOptions = {
  readonly listener: PgListener
  readonly channel?: string
  readonly bus: TriggerBus
  readonly logger?: Logger
}

export type PostgresBridge = {
  close(): Promise<void>
}

export const createPostgresBridge = async (opts: PostgresBridgeOptions): Promise<PostgresBridge> => {
  const channel = opts.channel ?? 'ingest_trigger'
  const { listener, bus, logger } = opts

  const onNotify = (payload: string): void => {
    let event: IngestTriggerEvent
    try {
      const parsed = JSON.parse(payload) as Record<string, unknown>
      event = {
        ...parsed,
        timestamp: new Date(parsed.timestamp as string),
        source: (parsed.source as IngestTriggerEvent['source']) || 'postgres',
      } as IngestTriggerEvent
    } catch (err) {
      logger?.warn('trigger/postgres: invalid JSON, discarding', {
        error: String(err),
        channel,
      })
      return
    }
    try {
      bus.publish(event)
    } catch (err) {
      logger?.error('trigger/postgres: publish to bus failed', {
        error: String(err),
        eventId: event.id,
      })
    }
  }

  await listener.listen(channel, onNotify)

  return {
    close: async () => {
      await listener.unlisten(channel)
    },
  }
}
