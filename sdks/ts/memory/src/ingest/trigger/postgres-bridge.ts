// SPDX-License-Identifier: Apache-2.0

/**
 * Optional PostgreSQL LISTEN/NOTIFY bridge. Listens on a configurable
 * channel and forwards parsed events to the local trigger bus. The
 * PostgreSQL client is injected via the PgListener interface -- no hard
 * dependency on any particular driver.
 *
 * Includes automatic reconnection with configurable exponential backoff
 * matching the Go bridge pattern.
 */

import type { Logger } from '../../llm/types.js'
import type { TriggerBus } from './types.js'
import { parseIngestTriggerEvent } from './types.js'

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
  /** Delay between reconnection attempts in milliseconds. Defaults to 5000. */
  readonly reconnectDelayMs?: number
  /** Maximum backoff delay in milliseconds. Defaults to 30000. */
  readonly maxReconnectDelayMs?: number
}

export type PostgresBridge = {
  close(): Promise<void>
}

const DEFAULT_RECONNECT_DELAY_MS = 5000
const MAX_RECONNECT_DELAY_MS = 30000

export const createPostgresBridge = async (opts: PostgresBridgeOptions): Promise<PostgresBridge> => {
  const channel = opts.channel ?? 'ingest_trigger'
  const { listener, bus, logger } = opts
  const baseDelay = opts.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS
  const maxDelay = opts.maxReconnectDelayMs ?? MAX_RECONNECT_DELAY_MS

  let closed = false
  let reconnectTimer: ReturnType<typeof setTimeout> | undefined

  const onNotify = (payload: string): void => {
    const result = parseIngestTriggerEvent(payload, 'postgres')
    if (!result.ok) {
      logger?.warn('trigger/postgres: invalid JSON, discarding', {
        error: result.error,
        channel,
      })
      return
    }
    try {
      bus.publish(result.event)
    } catch (err) {
      logger?.error('trigger/postgres: publish to bus failed', {
        error: String(err),
        eventId: result.event.id,
      })
    }
  }

  const connect = async (attempt: number): Promise<void> => {
    if (closed) return
    try {
      await listener.listen(channel, onNotify)
    } catch (err) {
      if (closed) return
      const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay)
      logger?.warn('trigger/postgres: listen error, reconnecting', {
        error: String(err),
        delay: String(delay),
      })
      await new Promise<void>((resolve) => {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = undefined
          resolve()
        }, delay)
      })
      return connect(attempt + 1)
    }
  }

  await connect(0)

  return {
    close: async () => {
      closed = true
      if (reconnectTimer !== undefined) {
        clearTimeout(reconnectTimer)
        reconnectTimer = undefined
      }
      await listener.unlisten(channel)
    },
  }
}
