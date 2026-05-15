// SPDX-License-Identifier: Apache-2.0

/**
 * Optional Redis pub/sub bridge. Listens on a configurable Redis channel
 * and forwards parsed events to the local trigger bus. The Redis client
 * is injected via the RedisSubscriber interface -- no hard dependency on
 * any particular Redis library.
 *
 * Includes automatic reconnection with configurable exponential backoff
 * matching the Go bridge pattern.
 */

import type { Logger } from '../../llm/types.js'
import type { TriggerBus } from './types.js'
import { parseIngestTriggerEvent } from './types.js'

/**
 * Abstraction over a Redis pub/sub client. Callers provide their own
 * implementation wrapping ioredis, node-redis, or any other library.
 */
export type RedisSubscriber = {
  subscribe(channel: string, onMessage: (message: string) => void): Promise<void>
  unsubscribe(channel: string): Promise<void>
}

export type RedisBridgeOptions = {
  readonly subscriber: RedisSubscriber
  readonly channel?: string
  readonly bus: TriggerBus
  readonly logger?: Logger
  /** Delay between reconnection attempts in milliseconds. Defaults to 5000. */
  readonly reconnectDelayMs?: number
  /** Maximum backoff delay in milliseconds. Defaults to 30000. */
  readonly maxReconnectDelayMs?: number
}

export type RedisBridge = {
  close(): Promise<void>
}

const DEFAULT_RECONNECT_DELAY_MS = 5000
const MAX_RECONNECT_DELAY_MS = 30000

export const createRedisBridge = async (opts: RedisBridgeOptions): Promise<RedisBridge> => {
  const channel = opts.channel ?? 'ingest:trigger'
  const { subscriber, bus, logger } = opts
  const baseDelay = opts.reconnectDelayMs ?? DEFAULT_RECONNECT_DELAY_MS
  const maxDelay = opts.maxReconnectDelayMs ?? MAX_RECONNECT_DELAY_MS

  let closed = false
  let reconnectTimer: ReturnType<typeof setTimeout> | undefined

  const onMessage = (message: string): void => {
    const result = parseIngestTriggerEvent(message, 'redis')
    if (!result.ok) {
      logger?.warn('trigger/redis: invalid JSON, discarding', {
        error: result.error,
        channel,
      })
      return
    }
    try {
      bus.publish(result.event)
    } catch (err) {
      logger?.error('trigger/redis: publish to bus failed', {
        error: String(err),
        eventId: result.event.id,
      })
    }
  }

  const connect = async (attempt: number): Promise<void> => {
    if (closed) return
    try {
      await subscriber.subscribe(channel, onMessage)
    } catch (err) {
      if (closed) return
      const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay)
      logger?.warn('trigger/redis: subscription error, reconnecting', {
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
      await subscriber.unsubscribe(channel)
    },
  }
}
