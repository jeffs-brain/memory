// SPDX-License-Identifier: Apache-2.0

/**
 * Optional Redis pub/sub bridge. Listens on a configurable Redis channel
 * and forwards parsed events to the local trigger bus. The Redis client
 * is injected via the RedisSubscriber interface — no hard dependency on
 * any particular Redis library.
 */

import type { Logger } from '../../llm/types.js'
import type { IngestTriggerEvent, TriggerBus } from './types.js'

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
}

export type RedisBridge = {
  close(): Promise<void>
}

export const createRedisBridge = async (opts: RedisBridgeOptions): Promise<RedisBridge> => {
  const channel = opts.channel ?? 'ingest:trigger'
  const { subscriber, bus, logger } = opts

  const onMessage = (message: string): void => {
    let event: IngestTriggerEvent
    try {
      const parsed = JSON.parse(message) as Record<string, unknown>
      event = {
        ...parsed,
        timestamp: new Date(parsed.timestamp as string),
        source: (parsed.source as IngestTriggerEvent['source']) || 'redis',
      } as IngestTriggerEvent
    } catch (err) {
      logger?.warn('trigger/redis: invalid JSON, discarding', {
        error: String(err),
        channel,
      })
      return
    }
    try {
      bus.publish(event)
    } catch (err) {
      logger?.error('trigger/redis: publish to bus failed', {
        error: String(err),
        eventId: event.id,
      })
    }
  }

  await subscriber.subscribe(channel, onMessage)

  return {
    close: async () => {
      await subscriber.unsubscribe(channel)
    },
  }
}
