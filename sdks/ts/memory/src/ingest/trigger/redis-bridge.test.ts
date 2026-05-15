// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { createEventBus } from './event-bus.js'
import { createRedisBridge } from './redis-bridge.js'
import type { RedisSubscriber } from './redis-bridge.js'
import type { IngestTriggerEvent } from './types.js'

const validEvent = (id: string): IngestTriggerEvent => ({
  id,
  brainId: 'brain-1',
  source: 'redis',
  payload: { kind: 'file', path: '/docs/readme.md' },
  timestamp: new Date(),
})

const mockRedisSubscriber = (messages: string[]): RedisSubscriber => {
  let handler: ((message: string) => void) | undefined
  return {
    subscribe: async (_channel: string, onMessage: (message: string) => void) => {
      handler = onMessage
      for (const msg of messages) {
        handler(msg)
      }
    },
    unsubscribe: async () => {
      handler = undefined
    },
  }
}

describe('redis-bridge', () => {
  it('valid JSON on channel -> parsed and published to bus', async () => {
    const bus = createEventBus()
    const received: IngestTriggerEvent[] = []
    bus.subscribe((event) => { received.push(event) })

    const event = validEvent('redis-1')
    const sub = mockRedisSubscriber([JSON.stringify(event)])

    const bridge = await createRedisBridge({ subscriber: sub, bus })
    await bus.close()

    expect(received).toHaveLength(1)
    expect(received[0].id).toBe('redis-1')
    await bridge.close()
  })

  it('invalid JSON on channel -> logged and discarded', async () => {
    const bus = createEventBus()
    const received: IngestTriggerEvent[] = []
    bus.subscribe((event) => { received.push(event) })

    const warn = vi.fn()
    const sub = mockRedisSubscriber(['not-json'])

    const bridge = await createRedisBridge({
      subscriber: sub,
      bus,
      logger: { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
    })
    await bus.close()

    expect(received).toHaveLength(0)
    expect(warn).toHaveBeenCalled()
    await bridge.close()
  })

  it('reconnects on subscribe error with backoff', async () => {
    const bus = createEventBus()
    const received: IngestTriggerEvent[] = []
    bus.subscribe((event) => { received.push(event) })

    const event = validEvent('redis-reconnect')
    let callCount = 0

    const failThenSucceedSubscriber: RedisSubscriber = {
      subscribe: async (_channel: string, onMessage: (message: string) => void) => {
        callCount++
        if (callCount === 1) {
          throw new Error('connection dropped')
        }
        // Second call succeeds and delivers the event.
        onMessage(JSON.stringify(event))
      },
      unsubscribe: async () => {},
    }

    const warn = vi.fn()
    const bridge = await createRedisBridge({
      subscriber: failThenSucceedSubscriber,
      bus,
      logger: { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
      reconnectDelayMs: 10,
    })

    await bus.close()

    expect(callCount).toBe(2)
    expect(received).toHaveLength(1)
    expect(received[0].id).toBe('redis-reconnect')
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('reconnecting'),
      expect.objectContaining({ error: expect.stringContaining('connection dropped') }),
    )
    await bridge.close()
  })
})
