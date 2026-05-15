// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { createEventBus } from './event-bus.js'
import { createPostgresBridge } from './postgres-bridge.js'
import type { PgListener } from './postgres-bridge.js'
import type { IngestTriggerEvent } from './types.js'

const validEvent = (id: string): IngestTriggerEvent => ({
  id,
  brainId: 'brain-1',
  source: 'postgres',
  payload: { kind: 'file', path: '/docs/readme.md' },
  timestamp: new Date(),
})

const mockPgListener = (payloads: string[]): PgListener => {
  let handler: ((payload: string) => void) | undefined
  return {
    listen: async (_channel: string, onNotify: (payload: string) => void) => {
      handler = onNotify
      for (const p of payloads) {
        handler(p)
      }
    },
    unlisten: async () => {
      handler = undefined
    },
  }
}

describe('postgres-bridge', () => {
  it('NOTIFY payload -> parsed and published to bus', async () => {
    const bus = createEventBus()
    const received: IngestTriggerEvent[] = []
    bus.subscribe((event) => { received.push(event) })

    const event = validEvent('pg-1')
    const listener = mockPgListener([JSON.stringify(event)])

    await createPostgresBridge({ listener, bus })
    await delay(50)

    expect(received).toHaveLength(1)
    expect(received[0].id).toBe('pg-1')
    await bus.close()
  })

  it('invalid JSON payload -> logged and discarded', async () => {
    const bus = createEventBus()
    const received: IngestTriggerEvent[] = []
    bus.subscribe((event) => { received.push(event) })

    const warn = vi.fn()
    const listener = mockPgListener(['{broken'])

    await createPostgresBridge({
      listener,
      bus,
      logger: { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
    })
    await delay(50)

    expect(received).toHaveLength(0)
    expect(warn).toHaveBeenCalled()
    await bus.close()
  })
})

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
