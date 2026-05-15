// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { createEventBus } from './event-bus.js'
import type { IngestTriggerEvent, TriggerBus } from './types.js'

const validEvent = (id: string): IngestTriggerEvent => ({
  id,
  brainId: 'brain-1',
  source: 'event-bus',
  payload: { kind: 'file', path: '/docs/readme.md' },
  timestamp: new Date(),
})

describe('event-bus', () => {
  it('publish -> subscriber receives event', async () => {
    const bus = createEventBus()
    const received: IngestTriggerEvent[] = []

    bus.subscribe((event) => {
      received.push(event)
    })

    bus.publish(validEvent('e1'))
    await delay(50)

    expect(received).toHaveLength(1)
    expect(received[0].id).toBe('e1')
    await bus.close()
  })

  it('multiple subscribers all receive the same event', async () => {
    const bus = createEventBus()
    const counts = [0, 0, 0]

    bus.subscribe(() => { counts[0]++ })
    bus.subscribe(() => { counts[1]++ })
    bus.subscribe(() => { counts[2]++ })

    bus.publish(validEvent('e2'))
    await delay(50)

    expect(counts).toEqual([1, 1, 1])
    await bus.close()
  })

  it('unsubscribe stops delivery', async () => {
    const bus = createEventBus()
    let firstCount = 0
    let secondCount = 0

    const unsub = bus.subscribe(() => { firstCount++ })
    bus.subscribe(() => { secondCount++ })

    bus.publish(validEvent('e3'))
    await delay(50)

    unsub()

    bus.publish(validEvent('e4'))
    await delay(50)

    expect(firstCount).toBe(1)
    expect(secondCount).toBe(2)
    await bus.close()
  })

  it('queue depth exceeded -> oldest events dropped, warning logged', async () => {
    const warn = vi.fn()
    const bus = createEventBus({
      maxQueueDepth: 2,
      logger: { debug: vi.fn(), info: vi.fn(), warn, error: vi.fn() },
    })

    // Subscribe with a slow handler so events back up in the queue.
    const received: string[] = []
    bus.subscribe(async (event) => {
      await delay(200)
      received.push(event.id)
    })

    // Publish 4 events rapidly. The first triggers an async flush that
    // blocks for 200ms; the remaining 3 pile up in the queue (depth 2),
    // causing at least one drop.
    bus.publish(validEvent('q1'))
    bus.publish(validEvent('q2'))
    bus.publish(validEvent('q3'))
    bus.publish(validEvent('q4'))

    expect(warn).toHaveBeenCalled()
    await bus.close()
  })

  it('malformed event (missing brainId) -> rejected with error', () => {
    const bus = createEventBus()

    expect(() =>
      bus.publish({
        id: '1',
        brainId: '',
        source: 'event-bus',
        payload: { kind: 'file', path: '/x' },
        timestamp: new Date(),
      }),
    ).toThrow('brainId')
  })

  it('malformed event (missing id) -> rejected', () => {
    const bus = createEventBus()
    expect(() =>
      bus.publish({
        id: '',
        brainId: 'b',
        source: 'event-bus',
        payload: { kind: 'file', path: '/x' },
        timestamp: new Date(),
      }),
    ).toThrow('id is required')
  })

  it('malformed event (invalid payload kind) -> rejected', () => {
    const bus = createEventBus()
    expect(() =>
      bus.publish({
        id: '1',
        brainId: 'b',
        source: 'event-bus',
        payload: { kind: 'banana' as 'file', path: '/x' },
        timestamp: new Date(),
      }),
    ).toThrow('invalid payload kind')
  })

  it('malformed event (file without path) -> rejected', () => {
    const bus = createEventBus()
    expect(() =>
      bus.publish({
        id: '1',
        brainId: 'b',
        source: 'event-bus',
        payload: { kind: 'file', path: '' },
        timestamp: new Date(),
      }),
    ).toThrow('file payload requires a path')
  })

  it('malformed event (url without url) -> rejected', () => {
    const bus = createEventBus()
    expect(() =>
      bus.publish({
        id: '1',
        brainId: 'b',
        source: 'event-bus',
        payload: { kind: 'url', url: '' },
        timestamp: new Date(),
      }),
    ).toThrow('url payload requires a URL')
  })

  it('malformed event (raw without content) -> rejected', () => {
    const bus = createEventBus()
    expect(() =>
      bus.publish({
        id: '1',
        brainId: 'b',
        source: 'event-bus',
        payload: { kind: 'raw', content: '' },
        timestamp: new Date(),
      }),
    ).toThrow('raw payload requires content')
  })

  it('close() drains pending events before resolving', async () => {
    const bus = createEventBus()
    const received: string[] = []

    bus.subscribe(async (event) => {
      await delay(10)
      received.push(event.id)
    })

    for (let i = 0; i < 5; i++) {
      bus.publish(validEvent(`drain-${i}`))
    }

    await bus.close()
    expect(received).toHaveLength(5)
  })

  it('publish after close throws', async () => {
    const bus = createEventBus()
    await bus.close()

    expect(() => bus.publish(validEvent('late'))).toThrow('closed')
  })

  it('handler error is logged but does not remove handler', async () => {
    const errorFn = vi.fn()
    const bus = createEventBus({
      logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: errorFn },
    })

    let callCount = 0
    bus.subscribe(() => {
      callCount++
      throw new Error('handler fail')
    })

    bus.publish(validEvent('err1'))
    bus.publish(validEvent('err2'))
    bus.publish(validEvent('err3'))
    await delay(100)

    expect(callCount).toBe(3)
    expect(errorFn).toHaveBeenCalledTimes(3)
    await bus.close()
  })
})

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
