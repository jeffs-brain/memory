// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import type { Logger } from '../../llm/types.js'
import type {
  DocumentDetectedEvent,
  IngestHookEvent,
  Plugin,
} from '../../memory/types.js'
import { fireDocumentDetected, fireIngestEnd, fireIngestStart } from './lifecycle.js'

const makeLogger = (): Logger & { calls: { method: string; msg: string }[] } => {
  const calls: { method: string; msg: string }[] = []
  return {
    calls,
    debug: (msg: string) => { calls.push({ method: 'debug', msg }) },
    info: (msg: string) => { calls.push({ method: 'info', msg }) },
    warn: (msg: string) => { calls.push({ method: 'warn', msg }) },
    error: (msg: string) => { calls.push({ method: 'error', msg }) },
  }
}

const sampleDetectedEvent: DocumentDetectedEvent = {
  brainId: 'brain-1',
  references: [{ kind: 'url', value: 'https://example.com', confidence: 0.9 }],
  sessionId: 'sess-1',
  actorId: 'actor-1',
}

const sampleIngestEvent: IngestHookEvent = {
  brainId: 'brain-1',
  path: 'raw/documents/readme.md',
  source: 'file',
  contentType: 'text/markdown',
  bytes: 1024,
}

describe('fireDocumentDetected', () => {
  it('fires onDocumentDetected on all plugins in order', async () => {
    const order: string[] = []
    const plugins: Plugin[] = [
      { name: 'a', onDocumentDetected: () => { order.push('a'); return true } },
      { name: 'b', onDocumentDetected: () => { order.push('b'); return true } },
      { name: 'c', onDocumentDetected: () => { order.push('c'); return true } },
    ]
    const logger = makeLogger()

    const result = await fireDocumentDetected(plugins, sampleDetectedEvent, logger)

    expect(result.cancelled).toBe(false)
    expect(order).toEqual(['a', 'b', 'c'])
  })

  it('returns cancelled=true when a plugin returns false', async () => {
    const plugins: Plugin[] = [
      { name: 'a', onDocumentDetected: () => true },
      { name: 'blocker', onDocumentDetected: () => false },
      { name: 'c', onDocumentDetected: () => true },
    ]
    const logger = makeLogger()

    const result = await fireDocumentDetected(plugins, sampleDetectedEvent, logger)

    expect(result.cancelled).toBe(true)
  })

  it('swallows plugin errors and continues (fail open)', async () => {
    const order: string[] = []
    const plugins: Plugin[] = [
      { name: 'a', onDocumentDetected: () => { order.push('a'); return true } },
      {
        name: 'crasher',
        onDocumentDetected: () => { throw new Error('boom') },
      },
      { name: 'c', onDocumentDetected: () => { order.push('c'); return true } },
    ]
    const logger = makeLogger()

    const result = await fireDocumentDetected(plugins, sampleDetectedEvent, logger)

    expect(result.cancelled).toBe(false)
    expect(order).toEqual(['a', 'c'])
    expect(logger.calls.some((c) => c.method === 'warn' && c.msg.includes('onDocumentDetected'))).toBe(true)
  })

  it('skips plugins without onDocumentDetected', async () => {
    const plugins: Plugin[] = [
      { name: 'bare' },
      { name: 'with-hook', onDocumentDetected: () => true },
    ]
    const logger = makeLogger()

    const result = await fireDocumentDetected(plugins, sampleDetectedEvent, logger)

    expect(result.cancelled).toBe(false)
  })

  it('supports async hooks', async () => {
    const plugins: Plugin[] = [
      {
        name: 'async-plugin',
        onDocumentDetected: async () => {
          await new Promise((r) => setTimeout(r, 5))
          return true
        },
      },
    ]
    const logger = makeLogger()

    const result = await fireDocumentDetected(plugins, sampleDetectedEvent, logger)

    expect(result.cancelled).toBe(false)
  })
})

describe('fireIngestStart', () => {
  it('fires onIngestStart on all plugins in registration order', async () => {
    const order: string[] = []
    const plugins: Plugin[] = [
      { name: 'a', onIngestStart: () => { order.push('a'); return true } },
      { name: 'b', onIngestStart: () => { order.push('b'); return true } },
    ]
    const logger = makeLogger()

    const result = await fireIngestStart(plugins, sampleIngestEvent, logger)

    expect(result.cancelled).toBe(false)
    expect(order).toEqual(['a', 'b'])
  })

  it('returns cancelled=true when a plugin returns false', async () => {
    const plugins: Plugin[] = [
      { name: 'a', onIngestStart: () => true },
      { name: 'blocker', onIngestStart: () => false },
      { name: 'c', onIngestStart: () => true },
    ]
    const logger = makeLogger()

    const result = await fireIngestStart(plugins, sampleIngestEvent, logger)

    expect(result.cancelled).toBe(true)
  })

  it('swallows errors and does not cancel', async () => {
    const plugins: Plugin[] = [
      {
        name: 'crasher',
        onIngestStart: () => { throw new Error('fail') },
      },
    ]
    const logger = makeLogger()

    const result = await fireIngestStart(plugins, sampleIngestEvent, logger)

    expect(result.cancelled).toBe(false)
    expect(logger.calls.some((c) => c.method === 'warn')).toBe(true)
  })

  it('skips plugins without onIngestStart', async () => {
    const plugins: Plugin[] = [{ name: 'bare' }]
    const logger = makeLogger()

    const result = await fireIngestStart(plugins, sampleIngestEvent, logger)

    expect(result.cancelled).toBe(false)
  })
})

describe('fireIngestEnd', () => {
  it('fires onIngestEnd in reverse registration order', async () => {
    const order: string[] = []
    const plugins: Plugin[] = [
      { name: 'a', onIngestEnd: () => { order.push('a') } },
      { name: 'b', onIngestEnd: () => { order.push('b') } },
      { name: 'c', onIngestEnd: () => { order.push('c') } },
    ]
    const logger = makeLogger()

    await fireIngestEnd(plugins, sampleIngestEvent, logger)

    expect(order).toEqual(['c', 'b', 'a'])
  })

  it('does not fire when no onIngestEnd defined', async () => {
    const plugins: Plugin[] = [{ name: 'bare' }]
    const logger = makeLogger()

    await fireIngestEnd(plugins, sampleIngestEvent, logger)

    expect(logger.calls).toHaveLength(0)
  })

  it('swallows errors and continues to remaining plugins', async () => {
    const order: string[] = []
    const plugins: Plugin[] = [
      { name: 'a', onIngestEnd: () => { order.push('a') } },
      {
        name: 'crasher',
        onIngestEnd: () => { throw new Error('end fail') },
      },
      { name: 'c', onIngestEnd: () => { order.push('c') } },
    ]
    const logger = makeLogger()

    await fireIngestEnd(plugins, sampleIngestEvent, logger)

    // Reverse order: c runs first, crasher throws (swallowed), then a
    expect(order).toEqual(['c', 'a'])
    expect(logger.calls.some((c) => c.method === 'warn' && c.msg.includes('onIngestEnd'))).toBe(true)
  })

  it('supports async hooks', async () => {
    const called = vi.fn()
    const plugins: Plugin[] = [
      {
        name: 'async-end',
        onIngestEnd: async () => {
          await new Promise((r) => setTimeout(r, 5))
          called()
        },
      },
    ]
    const logger = makeLogger()

    await fireIngestEnd(plugins, sampleIngestEvent, logger)

    expect(called).toHaveBeenCalledTimes(1)
  })
})
