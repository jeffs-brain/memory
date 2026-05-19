// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { createMemStore } from '../store/index.js'
import { SyncStateManager } from './sync.js'

describe('SyncStateManager', () => {
  it('sets and gets a cursor', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    await mgr.setCursor('slack', 'brain-1', {
      value: 'ts:1234567890.123456',
      updatedAt: new Date(),
      metadata: { channel: 'C123' },
    })

    const cursor = await mgr.getCursor('slack', 'brain-1')
    expect(cursor).toBeDefined()
    expect(cursor!.value).toBe('ts:1234567890.123456')
    expect(cursor!.updatedAt).toBeInstanceOf(Date)
    expect(cursor!.metadata).toEqual({ channel: 'C123' })
  })

  it('returns undefined for nonexistent cursor', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    const cursor = await mgr.getCursor('unknown', 'brain-1')
    expect(cursor).toBeUndefined()
  })

  it('clears a cursor', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    await mgr.setCursor('slack', 'brain-1', {
      value: 'cursor-abc',
      updatedAt: new Date(),
    })

    await mgr.clearCursor('slack', 'brain-1')
    const cursor = await mgr.getCursor('slack', 'brain-1')
    expect(cursor).toBeUndefined()
  })

  it('isolates cursors across connectors', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    await mgr.setCursor('slack', 'brain-1', {
      value: 'slack-cursor',
      updatedAt: new Date(),
    })
    await mgr.setCursor('gdrive', 'brain-1', {
      value: 'gdrive-cursor',
      updatedAt: new Date(),
    })

    const slack = await mgr.getCursor('slack', 'brain-1')
    const gdrive = await mgr.getCursor('gdrive', 'brain-1')

    expect(slack!.value).toBe('slack-cursor')
    expect(gdrive!.value).toBe('gdrive-cursor')
  })

  it('isolates cursors across brains', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    await mgr.setCursor('slack', 'brain-a', {
      value: 'cursor-a',
      updatedAt: new Date(),
    })
    await mgr.setCursor('slack', 'brain-b', {
      value: 'cursor-b',
      updatedAt: new Date(),
    })

    const a = await mgr.getCursor('slack', 'brain-a')
    const b = await mgr.getCursor('slack', 'brain-b')

    expect(a!.value).toBe('cursor-a')
    expect(b!.value).toBe('cursor-b')
  })

  it('overwrites cursor with generation increment', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    await mgr.setCursor('slack', 'brain-1', {
      value: 'v1',
      updatedAt: new Date(),
    })
    await mgr.setCursor('slack', 'brain-1', {
      value: 'v2',
      updatedAt: new Date(),
    })

    const cursor = await mgr.getCursor('slack', 'brain-1')
    expect(cursor!.value).toBe('v2')
  })

  it('clear is no-op for nonexistent cursor', async () => {
    const store = createMemStore()
    const mgr = new SyncStateManager(store)

    await expect(mgr.clearCursor('nonexistent', 'brain-1')).resolves.toBeUndefined()
  })
})
