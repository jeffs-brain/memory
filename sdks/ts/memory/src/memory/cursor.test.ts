import { describe, expect, it } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import { toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'

describe('StoreBackedCursorStore', () => {
  it('defaults to zero when no file exists', async () => {
    const store = createMemStore()
    const cursor = createStoreBackedCursorStore(store)
    expect(await cursor.get('alice')).toBe(0)
  })

  it('round-trips set and get', async () => {
    const store = createMemStore()
    const cursor = createStoreBackedCursorStore(store)
    await cursor.set('alice', 42)
    expect(await cursor.get('alice')).toBe(42)
  })

  it('isolates cursors per actor', async () => {
    const store = createMemStore()
    const cursor = createStoreBackedCursorStore(store)
    await cursor.set('alice', 5)
    await cursor.set('bob', 11)
    expect(await cursor.get('alice')).toBe(5)
    expect(await cursor.get('bob')).toBe(11)
  })

  it('persists cursors to the underlying store as JSON', async () => {
    const store = createMemStore()
    const cursor = createStoreBackedCursorStore(store)
    await cursor.set('alice', 7)
    const raw = (
      await store.read(toPath('memory/_cursors/alice.json'))
    ).toString('utf8')
    const parsed = JSON.parse(raw) as { cursor: number; written: string }
    expect(parsed.cursor).toBe(7)
    expect(typeof parsed.written).toBe('string')
  })

  it('fresh instance reads cursor written by a previous instance', async () => {
    const store = createMemStore()
    const first = createStoreBackedCursorStore(store)
    await first.set('alice', 99)
    const second = createStoreBackedCursorStore(store)
    expect(await second.get('alice')).toBe(99)
  })

  it('isolates cursors per session for the same actor', async () => {
    const store = createMemStore()
    const cursor = createStoreBackedCursorStore(store)
    await cursor.set('alice', 5, { sessionId: 'session-a' })
    await cursor.set('alice', 11, { sessionId: 'session-b' })
    expect(await cursor.get('alice', { sessionId: 'session-a' })).toBe(5)
    expect(await cursor.get('alice', { sessionId: 'session-b' })).toBe(11)
    expect(await cursor.get('alice')).toBe(0)
  })

  it('stores session cursors alongside the legacy actor cursor', async () => {
    const store = createMemStore()
    const cursor = createStoreBackedCursorStore(store)
    await cursor.set('alice', 7)
    await cursor.set('alice', 3, { sessionId: 'session-a' })
    const raw = (
      await store.read(toPath('memory/_cursors/alice.json'))
    ).toString('utf8')
    const parsed = JSON.parse(raw) as {
      cursor: number
      written: string
      sessions: Array<{ sessionId: string; cursor: number; written: string }>
    }
    expect(parsed.cursor).toBe(7)
    expect(typeof parsed.written).toBe('string')
    expect(parsed.sessions).toEqual([
      {
        sessionId: 'session-a',
        cursor: 3,
        written: expect.any(String),
      },
    ])
  })

  it('does not reuse the legacy actor cursor for session-scoped reads', async () => {
    const store = createMemStore()
    await store.write(
      toPath('memory/_cursors/alice.json'),
      Buffer.from(JSON.stringify({ cursor: 13, written: '2026-04-17T10:00:00.000Z' }), 'utf8'),
    )
    const cursor = createStoreBackedCursorStore(store)
    expect(await cursor.get('alice')).toBe(13)
    expect(await cursor.get('alice', { sessionId: 'session-a' })).toBe(0)
  })
})
