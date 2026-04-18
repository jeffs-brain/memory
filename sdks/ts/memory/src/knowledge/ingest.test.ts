import { describe, expect, it } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import { createIngest, hashContent, ingestedPath } from './ingest.js'
import { noopLogger } from '../llm/index.js'
import { LOG_PATH, readLog } from './log.js'

describe('ingest', () => {
  it('writes ingested/<hash>.md with matching content and hash', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const text = 'Hello Jeff — here is a note.'
    const result = await ingest(text)

    const expected = hashContent(Buffer.from(text, 'utf8'))
    expect(result.hash).toBe(expected)
    expect(result.path).toBe(ingestedPath(expected))
    expect(result.bytes).toBe(Buffer.byteLength(text, 'utf8'))

    const stored = await store.read(result.path)
    expect(stored.toString('utf8')).toBe(text)
    expect(hashContent(stored)).toBe(expected)

    const log = await readLog(store)
    expect(log).toContain('ingest')
    expect(log).toContain(expected.slice(0, 12))
    expect(log).toContain(String(result.path))
    expect(await store.exists(LOG_PATH)).toBe(true)
  })

  it('is idempotent for the same payload', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const a = await ingest('same bytes')
    const b = await ingest('same bytes')
    expect(b.skipped).toBe('duplicate')
    expect(b.hash).toBe(a.hash)
  })

  it('honours a custom name', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const result = await ingest('note body', { name: 'My Custom Note!' })
    expect(result.path).toBe(ingestedPath('my-custom-note'))
  })
})
