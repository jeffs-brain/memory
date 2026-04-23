import { describe, expect, it } from 'vitest'

import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { createMobileStore, toPath } from './index.js'

describe('createMobileStore', () => {
  it('applies batched writes without blocking itself and keeps the batch reason on emitted events', async () => {
    const store = await createMobileStore({
      root: '/brains/demo',
      adapter: createMemoryFileAdapter(),
    })
    const events: Array<{ readonly kind: string; readonly reason?: string }> = []
    store.subscribe((event) => {
      events.push({ kind: event.kind, reason: event.reason })
    })

    await Promise.race([
      store.batch({ reason: 'initial-sync' }, async (batch) => {
        await batch.write(toPath('notes/alpha.md'), 'alpha')
        await batch.append(toPath('notes/alpha.md'), ' beta')
        await batch.write(toPath('notes/beta.md'), 'beta')
        await batch.rename(toPath('notes/beta.md'), toPath('notes/final.md'))
        await batch.delete(toPath('notes/final.md'))
      }),
      new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('batch timed out')), 1000)
      }),
    ])

    expect(await store.read(toPath('notes/alpha.md'))).toBe('alpha beta')
    expect(await store.exists(toPath('notes/final.md'))).toBe(false)
    expect(events).toEqual([
      { kind: 'created', reason: 'initial-sync' },
      { kind: 'updated', reason: 'initial-sync' },
      { kind: 'created', reason: 'initial-sync' },
      { kind: 'renamed', reason: 'initial-sync' },
      { kind: 'deleted', reason: 'initial-sync' },
    ])

    await store.close()
  })

  it('lists recursively and excludes generated files unless requested', async () => {
    const store = await createMobileStore({
      root: '/brains/demo',
      adapter: createMemoryFileAdapter(),
    })

    await store.write(toPath('memory/global/alpha.md'), 'alpha')
    await store.write(toPath('memory/global/_index.md'), 'generated')
    await store.write(toPath('memory/project/jb/beta.md'), 'beta')

    const visible = await store.list(toPath('memory'), { recursive: true })
    expect(visible.map((entry) => entry.path)).toEqual([
      'memory/global/alpha.md',
      'memory/project/jb/beta.md',
    ])

    const allFiles = await store.list(toPath('memory'), {
      recursive: true,
      includeGenerated: true,
    })
    expect(allFiles.map((entry) => entry.path)).toEqual([
      'memory/global/_index.md',
      'memory/global/alpha.md',
      'memory/project/jb/beta.md',
    ])

    await store.close()
  })
})
