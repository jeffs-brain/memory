import { describe, expect, it } from 'vitest'
import { joinPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import { createDedup } from './dedup.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { WIKI_PREFIX } from './promote.js'

describe('dedup', () => {
  it('suggests a merge for near-identical articles', async () => {
    const store = createMemStore()
    const body = 'This is shared content that should match by body hash.'
    const a = serialiseFrontmatter(
      { title: 'Dutch Public Transit', summary: 's', tags: [], sources: [] },
      body,
    )
    const b = serialiseFrontmatter(
      { title: 'Dutch Public Transit!', summary: 's', tags: [], sources: [] },
      body,
    )
    await store.write(joinPath(WIKI_PREFIX, 'a.md'), Buffer.from(a, 'utf8'))
    await store.write(joinPath(WIKI_PREFIX, 'b.md'), Buffer.from(b, 'utf8'))

    const dedup = createDedup({ store })
    const report = await dedup()
    expect(report.suggestions).toHaveLength(1)
    const s = report.suggestions[0]
    expect(s).toBeDefined()
    expect(s?.merge).toHaveLength(1)
    expect(s?.reason === 'both' || s?.reason === 'content_hash' || s?.reason === 'title').toBe(true)
  })

  it('returns no suggestions when articles are unrelated', async () => {
    const store = createMemStore()
    const a = serialiseFrontmatter(
      { title: 'Trains', summary: 's', tags: [], sources: [] },
      'Trains run on tracks and are powered by overhead lines.',
    )
    const b = serialiseFrontmatter(
      { title: 'Bikes', summary: 's', tags: [], sources: [] },
      'Bikes are two-wheeled and powered by the rider.',
    )
    await store.write(joinPath(WIKI_PREFIX, 'a.md'), Buffer.from(a, 'utf8'))
    await store.write(joinPath(WIKI_PREFIX, 'b.md'), Buffer.from(b, 'utf8'))

    const dedup = createDedup({ store })
    const report = await dedup()
    expect(report.suggestions).toHaveLength(0)
  })
})
