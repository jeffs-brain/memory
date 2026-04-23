import { describe, expect, it } from 'vitest'

import { joinPath } from '../store/index.js'
import { createDedup } from './dedup.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { WIKI_PREFIX } from './promote.js'
import { createTestStore } from './test-helpers.js'

describe('dedup', () => {
  it('suggests a merge for near-identical articles', async () => {
    const store = await createTestStore()
    const body = 'This is shared content that should match by body hash.'
    await store.write(
      joinPath(WIKI_PREFIX, 'a.md'),
      serialiseFrontmatter(
        { title: 'Dutch Public Transit', summary: 's', tags: [], sources: [] },
        body,
      ),
    )
    await store.write(
      joinPath(WIKI_PREFIX, 'b.md'),
      serialiseFrontmatter(
        { title: 'Dutch Public Transit!', summary: 's', tags: [], sources: [] },
        body,
      ),
    )

    const report = await createDedup({ store })()
    expect(report.suggestions).toHaveLength(1)
    expect(report.suggestions[0]?.merge).toEqual(['wiki/b.md'])
  })
})
