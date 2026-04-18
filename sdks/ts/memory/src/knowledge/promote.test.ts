// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { noopLogger } from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { joinPath } from '../store/index.js'
import { DRAFTS_PREFIX } from './compile.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { parseLog, readLog } from './log.js'
import { WIKI_PREFIX, createPromote } from './promote.js'

describe('promote', () => {
  it('moves drafts/<slug>.md to wiki/<slug>.md inside a single batch', async () => {
    const store = createMemStore()
    const draftPath = joinPath(DRAFTS_PREFIX, 'foo.md')
    const content = serialiseFrontmatter(
      { title: 'Foo', summary: 'A foo', tags: [], sources: [] },
      'Body of foo.',
    )
    await store.write(draftPath, Buffer.from(content, 'utf8'))

    const promote = createPromote({ store, logger: noopLogger })
    const result = await promote('foo')

    expect(String(result.to)).toBe(`${WIKI_PREFIX}/foo.md`)
    expect(await store.exists(result.to)).toBe(true)
    expect(await store.exists(draftPath)).toBe(false)

    const wikiContent = (await store.read(result.to)).toString('utf8')
    expect(wikiContent).toBe(content)

    const log = await readLog(store)
    const entries = parseLog(log)
    const promoteEntries = entries.filter((e) => e.kind === 'promote')
    expect(promoteEntries).toHaveLength(1)
    expect(promoteEntries[0]?.title).toBe('foo.md')
  })

  it('throws when the draft is missing', async () => {
    const store = createMemStore()
    const promote = createPromote({ store, logger: noopLogger })
    await expect(promote('missing')).rejects.toThrow(/draft not found/)
  })

  it.each([
    '../escape',
    '/leading-slash',
    'wiki/transport',
    'drafts/transport',
    'Transport',
    'transport notes',
    'transport\\notes',
  ])('rejects an unsafe article path: %s', async (slug) => {
    const store = createMemStore()
    const promote = createPromote({ store, logger: noopLogger })
    await expect(promote(slug)).rejects.toThrow(/article path/)
  })
})
