import { describe, expect, it, vi } from 'vitest'
import { joinPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { createLint } from './lint.js'
import { WIKI_PREFIX } from './promote.js'

describe('lint', () => {
  it('passes a valid article', async () => {
    const store = createMemStore()
    const content = serialiseFrontmatter(
      { title: 'Valid', summary: 'All fields present', tags: [], sources: [] },
      `${makeWords(205)} [[valid]]`,
    )
    await store.write(joinPath(WIKI_PREFIX, 'valid.md'), Buffer.from(content, 'utf8'))

    const lint = createLint({ store })
    const report = await lint()
    expect(report.ok).toBe(true)
    expect(report.issues).toHaveLength(0)
  })

  it('fails when a required frontmatter field is missing', async () => {
    const store = createMemStore()
    // summary omitted → ends up as empty string after parse.
    const content = '---\ntitle: No Summary\ntags: []\nsources: []\n---\n\nBody.\n'
    await store.write(joinPath(WIKI_PREFIX, 'broken.md'), Buffer.from(content, 'utf8'))

    const lint = createLint({ store })
    const report = await lint()
    expect(report.ok).toBe(false)
    const missing = report.issues.find(
      (i) => i.kind === 'missing_frontmatter_field' && i.message.includes('summary'),
    )
    expect(missing).toBeDefined()
    expect(missing?.message).toContain('summary')
  })

  it('flags stub and zero-link articles while ignoring nested underscore files', async () => {
    vi.useFakeTimers()
    try {
      vi.setSystemTime(new Date('2026-04-17T10:00:00Z'))
      const store = createMemStore()

      const reference = serialiseFrontmatter(
        { title: 'Reference', summary: 'A long article with links', tags: [], sources: [] },
        `${makeWords(205)} [[reference]]`,
      )
      await store.write(joinPath(WIKI_PREFIX, 'reference.md'), Buffer.from(reference, 'utf8'))

      const stub = serialiseFrontmatter(
        { title: 'Stubby', summary: 'Short and linked', tags: [], sources: [] },
        'See [[reference]].',
      )
      await store.write(joinPath(WIKI_PREFIX, 'stubby.md'), Buffer.from(stub, 'utf8'))

      const zeroLink = serialiseFrontmatter(
        { title: 'Quiet', summary: 'Long but linkless', tags: [], sources: [] },
        makeWords(210),
      )
      await store.write(joinPath(WIKI_PREFIX, 'quiet.md'), Buffer.from(zeroLink, 'utf8'))

      const ignored = serialiseFrontmatter(
        { title: 'Shared Title', summary: 'Ignored underscore file', tags: [], sources: [] },
        'Should not be linted.',
      )
      await store.write(joinPath(WIKI_PREFIX, 'topic/_scratch.md'), Buffer.from(ignored, 'utf8'))

      const lint = createLint({ store })
      const report = await lint()

      expect(report.ok).toBe(false)
      expect(report.issues.some((issue) => issue.path === 'wiki/topic/_scratch.md')).toBe(false)
      expect(report.issues.some((issue) => issue.kind === 'stub_article' && issue.path === 'wiki/stubby.md')).toBe(true)
      expect(
        report.issues.some((issue) => issue.kind === 'zero_link_article' && issue.path === 'wiki/quiet.md'),
      ).toBe(true)
      expect(report.issues.some((issue) => issue.kind === 'zero_link_article' && issue.path === 'wiki/stubby.md')).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('flags stale sources when a source changes after the article', async () => {
    vi.useFakeTimers()
    try {
      const store = createMemStore()
      const articlePath = joinPath(WIKI_PREFIX, 'stale.md')
      const sourcePath = joinPath('ingested', 'source.md')

      vi.setSystemTime(new Date('2026-04-17T10:00:00Z'))
      const article = serialiseFrontmatter(
        {
          title: 'Stale Article',
          summary: 'References a later source',
          tags: [],
          sources: [sourcePath],
        },
        `${makeWords(205)} [[stale]]`,
      )
      await store.write(articlePath, Buffer.from(article, 'utf8'))

      vi.setSystemTime(new Date('2026-04-17T11:00:00Z'))
      await store.write(sourcePath, Buffer.from('Source content updated later.', 'utf8'))

      const lint = createLint({ store })
      const report = await lint()

      const stale = report.issues.find((issue) => issue.kind === 'stale_source' && issue.path === articlePath)
      expect(stale).toBeDefined()
      expect(stale?.message).toContain(sourcePath)
    } finally {
      vi.useRealTimers()
    }
  })
})

const makeWords = (count: number): string => {
  return Array.from({ length: count }, (_, index) => `word${index + 1}`).join(' ')
}
