import { describe, expect, it, vi } from 'vitest'

import { joinPath } from '../store/index.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { createLintFix } from './lint-fix.js'
import { createLint } from './lint.js'
import { parseLog, readLog } from './log.js'
import { serialiseProcessedMarker } from './processed.js'
import { createTestStore, makeWords } from './test-helpers.js'
import type { CompileResult } from './types.js'

describe('lint-fix', () => {
  it('reopens processed sources for stub articles and triggers compile when available', async () => {
    const store = await createTestStore()
    await store.write(joinPath('raw/documents', 'stubby-source.md'), 'Source material for stub.')
    await store.write(
      joinPath('raw/documents', '_processed', 'stubby-source.json'),
      serialiseProcessedMarker({
        sourcePath: 'raw/documents/stubby-source.md',
        contentHash: 'hash-stub',
        processedAt: '2026-04-18T10:00:00.000Z',
        writtenPaths: ['wiki/stubby.md'],
      }),
    )
    await store.write(
      joinPath('wiki', 'stubby.md'),
      serialiseFrontmatter(
        {
          title: 'Stubby',
          summary: 'Thin article',
          tags: [],
          sources: ['raw/documents/stubby-source.md'],
        },
        'Short note with [[reference]].',
      ),
    )
    await store.write(
      joinPath('wiki', 'reference.md'),
      serialiseFrontmatter(
        { title: 'Reference', summary: 'Reference article', tags: [], sources: [] },
        `${makeWords(205)} [[reference]]`,
      ),
    )

    const compile = vi.fn(
      async (): Promise<CompileResult> => ({
        plan: {
          articles: [],
          newArticles: [],
          updates: [],
          crossReferences: [],
          concepts: [],
          processedSources: [],
        },
        written: [],
      }),
    )

    const lintFix = createLintFix({ store, compile })
    const plan = await lintFix.buildPlan(await createLint({ store })())
    const result = await lintFix.applyPlan(plan)

    expect(result.reopenedSources).toEqual(['raw/documents/stubby-source.md'])
    expect(result.clearedMarkers).toEqual(['raw/documents/_processed/stubby-source.json'])
    expect(result.compileTriggered).toBe(true)
    expect(compile).toHaveBeenCalledTimes(1)
    expect(await store.exists(joinPath('raw/documents', '_processed', 'stubby-source.json'))).toBe(
      false,
    )

    const log = parseLog(await readLog(store))
    expect(log.some((entry) => entry.kind === 'lint.fix')).toBe(true)
  })
})
