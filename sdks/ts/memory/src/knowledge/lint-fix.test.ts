// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi } from 'vitest'
import { joinPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { createLintFix } from './lint-fix.js'
import { createLint } from './lint.js'
import { parseLog, readLog } from './log.js'
import { serialiseProcessedMarker } from './processed.js'
import type { CompileResult } from './types.js'

describe('lint-fix', () => {
  it('builds a bounded plan from lint output and honours dry-run mode', async () => {
    const store = createMemStore()
    await seedStubArticle(store)
    await seedDuplicateGroup(store)

    const lint = createLint({ store })
    const report = await lint()
    const lintFix = createLintFix({ store, compile: vi.fn(async () => emptyCompileResult()) })

    const plan = await lintFix.buildPlan(report, { maxStubRehydrates: 1, maxDuplicateGroups: 1 })

    expect(plan.summary.stubRehydrates).toBe(1)
    expect(plan.summary.duplicateGroups).toBe(1)
    expect(plan.actions.some((action) => action.kind === 'rehydrate_stub')).toBe(true)
    expect(plan.actions.some((action) => action.kind === 'archive_duplicate_title')).toBe(true)

    const result = await lintFix.applyPlan(plan, { dryRun: true })

    expect(result.dryRun).toBe(true)
    expect(result.applied).toEqual([])
    expect(result.clearedMarkers).toEqual([])
    expect(await store.exists(joinPath('raw/documents', '_processed', 'stubby-source.json'))).toBe(true)

    const archived = await store.read(joinPath('wiki', 'shared-title-b.md'))
    const archivedText = archived.toString('utf8')
    expect(archivedText).not.toContain('archived duplicate of [[shared-title-a]]')
  })

  it('reopens processed sources for stub articles and triggers compile when available', async () => {
    const store = createMemStore()
    await seedStubArticle(store)
    const compile = vi.fn(async () => emptyCompileResult())

    const lintFix = createLintFix({ store, compile })
    const plan = await lintFix.buildPlan(await createLint({ store })())
    const result = await lintFix.applyPlan(plan)

    expect(result.dryRun).toBe(false)
    expect(result.reopenedSources).toEqual(['raw/documents/stubby-source.md'])
    expect(result.clearedMarkers).toEqual(['raw/documents/_processed/stubby-source.json'])
    expect(result.compileTriggered).toBe(true)
    expect(compile).toHaveBeenCalledTimes(1)
    expect(await store.exists(joinPath('raw/documents', '_processed', 'stubby-source.json'))).toBe(false)

    const log = parseLog(await readLog(store))
    expect(log.some((entry) => entry.kind === 'lint.fix')).toBe(true)
  })

  it('archives non-canonical duplicate titles conservatively and clears the duplicate lint', async () => {
    const store = createMemStore()
    await seedDuplicateGroup(store)

    const lintFix = createLintFix({ store })
    const report = await createLint({ store })()
    const plan = await lintFix.buildPlan(report)
    const result = await lintFix.applyPlan(plan, { runCompile: false })

    expect(result.archivedDuplicates).toEqual(['wiki/shared-title-b.md'])
    expect(result.compileTriggered).toBe(false)

    const archivedRaw = await store.read(joinPath('wiki', 'shared-title-b.md'))
    const archived = archivedRaw.toString('utf8')
    expect(archived).toContain('title: Shared Title (archived duplicate)')
    expect(archived).toContain('archived: true')
    expect(archived).toContain('superseded_by: wiki/shared-title-a.md')
    expect(archived).toContain('archived duplicate of [[shared-title-a]]')
    expect(archived).toContain('Longer duplicate article body')

    const postLint = await createLint({ store })()
    expect(postLint.issues.some((issue) => issue.kind === 'duplicate_title')).toBe(false)
  })

  it('skips stub rehydration when the sources are already pending compile', async () => {
    const store = createMemStore()
    await seedStubArticle(store, { withProcessedMarker: false })

    const lintFix = createLintFix({ store })
    const plan = await lintFix.buildPlan(await createLint({ store })())

    expect(plan.actions.some((action) => action.kind === 'rehydrate_stub')).toBe(false)
    expect(
      plan.skipped.some(
        (item) => item.kind === 'rehydrate_stub' && item.reason === 'already_pending',
      ),
    ).toBe(true)
  })
})

const emptyCompileResult = (): CompileResult => ({
  plan: {
    articles: [],
    newArticles: [],
    updates: [],
    crossReferences: [],
    concepts: [],
    processedSources: [],
  },
  written: [],
})

const seedStubArticle = async (
  store: ReturnType<typeof createMemStore>,
  opts: { withProcessedMarker?: boolean } = {},
): Promise<void> => {
  await store.write(
    joinPath('raw/documents', 'stubby-source.md'),
    Buffer.from('Source material for the stub article.', 'utf8'),
  )
  if (opts.withProcessedMarker !== false) {
    await store.write(
      joinPath('raw/documents', '_processed', 'stubby-source.json'),
      Buffer.from(
        serialiseProcessedMarker({
          sourcePath: 'raw/documents/stubby-source.md',
          contentHash: 'hash-stub',
          processedAt: '2026-04-18T10:00:00.000Z',
          writtenPaths: ['wiki/stubby.md'],
        }),
        'utf8',
      ),
    )
  }
  const stubArticle = serialiseFrontmatter(
    {
      title: 'Stubby',
      summary: 'Thin article',
      tags: [],
      sources: ['raw/documents/stubby-source.md'],
    },
    'Short note with [[reference]].',
  )
  const referenceArticle = serialiseFrontmatter(
    { title: 'Reference', summary: 'Reference article', tags: [], sources: [] },
    `${makeWords(205)} [[reference]]`,
  )
  await store.write(joinPath('wiki', 'stubby.md'), Buffer.from(stubArticle, 'utf8'))
  await store.write(joinPath('wiki', 'reference.md'), Buffer.from(referenceArticle, 'utf8'))
}

const seedDuplicateGroup = async (store: ReturnType<typeof createMemStore>): Promise<void> => {
  const canonical = serialiseFrontmatter(
    { title: 'Shared Title', summary: 'Canonical summary', tags: [], sources: [] },
    `${makeWords(260)} [[reference]]`,
  )
  const duplicate = serialiseFrontmatter(
    { title: 'Shared Title', summary: 'Duplicate summary', tags: [], sources: [] },
    `Longer duplicate article body.\n\n${makeWords(210)} [[reference]]`,
  )
  await store.write(joinPath('wiki', 'shared-title-a.md'), Buffer.from(canonical, 'utf8'))
  await store.write(joinPath('wiki', 'shared-title-b.md'), Buffer.from(duplicate, 'utf8'))
  await store.write(
    joinPath('wiki', 'reference.md'),
    Buffer.from(
      serialiseFrontmatter(
        { title: 'Reference', summary: 'Reference article', tags: [], sources: [] },
        `${makeWords(205)} [[reference]]`,
      ),
      'utf8',
    ),
  )
}

const makeWords = (count: number): string =>
  Array.from({ length: count }, (_, index) => `word${index + 1}`).join(' ')
