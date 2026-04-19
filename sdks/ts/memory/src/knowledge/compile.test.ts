// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { noopLogger, type CompletionRequest, type CompletionResponse, type Provider } from '../llm/index.js'
import { joinPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import { archivedSourcePath } from './archive.js'
import { DRAFTS_PREFIX, createCompile } from './compile.js'
import { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'
import { createIngest, hashContent } from './ingest.js'
import { parseLog, readLog } from './log.js'
import { parseProcessedMarker, processedMarkerPath } from './processed.js'
import { WIKI_PREFIX } from './promote.js'

const makeProvider = (handler: (req: CompletionRequest) => string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub',
  supportsStructuredDecoding: () => false,
  stream: async function* () {
    yield { type: 'done', stopReason: '' as const }
  },
  complete: async (req) => {
    const resp: CompletionResponse = {
      content: handler(req),
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }
    return resp
  },
  structured: async () => '',
})

describe('compile', () => {
  it('writes new articles as drafts and updates existing wiki articles', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const a = await ingest('Note about trains in the Netherlands.')
    const b = await ingest('Note about Dutch trams and bikes.')

    const existingPath = joinPath(WIKI_PREFIX, 'dutch-transit.md')
    const existingContent = serialiseFrontmatter(
      {
        title: 'Dutch Public Transit',
        summary: 'Existing overview',
        tags: ['transport'],
        sources: ['raw/documents/legacy.md'],
        created: '2026-01-01T00:00:00.000Z',
      },
      'Existing body with [[cities/amsterdam]] and [[transport/rail]].',
    )
    await store.write(existingPath, Buffer.from(existingContent, 'utf8'))

    const provider = makeProvider((req) => {
      const system = req.system ?? ''
      if (system.includes('knowledge curator')) {
        return JSON.stringify({
          new_articles: [
            {
              slug: 'cycling-in-the-netherlands',
              title: 'Cycling in the Netherlands',
              summary: 'A guide to Dutch cycling.',
              source_hashes: [b.hash],
            },
          ],
          updates: [
            {
              path: 'dutch-transit.md',
              reason: 'Add tram and bike context from the new note.',
            },
          ],
          cross_references: [],
          concepts: ['transit', 'cycling'],
        })
      }
      if (system.includes('technical writer')) {
        return JSON.stringify({
          title: 'Cycling in the Netherlands',
          summary: 'A guide to Dutch cycling.',
          tags: ['cycling', 'netherlands'],
          body: '# Cycling in the Netherlands\n\nBikes are central to Dutch mobility.',
        })
      }
      return JSON.stringify({
        title: 'Dutch Public Transit',
        summary: 'Updated overview of trains, trams, and bikes.',
        tags: ['transport', 'netherlands'],
        body: '# Dutch Public Transit\n\nThe Dutch network links trains, trams, and bikes. See [[cities/amsterdam]] and [[transport/rail]].\n\n## Sources\n\n- Raw ingested notes.',
      })
    })

    const compile = createCompile({ store, provider, logger: noopLogger })
    const result = await compile()

    expect(result.plan.articles).toHaveLength(1)
    expect(result.plan.newArticles).toHaveLength(1)
    expect(result.plan.updates).toHaveLength(1)
    expect(result.written).toHaveLength(2)

    const paths = result.written.map(String).sort()
    const draftPath = `${DRAFTS_PREFIX}/cycling-in-the-netherlands.md`
    expect(paths).toContain(draftPath)
    expect(paths).toContain(`${WIKI_PREFIX}/dutch-transit.md`)

    const draftContent = await store.read(joinPath(DRAFTS_PREFIX, 'cycling-in-the-netherlands.md'))
    const draft = parseFrontmatter(draftContent.toString('utf8'))
    expect(draft.frontmatter.title).toBe('Cycling in the Netherlands')
    expect(draft.frontmatter.sources).toContain(archivedSourcePath(b.hash))

    const updatedContent = await store.read(existingPath)
    const updated = parseFrontmatter(updatedContent.toString('utf8'))
    expect(updated.frontmatter.title).toBe('Dutch Public Transit')
    expect(updated.frontmatter.summary).toBe('Updated overview of trains, trams, and bikes.')
    expect(updated.frontmatter.created).toBe('2026-01-01T00:00:00.000Z')
    expect(updated.frontmatter.modified).toBeDefined()
    expect(updated.frontmatter.sources).toContain('raw/documents/legacy.md')
    expect(updated.frontmatter.sources).toContain(archivedSourcePath(a.hash))
    expect(updated.body).toContain('trains, trams, and bikes')
    expect(updated.body).toContain('[[cities/amsterdam]]')
    expect((await store.read(archivedSourcePath(a.hash))).toString('utf8')).toBe(
      'Note about trains in the Netherlands.',
    )
    expect((await store.read(archivedSourcePath(b.hash))).toString('utf8')).toBe(
      'Note about Dutch trams and bikes.',
    )

    const log = await readLog(store)
    const entries = parseLog(log)
    expect(entries.filter((e) => e.kind === 'compile.plan')).toHaveLength(1)
    expect(entries.filter((e) => e.kind === 'compile.write')).toHaveLength(1)
    expect(entries.filter((e) => e.kind === 'compile.update')).toHaveLength(1)

    expect(a.hash).toBe(hashContent(Buffer.from('Note about trains in the Netherlands.', 'utf8')))
  })

  it('materialises cross-reference writes into existing wiki articles', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const a = await ingest('Note about trains in the Netherlands.')
    const b = await ingest('Note about Dutch trams and bikes.')

    const existingPath = joinPath(WIKI_PREFIX, 'dutch-transit.md')
    const existingContent = serialiseFrontmatter(
      {
        title: 'Dutch Public Transit',
        summary: 'Existing overview',
        tags: ['transport'],
        sources: ['raw/documents/legacy.md'],
        created: '2026-01-01T00:00:00.000Z',
      },
      'Existing body with [[cities/amsterdam]] and [[transport/rail]].',
    )
    await store.write(existingPath, Buffer.from(existingContent, 'utf8'))

    const crossReferencePath = joinPath(WIKI_PREFIX, 'netherlands-overview.md')
    const crossReferenceContent = serialiseFrontmatter(
      {
        title: 'Netherlands Overview',
        summary: 'Country overview',
        tags: ['netherlands'],
        sources: ['raw/documents/legacy-overview.md'],
      },
      'A broad overview of the Netherlands.',
    )
    await store.write(crossReferencePath, Buffer.from(crossReferenceContent, 'utf8'))

    const provider = makeProvider((req) => {
      const system = req.system ?? ''
      if (system.includes('knowledge curator')) {
        return JSON.stringify({
          new_articles: [
            {
              slug: 'cycling-in-the-netherlands',
              title: 'Cycling in the Netherlands',
              summary: 'A guide to Dutch cycling.',
              source_hashes: [b.hash],
            },
          ],
          updates: [
            {
              path: 'dutch-transit.md',
              reason: 'Add tram and bike context from the new note.',
            },
          ],
          cross_references: [
            {
              path: 'netherlands-overview.md',
              reason: 'Add a lightweight link to the cycling article.',
            },
          ],
          concepts: ['transit', 'cycling'],
        })
      }
      if (system.includes('technical writer')) {
        return JSON.stringify({
          title: 'Cycling in the Netherlands',
          summary: 'A guide to Dutch cycling.',
          tags: ['cycling', 'netherlands'],
          body: '# Cycling in the Netherlands\n\nBikes are central to Dutch mobility.',
        })
      }

      const prompt = req.messages.map((message) => message.content ?? '').join('\n')
      if (prompt.includes('lightweight link')) {
        return JSON.stringify({
          title: 'Netherlands Overview',
          summary: 'Overview with a cross-reference to cycling.',
          tags: ['netherlands'],
          body: '# Netherlands Overview\n\nThe Netherlands has extensive cycling infrastructure.\n\nSee also [[cycling-in-the-netherlands]].',
        })
      }

      return JSON.stringify({
        title: 'Dutch Public Transit',
        summary: 'Updated overview of trains, trams, and bikes.',
        tags: ['transport', 'netherlands'],
        body: '# Dutch Public Transit\n\nThe Dutch network links trains, trams, and bikes. See [[cities/amsterdam]] and [[transport/rail]].\n\n## Sources\n\n- Raw ingested notes.',
      })
    })

    const compile = createCompile({ store, provider, logger: noopLogger })
    const result = await compile()

    expect(result.plan.articles).toHaveLength(1)
    expect(result.plan.newArticles).toHaveLength(1)
    expect(result.plan.updates).toHaveLength(1)
    expect(result.plan.crossReferences).toHaveLength(1)
    expect(result.written).toHaveLength(3)

    const paths = result.written.map(String).sort()
    expect(paths).toContain(`${DRAFTS_PREFIX}/cycling-in-the-netherlands.md`)
    expect(paths).toContain(`${WIKI_PREFIX}/dutch-transit.md`)
    expect(paths).toContain(crossReferencePath)

    const crossReferenceUpdatedContent = await store.read(crossReferencePath)
    const crossReferenceUpdated = parseFrontmatter(crossReferenceUpdatedContent.toString('utf8'))
    expect(crossReferenceUpdated.frontmatter.title).toBe('Netherlands Overview')
    expect(crossReferenceUpdated.frontmatter.modified).toBeDefined()
    expect(crossReferenceUpdated.body).toContain('[[cycling-in-the-netherlands]]')

    const log = await readLog(store)
    const entries = parseLog(log)
    expect(entries.filter((e) => e.kind === 'compile.plan')).toHaveLength(1)
    expect(entries.filter((e) => e.kind === 'compile.write')).toHaveLength(1)
    expect(entries.filter((e) => e.kind === 'compile.update')).toHaveLength(1)
    expect(entries.filter((e) => e.kind === 'compile.crossref')).toHaveLength(1)
  })

  it('marks processed sources after a successful compile and skips them on later runs', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const updateSource = await ingest('Note about ferries linking Dutch neighbourhoods.')
    const createSource = await ingest('Note about protected cycle lanes in Utrecht.')

    const existingPath = joinPath(WIKI_PREFIX, 'dutch-transport.md')
    await store.write(
      existingPath,
      Buffer.from(
        serialiseFrontmatter(
          {
            title: 'Dutch Transport',
            summary: 'Existing transport overview',
            tags: ['transport'],
            sources: ['raw/documents/legacy.md'],
            created: '2026-01-01T00:00:00.000Z',
          },
          'Existing body with [[cities/utrecht]].',
        ),
        'utf8',
      ),
    )

    let plannerCalls = 0
    const provider = makeProvider((req) => {
      const system = req.system ?? ''
      if (system.includes('knowledge curator')) {
        plannerCalls += 1
        return JSON.stringify({
          new_articles: [
            {
              slug: 'utrecht-cycle-lanes',
              title: 'Utrecht Cycle Lanes',
              summary: 'Protected cycling routes in Utrecht.',
              source_hashes: [createSource.hash],
            },
          ],
          updates: [
            {
              path: 'dutch-transport.md',
              reason: 'Add ferry context from the latest note.',
              source_hashes: [updateSource.hash],
            },
          ],
          cross_references: [],
          concepts: ['transport', 'cycling'],
          processed_hashes: [updateSource.hash, createSource.hash],
        })
      }
      if (system.includes('technical writer')) {
        return JSON.stringify({
          title: 'Utrecht Cycle Lanes',
          summary: 'Protected cycling routes in Utrecht.',
          tags: ['cycling', 'utrecht'],
          body: '# Utrecht Cycle Lanes\n\nProtected lanes support daily travel.',
        })
      }
      return JSON.stringify({
        title: 'Dutch Transport',
        summary: 'Updated transport overview.',
        tags: ['transport'],
        body: '# Dutch Transport\n\nFerries complement rail and cycling. See [[cities/utrecht]].',
      })
    })

    const compile = createCompile({ store, provider, logger: noopLogger })
    const first = await compile()

    expect([...first.plan.processedSources].sort()).toEqual([createSource.hash, updateSource.hash].sort())

    const createdMarker = parseProcessedMarker(
      (await store.read(processedMarkerPath(createSource.hash))).toString('utf8'),
    )
    const updatedMarker = parseProcessedMarker(
      (await store.read(processedMarkerPath(updateSource.hash))).toString('utf8'),
    )

    expect(createdMarker?.sourcePath).toBe(`raw/documents/${createSource.hash}.md`)
    expect(createdMarker?.contentHash).toBe(hashContent(Buffer.from('Note about protected cycle lanes in Utrecht.', 'utf8')))
    expect(createdMarker?.writtenPaths).toContain(`${DRAFTS_PREFIX}/utrecht-cycle-lanes.md`)
    expect(updatedMarker?.sourcePath).toBe(`raw/documents/${updateSource.hash}.md`)
    expect(updatedMarker?.contentHash).toBe(
      hashContent(Buffer.from('Note about ferries linking Dutch neighbourhoods.', 'utf8')),
    )
    expect(updatedMarker?.writtenPaths).toContain(`${WIKI_PREFIX}/dutch-transport.md`)

    const updated = parseFrontmatter((await store.read(existingPath)).toString('utf8'))
    expect(updated.frontmatter.sources).toContain('raw/documents/legacy.md')
    expect(updated.frontmatter.sources).toContain(archivedSourcePath(updateSource.hash))
    expect(updated.frontmatter.sources).not.toContain(archivedSourcePath(createSource.hash))
    expect(await store.exists(archivedSourcePath(createSource.hash))).toBe(true)
    expect(await store.exists(archivedSourcePath(updateSource.hash))).toBe(true)

    const duplicate = await ingest('Note about protected cycle lanes in Utrecht.')
    expect(duplicate.skipped).toBe('duplicate')

    const second = await compile()
    expect(second.plan.articles).toHaveLength(0)
    expect(second.written).toHaveLength(0)
    expect(plannerCalls).toBe(1)
  })

  it('keeps unprocessed raw notes pending for a later compile run', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const firstSource = await ingest('Note about Dutch flood barriers.')
    const secondSource = await ingest('Note about Dutch river ferries.')

    let plannerCalls = 0
    const provider = makeProvider((req) => {
      const system = req.system ?? ''
      if (system.includes('knowledge curator')) {
        plannerCalls += 1
        return plannerCalls === 1
          ? JSON.stringify({
              new_articles: [
                {
                  slug: 'dutch-flood-barriers',
                  title: 'Dutch Flood Barriers',
                  summary: 'Overview of flood defences.',
                  source_hashes: [firstSource.hash],
                },
              ],
              updates: [],
              cross_references: [],
              concepts: ['water-management'],
              processed_hashes: [firstSource.hash],
            })
          : JSON.stringify({
              new_articles: [
                {
                  slug: 'dutch-river-ferries',
                  title: 'Dutch River Ferries',
                  summary: 'Overview of ferry crossings.',
                  source_hashes: [secondSource.hash],
                },
              ],
              updates: [],
              cross_references: [],
              concepts: ['transport'],
              processed_hashes: [secondSource.hash],
            })
      }
      return JSON.stringify({
        tags: ['netherlands'],
        body: 'Compiled body.',
      })
    })

    const compile = createCompile({ store, provider, logger: noopLogger })
    const first = await compile()
    expect(first.written.map(String)).toEqual([`${DRAFTS_PREFIX}/dutch-flood-barriers.md`])
    expect(await store.exists(processedMarkerPath(firstSource.hash))).toBe(true)
    expect(await store.exists(processedMarkerPath(secondSource.hash))).toBe(false)

    const second = await compile()
    expect(second.written.map(String)).toEqual([`${DRAFTS_PREFIX}/dutch-river-ferries.md`])
    expect(await store.exists(processedMarkerPath(secondSource.hash))).toBe(true)
    expect(plannerCalls).toBe(2)
  })

  it('returns an empty plan when there are no ingests', async () => {
    const store = createMemStore()
    const provider = makeProvider(() => '{"new_articles":[],"updates":[],"cross_references":[],"concepts":[]}')
    const compile = createCompile({ store, provider, logger: noopLogger })
    const result = await compile()
    expect(result.plan.articles).toHaveLength(0)
    expect(result.plan.newArticles).toHaveLength(0)
    expect(result.plan.updates).toHaveLength(0)
    expect(result.written).toHaveLength(0)
  })

  it('reconciles a planned new article into an update when the wiki article already exists', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const source = await ingest('New note about Dutch cycling policy.')

    const existingPath = joinPath(WIKI_PREFIX, 'cycling-in-the-netherlands.md')
    await store.write(
      existingPath,
      Buffer.from(
        serialiseFrontmatter(
          {
            title: 'Cycling in the Netherlands',
            summary: 'Existing article',
            tags: ['cycling'],
            sources: ['raw/documents/legacy.md'],
            created: '2026-01-01T00:00:00.000Z',
          },
          'Existing body.',
        ),
        'utf8',
      ),
    )

    const provider = makeProvider((req) => {
      if ((req.system ?? '').includes('knowledge curator')) {
        return JSON.stringify({
          new_articles: [
            {
              slug: 'cycling-in-the-netherlands',
              title: 'Cycling in the Netherlands',
              summary: 'Updated article',
              source_hashes: [source.hash],
            },
          ],
          updates: [],
          cross_references: [],
          concepts: ['cycling'],
        })
      }
      return JSON.stringify({
        title: 'Cycling in the Netherlands',
        summary: 'Updated article',
        tags: ['cycling', 'netherlands'],
        body: 'Updated body with the new source merged in.',
      })
    })

    const compile = createCompile({ store, provider, logger: noopLogger })
    const result = await compile()

    expect(result.written.map(String)).toEqual([`${WIKI_PREFIX}/cycling-in-the-netherlands.md`])
    expect(await store.exists(joinPath(DRAFTS_PREFIX, 'cycling-in-the-netherlands.md'))).toBe(
      false,
    )

    const updated = parseFrontmatter((await store.read(existingPath)).toString('utf8'))
    expect(updated.frontmatter.summary).toBe('Updated article')
    expect(updated.frontmatter.sources).toContain('raw/documents/legacy.md')
    expect(updated.frontmatter.sources).toContain(archivedSourcePath(source.hash))
    expect(updated.body).toContain('Updated body')

    const entries = parseLog(await readLog(store))
    expect(entries.filter((entry) => entry.kind === 'compile.write')).toHaveLength(0)
    expect(entries.filter((entry) => entry.kind === 'compile.update')).toHaveLength(1)
  })

  it('fails before batch write when an existing article file blocks a nested path', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    await ingest('Nested topic note.')
    await store.write(
      joinPath(DRAFTS_PREFIX, 'topic.md'),
      Buffer.from(
        serialiseFrontmatter(
          {
            title: 'Topic',
            summary: 'Blocking draft',
            tags: [],
            sources: [],
          },
          'Blocking body.',
        ),
        'utf8',
      ),
    )

    const provider = makeProvider((req) => {
      if ((req.system ?? '').includes('knowledge curator')) {
        return JSON.stringify({
          new_articles: [
            {
              slug: 'topic/detail',
              title: 'Topic Detail',
              summary: 'Nested article',
              source_hashes: ['seed'],
            },
          ],
          updates: [],
          cross_references: [],
          concepts: [],
        })
      }
      return JSON.stringify({
        title: 'Topic Detail',
        summary: 'Nested article',
        tags: [],
        body: 'Nested body.',
      })
    })

    const compile = createCompile({ store, provider, logger: noopLogger })
    await expect(compile()).rejects.toThrow(/path conflicts detected/)
  })
})
