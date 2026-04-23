import { describe, expect, it } from 'vitest'

import type { CompletionRequest, CompletionResponse, Provider } from '../llm/index.js'
import { joinPath } from '../store/index.js'
import { archivedSourcePath } from './archive.js'
import { DRAFTS_PREFIX, createCompile } from './compile.js'
import { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'
import { createIngest, hashContent } from './ingest.js'
import { parseLog, readLog } from './log.js'
import { WIKI_PREFIX } from './promote.js'
import { createTestStore } from './test-helpers.js'

const makeProvider = (handler: (req: CompletionRequest) => string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  supportsStructuredDecoding: () => false,
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  async complete(req) {
    const response: CompletionResponse = {
      content: handler(req),
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }
    return response
  },
  async structured() {
    return ''
  },
})

describe('compile', () => {
  it('writes new drafts and updates existing wiki articles', async () => {
    const store = await createTestStore()
    const ingest = createIngest({
      store,
      logger: {
        debug() {},
        info() {},
        warn() {},
        error() {},
      },
    })
    const first = await ingest('Note about trains in the Netherlands.')
    const second = await ingest('Note about Dutch trams and bikes.')

    const existingPath = joinPath(WIKI_PREFIX, 'dutch-transit.md')
    await store.write(
      existingPath,
      serialiseFrontmatter(
        {
          title: 'Dutch Public Transit',
          summary: 'Existing overview',
          tags: ['transport'],
          sources: ['raw/documents/legacy.md'],
          created: '2026-01-01T00:00:00.000Z',
        },
        'Existing body with [[cities/amsterdam]] and [[transport/rail]].',
      ),
    )

    const provider = makeProvider((req) => {
      const system = req.system ?? ''
      if (system.includes('knowledge curator')) {
        return JSON.stringify({
          new_articles: [
            {
              slug: 'cycling-in-the-netherlands',
              title: 'Cycling in the Netherlands',
              summary: 'A guide to Dutch cycling.',
              source_hashes: [second.hash],
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
        body: '# Dutch Public Transit\n\nThe Dutch network links trains, trams, and bikes. See [[cities/amsterdam]] and [[transport/rail]].',
      })
    })

    const compile = createCompile({
      store,
      provider,
      logger: {
        debug() {},
        info() {},
        warn() {},
        error() {},
      },
    })
    const result = await compile()

    expect(result.plan.newArticles).toHaveLength(1)
    expect(result.plan.updates).toHaveLength(1)
    expect(result.written.map(String).sort()).toEqual([
      `${DRAFTS_PREFIX}/cycling-in-the-netherlands.md`,
      `${WIKI_PREFIX}/dutch-transit.md`,
    ])

    const draft = parseFrontmatter(
      await store.read(joinPath(DRAFTS_PREFIX, 'cycling-in-the-netherlands.md')),
    )
    expect(draft.frontmatter.title).toBe('Cycling in the Netherlands')
    expect(draft.frontmatter.sources).toContain(archivedSourcePath(second.hash))

    const updated = parseFrontmatter(await store.read(existingPath))
    expect(updated.frontmatter.title).toBe('Dutch Public Transit')
    expect(updated.frontmatter.created).toBe('2026-01-01T00:00:00.000Z')
    expect(updated.frontmatter.modified).toBeDefined()
    expect(updated.frontmatter.sources).toContain('raw/documents/legacy.md')
    expect(updated.frontmatter.sources).toContain(archivedSourcePath(first.hash))
    expect(updated.body).toContain('trains, trams, and bikes')

    const entries = parseLog(await readLog(store))
    expect(entries.filter((entry) => entry.kind === 'compile.plan')).toHaveLength(1)
    expect(entries.filter((entry) => entry.kind === 'compile.write')).toHaveLength(1)
    expect(entries.filter((entry) => entry.kind === 'compile.update')).toHaveLength(1)
    expect(hashContent('Note about trains in the Netherlands.')).toBe(first.hash)
  })
})
