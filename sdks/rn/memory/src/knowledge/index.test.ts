import { describe, expect, it } from 'vitest'

import type { CompletionRequest, CompletionResponse, Provider } from '../llm/index.js'
import { joinPath } from '../store/index.js'
import { serialiseFrontmatter } from './frontmatter.js'
import { createKnowledge, createLintFix } from './index.js'
import { parseLog, readLog } from './log.js'
import { processedMarkerPath, serialiseProcessedMarker } from './processed.js'
import { createTestStore, makeWords } from './test-helpers.js'

const makeProvider = (handler: (req: CompletionRequest) => string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub',
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

describe('knowledge index', () => {
  it('exports lint-fix and wires it through createKnowledge', async () => {
    expect(typeof createLintFix).toBe('function')

    const store = await createTestStore()
    await store.write(joinPath('raw/documents', 'stubby-source.md'), 'Source material for stub.')
    await store.write(
      processedMarkerPath('stubby-source'),
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

    const knowledge = createKnowledge({
      store,
      provider: makeProvider(() =>
        JSON.stringify({
          new_articles: [],
          updates: [],
          cross_references: [],
          concepts: [],
          processed_hashes: ['stubby-source'],
        }),
      ),
    })

    const plan = await knowledge.lintFix.buildPlan(await knowledge.lint())
    expect(plan.actions[0]?.kind).toBe('rehydrate_stub')

    const result = await knowledge.lintFix.applyPlan(plan)
    expect(result.compileTriggered).toBe(true)
    expect(result.clearedMarkers).toEqual([processedMarkerPath('stubby-source')])
    expect(await store.exists(processedMarkerPath('stubby-source'))).toBe(true)

    const log = parseLog(await readLog(store))
    expect(log.some((entry) => entry.kind === 'lint.fix')).toBe(true)
    expect(log.some((entry) => entry.kind === 'compile.plan')).toBe(true)
  })
})
