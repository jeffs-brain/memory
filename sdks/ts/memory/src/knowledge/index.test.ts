import { describe, expect, it } from 'vitest'
import type { CompletionRequest, CompletionResponse, Provider } from '../llm/index.js'
import { joinPath } from '../store/index.js'
import { createMemStore } from '../store/memstore.js'
import {
  createKnowledge,
  createLintFix,
  parseLog,
  processedMarkerPath,
  readLog,
  serialiseFrontmatter,
  serialiseProcessedMarker,
} from './index.js'

const makeProvider = (handler: (req: CompletionRequest) => string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub',
  supportsStructuredDecoding: () => false,
  stream: async function* () {
    yield { type: 'done', stopReason: '' as const }
  },
  complete: async (req) => {
    const response: CompletionResponse = {
      content: handler(req),
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }
    return response
  },
  structured: async () => '',
})

describe('knowledge index', () => {
  it('exports lint-fix and wires it through createKnowledge', async () => {
    expect(typeof createLintFix).toBe('function')

    const store = createMemStore()
    const provider = makeProvider(() =>
      JSON.stringify({
        new_articles: [],
        updates: [],
        cross_references: [],
        concepts: [],
        processed_hashes: ['stubby-source'],
      }),
    )

    const knowledge = createKnowledge({ store, provider })
    await store.write(
      joinPath('ingested', 'stubby-source.md'),
      Buffer.from('Source material for the stub article.', 'utf8'),
    )
    await store.write(
      processedMarkerPath('stubby-source'),
      Buffer.from(
        serialiseProcessedMarker({
          sourcePath: 'ingested/stubby-source.md',
          contentHash: 'd8f9aa4ea36d5f7afcf286d5f43c278cead0f26a87ae7f0ef5b9e421e71ec674',
          processedAt: '2026-04-18T10:00:00.000Z',
          writtenPaths: ['wiki/stubby.md'],
        }),
        'utf8',
      ),
    )
    await store.write(
      joinPath('wiki', 'stubby.md'),
      Buffer.from(
        serialiseFrontmatter(
          {
            title: 'Stubby',
            summary: 'Thin article',
            tags: [],
            sources: ['ingested/stubby-source.md'],
          },
          'Short note with [[reference]].',
        ),
        'utf8',
      ),
    )
    await store.write(
      joinPath('wiki', 'reference.md'),
      Buffer.from(
        serialiseFrontmatter(
          {
            title: 'Reference',
            summary: 'Reference article',
            tags: [],
            sources: [],
          },
          `${makeWords(205)} [[reference]]`,
        ),
        'utf8',
      ),
    )

    const report = await knowledge.lint()
    const plan = await knowledge.lintFix.buildPlan(report)
    expect(plan.actions).toHaveLength(1)
    expect(plan.actions[0]?.kind).toBe('rehydrate_stub')

    const result = await knowledge.lintFix.applyPlan(plan)

    expect(result.compileTriggered).toBe(true)
    expect(result.clearedMarkers).toEqual([processedMarkerPath('stubby-source')])
    expect(result.compileResult?.plan.processedSources).toEqual(['stubby-source'])
    expect(await store.exists(processedMarkerPath('stubby-source'))).toBe(true)

    const log = parseLog(await readLog(store))
    expect(log.some((entry) => entry.kind === 'lint.fix')).toBe(true)
    expect(log.some((entry) => entry.kind === 'compile.plan')).toBe(true)
  })
})

const makeWords = (count: number): string =>
  Array.from({ length: count }, (_, index) => `word${index + 1}`).join(' ')
