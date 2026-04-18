import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { parseFrontmatter } from './frontmatter.js'
import { createMemory } from './index.js'
import { reflectionPath, scopeTopic } from './paths.js'
import { parseReflectionJson } from './reflect.js'
import type { Plugin } from './types.js'

const stubProvider = (content: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    throw new Error('not implemented')
  },
  complete: async (_req: CompletionRequest): Promise<CompletionResponse> => ({
    content,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async (_req: StructuredRequest) => content,
})

describe('reflect', () => {
  it('writes a reflection file with the expected frontmatter + body', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const provider = stubProvider(
      JSON.stringify({
        outcome: 'success',
        summary: 'Shipped the port.',
        retry_feedback: 'Start with store injection and keep the write path batched.',
        open_questions: ['What about plugins for Postgres?'],
        heuristics: [
          {
            rule: 'Always inject stores',
            context: 'memory pipelines',
            confidence: 'high',
            category: 'architecture',
            scope: 'global',
            anti_pattern: false,
          },
        ],
        should_record_episode: true,
      }),
    )

    const order: string[] = []
    const plugin: Plugin = {
      name: 'probe',
      onReflectionStart: () => {
        order.push('start')
      },
      onReflectionEnd: (ctx) => {
        order.push(`end:${ctx.result?.outcome ?? 'nil'}`)
      },
    }

    const mem = createMemory({
      store,
      provider,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      plugins: [plugin],
    })

    const res = await mem.reflect({
      messages: [{ role: 'user', content: 'task' }],
      sessionId: 'sess-123',
    })
    expect(res).toBeDefined()
    expect(res?.outcome).toBe('success')
    expect(res?.summary).toBe('Shipped the port.')
    expect(res?.retryFeedback).toBe(
      'Start with store injection and keep the write path batched.',
    )
    expect(res?.shouldRecordEpisode).toBe(true)
    expect(res?.heuristics).toHaveLength(1)
    expect(res?.openQuestions).toEqual(['What about plugins for Postgres?'])

    const path = reflectionPath('sess-123')
    const content = (await store.read(path)).toString('utf8')
    expect(content.startsWith('---\n')).toBe(true)
    expect(content).toContain('session_id: sess-123')
    expect(content).toContain('type: reflection')
    expect(content).toContain('should_record_episode: true')
    expect(content).toContain('## Summary')
    expect(content).toContain('## Retry feedback')
    expect(content).toContain('Start with store injection and keep the write path batched.')
    expect(content).toContain('## Open questions')
    expect(content).toContain('- What about plugins for Postgres?')
    expect(content).toContain('## Heuristics')
    expect(content).toContain('Always inject stores')
    expect(order).toEqual(['start', 'end:success'])

    const heuristicPath = scopeTopic(
      'global',
      'tenant-a',
      'heuristic-architecture-always-inject-stores.md',
    )
    const heuristicContent = (await store.read(heuristicPath)).toString('utf8')
    const heuristicNote = parseFrontmatter(heuristicContent)
    expect(heuristicNote.frontmatter.type).toBe('feedback')
    expect(heuristicNote.frontmatter.scope).toBe('global')
    expect(heuristicNote.frontmatter.confidence).toBe('high')
    expect(heuristicNote.frontmatter.tags).toEqual(
      expect.arrayContaining(['heuristic', 'architecture', 'high', 'pattern']),
    )
    expect(heuristicNote.body).toContain('Rule: Always inject stores')
    expect(heuristicNote.body).toContain('How to apply: Use this when memory pipelines.')
    expect((await store.read(scopeTopic('global', 'tenant-a', 'MEMORY.md'))).toString('utf8')).toContain(
      'heuristic-architecture-always-inject-stores.md',
    )
  })

  it('parses retry feedback and episode recording flags from either JSON casing', () => {
    expect(
      parseReflectionJson(
        JSON.stringify({
          outcome: 'partial',
          summary: 'Needs a second pass.',
          retryFeedback: 'Check the existing route contract first.',
          shouldRecordEpisode: true,
          heuristics: [],
        }),
      ),
    ).toMatchObject({
      outcome: 'partial',
      summary: 'Needs a second pass.',
      retryFeedback: 'Check the existing route contract first.',
      shouldRecordEpisode: true,
      heuristics: [],
      openQuestions: [],
    })

    expect(parseReflectionJson(JSON.stringify({ outcome: 'success', summary: 'ok' }))).toMatchObject({
      retryFeedback: '',
      shouldRecordEpisode: false,
    })
  })

  it('updates the same heuristic note on later sessions instead of creating duplicates', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    let callCount = 0
    const provider: Provider = {
      ...stubProvider(''),
      complete: async () => {
        callCount++
        return {
          content: JSON.stringify({
            outcome: 'success',
            summary: `Run ${callCount}`,
            heuristics: [
              {
                rule: 'Keep tests narrow',
                context: 'backend route changes',
                confidence: callCount === 1 ? 'low' : 'high',
                category: 'testing',
                scope: 'project',
                anti_pattern: false,
              },
            ],
          }),
          toolCalls: [],
          usage: { inputTokens: 0, outputTokens: 0 },
          stopReason: 'end_turn',
        }
      },
    }

    const mem = createMemory({
      store,
      provider,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    await mem.reflect({
      messages: [{ role: 'user', content: 'first' }],
      sessionId: 'sess-1',
      scope: 'project',
    })
    await mem.reflect({
      messages: [{ role: 'user', content: 'second' }],
      sessionId: 'sess-2',
      scope: 'project',
    })

    const entries = await store.list(toPath('memory/project/tenant-a'), { recursive: true })
    const heuristicFiles = entries.filter(
      (entry) => !entry.isDir && entry.path.endsWith('.md') && entry.path.includes('heuristic-testing-keep-tests-narrow.md'),
    )
    expect(heuristicFiles).toHaveLength(1)

    const heuristicPath = scopeTopic(
      'project',
      'tenant-a',
      'heuristic-testing-keep-tests-narrow.md',
    )
    const heuristicContent = (await store.read(heuristicPath)).toString('utf8')
    const heuristicNote = parseFrontmatter(heuristicContent)
    expect(heuristicNote.frontmatter.confidence).toBe('high')
    expect(heuristicNote.frontmatter.session_id).toBe('sess-2')
    expect(heuristicNote.body).toContain('Rule: Keep tests narrow')
  })
})
