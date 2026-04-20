// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { type Path, toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { createMemory } from './index.js'
import type { SearchIndex } from './types.js'

const dummyProvider = (): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  complete: async (_req: CompletionRequest): Promise<CompletionResponse> => ({
    content: '',
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async (_req: StructuredRequest) => '',
})

describe('contextualise', () => {
  it('injects top-N recalled memories into the prompt context', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)

    const paths: Path[] = [
      toPath('memory/project/tenant-a/one.md'),
      toPath('memory/project/tenant-a/two.md'),
    ]
    for (const p of paths) {
      const fm = [
        '---',
        `name: ${p}`,
        'type: project',
        'scope: project',
        'modified: 2026-04-17T00:00:00Z',
        '---',
        '',
        `body for ${p}`,
        '',
      ].join('\n')
      await store.write(p, Buffer.from(fm, 'utf8'))
    }

    const searchIndex: SearchIndex = {
      search: async () => [
        { path: paths[0] as Path, score: 0.9 },
        { path: paths[1] as Path, score: 0.6 },
      ],
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const ctx = await mem.contextualise({ message: 'tell me about auth', topK: 2 })
    expect(ctx.userMessage).toBe('tell me about auth')
    expect(ctx.memories).toHaveLength(2)
    expect(ctx.memories[0]?.path).toBe(paths[0])
    expect(ctx.systemReminder).toContain('<system-reminder>')
    expect(ctx.systemReminder).toContain('body for memory/project/tenant-a/one.md')
    expect(ctx.systemReminder).toContain('body for memory/project/tenant-a/two.md')
  })

  it('keeps project contextual recall broad via an explicit global fallback', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const seenScopes: Array<string | undefined> = []

    const projectPath = toPath('memory/project/tenant-a/repo-auth.md')
    const globalPath = toPath('memory/global/planning-style.md')
    await store.write(
      projectPath,
      Buffer.from(
        [
          '---',
          'name: repo auth',
          'type: project',
          'scope: project',
          'modified: 2026-04-17T00:00:00Z',
          '---',
          '',
          'Use OIDC for repo auth.',
          '',
        ].join('\n'),
        'utf8',
      ),
    )
    await store.write(
      globalPath,
      Buffer.from(
        [
          '---',
          'name: planning style',
          'type: reference',
          'scope: global',
          'modified: 2026-04-18T00:00:00Z',
          '---',
          '',
          'Prefer explicit plans and clear verification.',
          '',
        ].join('\n'),
        'utf8',
      ),
    )

    const searchIndex: SearchIndex = {
      search: async (_query, _embedding, opts) => {
        seenScopes.push(opts.scope)
        if (opts.scope === 'global') {
          return [{ path: globalPath, score: 0.8 }]
        }
        return [{ path: projectPath, score: 0.9 }]
      },
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const ctx = await mem.contextualise({ message: 'how should I tackle auth work?' })

    expect(seenScopes).toEqual(['project', 'global'])
    expect(ctx.memories.map((memory) => memory.path)).toEqual([projectPath, globalPath])
  })
})
