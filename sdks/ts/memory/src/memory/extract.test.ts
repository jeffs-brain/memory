// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Message,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { parseExtractionJson } from './extract.js'
import { buildFrontmatter } from './frontmatter.js'
import { createMemory } from './index.js'
import { scopeTopic } from './paths.js'
import type { Plugin } from './types.js'

const stubProvider = (content: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
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

const stubEmbedder = (): Embedder => ({
  name: () => 'stub',
  model: () => 'stub',
  dimension: () => 3,
  embed: async (texts) => texts.map(() => [1, 0, 0]),
})

const messages = (n: number): Message[] => {
  const out: Message[] = []
  for (let i = 0; i < n; i++) {
    out.push({ role: i % 2 === 0 ? 'user' : 'assistant', content: `msg ${i}` })
  }
  return out
}

describe('extract', () => {
  it('persists the notes returned by the provider under the expected path', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const provider = stubProvider(
      JSON.stringify({
        memories: [
          {
            action: 'create',
            filename: 'feedback-testing.md',
            name: 'Feedback on testing',
            description: 'User prefers integration tests',
            type: 'feedback',
            scope: 'global',
            content: 'Prefer integration tests over snapshots.',
            index_entry: '- feedback-testing.md: testing preference',
          },
          {
            action: 'create',
            filename: 'project-auth.md',
            name: 'Project auth',
            description: 'Auth provider choice',
            type: 'project',
            scope: 'project',
            content: 'Auth uses Lleverage OIDC.',
            index_entry: '- project-auth.md: auth choice',
          },
        ],
      }),
    )

    const hooks: string[] = []
    const plugin: Plugin = {
      name: 'probe',
      onExtractionStart: (ctx) => {
        hooks.push(`start:${ctx.messages.length}`)
      },
      onExtractionEnd: (ctx) => {
        hooks.push(`end:${ctx.extracted.length}`)
      },
    }

    const mem = createMemory({
      store,
      provider,
      embedder: stubEmbedder(),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      plugins: [plugin],
    })

    const result = await mem.extract({ messages: messages(8) })
    expect(result).toHaveLength(2)

    const globalPath = scopeTopic('global', 'tenant-a', 'feedback-testing.md')
    const projectPath = scopeTopic('project', 'tenant-a', 'project-auth.md')
    const globalContent = (await store.read(globalPath)).toString('utf8')
    const projectContent = (await store.read(projectPath)).toString('utf8')
    expect(globalContent).toContain('Feedback on testing')
    expect(globalContent).toContain('Prefer integration tests over snapshots')
    expect(projectContent).toContain('Auth uses Lleverage OIDC')

    // Cursor must have advanced to the full message count.
    expect(await cursorStore.get('tenant-a')).toBe(8)

    // Index files populated for each scope.
    expect((await store.read(toPath('memory/global/MEMORY.md'))).toString('utf8')).toContain(
      'feedback-testing.md',
    )
    expect(
      (await store.read(toPath('memory/project/tenant-a/MEMORY.md'))).toString('utf8'),
    ).toContain('project-auth.md')

    expect(hooks).toEqual(['start:8', 'end:2'])
  })

  it('skips provider call when new messages below the minimum', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    let called = false
    const provider: Provider = {
      ...stubProvider('{"memories":[]}'),
      complete: async () => {
        called = true
        return {
          content: '{"memories":[]}',
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
      extractMinMessages: 6,
    })
    const extracted = await mem.extract({ messages: messages(3) })
    expect(extracted).toEqual([])
    expect(called).toBe(false)
  })

  it('extracts a default two-message turn', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider(
        '{"memories":[{"action":"create","filename":"facts.md","name":"Facts","description":"facts","type":"project","scope":"project","content":"A fact.","index_entry":"- facts"}]}',
      ),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    const extracted = await mem.extract({ messages: messages(2) })

    expect(extracted).toHaveLength(1)
    expect(
      (await store.read(scopeTopic('project', 'tenant-a', 'facts.md'))).toString('utf8'),
    ).toContain('A fact.')
  })

  it('stamps superseded_by across scopes when the old note only exists elsewhere', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    await store.write(
      scopeTopic('global', 'tenant-a', 'passport-location.md'),
      Buffer.from(
        `${buildFrontmatter({
          extra: {},
          name: 'Passport location',
          description: 'Old passport storage location',
          type: 'user',
          scope: 'global',
          modified: '2026-04-18T10:00:00.000Z',
        })}The passports are in the hall drawer.\n`,
        'utf8',
      ),
    )

    const mem = createMemory({
      store,
      provider: stubProvider(
        JSON.stringify({
          memories: [
            {
              action: 'create',
              filename: 'passport-location-current.md',
              name: 'Current passport location',
              description: 'Updated passport storage location',
              type: 'project',
              scope: 'project',
              content: 'The passports are now in the bedroom safe.',
              index_entry: '- passport-location-current.md: bedroom safe',
              supersedes: 'passport-location.md',
            },
          ],
        }),
      ),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    await mem.extract({
      messages: [
        {
          role: 'user',
          content: 'Please record the updated passport location.',
        },
        {
          role: 'assistant',
          content: 'I have recorded it.',
        },
      ],
    })

    const superseded = (
      await store.read(scopeTopic('global', 'tenant-a', 'passport-location.md'))
    ).toString('utf8')

    expect(superseded).toContain('superseded_by: passport-location-current.md')
  })

  it('tracks extraction progress per session for the same actor', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    let calls = 0
    const provider: Provider = {
      ...stubProvider('{"memories":[]}'),
      complete: async () => {
        calls++
        return {
          content: '{"memories":[]}',
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
      extractMinMessages: 6,
    })

    expect(await mem.extract({ messages: messages(6), sessionId: 'session-a' })).toEqual([])
    expect(await mem.extract({ messages: messages(6), sessionId: 'session-b' })).toEqual([])
    expect(await mem.extract({ messages: messages(6), sessionId: 'session-a' })).toEqual([])

    expect(calls).toBe(2)
    expect(await cursorStore.get('tenant-a', { sessionId: 'session-a' })).toBe(6)
    expect(await cursorStore.get('tenant-a', { sessionId: 'session-b' })).toBe(6)
  })

  it('uses plain completion for extraction even when structured decoding exists', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    let completeCalls = 0
    let structuredCalls = 0
    const content = JSON.stringify({
      memories: [
        {
          action: 'create',
          filename: 'project-context.md',
          name: 'Project context',
          description: 'Important project context',
          type: 'project',
          scope: 'project',
          content: 'The project uses structured extraction.',
          index_entry: '- project-context.md: structured extraction',
        },
      ],
    })
    const provider: Provider = {
      ...stubProvider(content),
      supportsStructuredDecoding: () => true,
      complete: async () => {
        completeCalls++
        return {
          content,
          toolCalls: [],
          usage: { inputTokens: 0, outputTokens: 0 },
          stopReason: 'end_turn',
        }
      },
      structured: async () => {
        structuredCalls++
        return content
      },
    }
    const mem = createMemory({
      store,
      provider,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    const extracted = await mem.extract({ messages: messages(8) })

    expect(extracted).toHaveLength(1)
    expect(completeCalls).toBe(1)
    expect(structuredCalls).toBe(0)
  })

  it('repairs trailing commas in extraction JSON', () => {
    expect(parseExtractionJson('{"memories":[{"filename":"x.md","content":"hi",}]}')).toHaveLength(
      1,
    )
  })

  it('accepts a bare extraction array', () => {
    expect(parseExtractionJson('[{"filename":"x.md","content":"hi"}]')).toHaveLength(1)
  })

  it('includes bounded existing memories in the extraction prompt for project scope', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    await store.write(
      scopeTopic('global', 'tenant-a', 'feedback-testing.md'),
      Buffer.from(
        `${buildFrontmatter({
          extra: {},
          name: 'Testing feedback',
          description: 'Prefer integration tests',
          type: 'feedback',
          scope: 'global',
          modified: '2026-04-18T10:00:00.000Z',
        })}Prefer integration tests over snapshots.\n`,
        'utf8',
      ),
    )
    await store.write(
      scopeTopic('project', 'tenant-a', 'project-auth.md'),
      Buffer.from(
        `${buildFrontmatter({
          extra: {},
          name: 'Auth choice',
          description: 'Use OIDC for auth',
          type: 'project',
          scope: 'project',
          modified: '2026-04-18T11:00:00.000Z',
        })}The project uses OIDC.\n`,
        'utf8',
      ),
    )
    await store.write(
      toPath('memory/project/tenant-a/MEMORY.md'),
      Buffer.from('- ignore generated index\n', 'utf8'),
    )

    let prompt = ''
    const provider: Provider = {
      ...stubProvider('{"memories":[]}'),
      complete: async (req) => {
        prompt = req.messages[0]?.content ?? ''
        return {
          content: '{"memories":[]}',
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

    await mem.extract({ messages: messages(8), scope: 'project' })

    expect(prompt).toContain('## Existing memories')
    expect(prompt).toContain('### [project] project-auth.md')
    expect(prompt).toContain('### [global] feedback-testing.md')
    expect(prompt).toContain('Use OIDC for auth')
    expect(prompt).toContain('Prefer integration tests over snapshots.')
    expect(prompt).not.toContain('MEMORY.md')
  })

  it('limits existing-memory context to the requested global scope', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    await store.write(
      scopeTopic('global', 'tenant-a', 'global-preference.md'),
      Buffer.from(
        `${buildFrontmatter({
          extra: {},
          name: 'Global preference',
          description: 'Keep answers concise',
          type: 'feedback',
          scope: 'global',
          modified: '2026-04-18T10:00:00.000Z',
        })}Keep answers concise.\n`,
        'utf8',
      ),
    )
    await store.write(
      scopeTopic('project', 'tenant-a', 'project-note.md'),
      Buffer.from(
        `${buildFrontmatter({
          extra: {},
          name: 'Project note',
          description: 'Use Postgres',
          type: 'project',
          scope: 'project',
          modified: '2026-04-18T11:00:00.000Z',
        })}Use Postgres.\n`,
        'utf8',
      ),
    )

    let prompt = ''
    const provider: Provider = {
      ...stubProvider('{"memories":[]}'),
      complete: async (req) => {
        prompt = req.messages[0]?.content ?? ''
        return {
          content: '{"memories":[]}',
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

    await mem.extract({ messages: messages(8), scope: 'global' })

    expect(prompt).toContain('### [global] global-preference.md')
    expect(prompt).not.toContain('### [project] project-note.md')
  })

  it('adds replay temporal metadata for session-backed extraction', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider(
        JSON.stringify({
          memories: [
            {
              action: 'create',
              filename: 'project-context.md',
              name: 'Project context',
              description: 'Important project context',
              type: 'project',
              scope: 'project',
              content: 'The user ordered a parcel.',
              index_entry: '- project-context.md: parcel update',
            },
          ],
        }),
      ),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: messages(2),
      sessionId: 'session-a',
      sessionDate: '2023/09/30 (Sat) 18:36',
    })

    expect(extracted[0]).toMatchObject({
      sessionId: 'session-a',
      sessionDate: '2023-09-30',
      observedOn: '2023-09-30T18:36:00.000Z',
      modifiedOverride: '2023-09-30T18:36:00.000Z',
    })
    expect(extracted[0]?.content).toContain('[Observed on 2023/09/30 (Sat) 18:36]')

    const projectPath = scopeTopic('project', 'tenant-a', 'project-context.md')
    const content = (await store.read(projectPath)).toString('utf8')
    expect(content).toContain('session_id: session-a')
    expect(content).toContain('session_date: 2023-09-30')
    expect(content).toContain('observed_on: 2023-09-30T18:36:00.000Z')
    expect(content).toContain('[Date: 2023-09-30 Saturday September 2023]')
  })

  it('rewrites heuristic filenames using provider session metadata', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider(
        JSON.stringify({
          memories: [
            {
              action: 'create',
              filename: 'project-note.md',
              name: 'Project note',
              description: 'Keep the plan handy',
              type: 'project',
              scope: 'project',
              content: 'Keep the plan handy.',
              index_entry: '- project-note.md: keep the plan handy',
              sessionId: 'session-provider',
            },
          ],
        }),
      ),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'system',
          content: 'This conversation took place on 2023/07/15 (Sat) 22:42.',
        },
        {
          role: 'user',
          content:
            "I've been reading about the Amazon rainforest and its indigenous communities in National Geographic, and I just finished my fifth issue.",
        },
        {
          role: 'assistant',
          content: 'That sounds fascinating.',
        },
      ],
    })

    const heuristic = extracted.find((memory) => memory.content.includes('fifth issue'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.filename).toContain('session-provider')
    expect(heuristic?.sessionId).toBe('session-provider')
  })

  it('rewrites heuristic filenames using the system session id when not passed explicitly', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'system',
          content:
            'session_id: session-system\nThis conversation took place on 2023/07/15 (Sat) 22:42.',
        },
        {
          role: 'user',
          content:
            "I've been reading about the Amazon rainforest and its indigenous communities in National Geographic, and I just finished my fifth issue.",
        },
        {
          role: 'assistant',
          content: 'That sounds fascinating.',
        },
      ],
    })

    const heuristic = extracted.find((memory) => memory.content.includes('fifth issue'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.filename).toContain('session-system')
    expect(heuristic?.sessionId).toBe('session-system')
  })

  it('adds heuristic user fact notes for quantified user statements', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm thinking of getting a tune-up. My car was getting 30 miles per gallon in the city a few months ago.",
        },
        {
          role: 'assistant',
          content: 'A tune-up can help improve fuel efficiency.',
        },
      ],
      sessionId: 'session-a',
      sessionDate: '2023/05/22 (Mon) 19:18',
    })

    const heuristic = extracted.find((memory) => memory.content.includes('30 miles per gallon'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.scope).toBe('global')
    expect(heuristic?.type).toBe('user')
    expect(heuristic?.filename).toContain('2023-05-22')
    expect(heuristic?.filename).toContain('session-a')
  })

  it('adds heuristic user facts for cadence and storage statements', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: "I see Dr. Smith every week. I've been keeping the spare blankets under my bed.",
        },
        {
          role: 'assistant',
          content: 'Noted.',
        },
      ],
      sessionId: 'session-cadence',
      sessionDate: '2023/08/14 (Mon) 09:30',
    })

    const cadence = extracted.find((memory) =>
      memory.content.includes('I see Dr. Smith every week.'),
    )
    const storage = extracted.find((memory) =>
      memory.content.includes("I've been keeping the spare blankets under my bed."),
    )

    expect(cadence).toBeDefined()
    expect(cadence?.scope).toBe('global')
    expect(cadence?.type).toBe('user')
    expect(storage).toBeDefined()
    expect(storage?.scope).toBe('global')
    expect(storage?.type).toBe('user')
  })

  it('captures explicit bi-weekly cadence phrasing as a user fact', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: 'I meet with my coach bi-weekly to review my progress.',
        },
        {
          role: 'assistant',
          content: 'Noted.',
        },
      ],
      sessionId: 'session-biweekly',
      sessionDate: '2024/03/25 (Mon) 09:15',
    })

    const heuristic = extracted.find((memory) => memory.content.includes('bi-weekly'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.scope).toBe('global')
    expect(heuristic?.type).toBe('user')
    expect(heuristic?.filename).toContain('session-biweekly')
  })

  it('captures bandwidth and transfer-rate units as quantified user facts', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: "I've upgraded my line to 500 Mbps and the backup sync now runs at 1.5 GB/s.",
        },
        {
          role: 'assistant',
          content: 'That is a substantial increase.',
        },
      ],
      sessionId: 'session-bandwidth',
      sessionDate: '2023/08/14 (Mon) 09:30',
    })

    const bandwidth = extracted.find((memory) => memory.content.includes('500 Mbps'))

    expect(bandwidth).toBeDefined()
    expect(bandwidth?.scope).toBe('global')
    expect(bandwidth?.type).toBe('user')
    expect(bandwidth?.tags).toEqual(expect.arrayContaining(['500 Mbps', '1.5 GB/s']))
  })

  it('creates assistant row memories from weekday markdown tables', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: 'Please record the Sunday rota.',
        },
        {
          role: 'assistant',
          content:
            '| Day | 8 am - 4 pm | 4 pm - 12 am |\n| --- | --- | --- |\n| Sunday | Admon | Bex |\n| Monday | Carla | Dan |',
        },
      ],
      sessionId: 'session-rota',
      sessionDate: '2023/08/13 (Sun) 09:00',
    })

    const sunday = extracted.find((memory) => memory.content.includes('Sunday roster:'))

    expect(sunday).toBeDefined()
    expect(sunday?.scope).toBe('project')
    expect(sunday?.type).toBe('reference')
    expect(sunday?.name.startsWith('Reference:')).toBe(true)
    expect(sunday?.content).toContain('Sunday roster: Admon, 8 am - 4 pm')
    expect(sunday?.content).toContain('Bex, 4 pm - 12 am')
    expect(sunday?.content).toContain('[Date: 2023-08-13')
    expect(sunday?.filename).toContain('reference-2023-08-13')
  })

  it('treats ordinal progress updates as quantified user facts', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I've been reading about the Amazon rainforest and its indigenous communities in National Geographic, and I just finished my fifth issue.",
        },
        {
          role: 'assistant',
          content: 'That sounds fascinating.',
        },
      ],
      sessionId: 'session-ordinal',
      sessionDate: '2023/07/15 (Sat) 22:42',
    })

    const heuristic = extracted.find((memory) => memory.content.includes('fifth issue'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.filename).toContain('2023-07-15')
  })

  it('falls back to the system message for heuristic session dates', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'system',
          content: 'This conversation took place on 2023/07/15 (Sat) 22:42.',
        },
        {
          role: 'user',
          content:
            "I've been reading about the Amazon rainforest and its indigenous communities in National Geographic, and I just finished my fifth issue.",
        },
      ],
      sessionId: 'session-fallback',
    })

    const heuristic = extracted.find((memory) => memory.content.includes('fifth issue'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.filename).toContain('2023-07-15')
    expect(heuristic?.observedOn).toBe('2023-07-15T22:42:00.000Z')
  })

  it('adds searchable user facts for week-long and 10-day social-media breaks', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm taking a week-long break from social media. I also took a 10-day break from Instagram earlier this year.",
        },
        {
          role: 'assistant',
          content: 'Those breaks can make it easier to reset your habits.',
        },
      ],
      sessionId: 'session-breaks',
      sessionDate: '2023/06/01 (Thu) 08:15',
    })

    const weekLong = extracted.find((memory) =>
      memory.content.includes('week-long break from social media'),
    )
    const tenDay = extracted.find((memory) =>
      memory.content.includes('10-day break from Instagram'),
    )

    expect(weekLong).toBeDefined()
    expect(weekLong?.scope).toBe('global')
    expect(weekLong?.description).toContain('week-long break from social media')
    expect(weekLong?.indexEntry).toContain('week-long break from social media')

    expect(tenDay).toBeDefined()
    expect(tenDay?.scope).toBe('global')
    expect(tenDay?.description).toContain('10-day break from Instagram')
    expect(tenDay?.indexEntry).toContain('10-day break from Instagram')
  })

  it('adds heuristic user facts for relative-time number words', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm actually planning to buy a new phone charger, since I lost my old one at the gym about two weeks ago.",
        },
        {
          role: 'assistant',
          content: 'Buying a new charger makes sense.',
        },
      ],
      sessionId: 'session-charger',
      sessionDate: '2023/05/26 (Fri) 18:20',
    })

    const heuristic = extracted.find((memory) =>
      memory.content.includes('lost my old one at the gym about two weeks ago'),
    )
    expect(heuristic).toBeDefined()
    expect(heuristic?.observedOn).toBe('2023-05-12T00:00:00.000Z')
  })

  it('adds heuristic user facts for Airbnb booking lead times', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I've had a great experience with Airbnb in the past, like when I stayed in Haight-Ashbury for my best friend's wedding and had to book three months in advance.",
        },
        {
          role: 'assistant',
          content: 'That sounds like a memorable trip.',
        },
      ],
      sessionId: 'session-airbnb',
      sessionDate: '2023/05/27 (Sat) 03:04',
    })

    const heuristic = extracted.find(
      (memory) =>
        memory.content.includes('Airbnb') &&
        memory.content.includes('book three months in advance'),
    )
    expect(heuristic).toBeDefined()
    expect(heuristic?.filename).toContain('session-airbnb')
  })

  it('does not infer a pending task from a generic advice question', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: 'Can you tell me what are some things I should consider before making an offer?',
        },
        {
          role: 'assistant',
          content: 'I can walk you through the main factors to check.',
        },
      ],
      sessionId: 'session-offer',
      sessionDate: '2022/03/02 (Wed) 04:59',
    })

    expect(
      extracted.some((memory) =>
        memory.content.includes('still needs to consider before making an offer'),
      ),
    ).toBe(false)
  })

  it('preserves quantified user facts when later generic notes reuse the same filename', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const responses = [
      JSON.stringify({
        memories: [
          {
            action: 'create',
            filename: 'user-car-fuel-efficiency.md',
            name: "User's Car Fuel Efficiency",
            description: "User's car was getting 30 miles per gallon in the city a few months ago.",
            type: 'user',
            scope: 'global',
            content: "The user's car was getting 30 miles per gallon in the city a few months ago.",
            index_entry: "User's car was getting 30 miles per gallon in the city a few months ago.",
          },
        ],
      }),
      JSON.stringify({
        memories: [
          {
            action: 'update',
            filename: 'user-car-fuel-efficiency.md',
            name: "User's Car Fuel Efficiency",
            description: "User's car fuel efficiency in the city.",
            type: 'user',
            scope: 'global',
            content:
              'The user has been getting around 28 miles per gallon in the city with their car.',
            index_entry: "User's car fuel efficiency is 28 mpg in the city.",
          },
        ],
      }),
    ]
    let call = 0
    const provider: Provider = {
      ...stubProvider(responses[0] ?? '{"memories":[]}'),
      complete: async () => {
        const content = responses[call] ?? '{"memories":[]}'
        call++
        return {
          content,
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
      extractMinMessages: 2,
    })

    await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm thinking of getting a tune-up for my car soon. My car was getting 30 miles per gallon in the city a few months ago, so I'm hoping to get back to that.",
        },
        {
          role: 'assistant',
          content: 'A tune-up is a great way to improve fuel efficiency.',
        },
      ],
      sessionId: 'session-a',
      sessionDate: '2023/05/22 (Mon) 19:18',
    })

    await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm thinking of getting a new air filter for my car. I've been getting around 28 miles per gallon in the city lately.",
        },
        {
          role: 'assistant',
          content: 'A new air filter can help improve fuel efficiency.',
        },
      ],
      sessionId: 'session-b',
      sessionDate: '2023/05/25 (Thu) 14:56',
    })

    const files = await store.list(toPath('memory/global'), {
      recursive: true,
      includeGenerated: false,
    })
    const noteBodies = await Promise.all(
      files
        .filter((entry) => !entry.isDir && entry.path.endsWith('.md'))
        .map(async (entry) => (await store.read(entry.path)).toString('utf8')),
    )

    expect(noteBodies.some((body) => body.includes('30 miles per gallon'))).toBe(true)
    expect(noteBodies.some((body) => body.includes('28 miles per gallon'))).toBe(true)
    expect(
      files.some((entry) => !entry.isDir && entry.path.includes('user-fact-2023-05-22-session-a')),
    ).toBe(true)
  })

  it('adds heuristic user fact notes for milestone statements without quantities', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm planning to start learning about deep learning. By the way, I just completed my undergraduate degree in computer science.",
        },
        {
          role: 'assistant',
          content: 'That foundation will help with deep learning.',
        },
      ],
      sessionId: 'session-b',
      sessionDate: '2022/11/17 (Thu) 15:34',
    })

    const heuristic = extracted.find((memory) =>
      memory.content.includes('completed my undergraduate degree'),
    )
    expect(heuristic).toBeDefined()
    expect(heuristic?.scope).toBe('global')
    expect(heuristic?.type).toBe('user')
    expect(heuristic?.filename).toContain('milestone')
  })

  it('adds heuristic user fact notes for month-name dates', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "My close friend Rachel got engaged on May 15th, and we're already planning her bachelorette party.",
        },
        {
          role: 'assistant',
          content: 'That sounds exciting.',
        },
      ],
      sessionId: 'session-rachel',
      sessionDate: '2023/07/07 (Fri) 04:44',
    })

    const heuristic = extracted.find((memory) => memory.content.includes('May 15th'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.content.startsWith('[Date: 2023-05-15')).toBe(true)
    expect(heuristic?.observedOn).toBe('2023-05-15T00:00:00.000Z')
  })

  it('adds milestone notes for group joins with resolved dates', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            'I just joined a new book club group called "Page Turners" last week, where we discuss our favourite novels and share recommendations.',
        },
        {
          role: 'assistant',
          content: 'That sounds like a great group.',
        },
      ],
      sessionId: 'session-page-turners',
      sessionDate: '2023/05/25 (Thu) 01:50',
    })

    const heuristic = extracted.find((memory) => memory.content.includes('Page Turners'))
    expect(heuristic).toBeDefined()
    expect(heuristic?.content.startsWith('[Date: 2023-05-18')).toBe(true)
    expect(heuristic?.observedOn).toBe('2023-05-18T00:00:00.000Z')
  })

  it('reshapes appointment notes so the doctor type appears in the summary and index entry', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider(
        JSON.stringify({
          memories: [
            {
              action: 'create',
              filename: 'user-appointment.md',
              name: 'Appointment',
              description: 'Upcoming appointment',
              type: 'user',
              scope: 'global',
              content:
                'The user has a dermatologist appointment with Dr Patel next Tuesday at 3 pm.',
              index_entry: '- user-appointment.md: appointment',
            },
          ],
        }),
      ),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: "I've got a dermatologist appointment with Dr Patel next Tuesday at 3 pm.",
        },
        {
          role: 'assistant',
          content: 'That is useful to keep in mind.',
        },
      ],
      sessionId: 'session-health',
      sessionDate: '2023/06/06 (Tue) 09:00',
    })

    const appointment = extracted.find((memory) => memory.filename === 'user-appointment.md')

    expect(appointment).toBeDefined()
    expect(appointment?.description).toContain('dermatologist appointment')
    expect(appointment?.description).toContain('Dr Patel')
    expect(appointment?.indexEntry).toContain('dermatologist appointment')
    expect(appointment?.indexEntry).toContain('Dr Patel')
    expect(appointment?.tags).toEqual(
      expect.arrayContaining(['appointment', 'medical', 'dermatologist', 'next tuesday', '3 pm']),
    )
  })

  it('does not invent a medical appointment from a charity event near a hospital', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content: `I'm feeling a bit tired today, just got back from the "24-Hour Bike Ride" charity event, where I cycled for 4 hours non-stop to raise money for a local children's hospital.`,
        },
        {
          role: 'assistant',
          content: 'That sounds exhausting but worthwhile.',
        },
      ],
      sessionId: 'session-charity',
      sessionDate: '2023/02/14 (Tue) 19:50',
    })

    expect(extracted.some((memory) => memory.description.includes('medical appointment'))).toBe(
      false,
    )
  })

  it('adds heuristic event notes for religious services', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I'm glad I got to attend the Maundy Thursday service at the Episcopal Church, it was a beautiful and moving experience.",
        },
        {
          role: 'assistant',
          content: 'That sounds meaningful.',
        },
      ],
      sessionId: 'session-service',
      sessionDate: '2023/04/06 (Thu) 05:36',
    })

    const eventMemory = extracted.find((memory) =>
      memory.content.includes('Maundy Thursday service at the Episcopal Church'),
    )
    expect(eventMemory).toBeDefined()
    expect(eventMemory?.description).toContain('Episcopal Church')
    expect(eventMemory?.observedOn).toBe('2023-04-06T05:36:00.000Z')
  })

  it('adds heuristic preference notes for recommendation constraints', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            'Besides great views, I also like hotels with unique features, such as a rooftop pool or a hot tub on the balcony.',
        },
        {
          role: 'assistant',
          content: 'Those features make for a memorable stay.',
        },
      ],
      sessionId: 'session-c',
      sessionDate: '2023/05/27 (Sat) 18:36',
    })

    const heuristic = extracted.find((memory) =>
      memory.content.includes('The user prefers hotels with great views'),
    )
    expect(heuristic).toBeDefined()
    expect(heuristic?.scope).toBe('global')
    expect(heuristic?.type).toBe('user')
    expect(heuristic?.filename).toContain('user-preference-2023-05-27-session-c')
    expect(heuristic?.content).toContain('Evidence:')
  })

  it('adds durable preference notes for recommendation constraints like family-friendly and under a limit', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            'Can you recommend a family-friendly, light-hearted film for tonight, ideally under 100 minutes and without gore?',
        },
        {
          role: 'assistant',
          content: 'I will keep the suggestions gentle and short.',
        },
      ],
      sessionId: 'session-film',
      sessionDate: '2023/07/14 (Fri) 20:05',
    })

    const heuristic = extracted.find((memory) =>
      memory.content.includes('The user prefers films with these constraints'),
    )

    expect(heuristic).toBeDefined()
    expect(heuristic?.scope).toBe('global')
    expect(heuristic?.description).toContain('family-friendly')
    expect(heuristic?.description).toContain('light-hearted')
    expect(heuristic?.description).toContain('under 100 minutes')
    expect(heuristic?.content).toContain('without gore')
    expect(heuristic?.tags).toEqual(
      expect.arrayContaining(['recommendation', 'entertainment', 'film', 'tonight']),
    )
  })

  it('does not infer entertainment preferences from unrelated family-friendly searches', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            "I work in the city of Irvine, California. I'm looking for a family-friendly area with good schools and parks.",
        },
        {
          role: 'assistant',
          content: 'I can suggest a few neighbourhoods nearby.',
        },
      ],
      sessionId: 'session-irvine',
      sessionDate: '2022/03/02 (Wed) 22:25',
    })

    expect(
      extracted.some((memory) => memory.content.includes('prefers books with these constraints')),
    ).toBe(false)
  })

  it('adds heuristic preference notes for advanced technical interests', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const mem = createMemory({
      store,
      provider: stubProvider('{"memories":[]}'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      extractMinMessages: 2,
    })

    const extracted = await mem.extract({
      messages: [
        {
          role: 'user',
          content:
            'Can you give me an overview of the recent advancements in this field of deep learning for medical image analysis? Skip the basics as I am working in the field.',
        },
        {
          role: 'assistant',
          content: 'I will focus on recent developments and skip introductory material.',
        },
      ],
      sessionId: 'session-d',
      sessionDate: '2023/05/20 (Sat) 06:37',
    })

    const heuristic = extracted.find((memory) =>
      memory.content.includes(
        'The user prefers advanced publications, papers, and conferences on deep learning for medical image analysis',
      ),
    )
    expect(heuristic).toBeDefined()
    expect(heuristic?.scope).toBe('global')
    expect(heuristic?.type).toBe('user')
    expect(heuristic?.filename).toContain('user-preference-2023-05-20-session-d')
  })
})
