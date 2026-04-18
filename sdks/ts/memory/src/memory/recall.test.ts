import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { type Path, toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { createMemory } from './index.js'
import {
  isRecentMemoryQuery,
  isTimeSensitiveMemoryQuery,
  mergeRecallHits,
} from './recall.js'
import type { RecallHit, SearchIndex } from './types.js'

const dummyProvider = (): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield await Promise.reject(new Error('not implemented'))
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

const recordingEmbedder = (): { embedder: Embedder; seen: string[] } => {
  const seen: string[] = []
  const embedder: Embedder = {
    name: () => 'stub',
    model: () => 'stub',
    dimension: () => 3,
    embed: async (texts) => {
      for (const t of texts) seen.push(t)
      return texts.map(() => [1, 0, 0])
    },
  }
  return { embedder, seen }
}

const writeProjectNote = async (args: {
  readonly store: ReturnType<typeof createMemStore>
  readonly path: Path
  readonly name: string
  readonly body: string
  readonly description?: string
  readonly modified?: string
  readonly created?: string
  readonly sessionDate?: string
  readonly observedOn?: string
  readonly tags?: readonly string[]
  readonly scope?: 'project' | 'global' | 'agent'
  readonly type?: string
}): Promise<void> => {
  const scope = args.scope ?? 'project'
  const frontmatter = [
    '---',
    `name: ${args.name}`,
    `description: ${args.description ?? 'desc'}`,
    `type: ${args.type ?? (scope === 'global' ? 'reference' : 'project')}`,
    `scope: ${scope}`,
    ...(args.modified !== undefined ? [`modified: ${args.modified}`] : []),
    ...(args.created !== undefined ? [`created: ${args.created}`] : []),
    ...(args.sessionDate !== undefined ? [`session_date: ${args.sessionDate}`] : []),
    ...(args.observedOn !== undefined ? [`observed_on: ${args.observedOn}`] : []),
    ...(args.tags !== undefined && args.tags.length > 0
      ? [`tags: [${args.tags.map((tag) => `"${tag}"`).join(', ')}]`]
      : []),
    '---',
    '',
  ].join('\n')
  await args.store.write(args.path, Buffer.from(`${frontmatter}${args.body}\n`, 'utf8'))
}

const writeGlobalNote = async (args: {
  readonly store: ReturnType<typeof createMemStore>
  readonly path: Path
  readonly name: string
  readonly body: string
  readonly description?: string
  readonly modified?: string
  readonly created?: string
  readonly sessionDate?: string
  readonly observedOn?: string
  readonly tags?: readonly string[]
  readonly type?: string
}): Promise<void> =>
  writeProjectNote({
    ...args,
    scope: 'global',
    type: args.type ?? 'reference',
  })

const makeRecallHit = (args: {
  readonly path: string
  readonly score: number
  readonly content: string
  readonly sessionDate?: string
}): RecallHit => ({
  path: toPath(args.path),
  score: args.score,
  content: args.content,
  note: {
    path: toPath(args.path),
    name: args.path,
    description: '',
    type: 'project',
    scope: 'project',
    tags: [],
    content: args.content,
    ...(args.sessionDate !== undefined ? { sessionDate: args.sessionDate } : {}),
  },
})

describe('recall', () => {
  it('returns hits in descending score order and hydrates content', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder, seen } = recordingEmbedder()

    const alpha = toPath('memory/project/tenant-a/alpha.md')
    const beta = toPath('memory/project/tenant-a/beta.md')
    const gamma = toPath('memory/project/tenant-a/gamma.md')
    await writeProjectNote({ store, path: alpha, name: 'alpha', body: 'alpha body' })
    await writeProjectNote({ store, path: beta, name: 'beta', body: 'beta body' })
    await writeProjectNote({ store, path: gamma, name: 'gamma', body: 'gamma body' })

    const searchIndex: SearchIndex = {
      search: async (query, embedding) => {
        expect(embedding).toBeDefined()
        expect(query).toBe('what is known?')
        return [
          { path: beta, score: 0.4 },
          { path: gamma, score: 0.9 },
          { path: alpha, score: 0.7 },
        ]
      },
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({ query: 'what is known?' })
    expect(hits.map((h) => h.path)).toEqual([gamma, alpha, beta])
    expect(hits[0]?.content).toContain('gamma body')
    expect(seen).toEqual(['what is known?'])
  })

  it('mixes project and global candidate recall for project scope', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()
    const seenScopes: Array<{ scope: string | undefined; actorId: string | undefined }> = []

    const projectPath = toPath('memory/project/tenant-a/repo-conventions.md')
    const globalPath = toPath('memory/global/planning-style.md')
    await writeProjectNote({
      store,
      path: projectPath,
      name: 'Repo conventions',
      body: 'The repo uses strict types and small focused modules.',
    })
    await writeGlobalNote({
      store,
      path: globalPath,
      name: 'Planning style',
      body: 'Prefer explicit plans and closing out work thoroughly.',
    })

    const searchIndex: SearchIndex = {
      search: async (_query, _embedding, opts) => {
        seenScopes.push({ scope: opts.scope, actorId: opts.actorId })
        if (opts.scope === 'global') {
          return [{ path: globalPath, score: 0.86 }]
        }
        return [{ path: projectPath, score: 0.82 }]
      },
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({ query: 'how do I like to plan work in this repo?', k: 2 })

    expect(seenScopes).toEqual([
      { scope: 'project', actorId: 'tenant-a' },
      { scope: 'global', actorId: 'tenant-a' },
    ])
    expect(hits.map((hit) => hit.path)).toEqual(expect.arrayContaining([projectPath, globalPath]))
    expect(hits.find((hit) => hit.path === globalPath)?.note.scope).toBe('global')
  })

  it('reorders recalled hits chronologically for time-sensitive queries', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()

    const oldest = toPath('memory/project/tenant-a/oldest.md')
    const middle = toPath('memory/project/tenant-a/middle.md')
    const newest = toPath('memory/project/tenant-a/newest.md')
    await writeProjectNote({
      store,
      path: oldest,
      name: 'oldest',
      modified: '2024-01-10T10:00:00.000Z',
      body: 'oldest body',
    })
    await writeProjectNote({
      store,
      path: middle,
      name: 'middle',
      modified: '2024-02-10T10:00:00.000Z',
      body: 'middle body',
    })
    await writeProjectNote({
      store,
      path: newest,
      name: 'newest',
      modified: '2024-03-10T10:00:00.000Z',
      body: 'newest body',
    })

    const searchIndex: SearchIndex = {
      search: async () => [
        { path: newest, score: 0.9 },
        { path: middle, score: 0.7 },
        { path: oldest, score: 0.5 },
      ],
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({ query: 'what happened before March 2024?' })
    expect(hits.map((hit) => hit.path)).toEqual([oldest, middle, newest])
  })

  it('merges recent-query hits newest-first across scopes', () => {
    const hits = mergeRecallHits(
      [
        makeRecallHit({
          path: 'memory/project/tenant-a/older.md',
          score: 0.9,
          content: 'Older note',
          sessionDate: '2024-01-10',
        }),
        makeRecallHit({
          path: 'memory/global/tenant-a/newer.md',
          score: 0.7,
          content: 'Newer note',
          sessionDate: '2024-03-10',
        }),
      ],
      { query: 'what was the most recent appointment?' },
    )

    expect(isRecentMemoryQuery('what was the most recent appointment?')).toBe(true)
    expect(hits.map((hit) => hit.path)).toEqual([
      toPath('memory/global/tenant-a/newer.md'),
      toPath('memory/project/tenant-a/older.md'),
    ])
  })

  it('merges timeline-query hits chronologically across scopes', () => {
    const hits = mergeRecallHits(
      [
        makeRecallHit({
          path: 'memory/project/tenant-a/march.md',
          score: 0.9,
          content: 'March note',
          sessionDate: '2024-03-10',
        }),
        makeRecallHit({
          path: 'memory/global/tenant-a/january.md',
          score: 0.7,
          content: 'January note',
          sessionDate: '2024-01-10',
        }),
      ],
      { query: 'what happened before March 2024?' },
    )

    expect(hits.map((hit) => hit.path)).toEqual([
      toPath('memory/global/tenant-a/january.md'),
      toPath('memory/project/tenant-a/march.md'),
    ])
  })

  it('prefers recent concrete event notes over generic advice for recent queries', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()

    const guidance = toPath('memory/project/tenant-a/workshop-guidance.md')
    const januaryWorkshop = toPath('memory/project/tenant-a/workshop-january.md')
    const marchWorkshop = toPath('memory/project/tenant-a/workshop-march.md')

    await writeProjectNote({
      store,
      path: guidance,
      name: 'Workshop guidance',
      description: 'Generic workshop advice',
      body: 'Workshop best practices: always prepare an agenda and summary.',
    })
    await writeProjectNote({
      store,
      path: januaryWorkshop,
      name: 'Product team workshop in January',
      modified: '2024-01-08T09:00:00.000Z',
      body: 'Workshop with the product team about onboarding metrics.',
    })
    await writeProjectNote({
      store,
      path: marchWorkshop,
      name: 'Product team workshop in March',
      modified: '2024-03-18T09:00:00.000Z',
      body: 'Workshop with the product team about roadmap delivery.',
    })

    const searchIndex: SearchIndex = {
      search: async (_query, _embedding, opts) => {
        expect(opts.k).toBeGreaterThan(2)
        return [
          { path: guidance, score: 0.95 },
          { path: januaryWorkshop, score: 0.75 },
          { path: marchWorkshop, score: 0.7 },
        ]
      },
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const query = 'what was the most recent workshop with the product team?'
    const hits = await mem.recall({ query, k: 2 })

    expect(isTimeSensitiveMemoryQuery(query)).toBe(false)
    expect(hits.map((hit) => hit.path)).toEqual([marchWorkshop, januaryWorkshop])
  })

  it('keeps broader recall coverage for aggregate questions instead of near-duplicate hits', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()

    const genericBudget = toPath('memory/project/tenant-a/bike-budget.md')
    const januaryTyres = toPath('memory/project/tenant-a/bike-tyres-january.md')
    const januaryTyresRepeat = toPath('memory/project/tenant-a/bike-tyres-repeat.md')
    const februaryService = toPath('memory/project/tenant-a/bike-service-february.md')
    const marchLock = toPath('memory/project/tenant-a/bike-lock-march.md')

    await writeProjectNote({
      store,
      path: genericBudget,
      name: 'Bike budget advice',
      description: 'General guidance',
      body: 'Keep a bike maintenance budget and review it monthly.',
    })
    await writeProjectNote({
      store,
      path: januaryTyres,
      name: 'Bike tyres in January',
      modified: '2024-01-12T10:00:00.000Z',
      body: 'Spent £120 on bike tyres and tubes after punctures.',
    })
    await writeProjectNote({
      store,
      path: januaryTyresRepeat,
      name: 'Bike tyres again in January',
      modified: '2024-01-13T10:00:00.000Z',
      body: 'Bike tyres and tubes cost £118 after another puncture.',
    })
    await writeProjectNote({
      store,
      path: februaryService,
      name: 'Bike service in February',
      modified: '2024-02-20T12:00:00.000Z',
      body: 'Paid £80 for a bike service and brake pads.',
    })
    await writeProjectNote({
      store,
      path: marchLock,
      name: 'Bike lock in March',
      modified: '2024-03-05T12:00:00.000Z',
      body: 'Bought a bike lock for £45.',
    })

    const searchIndex: SearchIndex = {
      search: async (_query, _embedding, opts) => {
        expect(opts.k).toBeGreaterThan(3)
        return [
          { path: genericBudget, score: 0.97 },
          { path: januaryTyres, score: 0.95 },
          { path: januaryTyresRepeat, score: 0.94 },
          { path: februaryService, score: 0.7 },
          { path: marchLock, score: 0.69 },
        ]
      },
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({
      query: 'how much did I spend on bike expenses across different periods?',
      k: 3,
    })
    const recalledPaths = hits.map((hit) => hit.path)

    expect(recalledPaths).toContain(februaryService)
    expect(recalledPaths).toContain(marchLock)
    expect(recalledPaths).not.toContain(genericBudget)
    expect(
      recalledPaths.filter((path) => path === januaryTyres || path === januaryTyresRepeat),
    ).toHaveLength(1)
  })

  it('suppresses surfaced hits before mixed-scope recall results are hydrated', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()

    const projectPath = toPath('memory/project/tenant-a/repo-setup.md')
    const globalPath = toPath('memory/global/workflow.md')
    await writeProjectNote({
      store,
      path: projectPath,
      name: 'Repo setup',
      body: 'Project setup note.',
    })
    await writeGlobalNote({
      store,
      path: globalPath,
      name: 'Workflow',
      body: 'General workflow note.',
    })

    const searchIndex: SearchIndex = {
      search: async (_query, _embedding, opts) =>
        opts.scope === 'global'
          ? [{ path: globalPath, score: 0.7 }]
          : [
              { path: projectPath, score: 0.95 },
              { path: globalPath, score: 0.7 },
            ],
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({
      query: 'what should I remember?',
      k: 2,
      surfacedPaths: [projectPath],
    })

    expect(hits.map((hit) => hit.path)).toEqual([globalPath])
  })

  it('falls back to completion-based recall selection when structured decoding is unavailable', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()
    const completeCalls: CompletionRequest[] = []

    const alpha = toPath('memory/project/tenant-a/alpha.md')
    const beta = toPath('memory/project/tenant-a/beta.md')
    const gamma = toPath('memory/project/tenant-a/gamma.md')
    await writeProjectNote({ store, path: alpha, name: 'alpha', body: 'alpha body' })
    await writeProjectNote({ store, path: beta, name: 'beta', body: 'beta body' })
    await writeProjectNote({ store, path: gamma, name: 'gamma', body: 'gamma body' })

    const provider: Provider = {
      ...dummyProvider(),
      complete: async (req) => {
        completeCalls.push(req)
        return {
          content: JSON.stringify({ selected: [beta] }),
          toolCalls: [],
          usage: { inputTokens: 0, outputTokens: 0 },
          stopReason: 'end_turn',
        }
      },
    }

    const searchIndex: SearchIndex = {
      search: async () => [
        { path: alpha, score: 0.95 },
        { path: beta, score: 0.5 },
        { path: gamma, score: 0.4 },
      ],
    }

    const mem = createMemory({
      store,
      provider,
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({
      query: 'which memory should matter here?',
      k: 1,
      selector: 'auto',
    })

    expect(completeCalls).toHaveLength(1)
    expect(completeCalls[0]?.jsonMode).toBe(true)
    expect(hits.map((hit) => hit.path)).toEqual([beta])
  })

  it('follows wikilinks without resurfacing already surfaced memories', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const { embedder } = recordingEmbedder()

    const primary = toPath('memory/project/tenant-a/release-notes.md')
    const linkedProject = toPath('memory/project/tenant-a/deployment-checklist.md')
    const linkedGlobal = toPath('memory/global/working-style.md')
    await writeProjectNote({
      store,
      path: primary,
      name: 'Release notes',
      body: 'See [[deployment-checklist]] and [[global:working-style]] before shipping.',
    })
    await writeProjectNote({
      store,
      path: linkedProject,
      name: 'Deployment checklist',
      body: 'Check migrations and smoke tests.',
    })
    await writeGlobalNote({
      store,
      path: linkedGlobal,
      name: 'Working style',
      body: 'Prefer thorough plans and explicit verification.',
    })

    const searchIndex: SearchIndex = {
      search: async (_query, _embedding, opts) =>
        opts.scope === 'project' ? [{ path: primary, score: 0.9 }] : [],
    }

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })

    const hits = await mem.recall({
      query: 'what should I remember before the release?',
      k: 2,
      surfacedPaths: [linkedGlobal],
    })

    expect(hits.map((hit) => hit.path)).toEqual([primary, linkedProject])
  })

  it('falls back to embedding-aware recall when no search index is configured', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const embedder: Embedder = {
      name: () => 'stub',
      model: () => 'stub',
      dimension: () => 2,
      embed: async (texts) =>
        texts.map((text) =>
          text.includes('hedgehog') ? [1, 0] : text.includes('mealworms') ? [0.95, 0.05] : [0, 1],
        ),
    }

    await store.write(
      toPath('memory/project/tenant-a/hedgehog.md'),
      Buffer.from(
        [
          '---',
          'name: hedgehog',
          'description: desc',
          'type: project',
          'scope: project',
          '---',
          '',
          'Mealworms and quiet gardens.',
        ].join('\n'),
        'utf8',
      ),
    )
    await store.write(
      toPath('memory/project/tenant-a/trains.md'),
      Buffer.from(
        [
          '---',
          'name: trains',
          'description: desc',
          'type: project',
          'scope: project',
          '---',
          '',
          'High-speed rail corridors.',
        ].join('\n'),
        'utf8',
      ),
    )

    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder,
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    const hits = await mem.recall({ query: 'hedgehog' })
    expect(hits[0]?.path).toBe(toPath('memory/project/tenant-a/hedgehog.md'))
    expect(hits[0]?.content).toContain('Mealworms')
  })
})
