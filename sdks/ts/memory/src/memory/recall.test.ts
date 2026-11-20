// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Logger,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { StoreError } from '../store/errors.js'
import type { Store } from '../store/index.js'
import { type MemStore, createMemStore } from '../store/memstore.js'
import { type Path, toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { createMemory } from './index.js'
import { isRecentMemoryQuery, isTimeSensitiveMemoryQuery, mergeRecallHits } from './recall.js'
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

  it('keeps project-scope recall strict by default', async () => {
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

    expect(seenScopes).toEqual([{ scope: 'project', actorId: 'tenant-a' }])
    expect(hits.map((hit) => hit.path)).toEqual([projectPath])
  })

  it('includes explicit fallback scopes after the primary scope', async () => {
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

    const hits = await mem.recall({
      query: 'how do I like to plan work in this repo?',
      k: 2,
      fallbackScopes: ['global'],
    })

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

  it('keeps wikilink follow-ups inside the allowed scopes', async () => {
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
    })

    expect(hits.map((hit) => hit.path)).toEqual([primary, linkedProject])
  })

  it('follows fallback-scope wikilinks when the caller opts in', async () => {
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
      k: 3,
      fallbackScopes: ['global'],
    })

    expect(hits.map((hit) => hit.path)).toEqual([primary, linkedProject, linkedGlobal])
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

type StoreCounts = {
  read: number
  exists: number
  readMany: number
  readManyBatchSizes: number[]
}

type StoreOverrides = {
  readonly read?: (path: Path) => Promise<Buffer>
  readonly readMany?: (paths: readonly Path[]) => Promise<Map<Path, Buffer>>
}

const instrumentStore = (
  inner: MemStore,
  mode: 'batched' | 'serial',
  overrides: StoreOverrides = {},
): { readonly store: Store; readonly counts: StoreCounts } => {
  const counts: StoreCounts = { read: 0, exists: 0, readMany: 0, readManyBatchSizes: [] }
  const store: Store = {
    read: async (path) => {
      counts.read += 1
      return overrides.read !== undefined ? overrides.read(path) : inner.read(path)
    },
    write: (path, content) => inner.write(path, content),
    append: (path, content) => inner.append(path, content),
    delete: (path) => inner.delete(path),
    rename: (src, dst) => inner.rename(src, dst),
    exists: async (path) => {
      counts.exists += 1
      return inner.exists(path)
    },
    stat: (path) => inner.stat(path),
    list: (dir, opts) => inner.list(dir, opts),
    batch: (opts, fn) => inner.batch(opts, fn),
    subscribe: (sink) => inner.subscribe(sink),
    localPath: (path) => inner.localPath(path),
    close: () => inner.close(),
    ...(mode === 'batched'
      ? {
          readMany: async (paths: readonly Path[]) => {
            counts.readMany += 1
            counts.readManyBatchSizes.push(paths.length)
            return overrides.readMany !== undefined
              ? overrides.readMany(paths)
              : inner.readMany(paths)
          },
        }
      : {}),
  }
  return { store, counts }
}

const recordingLogger = (): { readonly logger: Logger; readonly warnings: string[] } => {
  const warnings: string[] = []
  return {
    warnings,
    logger: {
      debug: () => {},
      info: () => {},
      warn: (msg) => {
        warnings.push(msg)
      },
      error: () => {},
    },
  }
}

const releasePath = toPath('memory/project/tenant-a/release-notes.md')
const checklistPath = toPath('memory/project/tenant-a/deployment-checklist.md')
const deletedPath = toPath('memory/project/tenant-a/deleted.md')
const otherPaths = ['rollout', 'rollback', 'oncall'].map((name) =>
  toPath(`memory/project/tenant-a/${name}.md`),
)

const seedReleaseNotes = async (store: MemStore): Promise<SearchIndex> => {
  await writeProjectNote({
    store,
    path: releasePath,
    name: 'Release notes',
    body: 'See [[missing-topic]], [[deployment-checklist]] and [[global:working-style]].',
  })
  await writeProjectNote({
    store,
    path: checklistPath,
    name: 'Deployment checklist',
    body: 'Check migrations and smoke tests.',
  })
  await writeGlobalNote({
    store,
    path: toPath('memory/global/working-style.md'),
    name: 'Working style',
    body: 'Prefer explicit verification.',
  })
  for (const [index, path] of otherPaths.entries()) {
    await writeProjectNote({ store, path, name: `note ${index}`, body: `release body ${index}` })
  }
  return {
    search: async (_query, _embedding, opts) =>
      opts.scope === 'project'
        ? [
            { path: releasePath, score: 0.9 },
            { path: deletedPath, score: 0.8 },
            ...otherPaths.map((path, index) => ({ path, score: 0.7 - index * 0.1 })),
          ]
        : [],
  }
}

describe('recall store reads', () => {
  const recallWith = async (mode: 'batched' | 'serial', overrides: StoreOverrides = {}) => {
    const inner = createMemStore()
    const searchIndex = await seedReleaseNotes(inner)
    const { store, counts } = instrumentStore(inner, mode, overrides)
    const { logger, warnings } = recordingLogger()
    const mem = createMemory({
      store,
      provider: dummyProvider(),
      embedder: recordingEmbedder().embedder,
      cursorStore: createStoreBackedCursorStore(inner),
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
      logger,
    })
    return { mem, counts, warnings }
  }

  it('hydrates through one readMany and resolves wikilinks through another', async () => {
    const { mem, counts, warnings } = await recallWith('batched')
    const hits = await mem.recall({ query: 'release', k: 5 })

    expect(hits.map((hit) => hit.path)).toContain(checklistPath)
    expect(hits.map((hit) => hit.path)).not.toContain(deletedPath)
    expect(counts).toEqual({ read: 0, exists: 0, readMany: 2, readManyBatchSizes: [5, 2] })
    expect(warnings).toEqual([])
  })

  it('returns the same hits with or without readMany', async () => {
    const batched = await recallWith('batched')
    const serial = await recallWith('serial')
    for (const k of [1, 2, 5]) {
      const fromBatched = await batched.mem.recall({ query: 'release', k })
      const fromSerial = await serial.mem.recall({ query: 'release', k })
      expect(fromBatched).toStrictEqual(fromSerial)
    }
    expect(serial.counts.readMany).toBe(0)
    expect(serial.counts.read).toBeGreaterThan(0)
  })

  it('reads wikilink candidates in bounded batches and keeps link order', async () => {
    const inner = createMemStore()
    const links = Array.from({ length: 40 }, (_, index) => `[[topic-${index}]]`).join(' ')
    await writeProjectNote({ store: inner, path: releasePath, name: 'Hub', body: links })
    for (const index of [35, 38]) {
      await writeProjectNote({
        store: inner,
        path: toPath(`memory/project/tenant-a/topic-${index}.md`),
        name: `topic ${index}`,
        body: `topic body ${index}`,
      })
    }
    const searchIndex: SearchIndex = { search: async () => [{ path: releasePath, score: 0.9 }] }
    const results = await Promise.all(
      (['batched', 'serial'] as const).map(async (mode) => {
        const { store, counts } = instrumentStore(inner, mode)
        const mem = createMemory({
          store,
          provider: dummyProvider(),
          cursorStore: createStoreBackedCursorStore(inner),
          scope: 'project',
          actorId: 'tenant-a',
          searchIndex,
        })
        return { hits: await mem.recall({ query: 'hub', k: 5 }), counts }
      }),
    )
    const [batched, serial] = results
    expect(batched?.hits.map((hit) => hit.path)).toEqual([
      releasePath,
      toPath('memory/project/tenant-a/topic-35.md'),
      toPath('memory/project/tenant-a/topic-38.md'),
    ])
    expect(batched?.hits).toStrictEqual(serial?.hits)
    expect(batched?.counts.readManyBatchSizes).toEqual([1, 32, 8])
    expect(serial?.counts.exists).toBe(39)
  })

  it('rejects with the store error when no candidate could be read', async () => {
    const failure = new StoreError('backend unavailable')
    const { mem, warnings } = await recallWith('serial', {
      read: async () => {
        throw failure
      },
    })
    await expect(mem.recall({ query: 'release', k: 5 })).rejects.toBe(failure)
    expect(warnings).toEqual(Array(5).fill('memory: recall hydrate failed'))
  })

  it('keeps hits that hydrated when only some reads fail', async () => {
    const failing = otherPaths[0]
    const inner = createMemStore()
    const searchIndex = await seedReleaseNotes(inner)
    const { store } = instrumentStore(inner, 'serial', {
      read: async (path) => {
        if (path === failing) throw new StoreError('backend unavailable')
        return inner.read(path)
      },
    })
    const { logger, warnings } = recordingLogger()
    const mem = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore: createStoreBackedCursorStore(inner),
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
      logger,
    })
    const hits = await mem.recall({ query: 'release', k: 5 })
    expect(hits.map((hit) => hit.path)).not.toContain(failing)
    expect(hits.map((hit) => hit.path)).toContain(releasePath)
    expect(warnings).toEqual(['memory: recall hydrate failed'])
  })

  it('rejects with the store error when readMany fails', async () => {
    const failure = new StoreError('backend unavailable')
    const { mem } = await recallWith('batched', {
      readMany: async () => {
        throw failure
      },
    })
    await expect(mem.recall({ query: 'release', k: 5 })).rejects.toBe(failure)
  })

  it('still resolves an empty list when every candidate is missing', async () => {
    const store = createMemStore()
    const mem = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore: createStoreBackedCursorStore(store),
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex: { search: async () => [{ path: deletedPath, score: 0.9 }] },
    })
    await expect(mem.recall({ query: 'release' })).resolves.toEqual([])
  })
})

describe('recall cancellation', () => {
  const setup = async (embedder?: Embedder) => {
    const inner = createMemStore()
    const searchIndex = await seedReleaseNotes(inner)
    const { store, counts } = instrumentStore(inner, 'batched')
    const searches: number[] = []
    const mem = createMemory({
      store,
      provider: dummyProvider(),
      ...(embedder !== undefined ? { embedder } : {}),
      cursorStore: createStoreBackedCursorStore(inner),
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex: {
        search: async (query, embedding, opts) => {
          searches.push(1)
          return searchIndex.search(query, embedding, opts)
        },
      },
    })
    return { mem, counts, searches }
  }

  it('rejects at once with the reason when the signal is already aborted', async () => {
    const { embedder, seen } = recordingEmbedder()
    const { mem, counts, searches } = await setup(embedder)
    const reason = new Error('caller gave up')
    await expect(mem.recall({ query: 'release', signal: AbortSignal.abort(reason) })).rejects.toBe(
      reason,
    )
    expect(seen).toEqual([])
    expect(searches).toEqual([])
    expect(counts.readMany).toBe(0)
  })

  it('passes the signal to the embedder and rejects when it aborts mid-embed', async () => {
    const controller = new AbortController()
    const reason = new Error('caller gave up')
    let received: AbortSignal | undefined
    const embedder: Embedder = {
      name: () => 'stub',
      model: () => 'stub',
      dimension: () => 3,
      embed: (_texts, signal) => {
        received = signal
        return new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(signal.reason), { once: true })
          controller.abort(reason)
        })
      },
    }
    const { mem, counts, searches } = await setup(embedder)
    await expect(mem.recall({ query: 'release', signal: controller.signal })).rejects.toBe(reason)
    expect(received).toBe(controller.signal)
    expect(searches).toEqual([])
    expect(counts.readMany).toBe(0)
  })

  it('stops before hydrating when the signal aborts during the search', async () => {
    const controller = new AbortController()
    const reason = new Error('caller gave up')
    const inner = createMemStore()
    const searchIndex = await seedReleaseNotes(inner)
    const { store, counts } = instrumentStore(inner, 'batched')
    const mem = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore: createStoreBackedCursorStore(inner),
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex: {
        search: async (query, embedding, opts) => {
          controller.abort(reason)
          return searchIndex.search(query, embedding, opts)
        },
      },
    })
    await expect(mem.recall({ query: 'release', signal: controller.signal })).rejects.toBe(reason)
    expect(counts).toEqual({ read: 0, exists: 0, readMany: 0, readManyBatchSizes: [] })
  })

  it('stops before following wikilinks when the signal aborts during hydration', async () => {
    const controller = new AbortController()
    const reason = new Error('caller gave up')
    const inner = createMemStore()
    const searchIndex = await seedReleaseNotes(inner)
    const { store, counts } = instrumentStore(inner, 'batched', {
      readMany: async (paths) => {
        controller.abort(reason)
        return inner.readMany(paths)
      },
    })
    const mem = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore: createStoreBackedCursorStore(inner),
      scope: 'project',
      actorId: 'tenant-a',
      searchIndex,
    })
    await expect(mem.recall({ query: 'release', signal: controller.signal })).rejects.toBe(reason)
    expect(counts.readMany).toBe(1)
  })

  it('resolves as before when the signal never aborts', async () => {
    const { mem } = await setup(recordingEmbedder().embedder)
    const withSignal = await mem.recall({
      query: 'release',
      signal: new AbortController().signal,
    })
    const without = await mem.recall({ query: 'release' })
    expect(withSignal).toStrictEqual(without)
    expect(withSignal.length).toBeGreaterThan(0)
  })
})
