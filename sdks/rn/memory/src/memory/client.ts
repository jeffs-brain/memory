import { extractJson } from '../llm/structured.js'
import type { Embedder, Logger, Message, Provider } from '../llm/types.js'
import { noopLogger } from '../llm/types.js'
import type { Retrieval, RetrievalResult } from '../retrieval/index.js'
import type { SearchIndex } from '../search/index.js'
import { type Path, type Store, lastSegment, toPath } from '../store/index.js'
import { createConsolidate } from './consolidate.js'
import { createEpisodeRecorder } from './episodes.js'
import type { Frontmatter } from './frontmatter.js'
import { buildFrontmatter, parseFrontmatter } from './frontmatter.js'
import { type Scope, reflectionPath, scopeIndex, scopePrefix, scopeTopic } from './paths.js'
import { createStoreBackedProceduralStore } from './procedural-store.js'
import type {
  ConsolidationReport,
  CreateMemoryClientOptions,
  ExtractArgs,
  ExtractResult,
  MemoryClient,
  MemoryNoteType,
  PromptContext,
  RecallArgs,
  RecallHit,
  ReflectionResult,
  RememberArgs,
  StoredMemoryNote,
} from './types.js'

const DEFAULT_SCOPE: Scope = 'global'
const DEFAULT_ACTOR_ID = 'mobile'

const EXTRACT_SCHEMA = JSON.stringify({
  type: 'object',
  properties: {
    notes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          filename: { type: 'string' },
          name: { type: 'string' },
          description: { type: 'string' },
          type: { enum: ['user', 'feedback', 'project', 'reference'] },
          content: { type: 'string' },
          tags: {
            type: 'array',
            items: { type: 'string' },
          },
        },
        required: ['filename', 'name', 'description', 'type', 'content'],
      },
    },
  },
  required: ['notes'],
})

const EXTRACT_SYSTEM_PROMPT = `You extract durable memory notes from a chat.
Return only JSON.
Create notes only for stable facts, preferences, plans, or feedback that will still matter later.
Avoid ephemeral pleasantries and one-off chatter.
Descriptions must be one line and specific.
Filenames must be short, lowercase-friendly stems.`

const REFLECT_SCHEMA = JSON.stringify({
  type: 'object',
  properties: {
    outcome: { enum: ['success', 'partial', 'failure', 'unknown'] },
    summary: { type: 'string' },
    retryFeedback: { type: 'string' },
    shouldRecordEpisode: { type: 'boolean' },
    openQuestions: {
      type: 'array',
      items: { type: 'string' },
    },
    heuristics: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          rule: { type: 'string' },
          context: { type: 'string' },
          confidence: { enum: ['low', 'medium', 'high'] },
          category: { type: 'string' },
          scope: { enum: ['global', 'project', 'agent'] },
          antiPattern: { type: 'boolean' },
        },
        required: ['rule', 'context', 'confidence', 'category', 'scope', 'antiPattern'],
      },
    },
  },
  required: [
    'outcome',
    'summary',
    'retryFeedback',
    'shouldRecordEpisode',
    'openQuestions',
    'heuristics',
  ],
})

const REFLECT_SYSTEM_PROMPT = `You are a reflection agent. You analyse completed coding sessions to extract lasting wisdom.
Return only JSON.
Identify generalisable patterns, not a blow-by-blow summary.
Each heuristic must be actionable, scoped, and use British English.`

type ReflectionPayload = Omit<ReflectionResult, 'path'> & {
  readonly retryFeedback: string
  readonly shouldRecordEpisode: boolean
}

const toIso = (value?: string): string => value ?? new Date().toISOString()

const buildNoteContent = (frontmatter: Frontmatter, body: string): string => {
  const built = buildFrontmatter(frontmatter)
  const trimmed = body.trim()
  return trimmed === '' ? `${built}\n` : `${built}\n${trimmed}\n`
}

const parseStoredNote = (
  path: Path,
  raw: string,
  actorId: string,
  defaultScope: Scope,
): StoredMemoryNote => {
  const { frontmatter, body } = parseFrontmatter(raw)
  const noteType = normaliseNoteType(frontmatter.type)
  const scope = normaliseScope(frontmatter.scope, defaultScope)
  const created = frontmatter.created ?? new Date().toISOString()
  const modified = frontmatter.modified ?? created
  return {
    path,
    name: frontmatter.name ?? lastSegment(path).replace(/\.md$/i, ''),
    description: frontmatter.description ?? firstMeaningfulLine(body) ?? 'memory note',
    type: noteType,
    scope,
    actorId,
    tags: frontmatter.tags ?? [],
    content: body,
    created,
    modified,
    ...(frontmatter.session_id === undefined ? {} : { sessionId: frontmatter.session_id }),
    ...(frontmatter.session_date === undefined ? {} : { sessionDate: frontmatter.session_date }),
    ...(frontmatter.observed_on === undefined ? {} : { observedOn: frontmatter.observed_on }),
  }
}

const noteToFrontmatter = (note: StoredMemoryNote): Frontmatter => ({
  name: note.name,
  description: note.description,
  type: note.type,
  scope: note.scope,
  created: note.created,
  modified: note.modified,
  ...(note.tags.length === 0 ? {} : { tags: note.tags }),
  ...(note.sessionId === undefined ? {} : { session_id: note.sessionId }),
  ...(note.sessionDate === undefined ? {} : { session_date: note.sessionDate }),
  ...(note.observedOn === undefined ? {} : { observed_on: note.observedOn }),
  extra: {},
})

const normaliseScope = (value: string | undefined, fallback: Scope): Scope => {
  switch (value) {
    case 'global':
    case 'project':
    case 'agent':
      return value
    default:
      return fallback
  }
}

const normaliseNoteType = (value: string | undefined): MemoryNoteType => {
  switch (value) {
    case 'feedback':
    case 'project':
    case 'reference':
    case 'reflection':
      return value
    default:
      return 'user'
  }
}

const firstMeaningfulLine = (content: string): string | undefined => {
  for (const line of content.split('\n')) {
    const trimmed = line.trim()
    if (trimmed !== '') return trimmed
  }
  return undefined
}

const relativePath = (prefix: Path, path: Path): string => {
  const prefixValue = String(prefix)
  const value = String(path)
  if (!value.startsWith(`${prefixValue}/`)) return lastSegment(path)
  return value.slice(prefixValue.length + 1)
}

const buildIndexContent = (notes: readonly StoredMemoryNote[], prefix: Path): string => {
  const lines = [...notes]
    .sort((left, right) => {
      const timeDelta = Date.parse(right.modified) - Date.parse(left.modified)
      if (timeDelta !== 0) return timeDelta
      return left.path.localeCompare(right.path)
    })
    .map((note) => {
      return `- ${relativePath(prefix, note.path)}: ${note.description}`
    })
  return `${lines.join('\n')}\n`
}

const buildPromptMessages = (messages: readonly Message[]): string => {
  return messages
    .map((message) => `${message.role.toUpperCase()}: ${message.content ?? ''}`.trim())
    .join('\n')
}

const metadataFromNote = (note: StoredMemoryNote): Record<string, unknown> => ({
  type: note.type,
  scope: note.scope,
  actorId: note.actorId,
  tags: [...note.tags],
  created: note.created,
  modified: note.modified,
  ...(note.sessionId === undefined ? {} : { sessionId: note.sessionId }),
  ...(note.sessionDate === undefined ? {} : { sessionDate: note.sessionDate }),
  ...(note.observedOn === undefined ? {} : { observedOn: note.observedOn }),
})

const noteFromRetrievalResult = (result: RetrievalResult): StoredMemoryNote => {
  const metadata = result.metadata ?? {}
  const tags = Array.isArray(metadata.tags)
    ? metadata.tags.filter((value): value is string => typeof value === 'string')
    : []
  return {
    path: toPath(result.path),
    name: result.title,
    description: result.summary === '' ? 'memory note' : result.summary,
    type: normaliseNoteType(typeof metadata.type === 'string' ? metadata.type : undefined),
    scope: normaliseScope(
      typeof metadata.scope === 'string' ? metadata.scope : undefined,
      'global',
    ),
    actorId: typeof metadata.actorId === 'string' ? metadata.actorId : DEFAULT_ACTOR_ID,
    tags,
    content: result.content,
    created: typeof metadata.created === 'string' ? metadata.created : new Date().toISOString(),
    modified: typeof metadata.modified === 'string' ? metadata.modified : new Date().toISOString(),
    ...(typeof metadata.sessionId === 'string' ? { sessionId: metadata.sessionId } : {}),
    ...(typeof metadata.sessionDate === 'string' ? { sessionDate: metadata.sessionDate } : {}),
    ...(typeof metadata.observedOn === 'string' ? { observedOn: metadata.observedOn } : {}),
  }
}

export const createMemoryClient = (options: CreateMemoryClientOptions): MemoryClient => {
  const logger = options.logger ?? noopLogger
  const defaultScope = options.defaultScope ?? DEFAULT_SCOPE
  const defaultActorId = options.defaultActorId ?? DEFAULT_ACTOR_ID
  const consolidateMemory = createConsolidate({
    store: options.store,
    logger,
    defaultScope,
    defaultActorId,
    ...(options.provider === undefined ? {} : { provider: options.provider }),
  })
  const episodes = createEpisodeRecorder({
    store: options.store,
    logger,
    defaultScope,
    defaultActorId,
  })
  const proceduralStore = createStoreBackedProceduralStore(options.store)

  const remember = async (args: RememberArgs): Promise<StoredMemoryNote> => {
    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    const path = scopeTopic(scope, actorId, args.filename)

    let existing: StoredMemoryNote | undefined
    try {
      existing = parseStoredNote(path, await options.store.read(path), actorId, scope)
    } catch {
      existing = undefined
    }

    const note: StoredMemoryNote = {
      path,
      name: args.name,
      description: args.description,
      type: args.type ?? existing?.type ?? 'user',
      scope,
      actorId,
      tags: args.tags ?? existing?.tags ?? [],
      content: args.content,
      created: existing?.created ?? toIso(args.created),
      modified: toIso(args.modified),
      ...(args.sessionId === undefined
        ? existing?.sessionId === undefined
          ? {}
          : { sessionId: existing.sessionId }
        : { sessionId: args.sessionId }),
      ...(args.sessionDate === undefined
        ? existing?.sessionDate === undefined
          ? {}
          : { sessionDate: existing.sessionDate }
        : { sessionDate: args.sessionDate }),
      ...(args.observedOn === undefined
        ? existing?.observedOn === undefined
          ? {}
          : { observedOn: existing.observedOn }
        : { observedOn: args.observedOn }),
    }

    await options.store.write(path, buildNoteContent(noteToFrontmatter(note), note.content))
    await syncNoteToIndex(options.searchIndex, options.embedder, note, logger)
    await rebuildScopeIndex(options.store, scope, actorId)
    return note
  }

  const listNotes = async (
    args: { readonly scope?: Scope; readonly actorId?: string } = {},
  ): Promise<readonly StoredMemoryNote[]> => {
    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    const prefix = scopePrefix(scope, actorId)
    const files = await options.store.list(prefix, { recursive: true, includeGenerated: false })
    const notes: StoredMemoryNote[] = []
    for (const file of files) {
      if (file.isDir) continue
      if (!file.path.endsWith('.md') || lastSegment(file.path) === 'MEMORY.md') continue
      try {
        notes.push(parseStoredNote(file.path, await options.store.read(file.path), actorId, scope))
      } catch (error) {
        logger.warn('memory client: failed to parse note', {
          path: file.path,
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }
    return notes
  }

  const rebuildIndex = async (
    args: { readonly scope?: Scope; readonly actorId?: string } = {},
  ): Promise<void> => {
    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    const notes = await listNotes({ scope, actorId })
    const livePaths = new Set(notes.map((note) => note.path))
    for (const path of options.searchIndex.indexedPaths()) {
      if (path.startsWith(`${scopePrefix(scope, actorId)}/`) && !livePaths.has(path as Path)) {
        options.searchIndex.deleteByPath(path)
      }
    }
    for (const note of notes) {
      await syncNoteToIndex(options.searchIndex, options.embedder, note, logger)
    }
    await rebuildScopeIndex(options.store, scope, actorId)
  }

  const recall = async (args: RecallArgs): Promise<readonly RecallHit[]> => {
    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    const results = await options.retrieval.search({
      query: args.query,
      filters: {
        pathPrefix: `${scopePrefix(scope, actorId)}/`,
        scope,
      },
      ...(args.topK === undefined ? {} : { topK: args.topK }),
    })
    return results.map((result) => ({
      path: toPath(result.path),
      score: result.score,
      note: noteFromRetrievalResult(result),
    }))
  }

  const contextualise = async (
    args: RecallArgs & { readonly userMessage: string },
  ): Promise<PromptContext> => {
    const memories = await recall(args)
    const lines = memories.map((hit) => `- ${hit.note.name}: ${hit.note.description}`)
    return {
      userMessage: args.userMessage,
      memories,
      systemReminder:
        lines.length === 0
          ? ''
          : `Relevant memory:\n${lines.join('\n')}\n\nUse this only when it helps.`,
    }
  }

  const extract = async (args: ExtractArgs): Promise<ExtractResult> => {
    if (options.provider === undefined) {
      return { created: [], skipped: true, reason: 'no provider configured' }
    }
    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    const payload = await options.provider.structured({
      model: options.provider.modelName(),
      taskType: 'memory-extract',
      system: EXTRACT_SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: `Conversation:\n${buildPromptMessages(args.messages)}`,
        },
      ],
      schema: EXTRACT_SCHEMA,
      schemaName: 'memory_extract',
      maxRetries: 3,
    })

    let parsed: unknown
    try {
      parsed = JSON.parse(extractJson(payload))
    } catch (error) {
      logger.warn('memory client: extract payload parse failed', {
        error: error instanceof Error ? error.message : String(error),
      })
      return { created: [], skipped: true, reason: 'invalid extract payload' }
    }

    const notes = Array.isArray((parsed as { notes?: unknown }).notes)
      ? (parsed as { notes: unknown[] }).notes
      : []
    const created: StoredMemoryNote[] = []
    for (const candidate of notes) {
      if (!isExtractCandidate(candidate)) continue
      created.push(
        await remember({
          filename: candidate.filename,
          name: candidate.name,
          description: candidate.description,
          content: candidate.content,
          type: candidate.type,
          scope,
          actorId,
          ...(candidate.tags === undefined ? {} : { tags: candidate.tags }),
          ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
          ...(args.sessionDate === undefined ? {} : { sessionDate: args.sessionDate }),
          ...(args.observedOn === undefined ? {} : { observedOn: args.observedOn }),
        }),
      )
    }
    return { created, skipped: false }
  }

  const reflect = async (args: ExtractArgs): Promise<ReflectionResult | null> => {
    if (options.provider === undefined) return null
    const payload = await options.provider.structured({
      model: options.provider.modelName(),
      taskType: 'memory-reflect',
      system: REFLECT_SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content: `Conversation:\n${buildPromptMessages(args.messages)}`,
        },
      ],
      schema: REFLECT_SCHEMA,
      schemaName: 'memory_reflect',
      maxRetries: 3,
    })

    let parsed: unknown
    try {
      parsed = JSON.parse(extractJson(payload))
    } catch {
      return null
    }
    if (!isReflectionPayload(parsed)) return null
    const sessionId = args.sessionId ?? `reflection-${Date.now()}`
    const path = reflectionPath(sessionId)
    const content = [
      parsed.summary,
      '',
      'Outcome:',
      parsed.outcome,
      '',
      'Retry feedback:',
      parsed.retryFeedback === '' ? 'none' : parsed.retryFeedback,
      '',
      parsed.openQuestions.length === 0
        ? 'Open questions:\n- none'
        : `Open questions:\n${parsed.openQuestions.map((value) => `- ${value}`).join('\n')}`,
      '',
      parsed.heuristics.length === 0
        ? 'Heuristics:\n- none'
        : `Heuristics:\n${parsed.heuristics
            .map(
              (heuristic) =>
                `- ${heuristic.rule} (${heuristic.category}, ${heuristic.confidence}, ${heuristic.scope}, ${heuristic.antiPattern ? 'anti-pattern' : 'pattern'})`,
            )
            .join('\n')}`,
      '',
    ].join('\n')
    await options.store.write(
      path,
      buildNoteContent(
        {
          name: `Reflection ${sessionId}`,
          description: parsed.summary,
          type: 'reflection',
          created: toIso(),
          modified: toIso(),
          ...(args.sessionId === undefined ? {} : { session_id: args.sessionId }),
          extra: {
            outcome: parsed.outcome,
            should_record_episode: String(parsed.shouldRecordEpisode),
          },
        },
        content,
      ),
    )
    return {
      outcome: parsed.outcome,
      summary: parsed.summary,
      retryFeedback: parsed.retryFeedback,
      shouldRecordEpisode: parsed.shouldRecordEpisode,
      openQuestions: parsed.openQuestions,
      heuristics: parsed.heuristics,
      path,
    }
  }

  const consolidate = async (
    args: { readonly scope?: Scope; readonly actorId?: string } = {},
  ): Promise<ConsolidationReport> => {
    const report = await consolidateMemory(args)
    await rebuildIndex(args)
    return report
  }

  return {
    brainId: options.brainId,
    store: options.store,
    searchIndex: options.searchIndex,
    retrieval: options.retrieval,
    remember,
    forget: async (path) => {
      await options.store.delete(path)
      options.searchIndex.deleteByPath(path)
      const scope = path.includes('memory/project/')
        ? 'project'
        : path.includes('memory/agent/')
          ? 'agent'
          : 'global'
      const actorId = scope === 'global' ? defaultActorId : (path.split('/')[2] ?? defaultActorId)
      await rebuildScopeIndex(options.store, scope, actorId)
    },
    listNotes,
    rebuildIndex,
    recall,
    contextualise,
    extract,
    reflect,
    consolidate,
    recordEpisode: episodes.record,
    getEpisode: episodes.get,
    listEpisodes: episodes.list,
    queryEpisodes: episodes.query,
    persistProceduralRecords: async (args) =>
      await proceduralStore.persist({
        actorId: args.actorId ?? defaultActorId,
        records: args.records,
        ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
        ...(args.reason === undefined ? {} : { reason: args.reason }),
      }),
    detectAndPersistProceduralRecords: async (args) =>
      await proceduralStore.detectAndPersist({
        actorId: args.actorId ?? defaultActorId,
        messages: args.messages,
        ...(args.observedAt === undefined ? {} : { observedAt: args.observedAt }),
        ...(args.maxContextLength === undefined ? {} : { maxContextLength: args.maxContextLength }),
        ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
        ...(args.reason === undefined ? {} : { reason: args.reason }),
      }),
    listProceduralRecords: async (args = {}) =>
      await proceduralStore.list({
        actorId: args.actorId ?? defaultActorId,
        ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
        ...(args.tier === undefined ? {} : { tier: args.tier }),
        ...(args.outcome === undefined ? {} : { outcome: args.outcome }),
        ...(args.name === undefined ? {} : { name: args.name }),
        ...(args.tags === undefined ? {} : { tags: args.tags }),
        ...(args.since === undefined ? {} : { since: args.since }),
        ...(args.until === undefined ? {} : { until: args.until }),
        ...(args.limit === undefined ? {} : { limit: args.limit }),
        ...(args.sort === undefined ? {} : { sort: args.sort }),
      }),
    queryProceduralRecords: async (args) =>
      await proceduralStore.query({
        actorId: args.actorId ?? defaultActorId,
        text: args.text,
        ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
        ...(args.tier === undefined ? {} : { tier: args.tier }),
        ...(args.outcome === undefined ? {} : { outcome: args.outcome }),
        ...(args.name === undefined ? {} : { name: args.name }),
        ...(args.tags === undefined ? {} : { tags: args.tags }),
        ...(args.since === undefined ? {} : { since: args.since }),
        ...(args.until === undefined ? {} : { until: args.until }),
        ...(args.limit === undefined ? {} : { limit: args.limit }),
        ...(args.sort === undefined ? {} : { sort: args.sort }),
      }),
    close: async () => {
      await options.searchIndex.close()
      await options.store.close()
    },
  }
}

const syncNoteToIndex = async (
  searchIndex: SearchIndex,
  embedder: Embedder | undefined,
  note: StoredMemoryNote,
  logger: Logger,
): Promise<void> => {
  let embedding: number[] | undefined
  let embeddingModel: string | undefined
  if (embedder !== undefined) {
    try {
      const embedded = await embedder.embed([`${note.name}\n${note.description}\n${note.content}`])
      embedding = embedded[0]
      embeddingModel = embedder.model()
    } catch (error) {
      logger.warn('memory client: embedding failed, falling back to BM25 only', {
        path: note.path,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }

  searchIndex.upsertChunk({
    id: note.path,
    path: note.path,
    title: note.name,
    summary: note.description,
    tags: note.tags,
    content: note.content,
    metadata: metadataFromNote(note),
    ...(embedding === undefined ? {} : { embedding }),
    ...(embeddingModel === undefined ? {} : { embeddingModel }),
  })
}

const rebuildScopeIndex = async (store: Store, scope: Scope, actorId: string): Promise<void> => {
  const prefix = scopePrefix(scope, actorId)
  const files = await store.list(prefix, { recursive: true, includeGenerated: false })
  const notes: StoredMemoryNote[] = []
  for (const file of files) {
    if (file.isDir) continue
    if (!file.path.endsWith('.md') || lastSegment(file.path) === 'MEMORY.md') continue
    try {
      notes.push(parseStoredNote(file.path, await store.read(file.path), actorId, scope))
    } catch {
      // Ignore malformed notes while rebuilding the generated index.
    }
  }
  await store.write(scopeIndex(scope, actorId), buildIndexContent(notes, prefix))
}

const isExtractCandidate = (
  value: unknown,
): value is {
  readonly filename: string
  readonly name: string
  readonly description: string
  readonly type: MemoryNoteType
  readonly content: string
  readonly tags?: readonly string[]
} => {
  if (typeof value !== 'object' || value === null) return false
  const current = value as Record<string, unknown>
  return (
    typeof current.filename === 'string' &&
    typeof current.name === 'string' &&
    typeof current.description === 'string' &&
    typeof current.type === 'string' &&
    typeof current.content === 'string' &&
    (current.tags === undefined ||
      (Array.isArray(current.tags) && current.tags.every((entry) => typeof entry === 'string')))
  )
}

const isReflectionPayload = (value: unknown): value is ReflectionPayload => {
  if (typeof value !== 'object' || value === null) return false
  const current = value as Record<string, unknown>
  return (
    (current.outcome === 'success' ||
      current.outcome === 'partial' ||
      current.outcome === 'failure' ||
      current.outcome === 'unknown') &&
    typeof current.summary === 'string' &&
    typeof current.retryFeedback === 'string' &&
    typeof current.shouldRecordEpisode === 'boolean' &&
    Array.isArray(current.openQuestions) &&
    current.openQuestions.every((entry) => typeof entry === 'string') &&
    Array.isArray(current.heuristics) &&
    current.heuristics.every(isHeuristic)
  )
}

const isHeuristic = (value: unknown): boolean => {
  if (typeof value !== 'object' || value === null) return false
  const current = value as Record<string, unknown>
  return (
    typeof current.rule === 'string' &&
    typeof current.context === 'string' &&
    (current.confidence === 'low' ||
      current.confidence === 'medium' ||
      current.confidence === 'high') &&
    typeof current.category === 'string' &&
    (current.scope === 'global' || current.scope === 'project' || current.scope === 'agent') &&
    typeof current.antiPattern === 'boolean'
  )
}
