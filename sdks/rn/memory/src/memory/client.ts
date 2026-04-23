import { extractJson } from '../llm/structured.js'
import type { Embedder, Logger, Message, Provider } from '../llm/types.js'
import { noopLogger } from '../llm/types.js'
import type { Retrieval, RetrievalResult } from '../retrieval/index.js'
import type { SearchIndex } from '../search/index.js'
import { type Path, type Store, lastSegment, toPath } from '../store/index.js'
import { createConsolidate } from './consolidate.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { createEpisodeRecorder } from './episodes.js'
import type { Frontmatter } from './frontmatter.js'
import { buildFrontmatter, parseFrontmatter } from './frontmatter.js'
import {
  type Scope,
  ensureMarkdown,
  reflectionPath,
  scopeIndex,
  scopePrefix,
  scopeTopic,
} from './paths.js'
import {
  fireConsolidationEnd,
  fireConsolidationStart,
  fireExtractionEnd,
  fireExtractionStart,
  fireReflectionEnd,
  fireReflectionStart,
} from './plugins.js'
import { createStoreBackedProceduralStore } from './procedural-store.js'
import type {
  ConsolidationReport,
  ContextualiseInput,
  CreateMemoryClientOptions,
  ExtractArgs,
  ExtractResult,
  ExtractedMemory,
  LegacyContextualiseArgs,
  MemoryClient,
  MemoryNoteType,
  PromptContext,
  RecallArgs,
  RecallHit,
  RecallSelectorMode,
  ReflectionResult,
  RememberArgs,
  StoredMemoryNote,
} from './types.js'

const DEFAULT_SCOPE: Scope = 'global'
const DEFAULT_ACTOR_ID = 'mobile'
const DEFAULT_EXTRACT_MIN_MESSAGES = 1
const DEFAULT_EXTRACT_MAX_RECENT = 80
const DEFAULT_RECALL_TOP_K = 10
const DEFAULT_CONTEXTUALISE_TOP_K = 5
const DEFAULT_CONTEXTUALISE_SELECTOR: RecallSelectorMode = 'auto'
const RECALL_SELECTOR_MAX_TOKENS = 256
const RECALL_SELECTOR_TEMPERATURE = 0

const EXTRACT_SCHEMA = JSON.stringify({
  type: 'object',
  properties: {
    notes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          action: { enum: ['create', 'update'] },
          filename: { type: 'string' },
          name: { type: 'string' },
          description: { type: 'string' },
          type: { enum: ['user', 'feedback', 'project', 'reference'] },
          scope: { enum: ['global', 'project', 'agent'] },
          content: { type: 'string' },
          indexEntry: { type: 'string' },
          supersedes: { type: 'string' },
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
Filenames must be short, lowercase-friendly stems.
Optional fields are action, scope, indexEntry, and supersedes.
Default to action=create and the requested scope when those are omitted.`

const RECALL_SELECTOR_SCHEMA = JSON.stringify({
  type: 'object',
  properties: {
    selected: {
      type: 'array',
      items: { type: 'string' },
    },
  },
  required: ['selected'],
})

const RECALL_SELECTOR_SYSTEM_PROMPT = `You are selecting memories that will be useful to an AI assistant as it processes a user's query. You will be given the user's query and a list of available memory files with their filenames and descriptions.

Return a JSON object with a "selected" array of filenames for the memories that will clearly be useful. Only include memories you are certain will be helpful.

- If unsure whether a memory is relevant, do not include it.
- If no memories are relevant, return an empty array.

Respond with ONLY valid JSON, no other text.`

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
    ...(frontmatter.index_entry === undefined ? {} : { indexEntry: frontmatter.index_entry }),
    ...(frontmatter.supersedes === undefined ? {} : { supersedes: frontmatter.supersedes }),
    ...(frontmatter.session_id === undefined ? {} : { sessionId: frontmatter.session_id }),
    ...(frontmatter.session_date === undefined ? {} : { sessionDate: frontmatter.session_date }),
    ...(frontmatter.observed_on === undefined ? {} : { observedOn: frontmatter.observed_on }),
  }
}

const noteToFrontmatter = (note: StoredMemoryNote): Frontmatter => ({
  name: note.name,
  description: note.description,
  ...(note.indexEntry === undefined ? {} : { index_entry: note.indexEntry }),
  type: note.type,
  scope: note.scope,
  created: note.created,
  modified: note.modified,
  ...(note.supersedes === undefined ? {} : { supersedes: note.supersedes }),
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
    .map((note) => buildIndexLine(note, prefix))
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
  ...(note.indexEntry === undefined ? {} : { indexEntry: note.indexEntry }),
  ...(note.supersedes === undefined ? {} : { supersedes: note.supersedes }),
  ...(note.sessionId === undefined ? {} : { sessionId: note.sessionId }),
  ...(note.sessionDate === undefined ? {} : { sessionDate: note.sessionDate }),
  ...(note.observedOn === undefined ? {} : { observedOn: note.observedOn }),
})

const noteFromRetrievalResult = (result: RetrievalResult): StoredMemoryNote => {
  const metadata = result.metadata ?? {}
  const tags = Array.isArray(metadata.tags)
    ? metadata.tags.filter((value): value is string => typeof value === 'string')
    : []
  const fallbackScope = inferScopeFromPath(result.path)
  return {
    path: toPath(result.path),
    name: result.title,
    description: result.summary === '' ? 'memory note' : result.summary,
    type: normaliseNoteType(typeof metadata.type === 'string' ? metadata.type : undefined),
    scope: normaliseScope(
      typeof metadata.scope === 'string' ? metadata.scope : undefined,
      fallbackScope,
    ),
    actorId:
      typeof metadata.actorId === 'string'
        ? metadata.actorId
        : (inferActorIdFromPath(result.path) ?? DEFAULT_ACTOR_ID),
    tags,
    content: result.content,
    created: typeof metadata.created === 'string' ? metadata.created : new Date().toISOString(),
    modified: typeof metadata.modified === 'string' ? metadata.modified : new Date().toISOString(),
    ...(typeof metadata.indexEntry === 'string' ? { indexEntry: metadata.indexEntry } : {}),
    ...(typeof metadata.supersedes === 'string' ? { supersedes: metadata.supersedes } : {}),
    ...(typeof metadata.sessionId === 'string' ? { sessionId: metadata.sessionId } : {}),
    ...(typeof metadata.sessionDate === 'string' ? { sessionDate: metadata.sessionDate } : {}),
    ...(typeof metadata.observedOn === 'string' ? { observedOn: metadata.observedOn } : {}),
  }
}

export const createMemoryClient = (options: CreateMemoryClientOptions): MemoryClient => {
  const logger = options.logger ?? noopLogger
  const defaultScope = options.defaultScope ?? DEFAULT_SCOPE
  const defaultActorId = options.defaultActorId ?? DEFAULT_ACTOR_ID
  const plugins = options.plugins ?? []
  const cursorStore = options.cursorStore ?? createStoreBackedCursorStore(options.store)
  const extractMinMessages = normalisePositiveInt(
    options.extractMinMessages,
    DEFAULT_EXTRACT_MIN_MESSAGES,
  )
  const extractMaxRecent = normalisePositiveInt(
    options.extractMaxRecent,
    DEFAULT_EXTRACT_MAX_RECENT,
  )
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

  const rememberNote = async (
    args: RememberArgs,
    settings: {
      readonly rebuildGeneratedIndex?: boolean
    } = {},
  ): Promise<StoredMemoryNote> => {
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
      ...(args.indexEntry === undefined
        ? existing?.indexEntry === undefined
          ? {}
          : { indexEntry: existing.indexEntry }
        : { indexEntry: args.indexEntry }),
      ...(args.supersedes === undefined
        ? existing?.supersedes === undefined
          ? {}
          : { supersedes: existing.supersedes }
        : { supersedes: args.supersedes }),
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
    if (settings.rebuildGeneratedIndex !== false) {
      await rebuildScopeIndex(options.store, scope, actorId)
    }
    return note
  }

  const remember = async (args: RememberArgs): Promise<StoredMemoryNote> => {
    return await rememberNote(args)
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
    const k = normalisePositiveInt(args.k ?? args.topK, DEFAULT_RECALL_TOP_K)
    const candidateK =
      args.selector === 'auto' ? Math.max(DEFAULT_RECALL_TOP_K, k * 3) : Math.max(k, 1)
    const blockedPaths = new Set<string>([
      ...(args.excludedPaths ?? []).map(String),
      ...(args.surfacedPaths ?? []).map(String),
    ])
    const hitsByPath = new Map<string, RecallHit>()

    for (const currentScope of uniqueScopes(scope, args.fallbackScopes)) {
      const results = await options.retrieval.search({
        query: args.query,
        filters: {
          pathPrefix: `${scopePrefix(currentScope, actorId)}/`,
          scope: currentScope,
        },
        topK: candidateK,
      })

      for (const result of results) {
        if (blockedPaths.has(result.path)) continue
        const hit = recallHitFromRetrievalResult(result)
        const existing = hitsByPath.get(result.path)
        if (existing === undefined || hit.score > existing.score) {
          hitsByPath.set(result.path, hit)
        }
      }
    }

    const merged = [...hitsByPath.values()].sort((left, right) => {
      if (left.score !== right.score) return right.score - left.score
      return left.path.localeCompare(right.path)
    })
    const selected = await selectRecallHits(merged, {
      query: args.query,
      k,
      mode: args.selector ?? 'off',
      logger,
      ...(options.provider === undefined ? {} : { provider: options.provider }),
    })
    return selected.slice(0, k)
  }

  const contextualise = async (args: ContextualiseInput): Promise<PromptContext> => {
    const legacy = isLegacyContextualiseArgs(args)
    const scope = args.scope ?? defaultScope
    const message = legacy ? args.userMessage : args.message
    const query = legacy ? args.query : args.message
    const topK = normalisePositiveInt(
      legacy ? (args.topK ?? args.k) : args.topK,
      DEFAULT_CONTEXTUALISE_TOP_K,
    )
    const fallbackScopes =
      args.fallbackScopes ?? (scope === 'project' ? (['global'] as const) : undefined)
    const memories = await recall({
      query,
      k: topK,
      scope,
      ...(args.actorId === undefined ? {} : { actorId: args.actorId }),
      ...(fallbackScopes === undefined ? {} : { fallbackScopes }),
      ...(args.excludedPaths === undefined ? {} : { excludedPaths: args.excludedPaths }),
      ...(args.surfacedPaths === undefined ? {} : { surfacedPaths: args.surfacedPaths }),
      selector: args.selector ?? DEFAULT_CONTEXTUALISE_SELECTOR,
    })
    return {
      userMessage: message,
      memories,
      systemReminder: formatRecall(memories),
    }
  }

  const persistExtractedMemories = async (
    actorId: string,
    extracted: readonly ExtractedMemory[],
  ): Promise<readonly StoredMemoryNote[]> => {
    const created: StoredMemoryNote[] = []
    const touchedScopes = new Map<string, Scope>()

    for (const memory of extracted) {
      created.push(
        await rememberNote(
          {
            filename: memory.filename,
            name: memory.name,
            description: memory.description,
            content: memory.content,
            type: memory.type,
            scope: memory.scope,
            actorId,
            ...(memory.tags === undefined ? {} : { tags: memory.tags }),
            indexEntry: memory.indexEntry,
            ...(memory.supersedes === undefined ? {} : { supersedes: memory.supersedes }),
            ...(memory.sessionId === undefined ? {} : { sessionId: memory.sessionId }),
            ...(memory.sessionDate === undefined ? {} : { sessionDate: memory.sessionDate }),
            ...(memory.observedOn === undefined ? {} : { observedOn: memory.observedOn }),
          },
          { rebuildGeneratedIndex: false },
        ),
      )
      touchedScopes.set(memory.scope, memory.scope)
    }

    for (const touchedScope of touchedScopes.values()) {
      await rebuildScopeIndex(options.store, touchedScope, actorId)
    }

    return created
  }

  const runExtract = async (
    args: ExtractArgs,
    settings: {
      readonly persist: boolean
      readonly advanceCursor: boolean
    },
  ): Promise<{
    readonly extracted: readonly ExtractedMemory[]
    readonly created: readonly StoredMemoryNote[]
    readonly skipped: boolean
    readonly reason?: string
  }> => {
    if (options.provider === undefined) {
      return { extracted: [], created: [], skipped: true, reason: 'no provider configured' }
    }

    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    const cursorScope = args.sessionId === undefined ? undefined : { sessionId: args.sessionId }
    const cursor = settings.advanceCursor
      ? Math.min(await cursorStore.get(actorId, cursorScope), args.messages.length)
      : 0
    const unseenMessages = args.messages.slice(cursor)
    const windowedMessages =
      unseenMessages.length > extractMaxRecent
        ? unseenMessages.slice(-extractMaxRecent)
        : unseenMessages

    if (windowedMessages.length < extractMinMessages) {
      return {
        extracted: [],
        created: [],
        skipped: true,
        reason:
          unseenMessages.length === 0
            ? 'no new messages to extract'
            : 'extract threshold not reached',
      }
    }

    await fireExtractionStart(
      plugins,
      { actorId, scope, messages: windowedMessages, extracted: [] },
      logger,
    )

    let extracted: readonly ExtractedMemory[] = []
    try {
      const payload = await options.provider.structured({
        taskType: 'memory-extract',
        system: EXTRACT_SYSTEM_PROMPT,
        messages: [
          {
            role: 'user',
            content: `Conversation:\n${buildPromptMessages(windowedMessages)}`,
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
        return { extracted: [], created: [], skipped: true, reason: 'invalid extract payload' }
      }

      extracted = normaliseExtractedMemories(parsed, scope, args)
      const created = settings.persist ? await persistExtractedMemories(actorId, extracted) : []
      if (settings.advanceCursor) {
        await cursorStore.set(actorId, args.messages.length, cursorScope)
      }
      return { extracted, created, skipped: false }
    } catch (error) {
      logger.warn('memory client: extract provider call failed', {
        error: error instanceof Error ? error.message : String(error),
      })
      return { extracted: [], created: [], skipped: true, reason: 'extract request failed' }
    } finally {
      await fireExtractionEnd(
        plugins,
        { actorId, scope, messages: windowedMessages, extracted },
        logger,
      )
    }
  }

  const extract = async (args: ExtractArgs): Promise<ExtractResult> => {
    const result = await runExtract(args, { persist: true, advanceCursor: true })
    return {
      created: result.created,
      skipped: result.skipped,
      ...(result.reason === undefined ? {} : { reason: result.reason }),
    }
  }

  const previewExtract = async (args: ExtractArgs): Promise<readonly ExtractedMemory[]> => {
    const result = await runExtract(args, { persist: false, advanceCursor: false })
    return result.extracted
  }

  const reflect = async (args: ExtractArgs): Promise<ReflectionResult | null> => {
    if (options.provider === undefined) return null
    const scope = args.scope ?? defaultScope
    const actorId = args.actorId ?? defaultActorId
    let result: ReflectionResult | null = null

    await fireReflectionStart(plugins, { actorId, scope, messages: args.messages }, logger)

    try {
      const payload = await options.provider.structured({
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
      result = {
        outcome: parsed.outcome,
        summary: parsed.summary,
        retryFeedback: parsed.retryFeedback,
        shouldRecordEpisode: parsed.shouldRecordEpisode,
        openQuestions: parsed.openQuestions,
        heuristics: parsed.heuristics,
        path,
      }
      return result
    } finally {
      await fireReflectionEnd(
        plugins,
        {
          actorId,
          scope,
          messages: args.messages,
          ...(result === null ? {} : { result }),
        },
        logger,
      )
    }
  }

  const consolidate = async (
    args: { readonly scope?: Scope; readonly actorId?: string } = {},
  ): Promise<ConsolidationReport> => {
    const scope = args.scope ?? defaultScope
    let report: ConsolidationReport | undefined

    await fireConsolidationStart(plugins, { scope }, logger)
    try {
      report = await consolidateMemory(args)
      await rebuildIndex(args)
      return report
    } finally {
      await fireConsolidationEnd(
        plugins,
        { scope, ...(report === undefined ? {} : { report }) },
        logger,
      )
    }
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
    previewExtract,
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
    subscribe: (sink) => options.store.subscribe(sink),
    unsubscribe: (handle) => {
      handle()
    },
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

const buildIndexLine = (note: StoredMemoryNote, prefix: Path): string => {
  const entry = note.indexEntry?.trim()
  if (entry !== undefined && entry !== '') {
    return entry.startsWith('- ') ? entry : `- ${relativePath(prefix, note.path)}: ${entry}`
  }
  return `- ${relativePath(prefix, note.path)}: ${note.description}`
}

const inferScopeFromPath = (path: string): Scope => {
  if (path.startsWith('memory/project/')) return 'project'
  if (path.startsWith('memory/agent/')) return 'agent'
  return 'global'
}

const inferActorIdFromPath = (path: string): string | undefined => {
  const scope = inferScopeFromPath(path)
  if (scope === 'global') return undefined
  const segments = path.split('/')
  return segments[2]
}

const normalisePositiveInt = (value: number | undefined, fallback: number): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
  return Math.max(1, Math.trunc(value))
}

const uniqueScopes = (scope: Scope, fallbackScopes?: readonly Scope[]): readonly Scope[] => {
  const ordered = [scope, ...(fallbackScopes ?? [])]
  const deduped: Scope[] = []
  const seen = new Set<Scope>()
  for (const currentScope of ordered) {
    if (seen.has(currentScope)) continue
    seen.add(currentScope)
    deduped.push(currentScope)
  }
  return deduped
}

const recallHitFromRetrievalResult = (result: RetrievalResult): RecallHit => {
  const note = noteFromRetrievalResult(result)
  return {
    path: note.path,
    score: result.score,
    content: result.content,
    note,
  }
}

type ExtractCandidate = {
  readonly action?: 'create' | 'update'
  readonly filename: string
  readonly name: string
  readonly description: string
  readonly type: Exclude<MemoryNoteType, 'reflection'>
  readonly scope?: Scope
  readonly content: string
  readonly indexEntry?: string
  readonly supersedes?: string
  readonly tags?: readonly string[]
}

const extractCandidatesFromPayload = (value: unknown): readonly unknown[] => {
  if (typeof value !== 'object' || value === null) return []
  if (Array.isArray((value as { notes?: unknown }).notes)) {
    return (value as { notes: unknown[] }).notes
  }
  if (Array.isArray((value as { memories?: unknown }).memories)) {
    return (value as { memories: unknown[] }).memories
  }
  return []
}

const isExtractCandidate = (value: unknown): value is ExtractCandidate => {
  if (typeof value !== 'object' || value === null) return false
  const current = value as Record<string, unknown>
  return (
    (current.action === undefined || current.action === 'create' || current.action === 'update') &&
    typeof current.filename === 'string' &&
    typeof current.name === 'string' &&
    typeof current.description === 'string' &&
    (current.type === 'user' ||
      current.type === 'feedback' ||
      current.type === 'project' ||
      current.type === 'reference') &&
    (current.scope === undefined ||
      current.scope === 'global' ||
      current.scope === 'project' ||
      current.scope === 'agent') &&
    typeof current.content === 'string' &&
    (current.indexEntry === undefined || typeof current.indexEntry === 'string') &&
    (current.supersedes === undefined || typeof current.supersedes === 'string') &&
    (current.tags === undefined ||
      (Array.isArray(current.tags) && current.tags.every((entry) => typeof entry === 'string')))
  )
}

const normaliseExtractedMemories = (
  parsed: unknown,
  fallbackScope: Scope,
  args: ExtractArgs,
): readonly ExtractedMemory[] => {
  const extracted: ExtractedMemory[] = []
  for (const candidate of extractCandidatesFromPayload(parsed)) {
    if (!isExtractCandidate(candidate)) continue
    const filename = ensureMarkdown(candidate.filename)
    const description =
      candidate.description.trim() === '' ? candidate.name : candidate.description.trim()
    extracted.push({
      action: candidate.action ?? 'create',
      filename,
      name: candidate.name.trim(),
      description,
      type: candidate.type,
      content: candidate.content,
      indexEntry:
        candidate.indexEntry?.trim() === undefined || candidate.indexEntry.trim() === ''
          ? `- ${filename}: ${description}`
          : candidate.indexEntry.trim(),
      scope: candidate.scope ?? fallbackScope,
      ...(candidate.supersedes === undefined
        ? {}
        : { supersedes: ensureMarkdown(candidate.supersedes) }),
      ...(candidate.tags === undefined ? {} : { tags: candidate.tags }),
      ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
      ...(args.sessionDate === undefined ? {} : { sessionDate: args.sessionDate }),
      ...(args.observedOn === undefined ? {} : { observedOn: args.observedOn }),
    })
  }
  return extracted
}

const formatRecall = (memories: readonly RecallHit[]): string => {
  if (memories.length === 0) return ''
  return memories
    .map((memory) => {
      const scopeLabel = memory.note.scope === 'global' ? 'Global memory' : 'Memory'
      const parts = [
        '<system-reminder>',
        `${scopeLabel}: ${lastSegment(memory.path)}`,
        ...(memory.note.modified === undefined ? [] : [`_modified: ${memory.note.modified}_`]),
        '',
        memory.content,
        '</system-reminder>',
      ]
      return parts.join('\n')
    })
    .join('\n')
    .trim()
}

const isLegacyContextualiseArgs = (args: ContextualiseInput): args is LegacyContextualiseArgs =>
  'userMessage' in args

type RecallSelectorInput = {
  readonly query: string
  readonly k: number
  readonly provider?: Provider
  readonly logger: Logger
  readonly mode: RecallSelectorMode
}

type RecallSelectorCandidate = {
  readonly label: string
  readonly hit: RecallHit
}

const selectRecallHits = async (
  hits: readonly RecallHit[],
  input: RecallSelectorInput,
): Promise<readonly RecallHit[]> => {
  if (input.mode !== 'auto') return hits
  if (hits.length <= Math.max(1, input.k)) return hits
  if (input.provider === undefined) return hits

  const candidates = hits.slice(
    0,
    Math.min(hits.length, Math.max(DEFAULT_RECALL_TOP_K, input.k + 2)),
  )
  const labelled: readonly RecallSelectorCandidate[] = candidates.map((hit) => ({
    label: hit.path,
    hit,
  }))
  const userPrompt = buildRecallSelectorUserPrompt(input.query, labelled)

  let raw: string | undefined
  if (input.provider.supportsStructuredDecoding()) {
    try {
      raw = await input.provider.structured({
        taskType: 'memory-recall-selector',
        messages: [{ role: 'user', content: userPrompt }],
        system: RECALL_SELECTOR_SYSTEM_PROMPT,
        schema: RECALL_SELECTOR_SCHEMA,
        schemaName: 'memory_recall_selector',
        maxTokens: RECALL_SELECTOR_MAX_TOKENS,
        temperature: RECALL_SELECTOR_TEMPERATURE,
      })
    } catch (error) {
      input.logger.debug('memory client: structured recall selector failed', {
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }

  if (raw === undefined) {
    try {
      const response = await input.provider.complete({
        taskType: 'memory-recall-selector',
        messages: [{ role: 'user', content: userPrompt }],
        system: RECALL_SELECTOR_SYSTEM_PROMPT,
        maxTokens: RECALL_SELECTOR_MAX_TOKENS,
        temperature: RECALL_SELECTOR_TEMPERATURE,
        jsonMode: true,
      })
      raw = response.content
    } catch (error) {
      input.logger.warn('memory client: recall selector failed', {
        error: error instanceof Error ? error.message : String(error),
      })
      return hits
    }
  }

  const selectedLabels = parseRecallSelectorSelected(raw)
  if (selectedLabels.size === 0) return hits

  const selected: RecallHit[] = []
  const selectedPaths = new Set<string>()
  for (const candidate of labelled) {
    if (!selectedLabels.has(candidate.label)) continue
    selected.push(candidate.hit)
    selectedPaths.add(candidate.hit.path)
  }
  if (selected.length === 0) return hits
  if (selected.length >= input.k) return selected
  return [...selected, ...hits.filter((hit) => !selectedPaths.has(hit.path))]
}

const buildRecallSelectorUserPrompt = (
  query: string,
  candidates: readonly RecallSelectorCandidate[],
): string => {
  const parts = ['## User query', query, '', '## Available memories', '']
  for (const candidate of candidates) {
    const tags = candidate.hit.note.tags.length > 0 ? candidate.hit.note.tags.join(', ') : 'none'
    parts.push(`- filename: ${candidate.label}`)
    parts.push(`  name: ${candidate.hit.note.name}`)
    parts.push(`  description: ${candidate.hit.note.description || 'none'}`)
    parts.push(`  scope: ${candidate.hit.note.scope}`)
    parts.push(`  type: ${candidate.hit.note.type || 'unknown'}`)
    parts.push(`  tags: ${tags}`)
    parts.push('')
  }
  return parts.join('\n').trim()
}

const parseRecallSelectorSelected = (raw: string): ReadonlySet<string> => {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const selected = Array.isArray(parsed.selected) ? parsed.selected : []
    return new Set(
      selected
        .filter((value): value is string => typeof value === 'string')
        .map((value) => value.trim())
        .filter((value) => value !== ''),
    )
  } catch {
    return new Set()
  }
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
