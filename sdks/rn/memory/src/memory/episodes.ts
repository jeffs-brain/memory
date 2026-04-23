import { type Logger, type Message, noopLogger } from '../llm/types.js'
import { type Path, type Store, joinPath, toPath } from '../store/index.js'
import { buildFrontmatter, parseFrontmatter } from './frontmatter.js'
import { ensureMarkdown } from './paths.js'
import type { Scope } from './paths.js'
import type { Heuristic, ReflectionResult } from './types.js'

const EPISODES_PREFIX: Path = toPath('episodes')
const RECORD_REASON = 'record-episode'
const DEFAULT_LIST_LIMIT = 50
const DEFAULT_QUERY_LIMIT = 20
const MAX_LIMIT = 200
const TOKEN_PATTERN = /[a-z0-9][a-z0-9._:-]*/gi
const WRITE_SIGNAL_TOOL_NAMES = new Set(['write'])
const EDIT_SIGNAL_TOOL_NAMES = new Set(['edit'])

type StoredEpisodeHeuristic = {
  readonly rule: string
  readonly context: string
  readonly confidence: Heuristic['confidence']
  readonly category: string
  readonly scope: Scope
  readonly anti_pattern: boolean
}

type StoredEpisodeSignals = {
  readonly message_count: number
  readonly substantive_message_count: number
  readonly user_message_count: number
  readonly assistant_message_count: number
  readonly tool_message_count: number
  readonly tool_call_count: number
  readonly write_signal: boolean
  readonly edit_signal: boolean
  readonly tool_signal: boolean
}

type StoredEpisodePayload = {
  readonly session_id: string
  readonly actor_id: string
  readonly scope: Scope
  readonly summary: string
  readonly outcome: EpisodeOutcome
  readonly retry_feedback: string
  readonly should_record_episode: boolean
  readonly open_questions: readonly string[]
  readonly heuristics: readonly StoredEpisodeHeuristic[]
  readonly tags: readonly string[]
  readonly started_at?: string
  readonly ended_at?: string
  readonly signals: StoredEpisodeSignals
}

export type EpisodeOutcome = ReflectionResult['outcome']

export type EpisodeSignals = {
  readonly messageCount: number
  readonly substantiveMessageCount: number
  readonly userMessageCount: number
  readonly assistantMessageCount: number
  readonly toolMessageCount: number
  readonly toolCallCount: number
  readonly writeSignal: boolean
  readonly editSignal: boolean
  readonly toolSignal: boolean
}

export type EpisodeRecorderConfig = {
  readonly minMessages: number
  readonly minSubstantiveMessages: number
  readonly requireActionSignal: boolean
}

export type EpisodeRecord = {
  readonly path: Path
  readonly sessionId: string
  readonly actorId: string
  readonly scope: Scope
  readonly name: string
  readonly summary: string
  readonly outcome: EpisodeOutcome
  readonly retryFeedback: string
  readonly shouldRecordEpisode: boolean
  readonly openQuestions: readonly string[]
  readonly heuristics: readonly Heuristic[]
  readonly tags: readonly string[]
  readonly created?: string
  readonly modified?: string
  readonly startedAt?: string
  readonly endedAt?: string
  readonly signals: EpisodeSignals
}

export type EpisodeRecordArgs = {
  readonly sessionId: string
  readonly messages: readonly Message[]
  readonly reflection: Pick<
    ReflectionResult,
    'outcome' | 'summary' | 'retryFeedback' | 'shouldRecordEpisode' | 'openQuestions' | 'heuristics'
  >
  readonly actorId?: string
  readonly scope?: Scope
  readonly startedAt?: Date | string
  readonly endedAt?: Date | string
  readonly tags?: readonly string[]
}

export type EpisodeGateReason = 'passed' | 'below_threshold' | 'no_action_signal' | 'model_declined'

export type EpisodeGateDecision = {
  readonly allowed: boolean
  readonly reason: EpisodeGateReason
  readonly signals: EpisodeSignals
}

export type RecordEpisodeResult = EpisodeGateDecision & {
  readonly recorded: boolean
  readonly disposition?: 'created' | 'updated'
  readonly path?: Path
  readonly episode?: EpisodeRecord
}

export type EpisodeListOptions = {
  readonly actorId?: string
  readonly scope?: Scope
  readonly outcome?: EpisodeOutcome
  readonly sessionId?: string
  readonly tags?: readonly string[]
  readonly from?: Date | string
  readonly to?: Date | string
  readonly limit?: number
}

export type EpisodeQueryOptions = EpisodeListOptions & {
  readonly query: string
}

export type EpisodeQueryHit = EpisodeRecord & {
  readonly score: number
}

export type EpisodeRecorderDeps = {
  readonly store: Store
  readonly logger?: Logger
  readonly defaultScope: Scope
  readonly defaultActorId: string
  readonly config?: Partial<EpisodeRecorderConfig>
}

export type EpisodeRecorder = {
  evaluate(args: Pick<EpisodeRecordArgs, 'messages' | 'reflection'>): EpisodeGateDecision
  record(args: EpisodeRecordArgs): Promise<RecordEpisodeResult>
  get(sessionId: string): Promise<EpisodeRecord | undefined>
  list(opts?: EpisodeListOptions): Promise<readonly EpisodeRecord[]>
  query(opts: EpisodeQueryOptions): Promise<readonly EpisodeQueryHit[]>
}

export const defaultEpisodeRecorderConfig = (
  overrides: Partial<EpisodeRecorderConfig> = {},
): EpisodeRecorderConfig => ({
  minMessages: overrides.minMessages ?? 8,
  minSubstantiveMessages: overrides.minSubstantiveMessages ?? 1,
  requireActionSignal: overrides.requireActionSignal ?? true,
})

export const episodePath = (sessionId: string): Path =>
  joinPath(EPISODES_PREFIX, ensureMarkdown(sessionId))

export const createEpisodeRecorder = (deps: EpisodeRecorderDeps): EpisodeRecorder => {
  const logger = deps.logger ?? noopLogger
  const config = defaultEpisodeRecorderConfig(deps.config)

  const evaluate = (
    args: Pick<EpisodeRecordArgs, 'messages' | 'reflection'>,
  ): EpisodeGateDecision => {
    const signals = detectEpisodeSignals(args.messages)
    if (
      signals.messageCount < config.minMessages ||
      signals.substantiveMessageCount < config.minSubstantiveMessages
    ) {
      return { allowed: false, reason: 'below_threshold', signals }
    }
    if (args.reflection.shouldRecordEpisode !== true) {
      return { allowed: false, reason: 'model_declined', signals }
    }
    if (config.requireActionSignal && !signals.writeSignal && !signals.editSignal) {
      return { allowed: false, reason: 'no_action_signal', signals }
    }
    return { allowed: true, reason: 'passed', signals }
  }

  const get = async (sessionId: string): Promise<EpisodeRecord | undefined> => {
    const path = episodePath(sessionId)
    try {
      return parseEpisodeFile(path, await deps.store.read(path))
    } catch {
      return undefined
    }
  }

  const list = async (opts: EpisodeListOptions = {}): Promise<readonly EpisodeRecord[]> => {
    const paths = await listEpisodePaths(deps.store)
    const loaded = await Promise.all(
      paths.map(async (path) => {
        try {
          return parseEpisodeFile(path, await deps.store.read(path))
        } catch (error) {
          logger.warn('memory: failed to read episode', {
            path,
            error: error instanceof Error ? error.message : String(error),
          })
          return undefined
        }
      }),
    )

    return loaded
      .filter((episode): episode is EpisodeRecord => episode !== undefined)
      .filter((episode) => matchesEpisodeFilters(episode, opts))
      .sort(compareEpisodesNewestFirst)
      .slice(0, normaliseLimit(opts.limit, DEFAULT_LIST_LIMIT))
  }

  const query = async (opts: EpisodeQueryOptions): Promise<readonly EpisodeQueryHit[]> => {
    const base = await list({
      ...(opts.actorId === undefined ? {} : { actorId: opts.actorId }),
      ...(opts.scope === undefined ? {} : { scope: opts.scope }),
      ...(opts.outcome === undefined ? {} : { outcome: opts.outcome }),
      ...(opts.sessionId === undefined ? {} : { sessionId: opts.sessionId }),
      ...(opts.tags === undefined ? {} : { tags: opts.tags }),
      ...(opts.from === undefined ? {} : { from: opts.from }),
      ...(opts.to === undefined ? {} : { to: opts.to }),
      limit: MAX_LIMIT,
    })

    const queryTokens = tokenise(opts.query, 32)
    const trimmedQuery = opts.query.trim().toLowerCase()
    if (queryTokens.length === 0 && trimmedQuery === '') return []

    return base
      .map((episode) => ({
        episode,
        score: scoreEpisodeQuery(episode, trimmedQuery, queryTokens),
      }))
      .filter((entry) => entry.score > 0)
      .sort((left, right) =>
        left.score === right.score
          ? compareEpisodesNewestFirst(left.episode, right.episode)
          : right.score - left.score,
      )
      .slice(0, normaliseLimit(opts.limit, DEFAULT_QUERY_LIMIT))
      .map(({ episode, score }) => ({ ...episode, score }))
  }

  const record = async (args: EpisodeRecordArgs): Promise<RecordEpisodeResult> => {
    const gate = evaluate(args)
    if (!gate.allowed) {
      return { ...gate, recorded: false }
    }

    const actorId = args.actorId ?? deps.defaultActorId
    const scope = args.scope ?? deps.defaultScope
    const path = episodePath(args.sessionId)
    const now = new Date().toISOString()
    const startedAt = normaliseOptionalIsoTimestamp(args.startedAt)
    const endedAt = normaliseOptionalIsoTimestamp(args.endedAt)
    const summary = collapseWhitespace(args.reflection.summary)
    const name = `Episode ${args.sessionId}`
    const tags = buildEpisodeTags({
      scope,
      outcome: args.reflection.outcome,
      heuristics: args.reflection.heuristics,
      signals: gate.signals,
      ...(args.tags === undefined ? {} : { tags: args.tags }),
    })

    let storedEpisode: EpisodeRecord | undefined
    let disposition: 'created' | 'updated' = 'created'

    await deps.store.batch({ reason: RECORD_REASON }, async (batch) => {
      let created = now
      if (await batch.exists(path)) {
        disposition = 'updated'
        const existingRaw = await batch.read(path)
        const existing = parseEpisodeFile(path, existingRaw)
        created = existing?.created ?? parseFrontmatter(existingRaw).frontmatter.created ?? created
      }

      const payload = buildStoredEpisodePayload({
        sessionId: args.sessionId,
        actorId,
        scope,
        summary,
        outcome: args.reflection.outcome,
        retryFeedback: collapseWhitespace(args.reflection.retryFeedback ?? ''),
        shouldRecordEpisode: true,
        openQuestions: dedupeText(args.reflection.openQuestions),
        heuristics: args.reflection.heuristics,
        tags,
        signals: gate.signals,
        ...(startedAt === undefined ? {} : { startedAt }),
        ...(endedAt === undefined ? {} : { endedAt }),
      })

      const content = buildEpisodeFile({
        name,
        summary,
        scope,
        created,
        modified: now,
        sessionId: args.sessionId,
        sessionDate: deriveSessionDate(endedAt ?? startedAt ?? created),
        observedOn: deriveSessionDate(endedAt ?? startedAt ?? created),
        tags,
        actorId,
        outcome: args.reflection.outcome,
        signals: gate.signals,
        payload,
        ...(startedAt === undefined ? {} : { startedAt }),
        ...(endedAt === undefined ? {} : { endedAt }),
      })

      await batch.write(path, content)
      storedEpisode = parseEpisodeFile(path, content)
    })

    return {
      ...gate,
      recorded: true,
      disposition,
      path,
      ...(storedEpisode === undefined ? {} : { episode: storedEpisode }),
    }
  }

  return { evaluate, record, get, list, query }
}

const buildStoredEpisodePayload = (args: {
  readonly sessionId: string
  readonly actorId: string
  readonly scope: Scope
  readonly summary: string
  readonly outcome: EpisodeOutcome
  readonly retryFeedback: string
  readonly shouldRecordEpisode: boolean
  readonly openQuestions: readonly string[]
  readonly heuristics: readonly Heuristic[]
  readonly tags: readonly string[]
  readonly startedAt?: string
  readonly endedAt?: string
  readonly signals: EpisodeSignals
}): StoredEpisodePayload => ({
  session_id: args.sessionId,
  actor_id: args.actorId,
  scope: args.scope,
  summary: args.summary,
  outcome: args.outcome,
  retry_feedback: args.retryFeedback,
  should_record_episode: args.shouldRecordEpisode,
  open_questions: args.openQuestions,
  heuristics: args.heuristics.map((heuristic) => ({
    rule: heuristic.rule,
    context: heuristic.context,
    confidence: heuristic.confidence,
    category: heuristic.category,
    scope: heuristic.scope,
    anti_pattern: heuristic.antiPattern,
  })),
  tags: args.tags,
  ...(args.startedAt === undefined ? {} : { started_at: args.startedAt }),
  ...(args.endedAt === undefined ? {} : { ended_at: args.endedAt }),
  signals: {
    message_count: args.signals.messageCount,
    substantive_message_count: args.signals.substantiveMessageCount,
    user_message_count: args.signals.userMessageCount,
    assistant_message_count: args.signals.assistantMessageCount,
    tool_message_count: args.signals.toolMessageCount,
    tool_call_count: args.signals.toolCallCount,
    write_signal: args.signals.writeSignal,
    edit_signal: args.signals.editSignal,
    tool_signal: args.signals.toolSignal,
  },
})

const buildEpisodeFile = (args: {
  readonly name: string
  readonly summary: string
  readonly scope: Scope
  readonly created: string
  readonly modified: string
  readonly sessionId: string
  readonly sessionDate: string
  readonly observedOn: string
  readonly tags: readonly string[]
  readonly actorId: string
  readonly outcome: EpisodeOutcome
  readonly startedAt?: string
  readonly endedAt?: string
  readonly signals: EpisodeSignals
  readonly payload: StoredEpisodePayload
}): string => {
  const frontmatter = buildFrontmatter({
    name: args.name,
    description: args.summary,
    type: 'episode',
    scope: args.scope,
    created: args.created,
    modified: args.modified,
    source: 'episode',
    session_id: args.sessionId,
    session_date: args.sessionDate,
    observed_on: args.observedOn,
    tags: args.tags,
    extra: {
      actor_id: args.actorId,
      outcome: args.outcome,
      should_record_episode: 'true',
      message_count: String(args.signals.messageCount),
      substantive_message_count: String(args.signals.substantiveMessageCount),
      user_message_count: String(args.signals.userMessageCount),
      assistant_message_count: String(args.signals.assistantMessageCount),
      tool_message_count: String(args.signals.toolMessageCount),
      tool_call_count: String(args.signals.toolCallCount),
      write_signal: boolString(args.signals.writeSignal),
      edit_signal: boolString(args.signals.editSignal),
      tool_signal: boolString(args.signals.toolSignal),
      ...(args.startedAt === undefined ? {} : { started_at: args.startedAt }),
      ...(args.endedAt === undefined ? {} : { ended_at: args.endedAt }),
      heuristic_count: String(args.payload.heuristics.length),
      open_question_count: String(args.payload.open_questions.length),
    },
  })

  const body: string[] = [
    `# ${args.name}`,
    '',
    '## Summary',
    '',
    args.summary || '_no summary_',
    '',
    '## Outcome',
    '',
    args.outcome,
    '',
  ]

  if (args.payload.retry_feedback !== '') {
    body.push('## Retry feedback', '', args.payload.retry_feedback, '')
  }

  if (args.payload.open_questions.length > 0) {
    body.push('## Open questions', '')
    for (const question of args.payload.open_questions) {
      body.push(`- ${question}`)
    }
    body.push('')
  }

  if (args.payload.heuristics.length > 0) {
    body.push('## Heuristics', '')
    for (const heuristic of args.payload.heuristics) {
      const marker = heuristic.anti_pattern ? '[anti-pattern]' : '[pattern]'
      body.push(
        `- ${marker} ${heuristic.rule} _(context: ${heuristic.context || 'the same type of work'}; confidence: ${heuristic.confidence}; category: ${heuristic.category}; scope: ${heuristic.scope})_`,
      )
    }
    body.push('')
  }

  body.push(
    '## Signals',
    '',
    `- write_signal: ${boolString(args.signals.writeSignal)}`,
    `- edit_signal: ${boolString(args.signals.editSignal)}`,
    `- tool_signal: ${boolString(args.signals.toolSignal)}`,
    `- message_count: ${args.signals.messageCount}`,
    `- substantive_message_count: ${args.signals.substantiveMessageCount}`,
    `- tool_call_count: ${args.signals.toolCallCount}`,
    '',
    '## Episode data',
    '',
    '```json',
    `${JSON.stringify(args.payload, null, 2)}`,
    '```',
    '',
  )

  return `${frontmatter}\n${body.join('\n')}`
}

const parseEpisodeFile = (path: Path, raw: string): EpisodeRecord | undefined => {
  const { frontmatter, body } = parseFrontmatter(raw)
  const payload = parseEpisodePayload(body)
  const sessionId =
    frontmatter.session_id ?? payload?.session_id ?? path.split('/').pop()?.replace(/\.md$/i, '')
  if (sessionId === undefined || sessionId.trim() === '') return undefined

  const actorId = frontmatter.extra.actor_id ?? payload?.actor_id ?? ''
  const scope = isScope(frontmatter.scope) ? frontmatter.scope : (payload?.scope ?? 'project')
  const summary = collapseWhitespace(
    frontmatter.description ?? payload?.summary ?? firstMeaningfulBodyText(body),
  )
  const outcome = normaliseEpisodeOutcome(frontmatter.extra.outcome ?? payload?.outcome)
  const tags = dedupeStrings(frontmatter.tags ?? payload?.tags ?? [])
  const retryFeedback = collapseWhitespace(payload?.retry_feedback ?? '')
  const heuristics = (payload?.heuristics ?? []).map((heuristic) => ({
    rule: heuristic.rule,
    context: heuristic.context,
    confidence: heuristic.confidence,
    category: heuristic.category,
    scope: heuristic.scope,
    antiPattern: heuristic.anti_pattern,
  }))
  const signals = parseEpisodeSignals(frontmatter.extra, payload?.signals)
  const startedAt = normaliseOptionalIsoTimestamp(
    frontmatter.extra.started_at ?? payload?.started_at,
  )
  const endedAt = normaliseOptionalIsoTimestamp(frontmatter.extra.ended_at ?? payload?.ended_at)

  return {
    path,
    sessionId,
    actorId,
    scope,
    name: frontmatter.name ?? `Episode ${sessionId}`,
    summary,
    outcome,
    retryFeedback,
    shouldRecordEpisode:
      parseBoolean(frontmatter.extra.should_record_episode) ??
      payload?.should_record_episode ??
      true,
    openQuestions: payload?.open_questions ?? [],
    heuristics,
    tags,
    ...(frontmatter.created === undefined ? {} : { created: frontmatter.created }),
    ...(frontmatter.modified === undefined ? {} : { modified: frontmatter.modified }),
    ...(startedAt === undefined ? {} : { startedAt }),
    ...(endedAt === undefined ? {} : { endedAt }),
    signals,
  }
}

const parseEpisodePayload = (body: string): StoredEpisodePayload | undefined => {
  const match = body.match(/## Episode data\s+```json\s*([\s\S]*?)```/m)
  const json = match?.[1]?.trim()
  if (json === undefined || json === '') return undefined

  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch {
    return undefined
  }
  if (!isRecord(parsed)) return undefined

  const sessionId = firstString(parsed.session_id, parsed.sessionId)
  const actorId = firstString(parsed.actor_id, parsed.actorId)
  const scope = normaliseScope(firstString(parsed.scope))
  const summary = collapseWhitespace(firstString(parsed.summary) ?? '')
  const outcome = normaliseEpisodeOutcome(firstString(parsed.outcome))
  const retryFeedback = collapseWhitespace(
    firstString(parsed.retry_feedback, parsed.retryFeedback) ?? '',
  )
  const shouldRecordEpisode =
    parseBoolean(parsed.should_record_episode) ?? parseBoolean(parsed.shouldRecordEpisode) ?? false
  const openQuestions = toStringArray(parsed.open_questions, parsed.openQuestions)
  const heuristics = toHeuristicArray(parsed.heuristics)
  const tags = dedupeStrings(toStringArray(parsed.tags))
  const startedAt = normaliseOptionalIsoTimestamp(firstString(parsed.started_at, parsed.startedAt))
  const endedAt = normaliseOptionalIsoTimestamp(firstString(parsed.ended_at, parsed.endedAt))
  const signals = toStoredEpisodeSignals(parsed.signals)

  if (sessionId === undefined || actorId === undefined || signals === undefined) return undefined

  return {
    session_id: sessionId,
    actor_id: actorId,
    scope,
    summary,
    outcome,
    retry_feedback: retryFeedback,
    should_record_episode: shouldRecordEpisode,
    open_questions: openQuestions,
    heuristics,
    tags,
    ...(startedAt === undefined ? {} : { started_at: startedAt }),
    ...(endedAt === undefined ? {} : { ended_at: endedAt }),
    signals,
  }
}

const parseEpisodeSignals = (
  extra: Record<string, string>,
  payload: StoredEpisodeSignals | undefined,
): EpisodeSignals => ({
  messageCount: payload?.message_count ?? parseInteger(extra.message_count),
  substantiveMessageCount:
    payload?.substantive_message_count ?? parseInteger(extra.substantive_message_count),
  userMessageCount: payload?.user_message_count ?? parseInteger(extra.user_message_count),
  assistantMessageCount:
    payload?.assistant_message_count ?? parseInteger(extra.assistant_message_count),
  toolMessageCount: payload?.tool_message_count ?? parseInteger(extra.tool_message_count),
  toolCallCount: payload?.tool_call_count ?? parseInteger(extra.tool_call_count),
  writeSignal: payload?.write_signal ?? parseBoolean(extra.write_signal) ?? false,
  editSignal: payload?.edit_signal ?? parseBoolean(extra.edit_signal) ?? false,
  toolSignal: payload?.tool_signal ?? parseBoolean(extra.tool_signal) ?? false,
})

const toStoredEpisodeSignals = (value: unknown): StoredEpisodeSignals | undefined => {
  if (!isRecord(value)) return undefined
  return {
    message_count: normaliseCount(value.message_count, value.messageCount),
    substantive_message_count: normaliseCount(
      value.substantive_message_count,
      value.substantiveMessageCount,
    ),
    user_message_count: normaliseCount(value.user_message_count, value.userMessageCount),
    assistant_message_count: normaliseCount(
      value.assistant_message_count,
      value.assistantMessageCount,
    ),
    tool_message_count: normaliseCount(value.tool_message_count, value.toolMessageCount),
    tool_call_count: normaliseCount(value.tool_call_count, value.toolCallCount),
    write_signal: parseBoolean(value.write_signal) ?? parseBoolean(value.writeSignal) ?? false,
    edit_signal: parseBoolean(value.edit_signal) ?? parseBoolean(value.editSignal) ?? false,
    tool_signal: parseBoolean(value.tool_signal) ?? parseBoolean(value.toolSignal) ?? false,
  }
}

const toHeuristicArray = (value: unknown): readonly StoredEpisodeHeuristic[] => {
  if (!Array.isArray(value)) return []
  const out: StoredEpisodeHeuristic[] = []
  for (const item of value) {
    if (!isRecord(item)) continue
    out.push({
      rule: collapseWhitespace(firstString(item.rule) ?? ''),
      context: collapseWhitespace(firstString(item.context) ?? ''),
      confidence: normaliseConfidence(firstString(item.confidence)),
      category: collapseWhitespace(firstString(item.category) ?? ''),
      scope: normaliseScope(firstString(item.scope)),
      anti_pattern: parseBoolean(item.anti_pattern) ?? parseBoolean(item.antiPattern) ?? false,
    })
  }
  return out
}

const listEpisodePaths = async (store: Store): Promise<readonly Path[]> => {
  const entries = await store.list(EPISODES_PREFIX, { recursive: true })
  return entries
    .filter((entry) => !entry.isDir && entry.path.endsWith('.md'))
    .map((entry) => entry.path)
}

const matchesEpisodeFilters = (episode: EpisodeRecord, opts: EpisodeListOptions): boolean => {
  if (opts.actorId !== undefined && episode.actorId !== opts.actorId) return false
  if (opts.scope !== undefined && episode.scope !== opts.scope) return false
  if (opts.outcome !== undefined && episode.outcome !== opts.outcome) return false
  if (opts.sessionId !== undefined && episode.sessionId !== opts.sessionId) return false

  if (opts.tags !== undefined && opts.tags.length > 0) {
    const required = new Set(opts.tags.map(normaliseTag).filter((tag) => tag !== ''))
    if (required.size > 0) {
      const actual = new Set(episode.tags.map(normaliseTag))
      for (const tag of required) {
        if (!actual.has(tag)) return false
      }
    }
  }

  const timestamp = episodeTimestamp(episode)
  const from = toEpochMs(opts.from)
  const to = toEpochMs(opts.to)
  if (from !== undefined && timestamp < from) return false
  if (to !== undefined && timestamp > to) return false
  return true
}

const compareEpisodesNewestFirst = (left: EpisodeRecord, right: EpisodeRecord): number => {
  const delta = episodeTimestamp(right) - episodeTimestamp(left)
  if (delta !== 0) return delta
  return left.path.localeCompare(right.path)
}

const scoreEpisodeQuery = (
  episode: EpisodeRecord,
  trimmedQuery: string,
  queryTokens: readonly string[],
): number => {
  const summaryTokens = tokenise(episode.summary, 24)
  const tagTokens = tokenise(episode.tags.join(' '), 24)
  const heuristicTokens = tokenise(
    episode.heuristics
      .map((heuristic) =>
        [
          heuristic.rule,
          heuristic.context,
          heuristic.category,
          heuristic.confidence,
          heuristic.scope,
          heuristic.antiPattern ? 'anti-pattern' : 'pattern',
        ].join(' '),
      )
      .join(' '),
    48,
  )
  const supportTokens = tokenise(
    [
      episode.sessionId,
      episode.actorId,
      episode.outcome,
      episode.retryFeedback,
      ...episode.openQuestions,
    ].join(' '),
    48,
  )

  let score = 0
  score += countOverlap(queryTokens, summaryTokens) * 4
  score += countOverlap(queryTokens, tagTokens) * 3
  score += countOverlap(queryTokens, heuristicTokens) * 2
  score += countOverlap(queryTokens, supportTokens)

  if (trimmedQuery !== '') {
    const searchable = buildEpisodeSearchText(episode)
    if (searchable.includes(trimmedQuery)) score += 2
    if (episode.sessionId.toLowerCase() === trimmedQuery) score += 8
    if (episode.outcome.toLowerCase() === trimmedQuery) score += 2
  }

  return score
}

const buildEpisodeSearchText = (episode: EpisodeRecord): string =>
  [
    episode.path,
    episode.sessionId,
    episode.actorId,
    episode.summary,
    episode.outcome,
    episode.retryFeedback,
    ...episode.openQuestions,
    ...episode.tags,
    ...episode.heuristics.flatMap((heuristic) => [
      heuristic.rule,
      heuristic.context,
      heuristic.category,
      heuristic.confidence,
      heuristic.scope,
      heuristic.antiPattern ? 'anti-pattern' : 'pattern',
    ]),
  ]
    .join('\n')
    .toLowerCase()

const detectEpisodeSignals = (messages: readonly Message[]): EpisodeSignals => {
  let substantiveMessageCount = 0
  let userMessageCount = 0
  let assistantMessageCount = 0
  let toolMessageCount = 0
  let toolCallCount = 0
  let writeSignal = false
  let editSignal = false
  let toolSignal = false

  for (const message of messages) {
    if (message.role !== 'system') substantiveMessageCount += 1
    if (message.role === 'user') userMessageCount += 1
    if (message.role === 'assistant') assistantMessageCount += 1
    if (message.role === 'tool') toolMessageCount += 1

    const toolCallNames = extractToolCallNames(message)
    toolCallCount += toolCallNames.length
    if (message.role === 'tool' || toolCallNames.length > 0) toolSignal = true

    if (message.role === 'assistant') {
      for (const toolName of toolCallNames) {
        const normalised = toolName.trim().toLowerCase()
        if (WRITE_SIGNAL_TOOL_NAMES.has(normalised)) writeSignal = true
        if (EDIT_SIGNAL_TOOL_NAMES.has(normalised)) editSignal = true
      }
    }
  }

  return {
    messageCount: messages.length,
    substantiveMessageCount,
    userMessageCount,
    assistantMessageCount,
    toolMessageCount,
    toolCallCount,
    writeSignal,
    editSignal,
    toolSignal,
  }
}

const extractToolCallNames = (message: Message): string[] => {
  const names = [...(message.toolCalls ?? []).map((toolCall) => toolCall.name)]
  for (const block of message.blocks ?? []) {
    if (block.toolUse?.name !== undefined) names.push(block.toolUse.name)
  }
  return names
}

const buildEpisodeTags = (args: {
  readonly scope: Scope
  readonly outcome: EpisodeOutcome
  readonly tags?: readonly string[]
  readonly heuristics: readonly Heuristic[]
  readonly signals: EpisodeSignals
}): readonly string[] =>
  dedupeStrings([
    'episode',
    `scope-${slugify(args.scope)}`,
    `outcome-${slugify(args.outcome)}`,
    'significant',
    ...(args.signals.writeSignal ? ['signal-write'] : []),
    ...(args.signals.editSignal ? ['signal-edit'] : []),
    ...(args.signals.toolSignal ? ['signal-tool'] : []),
    ...args.heuristics.flatMap((heuristic) => [
      `heuristic-${slugify(heuristic.category)}`,
      `confidence-${slugify(heuristic.confidence)}`,
      ...(heuristic.antiPattern ? ['anti-pattern'] : ['pattern']),
    ]),
    ...(args.tags ?? []),
  ])

const tokenise = (text: string, limit: number): readonly string[] => {
  const matches = text.toLowerCase().match(TOKEN_PATTERN) ?? []
  const out: string[] = []
  const seen = new Set<string>()
  for (const match of matches) {
    const token = stemToken(match)
    if (token.length < 2) continue
    if (seen.has(token)) continue
    seen.add(token)
    out.push(token)
    if (out.length >= limit) break
  }
  return out
}

const stemToken = (token: string): string => {
  if (token.length > 5 && token.endsWith('ies')) return `${token.slice(0, -3)}y`
  if (token.length > 5 && token.endsWith('es')) return token.slice(0, -2)
  if (token.length > 4 && token.endsWith('s') && !token.endsWith('ss')) return token.slice(0, -1)
  return token
}

const countOverlap = (left: readonly string[], right: readonly string[]): number => {
  if (left.length === 0 || right.length === 0) return 0
  const rightSet = new Set(right)
  return left.reduce((count, token) => count + (rightSet.has(token) ? 1 : 0), 0)
}

const dedupeStrings = (values: readonly string[]): string[] => {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of values) {
    const tag = normaliseTag(raw)
    if (tag === '' || seen.has(tag)) continue
    seen.add(tag)
    out.push(tag)
  }
  return out
}

const dedupeText = (values: readonly string[]): string[] => {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of values) {
    const text = collapseWhitespace(raw)
    if (text === '') continue
    const key = text.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(text)
  }
  return out
}

const normaliseTag = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._:-]+/g, '-')
    .replace(/^-+|-+$/g, '')

const slugify = (value: string): string => {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return slug === '' ? 'episode' : slug.slice(0, 64).replace(/-+$/g, '')
}

const collapseWhitespace = (value: string): string => value.replace(/\s+/g, ' ').trim()

const firstMeaningfulBodyText = (body: string): string => {
  const stripped = body.replace(/## Episode data\s+```json[\s\S]*?```/m, '')
  for (const raw of stripped.split('\n')) {
    const line = raw.trim()
    if (line === '' || line.startsWith('#') || line.startsWith('- ')) continue
    return line
  }
  return ''
}

const boolString = (value: boolean): string => (value ? 'true' : 'false')

const parseBoolean = (value: unknown): boolean | undefined => {
  if (typeof value === 'boolean') return value
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim().toLowerCase()
  if (trimmed === 'true') return true
  if (trimmed === 'false') return false
  return undefined
}

const parseInteger = (value: string | undefined): number => {
  if (value === undefined) return 0
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 0
}

const normaliseCount = (...values: readonly unknown[]): number => {
  for (const value of values) {
    if (typeof value === 'number') {
      if (Number.isFinite(value) && value >= 0) return Math.floor(value)
      continue
    }
    if (typeof value === 'string') {
      const parsed = Number.parseInt(value, 10)
      if (Number.isFinite(parsed) && parsed >= 0) return parsed
    }
  }
  return 0
}

const firstString = (...values: readonly unknown[]): string | undefined => {
  for (const value of values) {
    if (typeof value !== 'string') continue
    const trimmed = value.trim()
    if (trimmed !== '') return trimmed
  }
  return undefined
}

const toStringArray = (...values: readonly unknown[]): string[] => {
  for (const value of values) {
    if (!Array.isArray(value)) continue
    return value
      .filter((item): item is string => typeof item === 'string')
      .map(collapseWhitespace)
      .filter((item) => item !== '')
  }
  return []
}

const normaliseEpisodeOutcome = (value: unknown): EpisodeOutcome => {
  if (value === 'success' || value === 'partial' || value === 'failure' || value === 'unknown') {
    return value
  }
  return 'unknown'
}

const isScope = (value: string | undefined): value is Scope =>
  value === 'global' || value === 'project' || value === 'agent'

const normaliseScope = (value: string | undefined): Scope => (isScope(value) ? value : 'project')

const normaliseConfidence = (value: string | undefined): Heuristic['confidence'] =>
  value === 'high' || value === 'medium' || value === 'low' ? value : 'low'

const normaliseOptionalIsoTimestamp = (value: Date | string | undefined): string | undefined => {
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? undefined : value.toISOString()
  }
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  if (trimmed === '') return undefined
  const parsed = Date.parse(trimmed)
  return Number.isFinite(parsed) ? new Date(parsed).toISOString() : undefined
}

const deriveSessionDate = (value: string): string => value.slice(0, 10)

const normaliseLimit = (value: number | undefined, fallback: number): number => {
  if (value === undefined) return fallback
  if (!Number.isFinite(value) || value <= 0) return fallback
  return Math.min(Math.floor(value), MAX_LIMIT)
}

const episodeTimestamp = (episode: EpisodeRecord): number =>
  toEpochMs(episode.endedAt) ??
  toEpochMs(episode.startedAt) ??
  toEpochMs(episode.modified) ??
  toEpochMs(episode.created) ??
  0

const toEpochMs = (value: Date | string | undefined): number | undefined => {
  if (value instanceof Date) {
    const time = value.getTime()
    return Number.isFinite(time) ? time : undefined
  }
  if (typeof value !== 'string') return undefined
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null
