// SPDX-License-Identifier: Apache-2.0

/**
 * Recall stage. Given a query string, use the injected SearchIndex +
 * Embedder to find relevant memory notes, then hydrate them from the
 * Store. Falls back to listing topics under the scope prefix + embedding
 * cosine scoring when no SearchIndex is configured so development and
 * test harnesses can exercise recall without a full retrieval pipeline.
 */

import type { Embedder, Logger, Provider } from '../llm/index.js'
import { ErrNotFound } from '../store/errors.js'
import type { Store } from '../store/index.js'
import { type Path, joinPath, lastSegment, validatePath } from '../store/path.js'
import { parseFrontmatter } from './frontmatter.js'
import {
  MEMORY_AGENT_PREFIX,
  MEMORY_GLOBAL_PREFIX,
  MEMORY_PROJECTS_PREFIX,
  scopePrefix,
} from './paths.js'
import { RECALL_SELECTOR_SYSTEM_PROMPT } from './prompts.js'
import type {
  MemoryNote,
  RecallHit,
  RecallOpts,
  RecallSelectorMode,
  Scope,
  SearchHit,
  SearchIndex,
} from './types.js'

const DEFAULT_K = 5
const MAX_CANDIDATE_K = 24
const CANDIDATE_FETCH_MULTIPLIER = 4
const AGGREGATE_FETCH_MULTIPLIER = 6
const RECALL_SELECTOR_MAX_TOKENS = 768
const RECALL_SELECTOR_TEMPERATURE = 0
const RECALL_SELECTOR_CANDIDATE_LIMIT = 12
const RECALL_PRIMARY_SCOPE_RESERVE = 2
const RECALL_WIKILINK_FOLLOW_UP_LIMIT = 2
const RECALL_WIKILINK_READ_BATCH = 32
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
const WIKILINK_PATTERN = /\[\[([^\]]+)\]\]/g
const TEMPORAL_SORT_PATTERNS = [
  /\b(?:today|yesterday|tomorrow|tonight)\b/i,
  /\blast\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i,
  /\b\d+\s+(?:day|days|week|weeks|month|months|year|years)\s+ago\b/i,
  /\b(?:oldest|earlier|before|after|between|since|compared|difference|timeline|history|trend)\b/i,
  /\bfirst\b/i,
  /\b(?:19|20)\d{2}\b/,
  /\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b/i,
  /\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i,
] as const
const RECENT_QUERY_PATTERNS = [
  /\b(?:latest|most recent|newest|updated|recent|recently|current|currently)\b/i,
  /\blast\s+time\b/i,
] as const
const AGGREGATE_QUERY_PATTERNS = [
  /\b(?:all|across|between|compare|comparison|different|history|timeline|pattern|patterns|period|periods|times|episodes|instances|list|summary|summarise|recap|types|kinds|how much|total|spent|expense|expenses|cost|costs|breaks|appointments|meetings|workshops)\b/i,
] as const
const CONCRETE_QUERY_PATTERNS = [
  /^(?:when|where|who|which)\b/i,
  /\b(?:how much|how many|spent|cost|costs|paid|before|after|between|compare|compared|difference|differences|happened|meeting|workshop|appointment|doctor|bill|expense|expenses|break)\b/i,
] as const
const GENERIC_ADVICE_PATTERNS = [
  /\b(?:tip|tips|advice|guidance|guideline|guidelines|best practice|best practices|principle|principles|always|never|generally|usually|remember to|consider|try to)\b/i,
] as const
const CONCRETE_CONTENT_PATTERNS = [
  /(?:£|\$|€)\s?\d+/i,
  /\b\d+(?:\.\d+)?\s?(?:hour|hours|day|days|week|weeks|month|months|year|years|km|mi|mile|miles|min|mins|minute|minutes)\b/i,
  /\b(?:19|20)\d{2}\b/i,
  /\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b/i,
  /\b(?:meeting|met|workshop|call|doctor|appointment|spent|paid|bought|travelled|visited|break|holiday|trip)\b/i,
] as const
const TOKEN_PATTERN = /[a-z0-9]+/gi
const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'been',
  'but',
  'by',
  'did',
  'do',
  'does',
  'for',
  'from',
  'had',
  'has',
  'have',
  'how',
  'i',
  'if',
  'in',
  'into',
  'is',
  'it',
  'its',
  'me',
  'my',
  'of',
  'on',
  'or',
  'our',
  'that',
  'the',
  'their',
  'them',
  'then',
  'there',
  'these',
  'they',
  'this',
  'those',
  'to',
  'was',
  'we',
  'were',
  'what',
  'when',
  'where',
  'which',
  'who',
  'with',
  'you',
  'your',
])

type RecallQuerySignals = {
  readonly timeline: boolean
  readonly recent: boolean
  readonly temporal: boolean
  readonly aggregate: boolean
  readonly concrete: boolean
}

type RankedRecallHit = {
  readonly hit: RecallHit
  readonly baseAdjustedScore: number
  readonly originalScore: number
  readonly timestamp: number
  readonly recencyScore: number
  readonly dateBucket?: string
  readonly signatureTokens: readonly string[]
  readonly topicTokens: readonly string[]
}

type RecallSearchScope = {
  readonly scope: Scope
  readonly actorId: string
  readonly primary: boolean
}

type ScopedSearchHit = SearchHit & {
  readonly sourceScope: Scope
  readonly primary: boolean
  readonly rank: number
}

export type RecallDeps = {
  readonly store: Store
  readonly provider?: Provider
  readonly embedder?: Embedder
  readonly searchIndex?: SearchIndex
  readonly logger: Logger
  readonly defaultScope: Scope
  readonly defaultActorId: string
}

export const createRecall = (deps: RecallDeps) => {
  return async (opts: RecallOpts): Promise<readonly RecallHit[]> => {
    const signal = opts.signal
    signal?.throwIfAborted()
    const k = opts.k ?? DEFAULT_K
    if (k <= 0) return []
    const scope = opts.scope ?? deps.defaultScope
    const actorId = opts.actorId ?? deps.defaultActorId
    const allowedScopes = resolveRecallAllowedScopes(scope, opts.fallbackScopes)
    const querySignals = classifyRecallQuery(opts.query)
    const excludedPaths = buildExcludedPathSet(opts)
    const candidateK = resolveCandidateK(k, querySignals, excludedPaths.size)

    const embedding = deps.embedder
      ? await embedSafe(deps.embedder, opts.query, deps.logger, signal)
      : undefined
    signal?.throwIfAborted()

    const hits = await searchRecallCandidates(
      deps,
      opts.query,
      embedding,
      allowedScopes,
      actorId,
      k,
      candidateK,
      excludedPaths,
      signal,
    )
    signal?.throwIfAborted()
    const filteredHits = filterExcludedSearchHits(hits, excludedPaths)
    const hydrated = await hydrateRecallHits(
      deps.store,
      filteredHits.slice(0, candidateK),
      scope,
      deps.logger,
      signal,
    )
    signal?.throwIfAborted()
    const selected = await maybeSelectRecallHits(hydrated, {
      query: opts.query,
      k,
      logger: deps.logger,
      mode: opts.selector ?? 'off',
      ...(deps.provider !== undefined ? { provider: deps.provider } : {}),
      ...(signal !== undefined ? { signal } : {}),
    })
    signal?.throwIfAborted()
    const expanded = await followWikilinkHits(selected, {
      store: deps.store,
      allowedScopes,
      actorId,
      excludedPaths,
      logger: deps.logger,
      ...(signal !== undefined ? { signal } : {}),
    })
    signal?.throwIfAborted()
    return rerankRecallHits(expanded, opts.query, k, querySignals, scope)
  }
}

export const isTimeSensitiveMemoryQuery = (query: string): boolean => {
  const trimmed = query.trim()
  if (trimmed === '') return false
  return TEMPORAL_SORT_PATTERNS.some((pattern) => pattern.test(trimmed))
}

export const isRecentMemoryQuery = (query: string): boolean => {
  const trimmed = query.trim()
  if (trimmed === '') return false
  return RECENT_QUERY_PATTERNS.some((pattern) => pattern.test(trimmed))
}

export const sortRecallHitsChronologically = (hits: readonly RecallHit[]): RecallHit[] =>
  [...hits]
    .map((hit, index) => ({
      hit,
      index,
      timestamp: noteTimestamp(hit.note),
    }))
    .sort((left, right) => {
      const leftMissing = !Number.isFinite(left.timestamp)
      const rightMissing = !Number.isFinite(right.timestamp)
      if (leftMissing && rightMissing) return left.index - right.index
      if (leftMissing) return 1
      if (rightMissing) return -1
      if (left.timestamp !== right.timestamp) return left.timestamp - right.timestamp
      return left.index - right.index
    })
    .map(({ hit }) => hit)

export type RecallHitSortMode = 'relevance' | 'recency' | 'relevance_then_recency'

export const mergeRecallHits = (
  hits: readonly RecallHit[],
  opts: {
    readonly query?: string
    readonly sort?: RecallHitSortMode
  } = {},
): RecallHit[] => {
  const best = new Map<string, RecallHit>()
  for (const hit of hits) {
    const current = best.get(hit.path)
    if (current === undefined || hit.score > current.score) {
      best.set(hit.path, hit)
    }
  }

  const merged = [...best.values()]
  if (
    opts.sort === 'recency' ||
    opts.sort === 'relevance_then_recency' ||
    isRecentMemoryQuery(opts.query ?? '')
  ) {
    return sortRecallHitsByRecency(merged)
  }
  if (isTimeSensitiveMemoryQuery(opts.query ?? '')) {
    return sortRecallHitsChronologically(merged)
  }
  return merged.sort((left, right) => right.score - left.score)
}

const sortRecallHitsByRecency = (hits: readonly RecallHit[]): RecallHit[] =>
  [...hits]
    .map((hit, index) => ({
      hit,
      index,
      timestamp: noteTimestamp(hit.note),
    }))
    .sort((left, right) => {
      const leftMissing = !Number.isFinite(left.timestamp)
      const rightMissing = !Number.isFinite(right.timestamp)
      if (leftMissing && rightMissing) return left.index - right.index
      if (leftMissing) return 1
      if (rightMissing) return -1
      if (left.timestamp !== right.timestamp) return right.timestamp - left.timestamp
      return left.index - right.index
    })
    .map(({ hit }) => hit)

const rerankRecallHits = (
  hits: readonly RecallHit[],
  query: string,
  k: number,
  querySignals: RecallQuerySignals,
  primaryScope: Scope,
): RecallHit[] => {
  if (hits.length <= 1) {
    return orderRecallHitsForQuery(hits, querySignals)
  }

  const queryTokens = tokenise(query)
  const maxScore = Math.max(0, ...hits.map((hit) => hit.score))
  const scoreScale = Math.min(Math.max(maxScore, 1), 4)
  const timestamps = hits
    .map((hit) => noteTimestamp(hit.note))
    .filter((timestamp) => Number.isFinite(timestamp))
  const oldestTimestamp = timestamps.length > 0 ? Math.min(...timestamps) : Number.NaN
  const newestTimestamp = timestamps.length > 0 ? Math.max(...timestamps) : Number.NaN

  const ranked = hits.map((hit) =>
    analyseRecallHit(
      hit,
      queryTokens,
      querySignals,
      scoreScale,
      maxScore,
      oldestTimestamp,
      newestTimestamp,
      primaryScope,
    ),
  )

  const selected: RecallHit[] = []
  const chosen: RankedRecallHit[] = []
  const remaining = [...ranked]

  while (selected.length < k && remaining.length > 0) {
    let nextIndex = 0
    let nextScore = Number.NEGATIVE_INFINITY

    for (let index = 0; index < remaining.length; index++) {
      const candidate = remaining[index]
      if (candidate === undefined) continue
      const selectionScore = scoreRecallCandidate(candidate, chosen, querySignals)
      if (selectionScore > nextScore) {
        nextIndex = index
        nextScore = selectionScore
      }
    }

    const [next] = remaining.splice(nextIndex, 1)
    if (next === undefined) continue
    const rerankedHit: RecallHit = {
      ...next.hit,
      score: nextScore,
    }
    selected.push(rerankedHit)
    chosen.push(next)
  }

  return orderRecallHitsForQuery(selected, querySignals)
}

const orderRecallHitsForQuery = (
  hits: readonly RecallHit[],
  querySignals: RecallQuerySignals,
): RecallHit[] => {
  if (querySignals.recent) return sortRecallHitsByRecency(hits)
  if (querySignals.timeline) return sortRecallHitsChronologically(hits)
  return [...hits]
}

const analyseRecallHit = (
  hit: RecallHit,
  queryTokens: readonly string[],
  querySignals: RecallQuerySignals,
  scoreScale: number,
  maxScore: number,
  oldestTimestamp: number,
  newestTimestamp: number,
  primaryScope: Scope,
): RankedRecallHit => {
  const searchableText = buildNoteSearchText(hit.note)
  const topicText = buildTopicText(hit.note)
  const searchableTokens = tokenise(searchableText, 48)
  const topicTokens = tokenise(topicText, 24)
  const queryMatches = countOverlap(queryTokens, searchableTokens)
  const titleMatches = countOverlap(queryTokens, topicTokens)
  const queryCoverage = queryTokens.length === 0 ? 0 : queryMatches / queryTokens.length
  const titleCoverage = queryTokens.length === 0 ? 0 : titleMatches / queryTokens.length
  const concreteScore = concreteNoteScore(hit.note)
  const genericPenalty = genericAdvicePenalty(hit.note)
  const timestamp = noteTimestamp(hit.note)
  const recencyScore =
    querySignals.recent && Number.isFinite(timestamp)
      ? normaliseRecency(timestamp, oldestTimestamp, newestTimestamp)
      : 0
  let bonus = queryCoverage * 1.4 + titleCoverage * 0.8 + Math.min(queryMatches, 3) * 0.15
  if (querySignals.temporal && Number.isFinite(timestamp)) bonus += 0.8
  if (querySignals.recent) bonus += recencyScore * 0.8
  if (querySignals.concrete) bonus += concreteScore * 0.55 - genericPenalty * 0.75
  else bonus -= genericPenalty * 0.1
  if (hit.note.scope === primaryScope) bonus += 0.25
  else bonus -= 0.1

  const baseAdjustedScore =
    hit.score + normaliseScore(hit.score, maxScore) * 0.35 + bonus * scoreScale

  return {
    hit,
    baseAdjustedScore,
    originalScore: hit.score,
    timestamp,
    recencyScore,
    ...(Number.isFinite(timestamp) ? { dateBucket: dateBucket(timestamp) } : {}),
    signatureTokens: searchableTokens,
    topicTokens,
  }
}

const scoreRecallCandidate = (
  candidate: RankedRecallHit,
  chosen: readonly RankedRecallHit[],
  querySignals: RecallQuerySignals,
): number => {
  if (chosen.length === 0) return candidate.baseAdjustedScore

  let score = candidate.baseAdjustedScore
  const maxSignatureSimilarity = Math.max(
    ...chosen.map((selected) =>
      jaccardSimilarity(candidate.signatureTokens, selected.signatureTokens),
    ),
  )
  score -=
    maxSignatureSimilarity * (querySignals.aggregate ? 1.1 : querySignals.temporal ? 0.75 : 0.35)

  if (querySignals.aggregate) {
    const maxTopicSimilarity = Math.max(
      ...chosen.map((selected) => jaccardSimilarity(candidate.topicTokens, selected.topicTokens)),
    )
    score -= maxTopicSimilarity * 0.45
    if (maxTopicSimilarity < 0.2) score += 0.25
  }

  if ((querySignals.aggregate || querySignals.temporal) && candidate.dateBucket !== undefined) {
    const hasMatchingDate = chosen.some((selected) => selected.dateBucket === candidate.dateBucket)
    if (!hasMatchingDate) score += 0.35
  }

  if (querySignals.recent) score += candidate.recencyScore * 0.4
  return score
}

const classifyRecallQuery = (query: string): RecallQuerySignals => {
  const trimmed = query.trim()
  const timeline = isTimeSensitiveMemoryQuery(trimmed)
  const recent = trimmed !== '' && RECENT_QUERY_PATTERNS.some((pattern) => pattern.test(trimmed))
  const temporal = timeline || recent
  const aggregate =
    trimmed !== '' &&
    (timeline || AGGREGATE_QUERY_PATTERNS.some((pattern) => pattern.test(trimmed)))
  const concrete =
    trimmed !== '' && (temporal || CONCRETE_QUERY_PATTERNS.some((pattern) => pattern.test(trimmed)))
  return { timeline, recent, temporal, aggregate, concrete }
}

const buildExcludedPathSet = (opts: RecallOpts): ReadonlySet<string> => {
  const excluded = new Set<string>()
  for (const path of opts.excludedPaths ?? []) excluded.add(path)
  for (const path of opts.surfacedPaths ?? []) excluded.add(path)
  return excluded
}

const resolveCandidateK = (
  k: number,
  querySignals: RecallQuerySignals,
  excludedCount: number,
): number => {
  const multiplier = querySignals.aggregate
    ? AGGREGATE_FETCH_MULTIPLIER
    : CANDIDATE_FETCH_MULTIPLIER
  const base = Math.max(k, k * multiplier)
  return Math.min(MAX_CANDIDATE_K, Math.max(k, base + excludedCount))
}

const resolveRecallAllowedScopes = (
  scope: Scope,
  fallbackScopes: readonly Scope[] | undefined,
): readonly Scope[] => {
  const allowed: Scope[] = [scope]
  const seen = new Set<Scope>(allowed)
  for (const fallbackScope of fallbackScopes ?? []) {
    if (!seen.has(fallbackScope)) {
      allowed.push(fallbackScope)
      seen.add(fallbackScope)
    }
  }
  return allowed
}

const resolveRecallSearchScopes = (
  allowedScopes: readonly Scope[],
  actorId: string,
): readonly RecallSearchScope[] => {
  return allowedScopes.map((scope, index) => ({
    scope,
    actorId,
    primary: index === 0,
  }))
}

const searchRecallCandidates = async (
  deps: RecallDeps,
  query: string,
  embedding: readonly number[] | undefined,
  allowedScopes: readonly Scope[],
  actorId: string,
  k: number,
  candidateK: number,
  excludedPaths: ReadonlySet<string>,
  signal: AbortSignal | undefined,
): Promise<readonly SearchHit[]> => {
  const searchScopes = resolveRecallSearchScopes(allowedScopes, actorId)
  const scopedHits = await Promise.all(
    searchScopes.map(async (searchScope) => {
      const hits = deps.searchIndex
        ? await deps.searchIndex.search(query, embedding, {
            k: candidateK,
            scope: searchScope.scope,
            actorId: searchScope.actorId,
          })
        : await fallbackSearch(
            deps.store,
            searchScope.scope,
            searchScope.actorId,
            query,
            candidateK,
            excludedPaths,
            deps.embedder,
            embedding,
            signal,
          )
      return {
        searchScope,
        hits,
      }
    }),
  )

  return mergeScopedSearchHits(scopedHits, k, candidateK)
}

const dedupeScopedSearchHits = (hits: readonly ScopedSearchHit[]): ScopedSearchHit[] => {
  const best = new Map<string, ScopedSearchHit>()
  for (const hit of hits) {
    const current = best.get(hit.path)
    if (current === undefined || hit.score > current.score) {
      best.set(hit.path, hit)
    }
  }
  return [...best.values()].sort((left, right) => {
    if (left.primary !== right.primary) return left.primary ? -1 : 1
    if (left.score !== right.score) return right.score - left.score
    return left.rank - right.rank
  })
}

const mergeScopedSearchHits = (
  results: readonly {
    readonly searchScope: RecallSearchScope
    readonly hits: readonly SearchHit[]
  }[],
  k: number,
  candidateK: number,
): readonly SearchHit[] => {
  const flattened = results.flatMap(({ searchScope, hits }) =>
    hits.map(
      (hit, index): ScopedSearchHit => ({
        ...hit,
        sourceScope: searchScope.scope,
        primary: searchScope.primary,
        rank: index,
      }),
    ),
  )
  if (flattened.length === 0) return []

  const deduped = dedupeScopedSearchHits(flattened)
  const primaryHits = deduped.filter((hit) => hit.primary)
  const secondaryHits = deduped.filter((hit) => !hit.primary)
  if (primaryHits.length === 0 || secondaryHits.length === 0) {
    return deduped.slice(0, candidateK).map(stripScopedSearchHit)
  }

  const primaryReserve = Math.min(
    primaryHits.length,
    Math.min(
      candidateK,
      Math.max(k, Math.min(k + RECALL_PRIMARY_SCOPE_RESERVE, Math.ceil(candidateK / 2))),
    ),
  )
  const selected: ScopedSearchHit[] = primaryHits.slice(0, primaryReserve)
  const seen = new Set(selected.map((hit) => hit.path))
  for (const hit of deduped) {
    if (selected.length >= candidateK) break
    if (seen.has(hit.path)) continue
    selected.push(hit)
    seen.add(hit.path)
  }

  return selected.map(stripScopedSearchHit)
}

const filterExcludedSearchHits = (
  hits: readonly SearchHit[],
  excludedPaths: ReadonlySet<string>,
): SearchHit[] => {
  if (excludedPaths.size === 0) return [...hits]
  return hits.filter((hit) => !excludedPaths.has(hit.path))
}

const stripScopedSearchHit = (hit: ScopedSearchHit): SearchHit => ({
  path: hit.path,
  score: hit.score,
})

type RecallDocumentRead =
  | { readonly ok: true; readonly content: Buffer }
  | { readonly ok: false; readonly error: unknown }

/**
 * Read every path once, through `store.readMany` when the store has it and
 * one `read` per path otherwise. Each result carries what `read` would have
 * returned or thrown for that path, so callers keep per-path semantics.
 */
const readRecallDocuments = async (
  store: Store,
  paths: readonly Path[],
  signal: AbortSignal | undefined,
): Promise<ReadonlyMap<Path, RecallDocumentRead>> => {
  const out = new Map<Path, RecallDocumentRead>()
  const unique = [...new Set(paths)]
  if (store.readMany === undefined) {
    for (const path of unique) {
      signal?.throwIfAborted()
      try {
        out.set(path, { ok: true, content: await store.read(path) })
      } catch (error) {
        out.set(path, { ok: false, error })
      }
    }
    return out
  }

  const valid: Path[] = []
  for (const path of unique) {
    try {
      validatePath(path)
      valid.push(path)
    } catch (error) {
      out.set(path, { ok: false, error })
    }
  }
  if (valid.length === 0) return out
  const contents = await store.readMany(valid)
  for (const path of valid) {
    const content = contents.get(path)
    out.set(
      path,
      content === undefined ? { ok: false, error: new ErrNotFound(path) } : { ok: true, content },
    )
  }
  return out
}

/**
 * Hydrate search hits in hit order. A missing note is skipped. Any other
 * failure is logged per path; when no hit could be hydrated and at least one
 * read failed for a reason other than a missing note, the first such error is
 * thrown so that a failed read is not reported as "nothing relevant".
 */
const hydrateRecallHits = async (
  store: Store,
  hits: readonly SearchHit[],
  scope: Scope,
  logger: Logger,
  signal: AbortSignal | undefined,
): Promise<RecallHit[]> => {
  if (hits.length === 0) return []
  const reads = await readRecallDocuments(
    store,
    hits.map((hit) => hit.path),
    signal,
  )
  const out: RecallHit[] = []
  const failures: unknown[] = []
  for (const hit of hits) {
    try {
      const read = reads.get(hit.path)
      if (read === undefined || !read.ok) throw read?.error ?? new ErrNotFound(hit.path)
      out.push(buildRecallHit(hit, read.content, scope))
    } catch (err) {
      if (err instanceof ErrNotFound) continue
      failures.push(err)
      logger.warn('memory: recall hydrate failed', {
        path: hit.path,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
  if (out.length === 0 && failures.length > 0) throw failures[0]
  return out
}

const buildRecallHit = (hit: SearchHit, content: Buffer, scope: Scope): RecallHit => {
  const raw = content.toString('utf8')
  const { frontmatter, body } = parseFrontmatter(raw)
  const noteScope = coerceScope(frontmatter.scope) ?? inferScopeFromPath(hit.path) ?? scope
  const note: MemoryNote = {
    path: hit.path,
    name: frontmatter.name ?? lastSegment(hit.path),
    description: frontmatter.description ?? '',
    type: frontmatter.type ?? '',
    scope: noteScope,
    tags: frontmatter.tags ?? [],
    content: body,
    ...(frontmatter.modified ? { modified: frontmatter.modified } : {}),
    ...(frontmatter.created ? { created: frontmatter.created } : {}),
    ...(frontmatter.session_id ? { sessionId: frontmatter.session_id } : {}),
    ...(frontmatter.session_date ? { sessionDate: frontmatter.session_date } : {}),
    ...(frontmatter.observed_on ? { observedOn: frontmatter.observed_on } : {}),
  }
  return { path: hit.path, score: hit.score, content: body, note }
}

const coerceScope = (value: unknown): Scope | undefined => {
  if (value === 'global' || value === 'project' || value === 'agent') {
    return value
  }
  return undefined
}

const inferScopeFromPath = (path: Path): Scope | undefined => {
  if (path === MEMORY_GLOBAL_PREFIX || path.startsWith(`${MEMORY_GLOBAL_PREFIX}/`)) {
    return 'global'
  }
  if (path === MEMORY_PROJECTS_PREFIX || path.startsWith(`${MEMORY_PROJECTS_PREFIX}/`)) {
    return 'project'
  }
  if (path === MEMORY_AGENT_PREFIX || path.startsWith(`${MEMORY_AGENT_PREFIX}/`)) {
    return 'agent'
  }
  return undefined
}

const noteTimestamp = (note: MemoryNote): number => {
  const raw = note.sessionDate ?? note.observedOn ?? note.modified ?? note.created
  if (raw === undefined || raw.trim() === '') return Number.POSITIVE_INFINITY
  const parsed = Date.parse(raw)
  return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY
}

const embedSafe = async (
  embedder: Embedder,
  text: string,
  logger: Logger,
  signal: AbortSignal | undefined,
): Promise<number[] | undefined> => {
  try {
    const [vec] = await embedder.embed([text], signal)
    return vec
  } catch (err) {
    signal?.throwIfAborted()
    logger.warn('memory: embed failed', {
      err: err instanceof Error ? err.message : String(err),
    })
    return undefined
  }
}

const fallbackSearch = async (
  store: Store,
  scope: Scope,
  actorId: string,
  query: string,
  k: number,
  excludedPaths: ReadonlySet<string>,
  embedder?: Embedder,
  queryEmbedding?: readonly number[],
  signal?: AbortSignal,
): Promise<readonly SearchHit[]> => {
  const prefix = scopePrefix(scope, actorId)
  let entries: readonly { path: Path; isDir: boolean }[]
  try {
    entries = await store.list(prefix, { recursive: true, includeGenerated: false })
  } catch (err) {
    if (err instanceof ErrNotFound) return []
    throw err
  }
  const q = query.toLowerCase()
  const candidates: { path: Path; content: string }[] = []
  for (const e of entries) {
    if (e.isDir) continue
    if (excludedPaths.has(e.path)) continue
    const name = lastSegment(e.path)
    if (!name.endsWith('.md') || name === 'MEMORY.md') continue
    try {
      candidates.push({ path: e.path, content: (await store.read(e.path)).toString('utf8') })
    } catch {}
  }

  let semanticScores = new Map<string, number>()
  if (embedder !== undefined && queryEmbedding !== undefined && candidates.length > 0) {
    try {
      const noteEmbeddings = await embedder.embed(
        candidates.map((candidate) => candidate.content),
        signal,
      )
      semanticScores = new Map(
        candidates.map((candidate, index) => [
          candidate.path,
          cosineSimilarity(queryEmbedding, noteEmbeddings[index]),
        ]),
      )
    } catch {
      semanticScores = new Map()
    }
  }

  const scored: SearchHit[] = []
  for (const candidate of candidates) {
    const lexical = q ? countOccurrences(candidate.content.toLowerCase(), q) : 0
    const semantic = semanticScores.get(candidate.path) ?? 0
    const score = lexical > 0 ? lexical * 10 + Math.max(semantic, 0) : Math.max(semantic, 0)
    if (score > 0) scored.push({ path: candidate.path, score })
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, k)
}

type RecallSelectorInput = {
  readonly query: string
  readonly k: number
  readonly provider?: Provider
  readonly logger: Logger
  readonly mode: RecallSelectorMode
  readonly signal?: AbortSignal
}

type RecallSelectorCandidate = {
  readonly label: string
  readonly hit: RecallHit
}

const maybeSelectRecallHits = async (
  hits: readonly RecallHit[],
  input: RecallSelectorInput,
): Promise<readonly RecallHit[]> => {
  if (input.mode !== 'auto') return hits
  if (hits.length <= Math.max(1, input.k)) return hits
  if (input.provider === undefined) return hits

  const candidateCount = Math.min(
    hits.length,
    Math.max(
      input.k + 2,
      Math.min(RECALL_SELECTOR_CANDIDATE_LIMIT, Math.max(input.k * 2, input.k)),
    ),
  )
  const candidates = hits.slice(0, candidateCount)
  const labelled = buildRecallSelectorCandidates(candidates)
  const userPrompt = buildRecallSelectorUserPrompt(input.query, labelled)

  const raw = await runRecallSelector(input, userPrompt)
  if (raw === undefined) return hits

  const selectedLabels = parseRecallSelectorSelected(raw)
  if (selectedLabels.size === 0) return hits

  const selectedPaths = new Set<string>()
  const selected: RecallHit[] = []
  for (const candidate of labelled) {
    if (!selectedLabels.has(candidate.label)) continue
    selected.push(candidate.hit)
    selectedPaths.add(candidate.hit.path)
  }
  if (selected.length === 0) return hits
  if (selected.length >= input.k) return selected
  return [...selected, ...hits.filter((hit) => !selectedPaths.has(hit.path))]
}

const runRecallSelector = async (
  input: RecallSelectorInput,
  userPrompt: string,
): Promise<string | undefined> => {
  if (input.provider === undefined) return undefined

  const request = {
    messages: [{ role: 'user', content: userPrompt }] as const,
    system: RECALL_SELECTOR_SYSTEM_PROMPT,
    maxTokens: RECALL_SELECTOR_MAX_TOKENS,
    temperature: RECALL_SELECTOR_TEMPERATURE,
  }

  if (input.provider.supportsStructuredDecoding()) {
    try {
      return await input.provider.structured(
        {
          ...request,
          schema: RECALL_SELECTOR_SCHEMA,
          schemaName: 'recall_selector',
        },
        input.signal,
      )
    } catch (err) {
      input.signal?.throwIfAborted()
      input.logger.debug('memory: structured recall selector failed, falling back to completion', {
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }

  try {
    const response = await input.provider.complete(
      {
        ...request,
        jsonMode: true,
      },
      input.signal,
    )
    return response.content
  } catch (err) {
    input.signal?.throwIfAborted()
    input.logger.warn('memory: recall selector failed', {
      err: err instanceof Error ? err.message : String(err),
    })
    return undefined
  }
}

const buildRecallSelectorCandidates = (
  hits: readonly RecallHit[],
): readonly RecallSelectorCandidate[] =>
  hits.map((hit) => ({
    label: hit.path,
    hit,
  }))

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

type WikilinkFollowUpInput = {
  readonly store: Store
  readonly allowedScopes: readonly Scope[]
  readonly actorId: string
  readonly excludedPaths: ReadonlySet<string>
  readonly logger: Logger
  readonly signal?: AbortSignal
}

type WikilinkCandidate = {
  readonly score: number
  /** Paths the link may resolve to, in the order scopes are tried. */
  readonly paths: readonly Path[]
}

type WikilinkResolution =
  | { readonly kind: 'unresolved' }
  | { readonly kind: 'resolved'; readonly path: Path; readonly read: RecallDocumentRead }

type WikilinkResolver = (candidate: WikilinkCandidate, index: number) => Promise<WikilinkResolution>

const followWikilinkHits = async (
  hits: readonly RecallHit[],
  input: WikilinkFollowUpInput,
): Promise<readonly RecallHit[]> => {
  if (hits.length === 0) return hits

  const candidates = hits.flatMap((hit) =>
    extractWikilinks(hit.content).map(
      (link): WikilinkCandidate => ({
        score: Math.max(hit.score * 0.85, 0.001),
        paths: wikilinkCandidatePaths(link, input.allowedScopes, input.actorId),
      }),
    ),
  )
  if (candidates.length === 0) return hits

  const resolve =
    input.store.readMany === undefined
      ? serialWikilinkResolver(input.store)
      : batchedWikilinkResolver(input.store, candidates, input.signal)
  const blockedPaths = new Set<string>([...input.excludedPaths, ...hits.map((hit) => hit.path)])
  const linkedHits: RecallHit[] = []
  const linkedScope = input.allowedScopes[0] ?? 'global'

  for (const [index, candidate] of candidates.entries()) {
    if (linkedHits.length >= RECALL_WIKILINK_FOLLOW_UP_LIMIT) break
    if (candidate.paths.length === 0) continue
    input.signal?.throwIfAborted()
    const resolution = await resolve(candidate, index)
    if (resolution.kind === 'unresolved' || blockedPaths.has(resolution.path)) continue
    try {
      if (!resolution.read.ok) throw resolution.read.error
      linkedHits.push(
        buildRecallHit(
          { path: resolution.path, score: candidate.score },
          resolution.read.content,
          linkedScope,
        ),
      )
      blockedPaths.add(resolution.path)
    } catch (err) {
      if (err instanceof ErrNotFound) continue
      input.logger.warn('memory: recall wikilink hydrate failed', {
        path: resolution.path,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }

  if (linkedHits.length === 0) return hits
  return [...hits, ...linkedHits]
}

/**
 * Resolve a link with one `exists` per scope tried, then one `read`, as stores
 * without `readMany` always have.
 */
const serialWikilinkResolver =
  (store: Store): WikilinkResolver =>
  async (candidate) => {
    for (const path of candidate.paths) {
      if (!(await store.exists(path))) continue
      try {
        return { kind: 'resolved', path, read: { ok: true, content: await store.read(path) } }
      } catch (error) {
        return { kind: 'resolved', path, read: { ok: false, error } }
      }
    }
    return { kind: 'unresolved' }
  }

/**
 * Resolve links from `readMany` results, reading the candidate paths of up to
 * {@link RECALL_WIKILINK_READ_BATCH} paths' worth of links per call, starting
 * at the first link not yet read. A link resolves to the first of its paths
 * that holds a document, as the serial resolver does.
 */
const batchedWikilinkResolver = (
  store: Store,
  candidates: readonly WikilinkCandidate[],
  signal: AbortSignal | undefined,
): WikilinkResolver => {
  const contents = new Map<Path, Buffer>()
  let readUpTo = 0
  return async (candidate, index) => {
    if (index >= readUpTo) {
      const batch = new Set<Path>()
      let next = index
      while (next < candidates.length) {
        const paths = candidates[next]?.paths ?? []
        if (batch.size > 0 && batch.size + paths.length > RECALL_WIKILINK_READ_BATCH) break
        for (const path of paths) batch.add(path)
        next += 1
      }
      readUpTo = next
      signal?.throwIfAborted()
      const read = (await store.readMany?.([...batch])) ?? new Map<Path, Buffer>()
      for (const [path, content] of read) contents.set(path, content)
    }
    for (const path of candidate.paths) {
      const content = contents.get(path)
      if (content !== undefined) return { kind: 'resolved', path, read: { ok: true, content } }
    }
    return { kind: 'unresolved' }
  }
}

const extractWikilinks = (content: string): readonly string[] => {
  const matches = [...content.matchAll(WIKILINK_PATTERN)]
  if (matches.length === 0) return []
  return matches.map((match) => match[1]?.trim() ?? '').filter((match) => match !== '')
}

const wikilinkCandidatePaths = (
  link: string,
  allowedScopes: readonly Scope[],
  actorId: string,
): readonly Path[] => {
  const rawTarget = link.includes('|') ? link.slice(0, link.indexOf('|')) : link
  const trimmedTarget = rawTarget.trim()
  if (trimmedTarget === '') return []

  const separatorIndex = trimmedTarget.indexOf(':')
  const explicitScope =
    separatorIndex === -1 ? undefined : coerceScope(trimmedTarget.slice(0, separatorIndex))
  if (explicitScope !== undefined) {
    if (!allowedScopes.includes(explicitScope)) return []
    const scopedTarget = normaliseWikilinkTarget(trimmedTarget.slice(separatorIndex + 1))
    return scopedTarget === '' ? [] : [topicPath(explicitScope, actorId, scopedTarget)]
  }

  const target = normaliseWikilinkTarget(trimmedTarget)
  if (target === '') return []
  return allowedScopes.map((scope) => topicPath(scope, actorId, target))
}

const topicPath = (scope: Scope, actorId: string, topic: string): Path =>
  joinPath(scopePrefix(scope, actorId), `${topic}.md`)

const normaliseWikilinkTarget = (target: string): string =>
  target
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9._-]/g, '')

const buildNoteSearchText = (note: MemoryNote): string =>
  [note.name, note.description, ...note.tags, note.content].join('\n')

const buildTopicText = (note: MemoryNote): string =>
  [note.name, note.description, ...note.tags].join('\n')

const concreteNoteScore = (note: MemoryNote): number => {
  const text = buildNoteSearchText(note)
  let score = 0
  if (Number.isFinite(noteTimestamp(note))) score += 1
  if (note.sessionId !== undefined && note.sessionId !== '') score += 0.35
  for (const pattern of CONCRETE_CONTENT_PATTERNS) {
    if (pattern.test(text)) score += 0.35
  }
  return score
}

const genericAdvicePenalty = (note: MemoryNote): number => {
  const text = buildNoteSearchText(note)
  const matchesGenericAdvice = GENERIC_ADVICE_PATTERNS.some((pattern) => pattern.test(text))
  if (!matchesGenericAdvice) return 0
  return Number.isFinite(noteTimestamp(note)) ? 0.25 : 1
}

const tokenise = (text: string, limit = 32): readonly string[] => {
  const matches = text.toLowerCase().match(TOKEN_PATTERN) ?? []
  const out: string[] = []
  const seen = new Set<string>()
  for (const match of matches) {
    const token = stemToken(match)
    if (token.length < 2) continue
    if (STOP_WORDS.has(token)) continue
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

const jaccardSimilarity = (left: readonly string[], right: readonly string[]): number => {
  if (left.length === 0 || right.length === 0) return 0
  const leftSet = new Set(left)
  const rightSet = new Set(right)
  let intersection = 0
  for (const token of leftSet) {
    if (rightSet.has(token)) intersection++
  }
  const union = leftSet.size + rightSet.size - intersection
  return union === 0 ? 0 : intersection / union
}

const normaliseScore = (score: number, maxScore: number): number => {
  if (maxScore <= 0) return 0
  return score / maxScore
}

const normaliseRecency = (
  timestamp: number,
  oldestTimestamp: number,
  newestTimestamp: number,
): number => {
  if (!Number.isFinite(timestamp)) return 0
  if (!Number.isFinite(oldestTimestamp) || !Number.isFinite(newestTimestamp)) return 0
  if (newestTimestamp <= oldestTimestamp) return 1
  return (timestamp - oldestTimestamp) / (newestTimestamp - oldestTimestamp)
}

const dateBucket = (timestamp: number): string => new Date(timestamp).toISOString().slice(0, 10)

const countOccurrences = (haystack: string, needle: string): number => {
  if (!needle) return 0
  let count = 0
  let idx = 0
  while (true) {
    const nextIdx = haystack.indexOf(needle, idx)
    if (nextIdx === -1) break
    count++
    idx = nextIdx + needle.length
  }
  return count
}

const cosineSimilarity = (
  left: readonly number[],
  right: readonly number[] | undefined,
): number => {
  if (right === undefined || left.length === 0 || right.length === 0) return 0
  const length = Math.min(left.length, right.length)
  let dot = 0
  let leftNorm = 0
  let rightNorm = 0
  for (let index = 0; index < length; index++) {
    const leftValue = left[index] ?? 0
    const rightValue = right[index] ?? 0
    dot += leftValue * rightValue
    leftNorm += leftValue * leftValue
    rightNorm += rightValue * rightValue
  }
  if (leftNorm === 0 || rightNorm === 0) return 0
  return dot / Math.sqrt(leftNorm * rightNorm)
}
