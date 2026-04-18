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
import { type Path, joinPath, lastSegment } from '../store/path.js'
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
    const k = opts.k ?? DEFAULT_K
    if (k <= 0) return []
    const scope = opts.scope ?? deps.defaultScope
    const actorId = opts.actorId ?? deps.defaultActorId
    const querySignals = classifyRecallQuery(opts.query)
    const excludedPaths = buildExcludedPathSet(opts)
    const candidateK = resolveCandidateK(k, querySignals, excludedPaths.size)

    const embedding = deps.embedder
      ? await embedSafe(deps.embedder, opts.query, deps.logger)
      : undefined

    const hits = await searchRecallCandidates(
      deps,
      opts.query,
      embedding,
      scope,
      actorId,
      k,
      candidateK,
      excludedPaths,
    )
    const filteredHits = filterExcludedSearchHits(hits, excludedPaths)
    const hydrated = await hydrateRecallHits(
      deps.store,
      filteredHits.slice(0, candidateK),
      scope,
      deps.logger,
    )
    const selected = await maybeSelectRecallHits(hydrated, {
      query: opts.query,
      k,
      logger: deps.logger,
      mode: opts.selector ?? 'off',
      ...(deps.provider !== undefined ? { provider: deps.provider } : {}),
    })
    const expanded = await followWikilinkHits(selected, {
      store: deps.store,
      scope,
      actorId,
      excludedPaths,
      logger: deps.logger,
    })
    return rerankRecallHits(expanded, opts.query, k, querySignals)
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
    analyseRecallHit(hit, queryTokens, querySignals, scoreScale, maxScore, oldestTimestamp, newestTimestamp),
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
  let bonus =
    queryCoverage * 1.4 +
    titleCoverage * 0.8 +
    Math.min(queryMatches, 3) * 0.15
  if (querySignals.temporal && Number.isFinite(timestamp)) bonus += 0.8
  if (querySignals.recent) bonus += recencyScore * 0.8
  if (querySignals.concrete) bonus += concreteScore * 0.55 - genericPenalty * 0.75
  else bonus -= genericPenalty * 0.1

  const baseAdjustedScore = hit.score + normaliseScore(hit.score, maxScore) * 0.35 + bonus * scoreScale

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
    ...chosen.map((selected) => jaccardSimilarity(candidate.signatureTokens, selected.signatureTokens)),
  )
  score -= maxSignatureSimilarity * (querySignals.aggregate ? 1.1 : querySignals.temporal ? 0.75 : 0.35)

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
    trimmed !== '' &&
    (temporal || CONCRETE_QUERY_PATTERNS.some((pattern) => pattern.test(trimmed)))
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
  const multiplier = querySignals.aggregate ? AGGREGATE_FETCH_MULTIPLIER : CANDIDATE_FETCH_MULTIPLIER
  const base = Math.max(k, k * multiplier)
  return Math.min(MAX_CANDIDATE_K, Math.max(k, base + excludedCount))
}

const resolveRecallSearchScopes = (
  scope: Scope,
  actorId: string,
): readonly RecallSearchScope[] => {
  if (scope !== 'project') {
    return [{ scope, actorId, primary: true }]
  }
  return [
    { scope: 'project', actorId, primary: true },
    { scope: 'global', actorId, primary: false },
  ]
}

const searchRecallCandidates = async (
  deps: RecallDeps,
  query: string,
  embedding: readonly number[] | undefined,
  scope: Scope,
  actorId: string,
  k: number,
  candidateK: number,
  excludedPaths: ReadonlySet<string>,
): Promise<readonly SearchHit[]> => {
  const searchScopes = resolveRecallSearchScopes(scope, actorId)
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

const hydrateRecallHits = async (
  store: Store,
  hits: readonly SearchHit[],
  scope: Scope,
  logger: Logger,
): Promise<RecallHit[]> => {
  const out: RecallHit[] = []
  for (const hit of hits) {
    try {
      const hydrated = await hydrateRecallHit(store, hit, scope)
      if (hydrated !== undefined) out.push(hydrated)
    } catch (err) {
      if (err instanceof ErrNotFound) continue
      logger.warn('memory: recall hydrate failed', {
        path: hit.path,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
  return out
}

const hydrateRecallHit = async (
  store: Store,
  hit: SearchHit,
  scope: Scope,
): Promise<RecallHit | undefined> => {
  const raw = (await store.read(hit.path)).toString('utf8')
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
): Promise<number[] | undefined> => {
  try {
    const [vec] = await embedder.embed([text])
    return vec
  } catch (err) {
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
      const noteEmbeddings = await embedder.embed(candidates.map((candidate) => candidate.content))
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
      return await input.provider.structured({
        ...request,
        schema: RECALL_SELECTOR_SCHEMA,
        schemaName: 'recall_selector',
      })
    } catch (err) {
      input.logger.debug('memory: structured recall selector failed, falling back to completion', {
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }

  try {
    const response = await input.provider.complete({
      ...request,
      jsonMode: true,
    })
    return response.content
  } catch (err) {
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
    const tags =
      candidate.hit.note.tags.length > 0 ? candidate.hit.note.tags.join(', ') : 'none'
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
  readonly scope: Scope
  readonly actorId: string
  readonly excludedPaths: ReadonlySet<string>
  readonly logger: Logger
}

const followWikilinkHits = async (
  hits: readonly RecallHit[],
  input: WikilinkFollowUpInput,
): Promise<readonly RecallHit[]> => {
  if (hits.length === 0) return hits

  const blockedPaths = new Set<string>([
    ...input.excludedPaths,
    ...hits.map((hit) => hit.path),
  ])
  const linkedHits: RecallHit[] = []

  for (const hit of hits) {
    if (linkedHits.length >= RECALL_WIKILINK_FOLLOW_UP_LIMIT) break
    const links = extractWikilinks(hit.content)
    for (const link of links) {
      if (linkedHits.length >= RECALL_WIKILINK_FOLLOW_UP_LIMIT) break
      const resolved = await resolveWikilinkPath(input.store, link, input.scope, input.actorId)
      if (resolved === undefined || blockedPaths.has(resolved)) continue
      try {
        const linked = await hydrateRecallHit(
          input.store,
          {
            path: resolved,
            score: Math.max(hit.score * 0.85, 0.001),
          },
          input.scope,
        )
        if (linked === undefined) continue
        linkedHits.push(linked)
        blockedPaths.add(resolved)
      } catch (err) {
        if (err instanceof ErrNotFound) continue
        input.logger.warn('memory: recall wikilink hydrate failed', {
          path: resolved,
          err: err instanceof Error ? err.message : String(err),
        })
      }
    }
  }

  if (linkedHits.length === 0) return hits
  return [...hits, ...linkedHits]
}

const extractWikilinks = (content: string): readonly string[] => {
  const matches = [...content.matchAll(WIKILINK_PATTERN)]
  if (matches.length === 0) return []
  return matches
    .map((match) => match[1]?.trim() ?? '')
    .filter((match) => match !== '')
}

const resolveWikilinkPath = async (
  store: Store,
  link: string,
  scope: Scope,
  actorId: string,
): Promise<Path | undefined> => {
  const rawTarget = link.includes('|') ? link.slice(0, link.indexOf('|')) : link
  const trimmedTarget = rawTarget.trim()
  if (trimmedTarget === '') return undefined

  if (trimmedTarget.startsWith('global:')) {
    const globalTarget = normaliseWikilinkTarget(trimmedTarget.slice('global:'.length))
    return resolveTopicPath(store, 'global', actorId, globalTarget)
  }

  if (scope === 'project') {
    const projectTarget = normaliseWikilinkTarget(trimmedTarget)
    return (
      (await resolveTopicPath(store, 'project', actorId, projectTarget)) ??
      (await resolveTopicPath(store, 'global', actorId, projectTarget))
    )
  }

  const target = normaliseWikilinkTarget(trimmedTarget)
  return resolveTopicPath(store, scope, actorId, target)
}

const resolveTopicPath = async (
  store: Store,
  scope: Scope,
  actorId: string,
  topic: string,
): Promise<Path | undefined> => {
  if (topic === '') return undefined
  const candidate = joinPath(scopePrefix(scope, actorId), `${topic}.md`)
  return (await store.exists(candidate)) ? candidate : undefined
}

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

const countOverlap = (
  left: readonly string[],
  right: readonly string[],
): number => {
  if (left.length === 0 || right.length === 0) return 0
  const rightSet = new Set(right)
  return left.reduce((count, token) => count + (rightSet.has(token) ? 1 : 0), 0)
}

const jaccardSimilarity = (
  left: readonly string[],
  right: readonly string[],
): number => {
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

const dateBucket = (timestamp: number): string =>
  new Date(timestamp).toISOString().slice(0, 10)

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
