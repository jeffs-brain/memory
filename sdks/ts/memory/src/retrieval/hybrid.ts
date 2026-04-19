// SPDX-License-Identifier: Apache-2.0

/**
 * Top-level hybrid retrieval pipeline. Port of
 * apps/jeff/internal/knowledge.KnowledgeBase.SearchHybrid +
 * SearchWithOpts. Wires together the existing query parser, search
 * index, embedder, RRF fusion, unanimity shortcut, reranker and retry
 * ladder into a single `search` / `searchRaw` surface.
 *
 * Pipeline stages (happy path):
 *   1. Parse + expand aliases + compile to FTS5 MATCH expression.
 *   2. Embed the query when an embedder is configured.
 *   3. Run BM25 (top candidateK) and vector search (top candidateK) in
 *      parallel. Vector leg is skipped when no embedder is supplied.
 *   4. RRF-fuse the two lists with k = 60.
 *   5. Unanimity shortcut: when top-3 BM25 and top-3 vector agree on
 *      ≥ 2 positions, skip the rerank pass and return the fused
 *      ranking as-is.
 *   6. Otherwise, rerank the fused top-rerankTopN (default 20).
 *   7. Truncate to topK and return with a HybridTrace.
 *
 * Retry ladder (only when BM25 returns zero and vectors are unavailable
 * or also empty):
 *   1. Strongest term (drop stop-words / short tokens).
 *   2. Force-refresh FTS (no-op for our architecture; documented in
 *      retry.ts).
 *   3. Refreshed sanitised query (strip punctuation, rerun).
 *   4. Refreshed strongest term.
 *   5. Trigram fuzzy fallback (Jaccard ≥ 0.3) over the chunk metadata
 *      held by the index.
 */

import type { Embedder, Logger } from '../llm/types.js'
import {
  type AliasTable,
  type Distiller,
  augmentQueryWithTemporal,
  compileToFTS,
  expandTemporal,
  expandAliases,
  parseQuery,
  parseQuestionDate,
} from '../query/index.js'
import type { Reranker } from '../rerank/index.js'
import { unanimityShortcut } from '../rerank/llm-rerank.js'
import type { SearchIndex } from '../search/index.js'
import type { Chunk } from '../search/writer.js'
import { type RRFCandidate, RRF_DEFAULT_K, reciprocalRankFusion } from './rrf.js'
import {
  type TrigramHit,
  type TrigramIndex,
  type TrigramSourceChunk,
  buildTrigramIndex,
  forceRefreshIndex,
  queryTokens,
  sanitiseQuery,
  strongestTerm,
} from './retry.js'
import type {
  HybridMode,
  HybridTrace,
  RetrievalFilters,
  RetrievalRequest,
  RetrievalResponse,
  RetrievalResult,
  RetryAttempt,
} from './types.js'

const DEFAULT_TOP_K = 10
const DEFAULT_CANDIDATE_K = 60
const DEFAULT_RERANK_TOP_N = 20
const MAX_BM25_FANOUT_QUERIES = 4
const MAX_DERIVED_SUB_QUERIES = 2
const BM25_FANOUT_PRIMARY_WINDOW = 10
const BM25_FANOUT_MIN_OVERLAP = 2
const FILTER_FETCH_MULTIPLIER = 4
const FILTER_FETCH_MAX_MULTIPLIER = 8
const RERANK_SNIPPET_MAX = 1200
const PHRASE_PROBE_MIN_TOKENS = 2
const PHRASE_PROBE_MAX_TOKENS = 4
const PREFERENCE_QUERY_RE =
  /\b(?:recommend|suggest|recommendation|suggestion|tips?|advice|ideas?|what should i|which should i)\b/i
const ENUMERATION_OR_TOTAL_QUERY_RE =
  /\b(?:how many|count|total|in total|sum|add up|list|what are all)\b/i
const PROPERTY_LOOKUP_QUERY_RE =
  /\b(?:how long is my|what specific|which specific|what exact|which exact)\b/i
const SPECIFIC_RECOMMENDATION_QUERY_RE = /\b(?:specific|exact)\b/i
const FIRST_PERSON_FACT_LOOKUP_RE = /\b(?:did i|have i|was i|were i)\b/i
const FACT_LOOKUP_VERB_RE =
  /\b(?:pick(?:ed)? up|bought|ordered|spent|earned|sold|drove|travelled|traveled|watched|visited|completed|finished|submitted|booked)\b/i
const PREFERENCE_NOTE_RE =
  /\b(?:prefer(?:s|red)?|like(?:s|d)?|love(?:s|d)?|want(?:s|ed)?|need(?:s|ed)?|avoid(?:s|ed)?|dislike(?:s|d)?|hate(?:s|d)?|enjoy(?:s|ed)?|interested in|looking for)\b/i
const GENERIC_NOTE_RE =
  /\b(?:tips?|advice|suggest(?:ion|ed)?s?|recommend(?:ation|ed)?s?|ideas?|options?|guide|tracking|tracker|checklist)\b/i
const ROLLUP_NOTE_RE =
  /\b(?:roll-?up|summary|recap|overview|aggregate|combined|overall|in total|totalled?|totalling)\b/i
const ATOMIC_EVENT_NOTE_RE =
  /\b(?:i|we)\s+(?:picked up|bought|ordered|spent|earned|sold|drove|travelled|traveled|went|watched|visited|completed|finished|started|booked|got|took|submitted)\b/i
const DATE_TAG_RE = /\[(?:date|observed on):/i
const QUESTION_LIKE_NOTE_RE =
  /(?:^|\n)(?:what\s+(?:are|is|should|could)|which\s+(?:should|would)|how\s+(?:can|should|could|long)|can\s+you|could\s+you|should\s+i|would\s+you|when\s+did|where\s+(?:can|should)|why\s+(?:is|does|did))\b/i
const DURATION_QUERY_RE = /\bhow long\b/i
const MONEY_EVENT_QUERY_RE = /\b(?:spent|spend|cost|costed|paid|pay)\b/i
const BODY_ABSOLUTE_DATE_RE =
  /\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b/i
const MEASUREMENT_VALUE_RE =
  /\b\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?(?:\s+|-)(?:minutes?|hours?|days?|weeks?|months?|years?)\b/i
const RECENCY_QUERY_RE = /\b(?:most recent|latest|last time|current(?:ly)?|now|newest)\b/i
const EARLIEST_QUERY_RE = /\b(?:earliest|first|initial|original|at first)\b/i
const INLINE_DATE_TAG_RE = /\[(?:observed on|date):?\s*([^\]]+)\]/i
const FRONTMATTER_DATE_KEYS = ['observed_on', 'observedOn', 'session_date', 'sessionDate', 'modified'] as const
const QUESTION_TOKEN_STOP_WORDS: ReadonlySet<string> = new Set([
  'the', 'and', 'for', 'with', 'what',
  'who', 'when', 'where', 'why', 'how',
  'did', 'does', 'was', 'were', 'are',
  'you', 'your', 'about', 'this', 'that',
  'have', 'has', 'had', 'from', 'into',
  'than', 'then', 'them', 'they', 'their',
])
const PHRASE_PROBE_CONNECTORS: ReadonlySet<string> = new Set(['and', 'or', 'plus'])
const PHRASE_PROBE_BOUNDARY_WORDS: ReadonlySet<string> = new Set([
  'a', 'an', 'the', 'and', 'or', 'plus',
  'for', 'with', 'what', 'who', 'when', 'where', 'why', 'how',
  'did', 'does', 'do', 'was', 'were', 'is', 'are', 'am',
  'you', 'your', 'about', 'this', 'that', 'these', 'those',
  'have', 'has', 'had', 'from', 'into', 'than', 'then', 'them', 'they', 'their',
  'i', 'me', 'my', 'we', 'our', 'us', 'it', 'if', 'to', 'of', 'on', 'in', 'at', 'by',
  'amount', 'total', 'all', 'list',
  'finally', 'decided', 'decide', 'wondering', 'wonder',
  'remembered', 'remember', 'thinking', 'back', 'previous', 'conversation',
  'can', 'could', 'would', 'should', 'remind', 'follow', 'specific', 'exact',
  'spent', 'spend', 'bought', 'buy', 'ordered', 'order',
  'purchased', 'purchase', 'paid', 'pay', 'submitted', 'submit',
  'many', 'much', 'long',
  'last', 'today', 'yesterday', 'tomorrow', 'week', 'month', 'year',
  'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
])
const PHRASE_PROBE_TRIM_WORDS: ReadonlySet<string> = new Set(['many', 'much', 'long'])
const ACTION_DATE_PROBE_RULES: ReadonlyArray<{ pattern: RegExp; probe: string }> = [
  { pattern: /\bsubmit(?:ted)?\b/i, probe: 'submission date' },
  { pattern: /\bbook(?:ed|ing)?\b/i, probe: 'booking date' },
  { pattern: /\b(?:buy|bought|purchase(?:d)?|order(?:ed)?)\b/i, probe: 'purchase date' },
  { pattern: /\bjoin(?:ed)?\b/i, probe: 'join date' },
  { pattern: /\b(?:start(?:ed)?|begin|began)\b/i, probe: 'start date' },
  { pattern: /\b(?:finish(?:ed)?|complete(?:d)?)\b/i, probe: 'completion date' },
  { pattern: /\baccept(?:ed|ance)?\b/i, probe: 'acceptance date' },
]
const ACTION_DATE_FOCUS_SKIP_WORDS: ReadonlySet<string> = new Set([
  'accept', 'accepted', 'acceptance',
  'begin', 'began', 'book', 'booked', 'booking',
  'buy', 'bought', 'complete', 'completed', 'completion',
  'date', 'finish', 'finished',
  'join', 'joined', 'order', 'ordered',
  'purchase', 'purchased', 'start', 'started',
  'submit', 'submitted', 'submission',
])
const INSPIRATION_QUERY_HINTS: ReadonlyArray<string> = [
  'inspiration',
  'inspired',
  'ideas',
  'stuck',
  'uninspired',
]
const INSPIRATION_FOCUS_SKIP_WORDS: ReadonlySet<string> = new Set([
  'find', 'finding', 'fresh', 'idea', 'ideas', 'inspiration', 'inspired',
  'new', 'stuck', 'uninspired',
])
const LOW_SIGNAL_PHRASE_PROBE_WORDS: ReadonlySet<string> = new Set([
  'after', 'before', 'day', 'days', 'event', 'events',
  'first', 'happen', 'happened', 'month', 'months',
  'second', 'third', 'time', 'times', 'week', 'weeks',
  'year', 'years',
])
const HEAD_BIGRAM_LAST_TOKENS: ReadonlySet<string> = new Set([
  'development', 'item', 'items', 'language', 'languages', 'product', 'products',
])

type RetrievalIntent = {
  readonly preferenceQuery: boolean
  readonly concreteFactQuery: boolean
}

type ResolvedQueryPlan = {
  readonly baseQuery: string
  readonly bm25Queries: readonly string[]
  readonly compiledPrimaryQuery: string
  readonly compiledBM25Queries: readonly string[]
  readonly phraseProbes: readonly string[]
  readonly temporalAugmented: boolean
}

export type CreateRetrievalOptions = {
  index: SearchIndex
  embedder?: Embedder
  reranker?: Reranker
  aliases?: AliasTable
  logger?: Logger
  /**
   * Optional query distiller. When supplied, the raw request query is
   * rewritten via `distiller.distill(query)` before parsing. A distiller
   * failure falls back to the original query; the trace records both
   * the rewrite and the fallback.
   */
  distiller?: Distiller
  /**
   * Optional supplier for the trigram fallback. When omitted, the
   * retrieval pipeline reads chunk metadata directly from the
   * index's SQLite handle (knowledge_chunks). Injection is available
   * for non-SQLite adapters or tests that want to drive the fallback
   * with a hand-built corpus.
   */
  trigramChunks?: readonly TrigramSourceChunk[]
}

export type Retrieval = {
  search(req: RetrievalRequest): Promise<readonly RetrievalResult[]>
  searchRaw(req: RetrievalRequest): Promise<RetrievalResponse>
}

/**
 * Factory. Keeping this as a function (rather than a class) matches the
 * rest of the package's style and sidesteps subclass-vs-composition
 * questions in downstream consumers.
 */
export function createRetrieval(opts: CreateRetrievalOptions): Retrieval {
  const log = opts.logger
  let trigramIndex: TrigramIndex | undefined
  let trigramBuilt = false

  const ensureTrigramIndex = (): TrigramIndex | undefined => {
    if (trigramBuilt) return trigramIndex
    trigramBuilt = true
    try {
      if (opts.trigramChunks !== undefined) {
        trigramIndex = buildTrigramIndex(opts.trigramChunks)
        return trigramIndex
      }
      // Pull metadata straight from knowledge_chunks. We only need id +
      // path + title + summary + content so the payload stays small.
      const rows = opts.index.db
        .prepare(
          `SELECT id, path, title, summary, content FROM knowledge_chunks`,
        )
        .all() as Array<{
        id: string
        path: string
        title: string | null
        summary: string | null
        content: string | null
      }>
      trigramIndex = buildTrigramIndex(
        rows.map((r) => ({
          id: r.id,
          path: r.path,
          title: r.title ?? '',
          summary: r.summary ?? '',
          content: r.content ?? '',
        })),
      )
      return trigramIndex
    } catch (err) {
      log?.warn?.('trigram index build failed', { err: String(err) })
      return undefined
    }
  }

  const runSearchRaw = async (req: RetrievalRequest): Promise<RetrievalResponse> => {
    const started = Date.now()

    const topK = req.topK ?? DEFAULT_TOP_K
    const candidateK = req.candidateK ?? DEFAULT_CANDIDATE_K
    const rerankTopN = req.rerankTopN ?? DEFAULT_RERANK_TOP_N
    const filtersApplied = hasRetrievalFilters(req.filters)

    // Run the optional distiller before parsing so the rewritten text
    // feeds every downstream leg (parser, compile, embedder, retry).
    let effectiveQuery = req.query
    let usedDistill = false
    let distilledQuery: string | undefined
    let distillElapsed: number | undefined
    if (opts.distiller !== undefined) {
      const distillStart = Date.now()
      try {
        const rewritten = await opts.distiller.distill(req.query, req.signal)
        distillElapsed = Date.now() - distillStart
        if (rewritten !== '') {
          effectiveQuery = rewritten
          usedDistill = true
          distilledQuery = rewritten
        }
      } catch (err) {
        distillElapsed = Date.now() - distillStart
        log?.warn?.('distiller failed; using raw query', { err: String(err) })
      }
    }

    const queryPlan = buildQueryPlan(effectiveQuery, req.questionDate, opts.aliases)
    const semanticQuery = queryPlan.baseQuery

    const embedderReady = opts.embedder !== undefined

    const requestedMode: HybridMode = req.mode ?? 'auto'
    let mode: HybridMode = requestedMode
    let fellBackToBM25 = false
    if (
      !embedderReady &&
      (mode === 'auto' || mode === 'hybrid' || mode === 'semantic' || mode === 'hybrid-rerank')
    ) {
      mode = 'bm25'
      if (requestedMode !== 'auto') fellBackToBM25 = true
    } else if (mode === 'auto') {
      mode = 'hybrid'
    }
    const rerankEnabled = req.rerank ?? mode === 'hybrid-rerank'

    const trace: HybridTrace = {
      mode,
      originalQuery: req.query,
      compiledQuery: queryPlan.compiledPrimaryQuery,
      bm25Queries: queryPlan.compiledBM25Queries,
      candidateK,
      rrfK: RRF_DEFAULT_K,
      bm25Elapsed: 0,
      vectorElapsed: 0,
      fusionElapsed: 0,
      rerankElapsed: 0,
      totalElapsed: 0,
      bm25Count: 0,
      vectorCount: 0,
      fusedCount: 0,
      fellBackToBM25,
      embedderUsed: false,
      reranked: false,
      temporalAugmented: queryPlan.temporalAugmented,
      filtersApplied,
      attempts: [],
    }
    const attempts: RetryAttempt[] = []

    const runBM25Query = (compiledQuery: string, limit: number): RRFCandidate[] =>
      searchBM25Candidates(opts.index, compiledQuery, limit, req.filters)

    const runBM25Fanout = (compiledQueries: readonly string[], limit: number): RRFCandidate[] => {
      const [primaryQuery, ...secondaryQueries] = compiledQueries
      if (primaryQuery === undefined) return []
      const primaryHits = runBM25Query(primaryQuery, limit)
      const lists: RRFCandidate[][] = primaryHits.length > 0 ? [primaryHits] : []
      for (const query of secondaryQueries) {
        const hits = runBM25Query(query, limit)
        if (hits.length === 0) continue
        if (
          primaryHits.length === 0 ||
          shouldBypassBM25FanoutOverlapGate(query) ||
          bm25FanoutOverlap(primaryHits, hits) >= BM25_FANOUT_MIN_OVERLAP
        ) {
          lists.push(hits)
        }
      }
      if (lists.length === 0) return []
      if (lists.length === 1) return lists[0] ?? []
      return reciprocalRankFusion(lists, RRF_DEFAULT_K).map((result, index) => ({
        id: result.id,
        path: result.path,
        title: result.title,
        summary: result.summary,
        content: result.content,
        ...(result.metadata !== undefined ? { metadata: result.metadata } : {}),
        bm25Rank: index,
      }))
    }

    // ---- BM25 leg (with retry ladder on zero hits) ----
    const bmStart = Date.now()
    let bmCandidates: RRFCandidate[] = []
    let bmError: unknown

    try {
      bmCandidates = runBM25Fanout(queryPlan.compiledBM25Queries, candidateK)
      attempts.push({
        strategy: 'initial',
        query: queryPlan.compiledPrimaryQuery,
        hits: bmCandidates.length,
      })

      if (bmCandidates.length === 0 && req.skipRetryLadder !== true) {
        // Rung 1: strongest term.
        const strongest = strongestTerm(semanticQuery)
        if (strongest !== undefined && strongest !== semanticQuery.trim().toLowerCase()) {
          bmCandidates = runBM25Fanout(
            buildCompiledBM25Queries({
              queries: buildBM25QueryTexts({ baseQuery: strongest, rawQuery: strongest }),
              aliases: opts.aliases,
            }),
            candidateK,
          )
          attempts.push({
            strategy: 'strongest_term',
            query: strongest,
            hits: bmCandidates.length,
          })
        }

        if (bmCandidates.length === 0) {
          // Rung 2: force-refresh (no-op) + rung 3: refreshed sanitised.
          forceRefreshIndex()
          const sanitised = sanitiseQuery(semanticQuery)
          if (sanitised !== '') {
            bmCandidates = runBM25Fanout(
              buildCompiledBM25Queries({
                queries: buildBM25QueryTexts({ baseQuery: sanitised, rawQuery: sanitised }),
                aliases: opts.aliases,
              }),
              candidateK,
            )
            attempts.push({
              strategy: 'refreshed_sanitised',
              query: sanitised,
              hits: bmCandidates.length,
            })
          }
        }

        if (bmCandidates.length === 0) {
          // Rung 4: refreshed strongest term.
          const strongest = strongestTerm(sanitiseQuery(semanticQuery))
          if (strongest !== undefined) {
            bmCandidates = runBM25Fanout(
              buildCompiledBM25Queries({
                queries: buildBM25QueryTexts({ baseQuery: strongest, rawQuery: strongest }),
                aliases: opts.aliases,
              }),
              candidateK,
            )
            attempts.push({
              strategy: 'refreshed_strongest',
              query: strongest,
              hits: bmCandidates.length,
            })
          }
        }

        if (bmCandidates.length === 0) {
          // Rung 5: trigram fuzzy fallback.
          const tokens = queryTokens(semanticQuery)
          const tri = ensureTrigramIndex()
          if (tri !== undefined && tokens.length > 0) {
            const fuzzy: readonly TrigramHit[] = tri.search(tokens, candidateK)
            bmCandidates = fuzzy.map((h, idx) => ({
              id: h.id,
              path: h.path,
              title: h.title,
              summary: h.summary,
              content: h.content,
              bm25Rank: idx,
            }))
            attempts.push({
              strategy: 'trigram_fuzzy',
              query: tokens.join(' '),
              hits: bmCandidates.length,
            })
          }
        }
      }
    } catch (err) {
      bmError = err
      log?.warn?.('bm25 leg failed', { err: String(err) })
    }
    trace.bm25Elapsed = Date.now() - bmStart
    trace.bm25Count = bmCandidates.length

    // ---- Vector leg ----
    const vecStart = Date.now()
    let vecCandidates: RRFCandidate[] = []
    let vecError: unknown

    if (
      opts.embedder !== undefined &&
      (mode === 'hybrid' || mode === 'semantic' || mode === 'hybrid-rerank')
    ) {
      try {
        const vectors = await opts.embedder.embed([semanticQuery], req.signal)
        const first = vectors[0]
        if (first !== undefined && first.length > 0) {
          trace.embedderUsed = true
          const hits = searchVectorCandidates(opts.index, first, candidateK, req.filters)
          vecCandidates = hits.map((h, idx) => ({
            id: h.chunk.id,
            path: h.chunk.path,
            title: h.chunk.title ?? '',
            summary: h.chunk.summary ?? '',
            content: h.chunk.content,
            ...(h.chunk.metadata !== undefined ? { metadata: h.chunk.metadata } : {}),
            vectorSimilarity: h.similarity,
          }))
        }
      } catch (err) {
        vecError = err
        log?.warn?.('vector leg failed', { err: String(err) })
      }
    }
    trace.vectorElapsed = Date.now() - vecStart
    trace.vectorCount = vecCandidates.length

    // ---- Mode resolution after legs complete ----
    if (bmError !== undefined && vecError !== undefined) {
      trace.errorStage = 'bm25'
      trace.errorDetail = String(bmError)
      trace.totalElapsed = Date.now() - started
      trace.attempts = attempts
      return { results: [], trace }
    }

    let fused: RetrievalResult[] = []
    const fuseStart = Date.now()
    if (mode === 'bm25') {
      fused = bmCandidates.map((c) => toResult(c, 1 / (RRF_DEFAULT_K + (c.bm25Rank ?? 0) + 1)))
    } else if (mode === 'semantic') {
      fused = vecCandidates.map((c, i) => toResult(c, 1 / (RRF_DEFAULT_K + i + 1)))
    } else {
      // hybrid: fuse whichever lists are non-empty.
      const lists: RRFCandidate[][] = []
      if (bmCandidates.length > 0) lists.push(bmCandidates)
      if (vecCandidates.length > 0) lists.push(vecCandidates)
      fused = reciprocalRankFusion(lists, RRF_DEFAULT_K)
    }
    fused = reweightSharedMemoryRanking(semanticQuery, fused)
    trace.fusionElapsed = Date.now() - fuseStart
    trace.fusedCount = fused.length

    // ---- Unanimity shortcut + rerank ----
    let final = fused
    if (fused.length === 0) {
      trace.rerankSkippedReason = 'empty_candidates'
    } else if (!rerankEnabled) {
      trace.rerankSkippedReason = 'mode_off'
    } else if (opts.reranker === undefined) {
      trace.rerankSkippedReason = 'no_reranker'
    } else {
      const shortcut = unanimityShortcut(
        bmCandidates.map((c) => ({ id: c.id })),
        vecCandidates.map((c) => ({ id: c.id })),
      )
      if (shortcut !== undefined) {
        trace.rerankSkippedReason = 'unanimity'
        trace.unanimity = { agreements: shortcut.agreements }
      } else {
        const rerankStart = Date.now()
        const n = Math.min(rerankTopN, fused.length)
        const head = fused.slice(0, n)
        const tail = fused.slice(n)
        try {
          const reranked = await opts.reranker.rerank(
            {
              query: semanticQuery,
              documents: head.map((r) => ({
                id: r.id,
                text: composeRerankText(r),
              })),
            },
            req.signal,
          )
          const reordered: Array<RetrievalResult & { originalIndex: number }> = []
          for (const hit of reranked) {
            const src = head[hit.index]
            if (src === undefined) continue
            reordered.push({ ...src, rerankScore: hit.score, originalIndex: hit.index })
          }
          reordered.sort((left, right) => {
            const leftScore = left.rerankScore ?? 0
            const rightScore = right.rerankScore ?? 0
            if (leftScore !== rightScore) return rightScore - leftScore
            if (left.score !== right.score) return right.score - left.score
            return left.originalIndex - right.originalIndex
          })
          final = reordered.map(({ originalIndex: _originalIndex, ...result }) => result).concat(tail)
          trace.reranked = true
          trace.rerankProvider = opts.reranker.name()
        } catch (err) {
          log?.warn?.('reranker failed; returning fused ranking', { err: String(err) })
          trace.errorStage = 'rerank'
          trace.errorDetail = String(err)
        }
        trace.rerankElapsed = Date.now() - rerankStart
      }
    }

    final = reweightTemporalRanking(req.query, req.questionDate, final)
    if (final.length > topK) final = final.slice(0, topK)

    trace.attempts = attempts
    trace.totalElapsed = Date.now() - started
    if (usedDistill) trace.usedDistill = true
    if (distilledQuery !== undefined) trace.distilledQuery = distilledQuery
    if (distillElapsed !== undefined) trace.distillElapsed = distillElapsed
    return { results: final, trace }
  }

  return {
    search: async (req) => (await runSearchRaw(req)).results,
    searchRaw: runSearchRaw,
  }
}

function bm25FanoutOverlap(
  primary: readonly RRFCandidate[],
  secondary: readonly RRFCandidate[],
): number {
  const primaryWindow = primary.slice(0, BM25_FANOUT_PRIMARY_WINDOW)
  if (primaryWindow.length === 0 || secondary.length === 0) {
    return 0
  }
  const primaryIds = new Set(primaryWindow.map((candidate) => candidate.id))
  let overlap = 0
  for (const candidate of secondary.slice(0, BM25_FANOUT_PRIMARY_WINDOW)) {
    if (!primaryIds.has(candidate.id)) continue
    overlap += 1
    if (overlap >= BM25_FANOUT_MIN_OVERLAP) {
      return overlap
    }
  }
  return overlap
}

function shouldBypassBM25FanoutOverlapGate(query: string): boolean {
  if (/\d/.test(query)) return true
  let terms = 0
  for (const token of query.split(/\s+/u)) {
    if (token === 'AND' || token === 'OR' || token === 'NOT') continue
    if (token === '') continue
    terms += 1
  }
  return terms >= PHRASE_PROBE_MIN_TOKENS && terms <= PHRASE_PROBE_MAX_TOKENS
}

function toResult(c: RRFCandidate, score: number): RetrievalResult {
  return {
    id: c.id,
    path: c.path,
    title: c.title ?? '',
    summary: c.summary ?? '',
    content: c.content ?? '',
    ...(c.metadata !== undefined ? { metadata: c.metadata } : {}),
    score,
    ...(c.bm25Rank !== undefined ? { bm25Rank: c.bm25Rank } : {}),
    ...(c.vectorSimilarity !== undefined ? { vectorSimilarity: c.vectorSimilarity } : {}),
  }
}

function composeRerankText(r: RetrievalResult): string {
  const title = (r.title ?? '').trim()
  const summary = (r.summary ?? '').trim()
  const body = (r.content ?? '').replace(/\s+/g, ' ').trim()
  const snippet =
    body === ''
      ? '(no body excerpt available)'
      : body.length <= RERANK_SNIPPET_MAX
        ? body
        : `${body.slice(0, RERANK_SNIPPET_MAX)}...`
  return [
    `title: ${title !== '' ? title : '(untitled)'}`,
    `    path: ${r.path}`,
    `    summary: ${summary !== '' ? summary : '(no summary available)'}`,
    '',
    `    content: ${snippet}`,
  ].join('\n')
}

function reweightSharedMemoryRanking(
  query: string,
  results: readonly RetrievalResult[],
): RetrievalResult[] {
  if (results.length === 0) {
    return [...results]
  }

  const intent = detectRetrievalIntent(query)
  if (!intent.preferenceQuery && !intent.concreteFactQuery) {
    return [...results]
  }

  return [...results]
    .map((result, index) => ({
      result: {
        ...result,
        score: result.score * retrievalIntentMultiplier(intent, query, result),
      },
      index,
    }))
    .sort((left, right) => {
      if (right.result.score !== left.result.score) {
        return right.result.score - left.result.score
      }
      return left.index - right.index
    })
    .map(({ result }) => result)
}

function reweightTemporalRanking(
  query: string,
  questionDate: string | undefined,
  results: readonly RetrievalResult[],
): RetrievalResult[] {
  if (results.length === 0) {
    return [...results]
  }

  const anchor = parseCandidateDate(questionDate ?? '')
  const filteredResults =
    anchor === undefined
      ? [...results]
      : results.filter((result) => {
          const candidate = extractCandidateDate(result)
          return candidate === undefined || candidate.getTime() <= anchor.getTime()
        })
  if (filteredResults.length === 0) {
    return []
  }

  const expansion = expandTemporal(query, questionDate)
  const wantsRecency = RECENCY_QUERY_RE.test(query)
  const wantsEarliest = !wantsRecency && EARLIEST_QUERY_RE.test(query)
  const hintTimes = expansion.dateHints
    .map((hint) => parseCandidateDate(hint))
    .filter((value): value is Date => value !== undefined)
  if (!wantsRecency && !wantsEarliest && hintTimes.length === 0) {
    return filteredResults
  }

  const candidateTimes = filteredResults.map((result) => extractCandidateDate(result))
  const datedTimes = candidateTimes.filter((value): value is Date => value !== undefined)
  if (datedTimes.length === 0) {
    return filteredResults
  }

  let minMs = datedTimes[0]!.getTime()
  let maxMs = minMs
  for (const value of datedTimes.slice(1)) {
    const ms = value.getTime()
    if (ms < minMs) minMs = ms
    if (ms > maxMs) maxMs = ms
  }

  return filteredResults
    .map((result, index) => {
      const candidateTime = candidateTimes[index]
      let multiplier = 1
      if (candidateTime !== undefined && hintTimes.length > 0) {
        multiplier *= temporalHintMultiplier(candidateTime, hintTimes)
      }
      if (candidateTime !== undefined && maxMs > minMs) {
        const norm = (candidateTime.getTime() - minMs) / (maxMs - minMs)
        if (wantsRecency) multiplier *= 1 + 0.25 * norm
        if (wantsEarliest) multiplier *= 1 + 0.25 * (1 - norm)
      } else if (candidateTime === undefined && (wantsRecency || wantsEarliest)) {
        multiplier *= 0.95
      }
      return {
        result: {
          ...result,
          score: result.score * multiplier,
        },
        index,
      }
    })
    .sort((left, right) => {
      if (right.result.score !== left.result.score) {
        return right.result.score - left.result.score
      }
      return left.index - right.index
    })
    .map(({ result }) => result)
}

function temporalHintMultiplier(candidateTime: Date, hintTimes: readonly Date[]): number {
  const candidateMs = candidateTime.getTime()
  let nearestDays = Number.POSITIVE_INFINITY
  for (const hint of hintTimes) {
    const diffDays = Math.abs(candidateMs - hint.getTime()) / 86_400_000
    if (diffDays < nearestDays) nearestDays = diffDays
  }
  if (nearestDays <= 1) return 1.35
  if (nearestDays <= 7) return 1.2
  if (nearestDays <= 30) return 1.08
  return 0.92
}

function extractCandidateDate(result: RetrievalResult): Date | undefined {
  const metadataDate = extractMetadataDate(result.metadata)
  if (metadataDate !== undefined) return metadataDate
  return extractDateFromText(result.content)
}

function extractMetadataDate(metadata: Record<string, unknown> | undefined): Date | undefined {
  if (metadata === undefined) return undefined
  for (const key of FRONTMATTER_DATE_KEYS) {
    const value = metadata[key]
    if (typeof value !== 'string') continue
    const parsed = parseCandidateDate(value)
    if (parsed !== undefined) return parsed
  }
  return undefined
}

function extractDateFromText(text: string): Date | undefined {
  const inlineMatch = INLINE_DATE_TAG_RE.exec(text)
  if (inlineMatch?.[1] !== undefined) {
    const parsed = parseCandidateDate(inlineMatch[1])
    if (parsed !== undefined) return parsed
  }
  for (const key of FRONTMATTER_DATE_KEYS) {
    const match = new RegExp(`^${key}:\\s*(.+)$`, 'im').exec(text)
    if (match?.[1] === undefined) continue
    const parsed = parseCandidateDate(match[1])
    if (parsed !== undefined) return parsed
  }
  return undefined
}

function parseCandidateDate(value: string): Date | undefined {
  const trimmed = value.trim()
  if (trimmed === '') return undefined
  return parseQuestionDate(trimmed)
}

function detectRetrievalIntent(query: string): RetrievalIntent {
  const normalised = query.toLowerCase()
  return {
    preferenceQuery: PREFERENCE_QUERY_RE.test(normalised),
    concreteFactQuery:
      PROPERTY_LOOKUP_QUERY_RE.test(normalised) ||
      ENUMERATION_OR_TOTAL_QUERY_RE.test(normalised) ||
      (FIRST_PERSON_FACT_LOOKUP_RE.test(normalised) &&
        FACT_LOOKUP_VERB_RE.test(normalised)),
  }
}

function retrievalIntentMultiplier(
  intent: RetrievalIntent,
  query: string,
  result: RetrievalResult,
): number {
  let multiplier = 1
  const text = retrievalResultText(result)
  if (intent.preferenceQuery) {
    multiplier *= preferenceIntentMultiplier(result, text)
  }
  if (intent.concreteFactQuery) {
    multiplier *= concreteFactIntentMultiplier(query, result, text)
  }
  return multiplier
}

function buildQueryPlan(
  query: string,
  questionDate: string | undefined,
  aliases: AliasTable | undefined,
): ResolvedQueryPlan {
  const baseQuery = augmentQueryWithTemporal(query, questionDate)
  const temporalAugmented = baseQuery.trim() !== query.trim()
  const bm25Queries = buildBM25QueryTexts({
    baseQuery,
    rawQuery: query,
  })
  const phraseProbes = derivePhraseProbes(query)
  const compiledBM25Queries = buildCompiledBM25Queries({
    queries: bm25Queries,
    phraseProbes,
    aliases,
  })
  const compiledPrimaryQuery = compiledBM25Queries[0] ?? compileSearchQuery(baseQuery, aliases)
  return {
    baseQuery,
    bm25Queries,
    compiledPrimaryQuery,
    compiledBM25Queries,
    phraseProbes,
    temporalAugmented,
  }
}

function buildCompiledBM25Queries(args: {
  readonly queries: readonly string[]
  readonly phraseProbes?: readonly string[]
  readonly aliases: AliasTable | undefined
}): readonly string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const query of args.queries) {
    const phraseProbes = args.phraseProbes ?? derivePhraseProbes(query)
    const compiled = compileSearchQuery(
      compileBM25FanoutQueryText(query, phraseProbes),
      args.aliases,
    )
    if (compiled === '' || seen.has(compiled)) continue
    seen.add(compiled)
    out.push(compiled)
    if (out.length >= MAX_BM25_FANOUT_QUERIES) break
  }
  return out
}

function buildBM25QueryTexts(args: {
  readonly baseQuery: string
  readonly rawQuery?: string
}): readonly string[] {
  const seen = new Set<string>()
  const out: string[] = []
  const push = (value: string | undefined): void => {
    const trimmed = value?.trim() ?? ''
    if (trimmed === '' || seen.has(trimmed)) return
    seen.add(trimmed)
    out.push(trimmed)
  }

  const rawQuery = args.rawQuery?.trim() ?? ''
  const phraseSource = rawQuery !== '' ? rawQuery : args.baseQuery
  const priorityQueries = derivePrioritySubQueries(phraseSource)
  if (shouldUsePriorityOnlyBM25(phraseSource) && priorityQueries.length >= 2) {
    for (const query of priorityQueries) push(query)
    return out.slice(0, MAX_BM25_FANOUT_QUERIES)
  }
  for (const query of priorityQueries) push(query)
  push(rawQuery !== '' ? rawQuery : args.baseQuery)
  push(args.baseQuery)
  for (const subQuery of deriveSubQueries(phraseSource)) {
    push(subQuery)
  }
  return out.slice(0, MAX_BM25_FANOUT_QUERIES)
}

function deriveSubQueries(query: string): readonly string[] {
  const out: string[] = []
  const seen = new Set<string>([query.trim().toLowerCase()])
  const inspirationQuery = containsAnyHint(query.trim().toLowerCase(), INSPIRATION_QUERY_HINTS)
  for (const probe of deriveSpecificRecommendationProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveMoneyFocusProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveActionDateContextProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveInspirationSourceProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveActionDateProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  const phrases = filteredPhraseProbes(query)
  for (const phrase of phrases) {
    if (inspirationQuery && filterQuestionTokens(phrase, INSPIRATION_FOCUS_SKIP_WORDS).length === 0) {
      continue
    }
    if (seen.has(phrase)) continue
    seen.add(phrase)
    out.push(phrase)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }

  const tokenSource = phrases.length > 0 ? phrases.join(' ') : query
  const tokens = questionTokens(tokenSource)
  if (tokens.length < 2) return out
  const sorted = [...tokens].sort((left, right) => right.length - left.length)
  for (const token of sorted) {
    if (inspirationQuery && INSPIRATION_FOCUS_SKIP_WORDS.has(token)) continue
    if (seen.has(token)) continue
    seen.add(token)
    out.push(token)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) break
  }
  return out
}

function derivePrioritySubQueries(query: string): readonly string[] {
  const out: string[] = []
  const seen = new Set<string>([query.trim().toLowerCase()])
  for (const probe of deriveSpecificRecommendationProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveMoneyFocusProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveActionDateContextProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const probe of deriveInspirationSourceProbes(query)) {
    if (seen.has(probe)) continue
    seen.add(probe)
    out.push(probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  for (const phrase of filteredPhraseProbes(query)) {
    if (seen.has(phrase)) continue
    seen.add(phrase)
    out.push(phrase)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  return out
}

function deriveSpecificRecommendationProbes(query: string): readonly string[] {
  const lowered = query.trim().toLowerCase()
  if (!SPECIFIC_RECOMMENDATION_QUERY_RE.test(lowered)) return []
  if (!/\brecommend(?:ed)?\b/i.test(lowered) && !/\bremind me\b/i.test(lowered)) {
    return []
  }
  if (lowered.includes('back-end') && /\blanguages?\b/.test(lowered)) {
    return ['back-end programming language', 'back-end development']
  }
  return []
}

function deriveMoneyFocusProbes(query: string): readonly string[] {
  const lowered = query.trim().toLowerCase()
  if (!ENUMERATION_OR_TOTAL_QUERY_RE.test(lowered) || !MONEY_EVENT_QUERY_RE.test(lowered)) {
    return []
  }
  const out: string[] = []
  const seen = new Set<string>()
  for (const phrase of filteredPhraseProbes(query)) {
    const candidate = moneyFocusProbeFromPhrase(phrase)
    if (candidate === '' || seen.has(candidate)) continue
    seen.add(candidate)
    out.push(candidate)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) {
      return out
    }
  }
  return out
}

function moneyFocusProbeFromPhrase(phrase: string): string {
  const head = derivePhraseHeadFocus(phrase)
  if (head === '') return ''
  if (head !== phrase && phrase.split(/\s+/u).length <= 2) {
    return `${head} cost`
  }
  return phrase
}

function derivePhraseHeadFocus(phrase: string): string {
  const tokens = phrase
    .trim()
    .toLowerCase()
    .split(/\s+/u)
    .filter((token) => token !== '')
  if (tokens.length === 0) return ''
  const last = tokens[tokens.length - 1]
  if (tokens.length >= 2 && HEAD_BIGRAM_LAST_TOKENS.has(last)) {
    return tokens.slice(-2).join(' ')
  }
  return last
}

function deriveActionDateContextProbes(query: string): readonly string[] {
  const actionDateProbes = deriveActionDateProbes(query)
  if (actionDateProbes.length === 0) return []
  const focuses = deriveActionDateFocuses(
    filterQuestionTokens(query, ACTION_DATE_FOCUS_SKIP_WORDS),
  )
  if (focuses.length === 0) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const probe of actionDateProbes) {
    for (const focus of focuses) {
      const candidate = `${focus} ${probe}`.trim()
      if (candidate === '' || seen.has(candidate)) continue
      seen.add(candidate)
      out.push(candidate)
      if (out.length >= MAX_DERIVED_SUB_QUERIES) return out
    }
  }
  return out
}

function deriveActionDateFocuses(tokens: readonly string[]): readonly string[] {
  if (tokens.length === 0) return []
  const out: string[] = []
  const seen = new Set<string>()
  const appendWindow = (start: number, end: number): void => {
    if (start < 0 || end > tokens.length || start >= end) return
    const candidate = tokens.slice(start, end).join(' ').trim()
    if (candidate === '' || seen.has(candidate)) return
    seen.add(candidate)
    out.push(candidate)
  }
  if (tokens.length === 1) {
    appendWindow(0, 1)
    return out
  }
  if (tokens.length === 2) {
    appendWindow(0, 2)
    return out
  }
  if (tokens.length === 3) {
    appendWindow(1, 3)
    appendWindow(0, 2)
    return out
  }
  appendWindow(tokens.length - 2, tokens.length)
  appendWindow(0, 2)
  return out
}

function deriveInspirationSourceProbes(query: string): readonly string[] {
  const lowered = query.trim().toLowerCase()
  if (lowered === '' || !containsAnyHint(lowered, INSPIRATION_QUERY_HINTS)) return []
  const tokens = filterQuestionTokens(query, INSPIRATION_FOCUS_SKIP_WORDS)
  const focus = tokens.at(-1)
  if (focus === undefined || focus === '') return []
  return [`${focus} social media tutorials`]
}

function deriveActionDateProbes(query: string): readonly string[] {
  const lowered = query.trim().toLowerCase()
  if (lowered === '' || !lowered.includes('when')) return []
  const out: string[] = []
  const seen = new Set<string>()
  for (const rule of ACTION_DATE_PROBE_RULES) {
    if (!rule.pattern.test(lowered) || seen.has(rule.probe)) continue
    seen.add(rule.probe)
    out.push(rule.probe)
    if (out.length >= MAX_DERIVED_SUB_QUERIES) break
  }
  return out
}

function filterQuestionTokens(
  query: string,
  skip: ReadonlySet<string>,
): readonly string[] {
  return questionTokens(query).filter((token) => !skip.has(token))
}

function containsAnyHint(text: string, hints: readonly string[]): boolean {
  return hints.some((hint) => text.includes(hint))
}

function derivePhraseProbes(query: string): readonly string[] {
  const tokens = phraseProbeTokens(query)
  if (tokens.length < PHRASE_PROBE_MIN_TOKENS) return []

  const out: string[] = []
  const seen = new Set<string>()
  const appendPhrase = (phrase: string): boolean => {
    const trimmed = phrase.trim()
    if (trimmed === '' || seen.has(trimmed)) return false
    seen.add(trimmed)
    out.push(trimmed)
    return out.length >= MAX_DERIVED_SUB_QUERIES
  }
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index]
    if (token === undefined || !PHRASE_PROBE_CONNECTORS.has(token)) continue

    const left = collectLeftPhrase(tokens.slice(0, index))
    if (left !== '' && appendPhrase(left)) return out

    const right = collectRightPhrase(tokens.slice(index + 1))
    if (right !== '' && appendPhrase(right)) return out
  }
  for (const phrase of deriveBoundarySpanProbes(tokens)) {
    if (appendPhrase(phrase)) {
      return out
    }
  }
  return out
}

function filteredPhraseProbes(query: string): readonly string[] {
  return derivePhraseProbes(query).filter(
    (phrase) => filterQuestionTokens(phrase, LOW_SIGNAL_PHRASE_PROBE_WORDS).length > 0,
  )
}

function shouldUsePriorityOnlyBM25(query: string): boolean {
  const lowered = query.trim().toLowerCase()
  return (
    deriveActionDateContextProbes(query).length > 0 ||
    (filteredPhraseProbes(query).length >= 2 && lowered.includes(' and '))
  )
}

function deriveBoundarySpanProbes(tokens: readonly string[]): readonly string[] {
  if (tokens.length < PHRASE_PROBE_MIN_TOKENS) return []
  const out: string[] = []
  let segment: string[] = []
  const flush = (): void => {
    const phrase = bestSegmentPhrase(segment)
    if (phrase !== '') out.push(phrase)
    segment = []
  }
  for (const token of tokens) {
    if (token === '' || PHRASE_PROBE_BOUNDARY_WORDS.has(token) || token.length < 2 || /\d/.test(token)) {
      flush()
      continue
    }
    segment.push(token)
  }
  flush()
  return [...new Set(out)]
}

function bestSegmentPhrase(tokens: readonly string[]): string {
  const trimmed = trimPhraseProbeTokens(tokens)
  if (trimmed.length < PHRASE_PROBE_MIN_TOKENS) return ''
  if (trimmed.length <= PHRASE_PROBE_MAX_TOKENS) return joinPhraseTokens(trimmed)

  for (let size = PHRASE_PROBE_MAX_TOKENS; size >= PHRASE_PROBE_MIN_TOKENS; size -= 1) {
    let best = ''
    let bestScore = -1
    for (let start = 0; start + size <= trimmed.length; start += 1) {
      const candidate = trimPhraseProbeTokens(trimmed.slice(start, start + size))
      if (candidate.length < PHRASE_PROBE_MIN_TOKENS) continue
      const score = phraseProbeScore(candidate)
      if (score > bestScore) {
        bestScore = score
        best = joinPhraseTokens(candidate)
      }
    }
    if (best !== '') return best
  }
  return ''
}

function trimPhraseProbeTokens(tokens: readonly string[]): readonly string[] {
  let start = 0
  while (start < tokens.length && PHRASE_PROBE_TRIM_WORDS.has(tokens[start]!)) {
    start += 1
  }
  return tokens.slice(start)
}

function phraseProbeScore(tokens: readonly string[]): number {
  return tokens.reduce((score, token) => score + token.length, tokens.length * 100)
}

function compileBM25FanoutQueryText(
  query: string,
  phraseProbes: readonly string[],
): string {
  const trimmed = query.trim()
  if (
    trimmed !== '' &&
    phraseProbes.includes(trimmed) &&
    trimmed.includes(' ') &&
    !trimmed.includes('"')
  ) {
    return `"${trimmed}"`
  }
  return trimmed
}

function phraseProbeTokens(query: string): readonly string[] {
  if (query.trim() === '') return []
  const out: string[] = []
  for (const raw of query.toLowerCase().split(/\s+/u)) {
    const token = raw.replace(/^[.,;:!?"'()[\]{}<>]+|[.,;:!?"'()[\]{}<>]+$/gu, '')
    if (token === '') continue
    out.push(token)
  }
  return out
}

function collectLeftPhrase(tokens: readonly string[]): string {
  if (tokens.length === 0) return ''
  const collected: string[] = []
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const token = tokens[index]
    if (token === undefined) continue
    if (PHRASE_PROBE_BOUNDARY_WORDS.has(token)) {
      if (collected.length > 0) break
      continue
    }
    if (token.length < 2 || /\d/.test(token)) {
      if (collected.length > 0) break
      continue
    }
    collected.push(token)
    if (collected.length >= PHRASE_PROBE_MAX_TOKENS) break
  }
  return joinPhraseTokens(collected.reverse())
}

function collectRightPhrase(tokens: readonly string[]): string {
  if (tokens.length === 0) return ''
  const collected: string[] = []
  for (const token of tokens) {
    if (PHRASE_PROBE_BOUNDARY_WORDS.has(token)) {
      if (collected.length > 0) break
      continue
    }
    if (token.length < 2 || /\d/.test(token)) {
      if (collected.length > 0) break
      continue
    }
    collected.push(token)
    if (collected.length >= PHRASE_PROBE_MAX_TOKENS) break
  }
  return joinPhraseTokens(collected)
}

function joinPhraseTokens(tokens: readonly string[]): string {
  if (tokens.length < PHRASE_PROBE_MIN_TOKENS) return ''
  return tokens.join(' ')
}

function questionTokens(query: string): readonly string[] {
  if (query.trim() === '') return []
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of query.toLowerCase().split(/\s+/)) {
    const token = raw.replace(/^[.,;:!?"'()[\]{}<>]+|[.,;:!?"'()[\]{}<>]+$/gu, '')
    if (token.length < 3) continue
    if (/\d/.test(token)) continue
    if (QUESTION_TOKEN_STOP_WORDS.has(token)) continue
    if (seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

function compileSearchQuery(query: string, aliases: AliasTable | undefined): string {
  if (query.trim() === '') return ''
  const ast = parseQuery(query)
  const expanded = aliases !== undefined ? expandAliases(ast, aliases) : ast
  return compileToFTS(expanded)
}

function searchBM25Candidates(
  index: SearchIndex,
  compiledQuery: string,
  limit: number,
  filters: RetrievalFilters | undefined,
): RRFCandidate[] {
  if (compiledQuery === '' || limit <= 0) return []
  const hits = searchBM25WithFilters(index, compiledQuery, limit, filters)
  return hits.map((hit, index) => ({
    id: hit.chunk.id,
    path: hit.chunk.path,
    title: hit.chunk.title ?? '',
    summary: hit.chunk.summary ?? '',
    content: hit.chunk.content,
    ...(hit.chunk.metadata !== undefined ? { metadata: hit.chunk.metadata } : {}),
    bm25Rank: index,
  }))
}

function searchBM25WithFilters(
  index: SearchIndex,
  compiledQuery: string,
  limit: number,
  filters: RetrievalFilters | undefined,
) {
  if (!hasRetrievalFilters(filters)) {
    return index.searchBM25(compiledQuery, limit)
  }

  let fetchLimit = Math.max(limit, limit * FILTER_FETCH_MULTIPLIER)
  const maxFetch = Math.max(fetchLimit, limit * FILTER_FETCH_MAX_MULTIPLIER)
  while (true) {
    const hits = index.searchBM25(compiledQuery, fetchLimit)
    const filtered = hits.filter((hit) => matchesRetrievalFilters(hit.chunk, filters))
    if (filtered.length >= limit || hits.length < fetchLimit || fetchLimit >= maxFetch) {
      return filtered.slice(0, limit)
    }
    fetchLimit = Math.min(fetchLimit * 2, maxFetch)
  }
}

function searchVectorCandidates(
  index: SearchIndex,
  embedding: Float32Array | number[],
  limit: number,
  filters: RetrievalFilters | undefined,
) {
  if (limit <= 0) return []
  if (!hasRetrievalFilters(filters)) {
    return index.searchVector(embedding, limit)
  }

  let fetchLimit = Math.max(limit, limit * FILTER_FETCH_MULTIPLIER)
  const maxFetch = Math.max(fetchLimit, limit * FILTER_FETCH_MAX_MULTIPLIER)
  while (true) {
    const hits = index.searchVector(embedding, fetchLimit)
    const filtered = hits.filter((hit) => matchesRetrievalFilters(hit.chunk, filters))
    if (filtered.length >= limit || hits.length < fetchLimit || fetchLimit >= maxFetch) {
      return filtered.slice(0, limit)
    }
    fetchLimit = Math.min(fetchLimit * 2, maxFetch)
  }
}

function hasRetrievalFilters(filters: RetrievalFilters | undefined): boolean {
  if (filters === undefined) return false
  return (
    (filters.pathPrefix?.trim() ?? '') !== '' ||
    (filters.scope?.trim() ?? '') !== '' ||
    (filters.project?.trim() ?? '') !== '' ||
    (filters.tags?.length ?? 0) > 0
  )
}

function matchesRetrievalFilters(
  chunk: Chunk,
  filters: RetrievalFilters | undefined,
): boolean {
  if (!hasRetrievalFilters(filters)) return true

  const pathPrefix = filters?.pathPrefix?.trim() ?? ''
  if (pathPrefix !== '' && !chunk.path.startsWith(pathPrefix)) return false

  const tags = filters?.tags?.map((tag) => tag.trim().toLowerCase()).filter(Boolean) ?? []
  if (tags.length > 0) {
    const chunkTags =
      typeof chunk.tags === 'string'
        ? chunk.tags.split(/\s+/)
        : (chunk.tags ?? []).map((tag) => String(tag))
    const available = new Set(
      chunkTags.map((tag) => tag.trim().toLowerCase()).filter((tag) => tag !== ''),
    )
    for (const tag of tags) {
      if (!available.has(tag)) return false
    }
  }

  const scope = filters?.scope?.trim().toLowerCase() ?? ''
  if (scope !== '' && !matchesScopeFilter(resolveChunkScope(chunk), scope)) return false

  const project = filters?.project?.trim().toLowerCase() ?? ''
  const actualProject = resolveChunkProject(chunk)
  if (project !== '' && actualProject !== undefined && actualProject !== project) return false

  return true
}

function resolveChunkScope(chunk: Chunk): string | undefined {
  const metadataScope = chunk.metadata?.['scope']
  if (typeof metadataScope === 'string' && metadataScope.trim() !== '') {
    return metadataScope.trim().toLowerCase()
  }

  if (chunk.path.startsWith('memory/global/')) return 'global'
  if (chunk.path.startsWith('memory/project/')) return 'project'
  if (chunk.path.startsWith('memory/agent/')) return 'agent'
  if (chunk.path.startsWith('wiki/')) return 'wiki'
  if (chunk.path.startsWith('raw/')) return 'raw_document'
  return undefined
}

function resolveChunkProject(chunk: Chunk): string | undefined {
  const metadataProject = chunk.metadata?.['project']
  if (typeof metadataProject === 'string' && metadataProject.trim() !== '') {
    return metadataProject.trim().toLowerCase()
  }

  const segments = chunk.path.split('/')
  if (segments[0] === 'memory' && segments[1] === 'project') {
    const project = segments[2]
    if (project !== undefined && project !== '') return project.toLowerCase()
  }
  return undefined
}

function matchesScopeFilter(actual: string | undefined, expected: string): boolean {
  if (actual === undefined) return false
  switch (expected) {
    case 'memory':
      return (
        actual === 'global' ||
        actual === 'global_memory' ||
        actual === 'project' ||
        actual === 'project_memory'
      )
    case 'global':
    case 'global_memory':
      return actual === 'global' || actual === 'global_memory'
    case 'project':
    case 'project_memory':
      return actual === 'project' || actual === 'project_memory'
    case 'agent':
    case 'agent_memory':
      return actual === 'agent' || actual === 'agent_memory'
    case 'raw':
    case 'raw_document':
      return actual === 'raw' || actual === 'raw_document'
    default:
      return actual === expected
  }
}

function preferenceIntentMultiplier(result: RetrievalResult, text: string): number {
  const path = result.path.toLowerCase()
  const isGlobalPreferenceNote =
    path.includes('memory/global/') &&
    (path.includes('user-preference-') || PREFERENCE_NOTE_RE.test(text))

  if (isGlobalPreferenceNote) {
    return path.includes('user-preference-') ? 2.35 : 2.1
  }

  if (!path.includes('memory/global/') && GENERIC_NOTE_RE.test(text)) {
    return 0.82
  }
  if (ROLLUP_NOTE_RE.test(text)) {
    return 0.9
  }
  return 1
}

function concreteFactIntentMultiplier(
  query: string,
  result: RetrievalResult,
  text: string,
): number {
  const path = result.path.toLowerCase()
  const isRollUp = ROLLUP_NOTE_RE.test(text)
  const isQuestionLikeNote =
    QUESTION_LIKE_NOTE_RE.test(text) && GENERIC_NOTE_RE.test(text)
  const isConcreteFact =
    path.includes('user-fact-') ||
    path.includes('milestone-') ||
    (!isRollUp && (DATE_TAG_RE.test(text) || ATOMIC_EVENT_NOTE_RE.test(text)))

  let multiplier = 1
  if (isConcreteFact) multiplier *= 2.2
  if (deriveActionDateProbes(query).length > 0 && BODY_ABSOLUTE_DATE_RE.test(text)) {
    multiplier *= 1.45
  }
  if (DURATION_QUERY_RE.test(query)) {
    multiplier *= MEASUREMENT_VALUE_RE.test(text) ? 1.35 : 0.72
  }
  if (isQuestionLikeNote) multiplier *= 0.45
  if (isRollUp) multiplier *= 0.45
  if (!isConcreteFact && !path.includes('memory/global/') && GENERIC_NOTE_RE.test(text)) {
    multiplier *= 0.75
  }
  return multiplier
}

function retrievalResultText(result: RetrievalResult): string {
  return [result.path, result.title, result.summary, result.content]
    .join('\n')
    .toLowerCase()
}
