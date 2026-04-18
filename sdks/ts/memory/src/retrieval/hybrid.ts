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
  compileToFTS,
  expandAliases,
  parseQuery,
} from '../query/index.js'
import type { Reranker } from '../rerank/index.js'
import { unanimityShortcut } from '../rerank/llm-rerank.js'
import type { SearchIndex } from '../search/index.js'
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
  RetrievalRequest,
  RetrievalResponse,
  RetrievalResult,
  RetryAttempt,
} from './types.js'

const DEFAULT_TOP_K = 10
const DEFAULT_CANDIDATE_K = 60
const DEFAULT_RERANK_TOP_N = 20
const RERANK_SNIPPET_MAX = 280
const PREFERENCE_QUERY_RE =
  /\b(?:recommend|suggest|recommendation|suggestion|tips?|advice|ideas?|what should i|which should i)\b/i
const ENUMERATION_OR_TOTAL_QUERY_RE =
  /\b(?:how many|count|total|in total|sum|add up|list|what are all)\b/i
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

type RetrievalIntent = {
  readonly preferenceQuery: boolean
  readonly concreteFactQuery: boolean
}

export type CreateRetrievalOptions = {
  index: SearchIndex
  embedder?: Embedder
  reranker?: Reranker
  aliases?: AliasTable
  logger?: Logger
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
    const rerankEnabled = req.rerank ?? true

    const ast = parseQuery(req.query)
    const expanded = opts.aliases !== undefined ? expandAliases(ast, opts.aliases) : ast
    const compiled = compileToFTS(expanded)

    const embedderReady = opts.embedder !== undefined

    const requestedMode: HybridMode = req.mode ?? 'auto'
    let mode: HybridMode = requestedMode
    let fellBackToBM25 = false
    if (!embedderReady && (mode === 'auto' || mode === 'hybrid' || mode === 'semantic')) {
      mode = 'bm25'
      if (requestedMode !== 'auto') fellBackToBM25 = true
    } else if (mode === 'auto') {
      mode = 'hybrid'
    }

    const trace: HybridTrace = {
      mode,
      originalQuery: req.query,
      compiledQuery: compiled,
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
      attempts: [],
    }
    const attempts: RetryAttempt[] = []

    const runBM25 = (query: string): RRFCandidate[] => {
      if (query === '') return []
      const ast = parseQuery(query)
      const expanded = opts.aliases !== undefined ? expandAliases(ast, opts.aliases) : ast
      const compiled = compileToFTS(expanded)
      if (compiled === '') return []
      const hits = opts.index.searchBM25(compiled, candidateK)
      return hits.map((h, idx) => ({
        id: h.chunk.id,
        path: h.chunk.path,
        title: h.chunk.title ?? '',
        summary: h.chunk.summary ?? '',
        content: h.chunk.content,
        bm25Rank: idx,
      }))
    }

    // ---- BM25 leg (with retry ladder on zero hits) ----
    const bmStart = Date.now()
    let bmCandidates: RRFCandidate[] = []
    let bmError: unknown

    try {
      bmCandidates = runBM25(compiled)
      attempts.push({ strategy: 'initial', query: compiled, hits: bmCandidates.length })

      if (bmCandidates.length === 0 && req.skipRetryLadder !== true) {
        // Rung 1: strongest term.
        const strongest = strongestTerm(req.query)
        if (strongest !== undefined && strongest !== req.query.trim().toLowerCase()) {
          bmCandidates = runBM25(strongest)
          attempts.push({
            strategy: 'strongest_term',
            query: strongest,
            hits: bmCandidates.length,
          })
        }

        if (bmCandidates.length === 0) {
          // Rung 2: force-refresh (no-op) + rung 3: refreshed sanitised.
          forceRefreshIndex()
          const sanitised = sanitiseQuery(req.query)
          if (sanitised !== '') {
            bmCandidates = runBM25(sanitised)
            attempts.push({
              strategy: 'refreshed_sanitised',
              query: sanitised,
              hits: bmCandidates.length,
            })
          }
        }

        if (bmCandidates.length === 0) {
          // Rung 4: refreshed strongest term.
          const strongest = strongestTerm(sanitiseQuery(req.query))
          if (strongest !== undefined) {
            bmCandidates = runBM25(strongest)
            attempts.push({
              strategy: 'refreshed_strongest',
              query: strongest,
              hits: bmCandidates.length,
            })
          }
        }

        if (bmCandidates.length === 0) {
          // Rung 5: trigram fuzzy fallback.
          const tokens = queryTokens(req.query)
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

    if (opts.embedder !== undefined && (mode === 'hybrid' || mode === 'semantic')) {
      try {
        const vectors = await opts.embedder.embed([req.query], req.signal)
        const first = vectors[0]
        if (first !== undefined && first.length > 0) {
          trace.embedderUsed = true
          const hits = opts.index.searchVector(first, candidateK)
          vecCandidates = hits.map((h, idx) => ({
            id: h.chunk.id,
            path: h.chunk.path,
            title: h.chunk.title ?? '',
            summary: h.chunk.summary ?? '',
            content: h.chunk.content,
            vectorSimilarity: h.similarity,
            bm25Rank: idx,
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
    fused = reweightSharedMemoryRanking(req.query, fused)
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
              query: req.query,
              documents: head.map((r) => ({
                id: r.id,
                text: composeRerankText(r),
              })),
            },
            req.signal,
          )
          const reordered: RetrievalResult[] = []
          for (const hit of reranked) {
            const src = head[hit.index]
            if (src === undefined) continue
            reordered.push({ ...src, rerankScore: hit.score })
          }
          final = reordered.concat(tail)
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

    if (final.length > topK) final = final.slice(0, topK)

    trace.attempts = attempts
    trace.totalElapsed = Date.now() - started
    return { results: final, trace }
  }

  return {
    search: async (req) => (await runSearchRaw(req)).results,
    searchRaw: runSearchRaw,
  }
}

function toResult(c: RRFCandidate, score: number): RetrievalResult {
  return {
    id: c.id,
    path: c.path,
    title: c.title ?? '',
    summary: c.summary ?? '',
    content: c.content ?? '',
    score,
    ...(c.bm25Rank !== undefined ? { bm25Rank: c.bm25Rank } : {}),
    ...(c.vectorSimilarity !== undefined ? { vectorSimilarity: c.vectorSimilarity } : {}),
  }
}

function composeRerankText(r: RetrievalResult): string {
  const title = (r.title ?? '').trim()
  const summary = (r.summary ?? '').trim()
  if (title !== '' && summary !== '') return `${title}\n${summary}`
  if (title !== '') return title
  if (summary !== '') return summary
  const body = (r.content ?? '').replace(/\s+/g, ' ').trim()
  return body.length <= RERANK_SNIPPET_MAX ? body : `${body.slice(0, RERANK_SNIPPET_MAX)}...`
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
        score: result.score * retrievalIntentMultiplier(intent, result),
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

function detectRetrievalIntent(query: string): RetrievalIntent {
  const normalised = query.toLowerCase()
  return {
    preferenceQuery: PREFERENCE_QUERY_RE.test(normalised),
    concreteFactQuery:
      ENUMERATION_OR_TOTAL_QUERY_RE.test(normalised) ||
      (FIRST_PERSON_FACT_LOOKUP_RE.test(normalised) &&
        FACT_LOOKUP_VERB_RE.test(normalised)),
  }
}

function retrievalIntentMultiplier(
  intent: RetrievalIntent,
  result: RetrievalResult,
): number {
  let multiplier = 1
  const text = retrievalResultText(result)
  if (intent.preferenceQuery) {
    multiplier *= preferenceIntentMultiplier(result, text)
  }
  if (intent.concreteFactQuery) {
    multiplier *= concreteFactIntentMultiplier(result, text)
  }
  return multiplier
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

function concreteFactIntentMultiplier(result: RetrievalResult, text: string): number {
  const path = result.path.toLowerCase()
  const isRollUp = ROLLUP_NOTE_RE.test(text)
  const isConcreteFact =
    path.includes('user-fact-') ||
    path.includes('milestone-') ||
    (!isRollUp && (DATE_TAG_RE.test(text) || ATOMIC_EVENT_NOTE_RE.test(text)))

  let multiplier = 1
  if (isConcreteFact) multiplier *= 2.2
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
