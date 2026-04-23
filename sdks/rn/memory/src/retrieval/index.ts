import { compileToFts, expandAliases, parseQuery } from '../query/index.js'
import { composeLLMRerankDocument, unanimityShortcut } from '../rerank/llm-rerank.js'
import { buildTrigramIndex, queryTokens, sanitiseQuery, strongestTerm } from './retry.js'
import { RRF_DEFAULT_K, reciprocalRankFusion } from './rrf.js'
import type {
  CreateRetrievalOptions,
  HybridMode,
  Retrieval,
  RetrievalFilters,
  RetrievalRequest,
  RetrievalResponse,
  RetrievalResult,
  RetryAttempt,
} from './types.js'

export type {
  CreateRetrievalOptions,
  HybridMode,
  HybridTrace,
  Retrieval,
  RetrievalFilters,
  RetrievalRequest,
  RetrievalResponse,
  RetrievalResult,
  RetryAttempt,
} from './types.js'
export { reciprocalRankFusion, RRF_DEFAULT_K } from './rrf.js'
export {
  buildTrigramIndex,
  computeTrigrams,
  queryTokens,
  sanitiseQuery,
  slugTextFor,
  strongestTerm,
  type TrigramChunk,
  type TrigramHit,
  type TrigramIndex,
} from './retry.js'

const DEFAULT_TOP_K = 10
const DEFAULT_CANDIDATE_K = 60
const DEFAULT_RERANK_TOP_N = 20

export const createRetrieval = (options: CreateRetrievalOptions): Retrieval => {
  let trigramIndex: ReturnType<typeof buildTrigramIndex> | null | undefined

  const getTrigramIndex = (): ReturnType<typeof buildTrigramIndex> | null => {
    if (trigramIndex !== undefined) return trigramIndex
    const chunks = options.index.indexedChunks?.() ?? []
    trigramIndex = chunks.length === 0 ? null : buildTrigramIndex(chunks)
    return trigramIndex
  }

  const searchRaw = async (request: RetrievalRequest): Promise<RetrievalResponse> => {
    const started = Date.now()
    const topK = request.topK ?? DEFAULT_TOP_K
    const candidateK = request.candidateK ?? DEFAULT_CANDIDATE_K
    const rrfK = options.rrfK ?? RRF_DEFAULT_K
    const rerankTopN = request.rerankTopN ?? DEFAULT_RERANK_TOP_N
    const requestedMode = request.mode ?? options.defaultMode ?? 'auto'
    const mode = resolveMode(requestedMode, options.embedder)

    const parsed = parseQuery(request.query)
    const expanded = options.aliases === undefined ? parsed : expandAliases(parsed, options.aliases)
    const compiledQuery = compileToFts(expanded)

    const runBm25 = (expression: string) =>
      options.index
        .searchBm25Compiled(expression, candidateK)
        .filter((result) => matchesFilters(result.chunk, request.filters))

    const bm25Results = mode === 'semantic' ? [] : runBm25(compiledQuery)

    let vectorResults: ReadonlyArray<{
      readonly chunk: {
        readonly id: string
        readonly path: string
        readonly title?: string
        readonly summary?: string
        readonly content: string
        readonly metadata?: Readonly<Record<string, unknown>>
      }
      readonly similarity: number
    }> = []

    if (mode !== 'bm25' && options.embedder !== undefined) {
      const embeddings = await options.embedder.embed([request.query], request.signal)
      const embedding = embeddings[0]
      if (embedding !== undefined) {
        vectorResults = options.index
          .searchVector(embedding, candidateK)
          .filter((result) => matchesFilters(result.chunk, request.filters))
      }
    }

    let fused = fuseResults(mode, bm25Results, vectorResults, rrfK)
    const attempts: RetryAttempt[] = [
      { strategy: 'initial', query: compiledQuery, hits: fused.length },
    ]

    if (fused.length === 0 && mode !== 'semantic' && request.skipRetryLadder !== true) {
      const strongest = strongestTerm(request.query)
      if (strongest !== undefined) {
        const strongestExpression = compileToFts(parseQuery(strongest))
        const strongestResults = runBm25(strongestExpression)
        attempts.push({
          strategy: 'strongest_term',
          query: strongestExpression,
          hits: strongestResults.length,
        })
        if (strongestResults.length > 0) {
          fused = strongestResults.map((result, index) => toBm25Result(result, index, rrfK))
        }
      }

      if (fused.length === 0) {
        const sanitised = sanitiseQuery(request.query)
        const sanitisedExpression = compileToFts(parseQuery(sanitised))
        if (sanitisedExpression !== '' && sanitisedExpression !== compiledQuery) {
          const sanitisedResults = runBm25(sanitisedExpression)
          attempts.push({
            strategy: 'sanitised',
            query: sanitisedExpression,
            hits: sanitisedResults.length,
          })
          if (sanitisedResults.length > 0) {
            fused = sanitisedResults.map((result, index) => toBm25Result(result, index, rrfK))
          }
        }
      }

      if (fused.length === 0) {
        const trigramTokens = queryTokens(request.query)
        const fuzzy = getTrigramIndex()?.search(trigramTokens, candidateK, request.filters) ?? []
        attempts.push({
          strategy: 'trigram_fuzzy',
          query: trigramTokens.join(' '),
          hits: fuzzy.length,
        })
        if (fuzzy.length > 0) {
          fused = fuzzy.map((result) => ({
            id: result.id,
            path: result.path,
            title: result.title ?? '',
            summary: result.summary ?? '',
            content: result.content,
            ...(result.metadata === undefined ? {} : { metadata: result.metadata }),
            score: result.similarity,
          }))
        }
      }
    }

    let final = fused
    let reranked = false
    let rerankElapsed = 0
    let rerankSkippedReason:
      | 'unanimity'
      | 'no_reranker'
      | 'empty_candidates'
      | 'mode_off'
      | undefined
    let rerankProvider: string | undefined
    let unanimity: { readonly agreements: number } | undefined
    const rerankEnabled = request.rerank ?? mode === 'hybrid-rerank'

    if (fused.length === 0) {
      rerankSkippedReason = 'empty_candidates'
    } else if (!rerankEnabled) {
      rerankSkippedReason = 'mode_off'
    } else if (options.reranker === undefined) {
      rerankSkippedReason = 'no_reranker'
    } else {
      const shortcut = unanimityShortcut(
        bm25Results.map((result) => ({ id: result.chunk.id })),
        vectorResults.map((result) => ({ id: result.chunk.id })),
      )
      if (shortcut !== undefined) {
        rerankSkippedReason = 'unanimity'
        unanimity = { agreements: shortcut.agreements }
      } else {
        const rerankStarted = Date.now()
        try {
          const headCount = Math.min(rerankTopN, fused.length)
          const head = fused.slice(0, headCount)
          const tail = fused.slice(headCount)
          const rerankedResults = await options.reranker.rerank(
            {
              query: request.query,
              documents: head.map((result, index) => ({
                id: result.id,
                text: composeLLMRerankDocument({
                  id: index,
                  path: result.path,
                  title: result.title,
                  summary: result.summary,
                  content: result.content,
                }),
              })),
            },
            request.signal,
          )

          const reordered: Array<
            RetrievalResult & { readonly rerankScore: number; readonly originalIndex: number }
          > = []
          for (const entry of rerankedResults) {
            const source = head[entry.index]
            if (source === undefined) continue
            reordered.push({
              ...source,
              rerankScore: entry.score,
              originalIndex: entry.index,
            })
          }

          reordered.sort((left, right) => {
            if (left.rerankScore !== right.rerankScore) return right.rerankScore - left.rerankScore
            if (left.score !== right.score) return right.score - left.score
            return left.originalIndex - right.originalIndex
          })

          const rerankedHead: RetrievalResult[] = reordered.map(
            ({ originalIndex: _originalIndex, ...result }) => ({
              ...result,
              score: result.rerankScore ?? result.score,
            }),
          )
          final = [...rerankedHead, ...tail]
          reranked = true
          rerankProvider = options.reranker.name()
        } catch {
          rerankProvider = options.reranker.name()
        }
        rerankElapsed = Date.now() - rerankStarted
      }
    }

    return {
      results: final.slice(0, topK),
      trace: {
        mode,
        originalQuery: request.query,
        compiledQuery,
        candidateK,
        rrfK,
        rerankElapsed,
        totalElapsed: Date.now() - started,
        bm25Count: bm25Results.length,
        vectorCount: vectorResults.length,
        fusedCount: final.length,
        reranked,
        embedderUsed: options.embedder !== undefined,
        filtersApplied: request.filters !== undefined,
        ...(rerankSkippedReason === undefined ? {} : { rerankSkippedReason }),
        ...(rerankProvider === undefined ? {} : { rerankProvider }),
        ...(unanimity === undefined ? {} : { unanimity }),
        attempts,
      },
    }
  }

  return {
    search: async (request) => (await searchRaw(request)).results,
    searchRaw,
  }
}

const toBm25Result = (
  result: {
    readonly chunk: {
      readonly id: string
      readonly path: string
      readonly title?: string
      readonly summary?: string
      readonly content: string
      readonly metadata?: Readonly<Record<string, unknown>>
    }
    readonly score: number
  },
  index: number,
  rrfK: number,
): RetrievalResult => ({
  id: result.chunk.id,
  path: result.chunk.path,
  title: result.chunk.title ?? '',
  summary: result.chunk.summary ?? '',
  content: result.chunk.content,
  ...(result.chunk.metadata === undefined ? {} : { metadata: result.chunk.metadata }),
  score: 1 / (rrfK + index + 1),
  bm25Rank: result.score,
})

const toVectorResult = (
  result: {
    readonly chunk: {
      readonly id: string
      readonly path: string
      readonly title?: string
      readonly summary?: string
      readonly content: string
      readonly metadata?: Readonly<Record<string, unknown>>
    }
    readonly similarity: number
  },
  index: number,
  rrfK: number,
): RetrievalResult => ({
  id: result.chunk.id,
  path: result.chunk.path,
  title: result.chunk.title ?? '',
  summary: result.chunk.summary ?? '',
  content: result.chunk.content,
  ...(result.chunk.metadata === undefined ? {} : { metadata: result.chunk.metadata }),
  score: 1 / (rrfK + index + 1),
  vectorSimilarity: result.similarity,
})

const fuseResults = (
  mode: HybridMode,
  bm25Results: ReadonlyArray<{
    readonly chunk: {
      readonly id: string
      readonly path: string
      readonly title?: string
      readonly summary?: string
      readonly content: string
      readonly metadata?: Readonly<Record<string, unknown>>
    }
    readonly score: number
  }>,
  vectorResults: ReadonlyArray<{
    readonly chunk: {
      readonly id: string
      readonly path: string
      readonly title?: string
      readonly summary?: string
      readonly content: string
      readonly metadata?: Readonly<Record<string, unknown>>
    }
    readonly similarity: number
  }>,
  rrfK: number,
): RetrievalResult[] => {
  if (mode === 'bm25') {
    return bm25Results.map((result, index) => toBm25Result(result, index, rrfK))
  }
  if (mode === 'semantic') {
    return vectorResults.map((result, index) => toVectorResult(result, index, rrfK))
  }

  return reciprocalRankFusion(
    [
      bm25Results.map((result) => ({
        id: result.chunk.id,
        path: result.chunk.path,
        title: result.chunk.title ?? '',
        summary: result.chunk.summary ?? '',
        content: result.chunk.content,
        ...(result.chunk.metadata === undefined ? {} : { metadata: result.chunk.metadata }),
        bm25Rank: result.score,
      })),
      vectorResults.map((result) => ({
        id: result.chunk.id,
        path: result.chunk.path,
        title: result.chunk.title ?? '',
        summary: result.chunk.summary ?? '',
        content: result.chunk.content,
        ...(result.chunk.metadata === undefined ? {} : { metadata: result.chunk.metadata }),
        vectorSimilarity: result.similarity,
      })),
    ],
    rrfK,
  )
}

const resolveMode = (
  requestedMode: HybridMode,
  embedder: CreateRetrievalOptions['embedder'],
): HybridMode => {
  if (requestedMode === 'auto') {
    return embedder === undefined ? 'bm25' : 'hybrid'
  }
  if (
    (requestedMode === 'semantic' ||
      requestedMode === 'hybrid' ||
      requestedMode === 'hybrid-rerank') &&
    embedder === undefined
  ) {
    return 'bm25'
  }
  return requestedMode
}

const matchesFilters = (
  chunk: {
    readonly path: string
    readonly tags?: readonly string[] | string
    readonly metadata?: Readonly<Record<string, unknown>>
  },
  filters: RetrievalFilters | undefined,
): boolean => {
  if (filters === undefined) return true
  if (
    filters.paths !== undefined &&
    filters.paths.length > 0 &&
    !filters.paths.includes(chunk.path)
  ) {
    return false
  }
  if (filters.pathPrefix !== undefined && !chunk.path.startsWith(filters.pathPrefix)) {
    return false
  }

  const metadataTags = Array.isArray(chunk.metadata?.tags)
    ? chunk.metadata.tags.filter((value): value is string => typeof value === 'string')
    : typeof chunk.tags === 'string'
      ? chunk.tags.split(/\s+/)
      : (chunk.tags ?? [])

  if (filters.tags !== undefined && filters.tags.length > 0) {
    const tagSet = new Set(metadataTags)
    if (!filters.tags.every((tag) => tagSet.has(tag))) return false
  }

  if (filters.scope !== undefined) {
    const scope = typeof chunk.metadata?.scope === 'string' ? chunk.metadata.scope : undefined
    if (scope !== filters.scope) return false
  }

  if (filters.project !== undefined) {
    const project = typeof chunk.metadata?.project === 'string' ? chunk.metadata.project : undefined
    if (project !== filters.project) return false
  }

  return true
}
