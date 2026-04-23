import { compileToFts, expandAliases, parseQuery } from '../query/index.js'
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

export const createRetrieval = (options: CreateRetrievalOptions): Retrieval => {
  let trigramIndex: ReturnType<typeof buildTrigramIndex> | null | undefined

  const getTrigramIndex = (): ReturnType<typeof buildTrigramIndex> | null => {
    if (trigramIndex !== undefined) return trigramIndex
    const chunks = options.index.indexedChunks?.() ?? []
    trigramIndex = chunks.length === 0 ? null : buildTrigramIndex(chunks)
    return trigramIndex
  }

  const searchRaw = async (request: RetrievalRequest): Promise<RetrievalResponse> => {
    const topK = request.topK ?? DEFAULT_TOP_K
    const candidateK = request.candidateK ?? DEFAULT_CANDIDATE_K
    const requestedMode = request.mode ?? options.defaultMode ?? 'auto'
    const mode = resolveMode(requestedMode, options.embedder)

    const parsed = parseQuery(request.query)
    const expanded = options.aliases === undefined ? parsed : expandAliases(parsed, options.aliases)
    const compiledQuery = compileToFts(expanded)

    const runBm25 = (expr: string) =>
      options.index
        .searchBm25Compiled(expr, candidateK)
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

    let fused = fuseResults(mode, bm25Results, vectorResults, options.rrfK ?? RRF_DEFAULT_K)
    const attempts: RetryAttempt[] = [
      {
        strategy: 'initial' as const,
        query: compiledQuery,
        hits: fused.length,
      },
    ]

    if (fused.length === 0 && mode !== 'semantic' && request.skipRetryLadder !== true) {
      const strongest = strongestTerm(request.query)
      if (strongest !== undefined) {
        const strongestExpr = compileToFts(parseQuery(strongest))
        const strongestResults = runBm25(strongestExpr)
        attempts.push({
          strategy: 'strongest_term',
          query: strongestExpr,
          hits: strongestResults.length,
        })
        if (strongestResults.length > 0) {
          fused = strongestResults.map((result, index) => toBm25Result(result, index))
        }
      }

      if (fused.length === 0) {
        const sanitised = sanitiseQuery(request.query)
        const sanitisedExpr = compileToFts(parseQuery(sanitised))
        if (sanitisedExpr !== '' && sanitisedExpr !== compiledQuery) {
          const sanitisedResults = runBm25(sanitisedExpr)
          attempts.push({
            strategy: 'sanitised',
            query: sanitisedExpr,
            hits: sanitisedResults.length,
          })
          if (sanitisedResults.length > 0) {
            fused = sanitisedResults.map((result, index) => toBm25Result(result, index))
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

    return {
      results: fused.slice(0, topK),
      trace: {
        mode,
        originalQuery: request.query,
        compiledQuery,
        candidateK,
        rrfK: options.rrfK ?? RRF_DEFAULT_K,
        bm25Count: bm25Results.length,
        vectorCount: vectorResults.length,
        fusedCount: fused.length,
        embedderUsed: options.embedder !== undefined,
        filtersApplied: request.filters !== undefined,
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
): RetrievalResult => ({
  id: result.chunk.id,
  path: result.chunk.path,
  title: result.chunk.title ?? '',
  summary: result.chunk.summary ?? '',
  content: result.chunk.content,
  ...(result.chunk.metadata === undefined ? {} : { metadata: result.chunk.metadata }),
  score: 1 / (RRF_DEFAULT_K + index + 1),
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
): RetrievalResult => ({
  id: result.chunk.id,
  path: result.chunk.path,
  title: result.chunk.title ?? '',
  summary: result.chunk.summary ?? '',
  content: result.chunk.content,
  ...(result.chunk.metadata === undefined ? {} : { metadata: result.chunk.metadata }),
  score: 1 / (RRF_DEFAULT_K + index + 1),
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
    return bm25Results.map((result, index) => toBm25Result(result, index))
  }
  if (mode === 'semantic') {
    return vectorResults.map((result, index) => toVectorResult(result, index))
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
  if ((requestedMode === 'semantic' || requestedMode === 'hybrid') && embedder === undefined) {
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
  return true
}
