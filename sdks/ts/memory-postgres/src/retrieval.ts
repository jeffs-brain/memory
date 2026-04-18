import type { Embedder } from '@jeffs-brain/memory/llm'
import { augmentQueryWithTemporal, type AliasTable } from '@jeffs-brain/memory/query'
import type { Reranker } from '@jeffs-brain/memory/rerank'
import {
  RRF_DEFAULT_K,
  buildTrigramIndex,
  forceRefreshIndex,
  queryTokens,
  reciprocalRankFusion,
  sanitiseQuery as retrySanitiseQuery,
  strongestTerm as retryStrongestTerm,
  type TrigramIndex,
  type TrigramSourceChunk,
} from '@jeffs-brain/memory/retrieval'
import {
  createPostgresSearchIndex,
  type PostgresBM25Result,
  type PostgresChunk,
  type PostgresVectorResult,
} from './search.js'
import type { PgSql } from './store.js'

const DEFAULT_CANDIDATE_LIMIT = 60
const DEFAULT_RERANK_TOP_N = 20
const DEFAULT_UNANIMITY_AGREE_MIN = 2
const RERANK_SNIPPET_MAX = 280

export type PostgresRetrievalMode = 'lexical' | 'semantic' | 'hybrid'
export type PostgresRetrievalSort = 'relevance' | 'recency' | 'relevance_then_recency'
export type PostgresRetrievalPrimitive = string | number | boolean | null

export type PostgresRetrievalFilterOperator = {
  readonly eq?: PostgresRetrievalPrimitive | undefined
  readonly ne?: PostgresRetrievalPrimitive | undefined
  readonly in?: readonly PostgresRetrievalPrimitive[] | undefined
  readonly nin?: readonly PostgresRetrievalPrimitive[] | undefined
  readonly gt?: string | number | undefined
  readonly gte?: string | number | undefined
  readonly lt?: string | number | undefined
  readonly lte?: string | number | undefined
}

export type PostgresRetrievalFilter =
  | PostgresRetrievalPrimitive
  | PostgresRetrievalFilterOperator

export type PostgresRetrievalFilters = Readonly<Record<string, PostgresRetrievalFilter>>

export type PostgresRetrievalScoreMap = {
  readonly lexical?: number
  readonly semantic?: number
  readonly rerank?: number
}

type InternalCandidate = {
  readonly chunkId: bigint
  readonly chunkKey: string
  readonly documentId: string
  readonly path: string
  readonly text: string
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
  readonly lexicalScore?: number
  readonly semanticScore?: number
  readonly bm25Rank?: number
}

export type PostgresRetrievedChunk = {
  readonly chunkId: string
  readonly documentId: string
  readonly path: string
  readonly score: number
  readonly text: string
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
  readonly componentScores?: PostgresRetrievalScoreMap
}

export type PostgresRetrievalAttempt = {
  readonly strategy:
    | 'initial'
    | 'strongest_term'
    | 'refreshed_sanitised'
    | 'refreshed_strongest'
    | 'trigram_fuzzy'
  readonly query: string
  readonly hits: number
}

export type PostgresRetrievalTrace = {
  readonly requestedMode: PostgresRetrievalMode
  readonly effectiveMode: PostgresRetrievalMode
  readonly fellBackToLexical: boolean
  readonly reranked: boolean
  readonly reranker: string | null
  readonly attempts: readonly PostgresRetrievalAttempt[]
  readonly lexicalHits: number
  readonly semanticHits: number
  readonly filteredOut: number
  readonly returned: number
  readonly sort: PostgresRetrievalSort
  readonly filtersApplied: boolean
  readonly includeSuperseded: boolean
}

export type PostgresRetrievalResponse = {
  readonly chunks: readonly PostgresRetrievedChunk[]
  readonly tookMs: number
  readonly trace: PostgresRetrievalTrace
}

export type PostgresRetrievalRequest = {
  readonly query: string
  readonly questionDate?: string
  readonly limit: number
  readonly mode?: PostgresRetrievalMode
  readonly sort?: PostgresRetrievalSort
  readonly filters?: PostgresRetrievalFilters
  readonly includeSuperseded?: boolean
  readonly rerank?: boolean
  readonly candidateLimit?: number
  readonly rerankTopN?: number
  readonly documentIdAllowlist?: ReadonlySet<string>
}

export type PostgresSearchLike = {
  readonly searchBM25: (
    query: string,
    limit: number,
  ) => Promise<readonly PostgresBM25Result[]>
  readonly hasEmbeddings?: () => Promise<boolean>
  readonly searchVector: (
    embedding: Float32Array | number[],
    limit: number,
  ) => Promise<readonly PostgresVectorResult[]>
  readonly getTrigramChunks?: () => Promise<readonly TrigramSourceChunk[]>
  readonly getChunk: (chunkId: bigint) => Promise<PostgresChunk | undefined>
}

export type PostgresRetriever = {
  readonly retrieve: (
    req: PostgresRetrievalRequest,
  ) => Promise<PostgresRetrievalResponse>
}

export type PostgresRetrieverFactoryInput = {
  readonly pg: PgSql
  readonly tenantId: string
  readonly brainId: string
  readonly env: NodeJS.ProcessEnv
}

export type PostgresRetrieverFactory = (
  input: PostgresRetrieverFactoryInput,
) => PostgresRetriever

export type PostgresEmbedderFactory = (env: NodeJS.ProcessEnv) => Embedder

export type PostgresRerankerFactory = (env: NodeJS.ProcessEnv) => Reranker | undefined

export type CreatePostgresRetrieverOptions = PostgresRetrieverFactoryInput & {
  readonly aliases?: AliasTable
  readonly indexFactory?: (input: {
    readonly pg: PgSql
    readonly tenantId: string
    readonly brainId: string
    readonly aliases?: AliasTable
  }) => PostgresSearchLike
  readonly embedderFactory?: PostgresEmbedderFactory
  readonly rerankerFactory?: PostgresRerankerFactory
}

const isDefined = <T>(value: T | undefined): value is T => value !== undefined

const toScoreMap = (input: {
  readonly lexical: number | undefined
  readonly semantic: number | undefined
  readonly rerank: number | undefined
}): PostgresRetrievalScoreMap | undefined => {
  const scores: {
    lexical?: number
    semantic?: number
    rerank?: number
  } = {}
  if (input.lexical !== undefined) scores.lexical = input.lexical
  if (input.semantic !== undefined) scores.semantic = input.semantic
  if (input.rerank !== undefined) scores.rerank = input.rerank
  return Object.keys(scores).length === 0 ? undefined : scores
}

const toRetrievedChunk = (input: {
  readonly chunkId: string
  readonly documentId: string
  readonly path: string
  readonly score: number
  readonly text: string
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
  readonly componentScores?: PostgresRetrievalScoreMap
}): PostgresRetrievedChunk => ({
  chunkId: input.chunkId,
  documentId: input.documentId,
  path: input.path,
  score: input.score,
  text: input.text,
  ...(input.metadata !== undefined ? { metadata: input.metadata } : {}),
  ...(input.componentScores !== undefined
    ? { componentScores: input.componentScores }
    : {}),
})

const composeRerankText = (text: string): string => {
  const body = text.replace(/\s+/g, ' ').trim()
  if (body.length <= RERANK_SNIPPET_MAX) return body
  return `${body.slice(0, RERANK_SNIPPET_MAX)}...`
}

const hasFilters = (filters: PostgresRetrievalFilters | undefined): boolean =>
  filters !== undefined && Object.keys(filters).length > 0

const isOperatorObject = (
  value: PostgresRetrievalFilter,
): value is PostgresRetrievalFilterOperator =>
  value !== null && typeof value === 'object' && !Array.isArray(value)

const compareFilterValue = (
  left: PostgresRetrievalPrimitive | undefined,
  right: string | number,
): number | undefined => {
  if (left === undefined || left === null) return undefined
  if (typeof left === 'number' && typeof right === 'number') return left - right
  if (typeof left === 'string' && typeof right === 'string') {
    return left.localeCompare(right)
  }
  return undefined
}

const matchesFilterField = (
  actual: PostgresRetrievalPrimitive | undefined,
  filter: PostgresRetrievalFilter,
): boolean => {
  if (!isOperatorObject(filter)) return actual === filter
  if (filter.eq !== undefined && actual !== filter.eq) return false
  if (filter.ne !== undefined && actual === filter.ne) return false
  if (Array.isArray(filter.in) && !filter.in.includes(actual ?? null)) return false
  if (Array.isArray(filter.nin) && filter.nin.includes(actual ?? null)) return false
  if (filter.gt !== undefined) {
    const result = compareFilterValue(actual, filter.gt)
    if (result === undefined || result <= 0) return false
  }
  if (filter.gte !== undefined) {
    const result = compareFilterValue(actual, filter.gte)
    if (result === undefined || result < 0) return false
  }
  if (filter.lt !== undefined) {
    const result = compareFilterValue(actual, filter.lt)
    if (result === undefined || result >= 0) return false
  }
  if (filter.lte !== undefined) {
    const result = compareFilterValue(actual, filter.lte)
    if (result === undefined || result > 0) return false
  }
  return true
}

const matchesRetrievalFilters = (
  metadata: Readonly<Record<string, string | number | boolean | null>> | undefined,
  filters: PostgresRetrievalFilters | undefined,
): boolean => {
  if (filters === undefined || Object.keys(filters).length === 0) return true
  return Object.entries(filters).every(([key, filter]) =>
    matchesFilterField(metadata?.[key], filter),
  )
}

const isSupersededChunk = (
  metadata: Readonly<Record<string, string | number | boolean | null>> | undefined,
): boolean =>
  (typeof metadata?.superseded_by === 'string' && metadata.superseded_by.trim() !== '') ||
  (typeof metadata?.supersededBy === 'string' && metadata.supersededBy.trim() !== '')

const metadataTimestamp = (
  metadata: Readonly<Record<string, string | number | boolean | null>> | undefined,
): number => {
  const raw = metadata?.modified ?? metadata?.created
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw
  if (typeof raw === 'string') {
    const parsed = Date.parse(raw)
    if (Number.isFinite(parsed)) return parsed
  }
  return Number.NEGATIVE_INFINITY
}

const compareChunkKeys = (left: string, right: string): number =>
  left === right ? 0 : left < right ? -1 : 1

const compareDesc = (left: number, right: number): number =>
  left === right ? 0 : right > left ? 1 : -1

const sortRetrievedChunks = (
  chunks: readonly PostgresRetrievedChunk[],
  sort: PostgresRetrievalSort,
): PostgresRetrievedChunk[] => {
  if (sort === 'relevance') return [...chunks]
  return [...chunks].sort((left, right) => {
    const leftTime = metadataTimestamp(left.metadata)
    const rightTime = metadataTimestamp(right.metadata)
    if (sort === 'recency') {
      const timeDiff = compareDesc(leftTime, rightTime)
      if (timeDiff !== 0) return timeDiff
      const scoreDiff = compareDesc(left.score, right.score)
      if (scoreDiff !== 0) return scoreDiff
      return compareChunkKeys(left.chunkId, right.chunkId)
    }
    const scoreDiff = compareDesc(left.score, right.score)
    if (scoreDiff !== 0) return scoreDiff
    const timeDiff = compareDesc(leftTime, rightTime)
    if (timeDiff !== 0) return timeDiff
    return compareChunkKeys(left.chunkId, right.chunkId)
  })
}

const unanimityShortcut = (
  bm25: readonly InternalCandidate[],
  vector: readonly InternalCandidate[],
  agreeMin: number = DEFAULT_UNANIMITY_AGREE_MIN,
): boolean => {
  const window = 3
  if (bm25.length < window || vector.length < window) return false
  let agreements = 0
  for (let i = 0; i < window; i++) {
    const left = bm25[i]
    const right = vector[i]
    if (left !== undefined && right !== undefined && left.chunkKey === right.chunkKey) {
      agreements++
    }
  }
  return agreements >= agreeMin
}

const defaultIndexFactory = (input: {
  readonly pg: PgSql
  readonly tenantId: string
  readonly brainId: string
  readonly aliases?: AliasTable
}): PostgresSearchLike =>
  createPostgresSearchIndex({
    sql: input.pg,
    tenantId: input.tenantId,
    brainId: input.brainId,
    ...(input.aliases !== undefined ? { aliases: input.aliases } : {}),
  })

const sameQuery = (left: string, right: string): boolean =>
  left.trim().toLowerCase() === right.trim().toLowerCase()

export const createPostgresRetriever = (
  opts: CreatePostgresRetrieverOptions,
): PostgresRetriever => {
  const index = (opts.indexFactory ?? defaultIndexFactory)({
    pg: opts.pg,
    tenantId: opts.tenantId,
    brainId: opts.brainId,
    ...(opts.aliases !== undefined ? { aliases: opts.aliases } : {}),
  })
  const embedderFactory = opts.embedderFactory
  const rerankerFactory = opts.rerankerFactory
  let trigramIndex: TrigramIndex | undefined
  let trigramIndexLoaded = false

  const ensureTrigramIndex = async (): Promise<TrigramIndex | undefined> => {
    if (trigramIndexLoaded) return trigramIndex
    trigramIndexLoaded = true
    try {
      const chunks = await index.getTrigramChunks?.()
      if (chunks === undefined || chunks.length === 0) return undefined
      trigramIndex = buildTrigramIndex(chunks)
      return trigramIndex
    } catch {
      return undefined
    }
  }

  const hydrate = async (
    hits: readonly (
      | { readonly chunkId: bigint; readonly score: number; readonly kind: 'lexical'; readonly rank: number }
      | { readonly chunkId: bigint; readonly score: number; readonly kind: 'semantic'; readonly rank: number }
    )[],
    documentIdAllowlist?: ReadonlySet<string>,
  ): Promise<InternalCandidate[]> => {
    const rows = await Promise.all(
      hits.map(async (hit) => {
        const chunk = await index.getChunk(hit.chunkId)
        if (chunk === undefined) return undefined
        if (documentIdAllowlist !== undefined && !documentIdAllowlist.has(chunk.documentId)) {
          return undefined
        }
        return {
          chunkId: chunk.chunkId,
          chunkKey: chunk.chunkId.toString(),
          documentId: chunk.documentId,
          path: chunk.path ?? chunk.documentId,
          text: chunk.content,
          ...(chunk.metadata !== undefined ? { metadata: chunk.metadata } : {}),
          ...(hit.kind === 'lexical'
            ? {
                lexicalScore: hit.score,
                bm25Rank: hit.rank,
              }
            : {
                semanticScore: hit.score,
              }),
        } satisfies InternalCandidate
      }),
    )
    return rows.filter(isDefined)
  }

  return {
    retrieve: async (req): Promise<PostgresRetrievalResponse> => {
      const startedAt = Date.now()
      const requestedMode = req.mode ?? 'hybrid'
      const sort = req.sort ?? 'relevance'
      const filtersApplied = hasFilters(req.filters)
      const includeSuperseded = req.includeSuperseded === true
      const lexicalQuery = augmentQueryWithTemporal(req.query, req.questionDate)
      let candidateLimit = Math.max(req.limit, req.candidateLimit ?? DEFAULT_CANDIDATE_LIMIT)
      if (filtersApplied || !includeSuperseded || sort !== 'relevance') {
        candidateLimit = Math.max(candidateLimit, req.limit * 8)
      }
      const rerankTopN = req.rerankTopN ?? DEFAULT_RERANK_TOP_N

      let effectiveMode: PostgresRetrievalMode = requestedMode
      let fellBackToLexical = false
      let embedding: number[] | undefined
      const attempts: PostgresRetrievalAttempt[] = []

      if (requestedMode !== 'lexical') {
        try {
          const hasEmbeddings = await index.hasEmbeddings?.()
          if (hasEmbeddings === false) {
            effectiveMode = 'lexical'
            fellBackToLexical = true
          }
        } catch {
          effectiveMode = 'lexical'
          fellBackToLexical = true
        }
      }

      if (effectiveMode !== 'lexical') {
        try {
          if (embedderFactory === undefined) throw new Error('embedder not configured')
          const embedder = embedderFactory(opts.env)
          const vectors = await embedder.embed([req.query])
          const first = vectors[0]
          if (first !== undefined && first.length > 0) {
            embedding = first
          } else {
            effectiveMode = 'lexical'
            fellBackToLexical = true
          }
        } catch {
          effectiveMode = 'lexical'
          fellBackToLexical = true
        }
      }

      const runBm25WithRetry = async (): Promise<readonly PostgresBM25Result[]> => {
        if (effectiveMode === 'semantic') return []

        const runAttempt = async (
          strategy: PostgresRetrievalAttempt['strategy'],
          query: string,
        ): Promise<readonly PostgresBM25Result[]> => {
          const hits = await index.searchBM25(query, candidateLimit)
          attempts.push({ strategy, query, hits: hits.length })
          return hits
        }

        let hits = await runAttempt('initial', lexicalQuery)
        if (hits.length > 0) return hits

        const strongest = retryStrongestTerm(lexicalQuery)
        if (strongest !== undefined && !sameQuery(strongest, lexicalQuery)) {
          hits = await runAttempt('strongest_term', strongest)
          if (hits.length > 0) return hits
        }

        forceRefreshIndex()
        const sanitised = retrySanitiseQuery(lexicalQuery)
        if (sanitised !== '' && !sameQuery(sanitised, lexicalQuery)) {
          hits = await runAttempt('refreshed_sanitised', sanitised)
          if (hits.length > 0) return hits
        }

        const strongestSanitised = retryStrongestTerm(sanitised)
        if (
          strongestSanitised !== undefined &&
          !sameQuery(strongestSanitised, lexicalQuery) &&
          !sameQuery(strongestSanitised, sanitised)
        ) {
          hits = await runAttempt('refreshed_strongest', strongestSanitised)
          if (hits.length > 0) return hits
        }

        const trigramTokens = queryTokens(lexicalQuery)
        if (trigramTokens.length > 0) {
          const trigram = await ensureTrigramIndex()
          if (trigram !== undefined) {
            const fuzzy = trigram.search(trigramTokens, candidateLimit)
            attempts.push({
              strategy: 'trigram_fuzzy',
              query: trigramTokens.join(' '),
              hits: fuzzy.length,
            })
            return fuzzy.map((hit) => ({
              chunkId: BigInt(hit.id),
              score: hit.similarity,
            }))
          }
        }

        return hits
      }

      const bm25Promise = runBm25WithRetry()
      const vectorPromise =
        effectiveMode === 'lexical' || embedding === undefined
          ? Promise.resolve([] as const)
          : index.searchVector(embedding, candidateLimit)

      const [bm25Hits, vectorHits] = await Promise.all([bm25Promise, vectorPromise])
      const [bm25Candidates, vectorCandidates] = await Promise.all([
        hydrate(
          bm25Hits.map((hit, rank) => ({
            chunkId: hit.chunkId,
            score: hit.score,
            kind: 'lexical' as const,
            rank,
          })),
          req.documentIdAllowlist,
        ),
        hydrate(
          vectorHits.map((hit, rank) => ({
            chunkId: hit.chunkId,
            score: hit.score,
            kind: 'semantic' as const,
            rank,
            })),
          req.documentIdAllowlist,
        ),
      ])
      const lexicalHits = bm25Candidates.length
      const semanticHits = vectorCandidates.length

      const merged = new Map<string, InternalCandidate>()
      for (const candidate of [...bm25Candidates, ...vectorCandidates]) {
        const existing = merged.get(candidate.chunkKey)
        if (existing === undefined) {
          merged.set(candidate.chunkKey, candidate)
          continue
        }
        merged.set(candidate.chunkKey, {
          ...existing,
          ...(existing.text === '' && candidate.text !== '' ? { text: candidate.text } : {}),
          ...(existing.lexicalScore === undefined && candidate.lexicalScore !== undefined
            ? {
                lexicalScore: candidate.lexicalScore,
                bm25Rank: candidate.bm25Rank,
              }
            : {}),
          ...(existing.semanticScore === undefined && candidate.semanticScore !== undefined
            ? { semanticScore: candidate.semanticScore }
            : {}),
        })
      }

      const fusedBase = reciprocalRankFusion(
        [
          bm25Candidates.map((candidate) => ({
            id: candidate.chunkKey,
            path: candidate.path,
            content: candidate.text,
            ...(candidate.bm25Rank !== undefined ? { bm25Rank: candidate.bm25Rank } : {}),
          })),
          vectorCandidates.map((candidate) => ({
            id: candidate.chunkKey,
            path: candidate.path,
            content: candidate.text,
            ...(candidate.semanticScore !== undefined
              ? { vectorSimilarity: candidate.semanticScore }
              : {}),
          })),
        ].filter((list) => list.length > 0),
        RRF_DEFAULT_K,
      )

      let chunks = fusedBase
        .map((entry) => {
          const source = merged.get(entry.id)
          if (source === undefined) return undefined
          const componentScores = toScoreMap({
            lexical: source.lexicalScore,
            semantic: source.semanticScore,
            rerank: undefined,
          })
          return toRetrievedChunk({
            chunkId: source.chunkKey,
            documentId: source.documentId,
            path: source.path,
            score: entry.score,
            text: source.text,
            ...(source.metadata !== undefined ? { metadata: source.metadata } : {}),
            ...(componentScores !== undefined ? { componentScores } : {}),
          })
        })
        .filter(isDefined)

      const unfilteredCount = chunks.length
      chunks = chunks.filter(
        (chunk) =>
          matchesRetrievalFilters(chunk.metadata, req.filters) &&
          (includeSuperseded || !isSupersededChunk(chunk.metadata)),
      )
      const filteredOut = unfilteredCount - chunks.length

      let reranked = false
      let rerankerName: string | null = null
      const reranker =
        req.rerank === false || rerankerFactory === undefined ? undefined : rerankerFactory(opts.env)
      if (
        reranker !== undefined &&
        chunks.length > 0 &&
        !unanimityShortcut(bm25Candidates, vectorCandidates)
      ) {
        const head = chunks.slice(0, Math.min(rerankTopN, chunks.length))
        const tail = chunks.slice(head.length)
        try {
          const rerankedHead = await reranker.rerank({
            query: req.query,
            documents: head.map((chunk) => ({
              id: chunk.chunkId,
              text: composeRerankText(chunk.text),
            })),
          })
          chunks = rerankedHead
            .map((hit) => {
              const source = head[hit.index]
              if (source === undefined) return undefined
              const componentScores = toScoreMap({
                lexical: source.componentScores?.lexical,
                semantic: source.componentScores?.semantic,
                rerank: hit.score,
              })
              return toRetrievedChunk({
                ...source,
                ...(componentScores !== undefined ? { componentScores } : {}),
              })
            })
            .filter(isDefined)
            .concat(tail)
          reranked = true
          rerankerName = reranker.name()
        } catch {
          reranked = false
          rerankerName = null
        }
      }

      chunks = sortRetrievedChunks(chunks, sort)

      if (chunks.length > req.limit) {
        chunks = chunks.slice(0, req.limit)
      }

      return {
        chunks,
        tookMs: Date.now() - startedAt,
        trace: {
          requestedMode,
          effectiveMode,
          fellBackToLexical,
          reranked,
          reranker: rerankerName,
          attempts,
          lexicalHits,
          semanticHits,
          filteredOut,
          returned: chunks.length,
          sort,
          filtersApplied,
          includeSuperseded,
        },
      }
    },
  }
}
