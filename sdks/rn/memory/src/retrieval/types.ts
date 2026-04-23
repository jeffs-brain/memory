import type { AliasTable } from '../query/index.js'
import type { Reranker } from '../rerank/index.js'

export type HybridMode = 'auto' | 'bm25' | 'semantic' | 'hybrid' | 'hybrid-rerank'

export type RetrievalFilters = {
  readonly paths?: readonly string[]
  readonly pathPrefix?: string
  readonly tags?: readonly string[]
  readonly scope?: string
  readonly project?: string
}

export type RetrievalRequest = {
  readonly query: string
  readonly topK?: number
  readonly candidateK?: number
  readonly mode?: HybridMode
  readonly filters?: RetrievalFilters
  readonly rerank?: boolean
  readonly rerankTopN?: number
  readonly skipRetryLadder?: boolean
  readonly signal?: AbortSignal
}

export type RetrievalResult = {
  readonly id: string
  readonly path: string
  readonly title: string
  readonly summary: string
  readonly content: string
  readonly metadata?: Record<string, unknown>
  readonly score: number
  readonly bm25Rank?: number
  readonly vectorSimilarity?: number
  readonly rerankScore?: number
}

export type RetryAttempt = {
  readonly strategy:
    | 'initial'
    | 'strongest_term'
    | 'sanitised'
    | 'refreshed_sanitised'
    | 'refreshed_strongest'
    | 'trigram_fuzzy'
  readonly query: string
  readonly hits: number
}

export type HybridTrace = {
  readonly mode: HybridMode
  readonly originalQuery: string
  readonly compiledQuery: string
  readonly candidateK: number
  readonly rrfK: number
  readonly rerankElapsed: number
  readonly totalElapsed: number
  readonly bm25Count: number
  readonly vectorCount: number
  readonly fusedCount: number
  readonly reranked: boolean
  readonly embedderUsed: boolean
  readonly filtersApplied: boolean
  readonly rerankSkippedReason?: 'unanimity' | 'no_reranker' | 'empty_candidates' | 'mode_off'
  readonly rerankProvider?: string
  readonly unanimity?: { readonly agreements: number }
  readonly attempts: readonly RetryAttempt[]
}

export type RetrievalResponse = {
  readonly results: readonly RetrievalResult[]
  readonly trace: HybridTrace
}

export type Retrieval = {
  search(request: RetrievalRequest): Promise<readonly RetrievalResult[]>
  searchRaw(request: RetrievalRequest): Promise<RetrievalResponse>
}

export type RetrievalEmbedder = {
  embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]>
}

export type CreateRetrievalOptions = {
  readonly index: {
    indexedChunks?(): ReadonlyArray<{
      readonly id: string
      readonly path: string
      readonly title?: string
      readonly summary?: string
      readonly content: string
      readonly tags?: readonly string[] | string
      readonly metadata?: Readonly<Record<string, unknown>>
    }>
    searchBm25Compiled(
      expr: string,
      limit: number,
    ): ReadonlyArray<{
      readonly chunk: {
        readonly id: string
        readonly path: string
        readonly title?: string
        readonly summary?: string
        readonly content: string
        readonly tags?: readonly string[] | string
        readonly metadata?: Readonly<Record<string, unknown>>
      }
      readonly score: number
    }>
    searchVector(
      embedding: Float32Array | readonly number[],
      limit: number,
    ): ReadonlyArray<{
      readonly chunk: {
        readonly id: string
        readonly path: string
        readonly title?: string
        readonly summary?: string
        readonly content: string
        readonly tags?: readonly string[] | string
        readonly metadata?: Readonly<Record<string, unknown>>
      }
      readonly similarity: number
    }>
  }
  readonly embedder?: RetrievalEmbedder
  readonly reranker?: Reranker
  readonly aliases?: AliasTable
  readonly rrfK?: number
  readonly defaultMode?: HybridMode
}
