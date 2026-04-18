// SPDX-License-Identifier: Apache-2.0

/**
 * Public types for the hybrid retrieval pipeline. Mirrors the shape of
 * apps/jeff/internal/knowledge.HybridTrace + SearchTrace so consumers
 * can reason about every stage (BM25 leg, vector leg, fusion, rerank,
 * retry ladder) from a single returned trace.
 */

import type { AliasTable } from '../query/index.js'
import type { BM25Result, VectorResult } from '../search/reader.js'

/**
 * Hybrid retrieval mode selector. `auto` picks hybrid when an embedder
 * is supplied, otherwise falls back to BM25 only. The explicit modes
 * force a specific shape regardless of what is available.
 */
export type HybridMode = 'auto' | 'bm25' | 'semantic' | 'hybrid'

/**
 * Retrieval request parameters. `topK` is the final result count; every
 * other knob is optional and defaults to Jeff's production shape.
 */
export type RetrievalRequest = {
  /** The raw user query string (unsanitised, normalised downstream). */
  query: string
  /** Final result count (default 10). */
  topK?: number
  /** Per-retriever candidate slate size before fusion (default 60). */
  candidateK?: number
  /** Override mode selection (default: auto). */
  mode?: HybridMode
  /** Enable the reranker pass when a reranker is configured (default true). */
  rerank?: boolean
  /** Number of top fused candidates to rerank (default 20). */
  rerankTopN?: number
  /** Skip the retry ladder even when BM25 returns zero (default false). */
  skipRetryLadder?: boolean
  /** Abort signal propagated into embedder / rerank calls. */
  signal?: AbortSignal
}

/**
 * A single retrieval hit. `chunk` is the hydrated chunk from the index;
 * `score` is the fused RRF score (post-rerank when rerank ran). The
 * original BM25 rank and vector similarity are carried through for
 * downstream observability.
 */
export type RetrievalResult = {
  readonly id: string
  readonly path: string
  readonly title: string
  readonly summary: string
  readonly content: string
  readonly score: number
  readonly bm25Rank?: number
  readonly vectorSimilarity?: number
  readonly rerankScore?: number
}

/**
 * HybridTrace records every decision the pipeline made so --explain
 * style reporting can reconstruct the run. All elapsed fields are in
 * milliseconds (wall clock).
 */
export type HybridTrace = {
  mode: HybridMode
  originalQuery: string
  compiledQuery: string
  candidateK: number
  rrfK: number
  bm25Elapsed: number
  vectorElapsed: number
  fusionElapsed: number
  rerankElapsed: number
  totalElapsed: number
  bm25Count: number
  vectorCount: number
  fusedCount: number
  fellBackToBM25: boolean
  embedderUsed: boolean
  reranked: boolean
  rerankSkippedReason?: 'unanimity' | 'no_reranker' | 'empty_candidates' | 'mode_off'
  rerankProvider?: string
  attempts: readonly RetryAttempt[]
  unanimity?: { agreements: number }
  errorStage?: string
  errorDetail?: string
  /** True when a distiller rewrote the query before parsing. */
  usedDistill?: boolean
  /** Distilled query text when distillation ran. */
  distilledQuery?: string
  /** Elapsed wall-clock for the distill step in milliseconds. */
  distillElapsed?: number
}

/**
 * One pass through the retry ladder. The retrieval pipeline records an
 * entry per strategy tried so callers can see what actually hit the
 * index before results came back.
 */
export type RetryAttempt = {
  strategy:
    | 'initial'
    | 'strongest_term'
    | 'refreshed_sanitised'
    | 'refreshed_strongest'
    | 'trigram_fuzzy'
  query: string
  hits: number
}

/**
 * RetrievalResponse bundles the ranked hits with the trace. Returned by
 * the `searchRaw` method so callers that care about observability do
 * not need a second round-trip to reconstruct what happened.
 */
export type RetrievalResponse = {
  results: readonly RetrievalResult[]
  trace: HybridTrace
}

/** Re-export for convenience. */
export type { AliasTable, BM25Result, VectorResult }
