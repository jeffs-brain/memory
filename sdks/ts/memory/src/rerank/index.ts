// SPDX-License-Identifier: Apache-2.0

/**
 * Rerank contracts used by the retrieval pipeline. The interface is a
 * superset of the llm/Reranker shape: callers pass the query plus the
 * raw document payloads and receive sorted indices + scores back.
 * Cross-encoder and LLM backends both implement this contract.
 */

export type RerankDoc = {
  /** Stable identifier carried through for callers to re-key results. */
  id: string
  /** Text payload shipped to the reranker (e.g. `title\nsummary`). */
  text: string
}

export type RerankRequest = {
  query: string
  documents: readonly RerankDoc[]
}

export type RerankResult = {
  /** Original index into RerankRequest.documents, sorted by descending score. */
  index: number
  /** Opaque score; larger is better. */
  score: number
  /** The document id, echoed for convenience. */
  id: string
}

/**
 * Reranker is the thin contract consumed by the retrieval pipeline.
 * Implementations must be pure with respect to their inputs: the same
 * request should always produce the same ordering so the pipeline can
 * cache or log results safely.
 */
export type Reranker = {
  name(): string
  isAvailable?(signal?: AbortSignal): Promise<boolean>
  rerank(req: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]>
}

export { CrossEncoderReranker, type CrossEncoderRerankerConfig } from './crossencoder.js'
export { AutoReranker, type AutoRerankerConfig } from './auto.js'
export {
  LLMReranker,
  unanimityShortcut,
  DEFAULT_RERANK_BATCH_SIZE,
  DEFAULT_RERANK_PARALLELISM,
  DEFAULT_UNANIMITY_AGREE_MIN,
  type LLMRerankerConfig,
  type UnanimityCandidate,
  type UnanimityShortcut,
} from './llm-rerank.js'
export { runBatches, type BatchRunnerOptions } from './batch.js'
export {
  DEFAULT_SHARED_RERANK_CONCURRENCY,
  runWithSharedRerankConcurrency,
} from './concurrency.js'
