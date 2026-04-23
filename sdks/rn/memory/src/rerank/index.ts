export type RerankDoc = {
  readonly id: string
  readonly text: string
}

export type RerankRequest = {
  readonly query: string
  readonly documents: readonly RerankDoc[]
}

export type RerankResult = {
  readonly index: number
  readonly score: number
  readonly id: string
}

export type Reranker = {
  name(): string
  isAvailable?(signal?: AbortSignal): Promise<boolean>
  rerank(request: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]>
}

export { AutoReranker, type AutoRerankerConfig } from './auto.js'
export {
  LLMReranker,
  composeLLMRerankDocument,
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
