// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import type { IngestMode } from './types.js'
import type { LMEUpstreamBundleName, LMEUpstreamDatasetName } from './upstream.js'

export type LMECacheKeyInput = {
  readonly datasetName: LMEUpstreamDatasetName
  readonly datasetBundle: LMEUpstreamBundleName
  readonly datasetSha256: string
  readonly sampleSignature?: string
  readonly ingestMode: IngestMode
  readonly retrievalMode: 'bm25' | 'semantic' | 'hybrid' | 'hybrid-rerank'
  readonly rerank: boolean
  readonly topK: number
  readonly candidateK: number
  readonly rerankTopN: number
  readonly ingestConcurrency: number
  readonly judgeConcurrency: number
  readonly embeddingBatchSize: number
  readonly readerBudgetChars: number
  readonly extractModel?: string
  readonly readerModel?: string
  readonly judgeModel?: string
  readonly embedder?: string
  readonly reranker?: string
}

export const lmeCacheKeySeed = (input: LMECacheKeyInput): string =>
  [
    `dataset_bundle=${input.datasetBundle}`,
    `dataset_name=${input.datasetName}`,
    `dataset_sha256=${input.datasetSha256}`,
    `sample_signature=${input.sampleSignature ?? ''}`,
    `ingest_mode=${input.ingestMode}`,
    `retrieval_mode=${input.retrievalMode}`,
    `rerank=${String(input.rerank)}`,
    `top_k=${String(input.topK)}`,
    `candidate_k=${String(input.candidateK)}`,
    `rerank_top_n=${String(input.rerankTopN)}`,
    `ingest_concurrency=${String(input.ingestConcurrency)}`,
    `judge_concurrency=${String(input.judgeConcurrency)}`,
    `embedding_batch_size=${String(input.embeddingBatchSize)}`,
    `reader_budget_chars=${String(input.readerBudgetChars)}`,
    `extract_model=${input.extractModel ?? ''}`,
    `reader_model=${input.readerModel ?? ''}`,
    `judge_model=${input.judgeModel ?? ''}`,
    `embedder=${input.embedder ?? ''}`,
    `reranker=${input.reranker ?? ''}`,
  ].join('\n')

export const createLMECacheKey = (input: LMECacheKeyInput): string =>
  `lme-${createHash('sha256').update(lmeCacheKeySeed(input)).digest('hex').slice(0, 24)}`
