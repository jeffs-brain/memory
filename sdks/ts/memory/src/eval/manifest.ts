import type { LoadedDataset } from './dataset.js'
import { createLMECacheKey, type LMECacheKeyInput } from './cache-key.js'
import type { IngestMode } from './types.js'
import type {
  LMEOfficialRepoFetchResult,
  LMEUpstreamBundleName,
  LMEUpstreamDatasetName,
  LMEUpstreamFetchMetadata,
} from './upstream.js'

export type LMERunConfig = {
  readonly ingestMode: IngestMode
  readonly retrievalMode: 'bm25' | 'semantic' | 'hybrid'
  readonly rerank: boolean
  readonly topK: number
  readonly candidateK: number
  readonly rerankTopN: number
  readonly judgeConcurrency: number
  readonly ingestConcurrency: number
  readonly embeddingBatchSize: number
  readonly readerBudgetChars: number
  readonly questionCategories?: readonly string[]
}

export type LMESampleManifest = {
  readonly size: number
  readonly seed: number
  readonly signature: string
}

export type LMEModelManifest = {
  readonly extractProvider?: string
  readonly extractModel?: string
  readonly readerProvider?: string
  readonly readerModel?: string
  readonly judgeProvider?: string
  readonly judgeModel?: string
  readonly embedder?: string
  readonly reranker?: string
}

export type LMECacheManifest = {
  readonly key: string
  readonly brainRoot: string
  readonly brainCacheHit: boolean
  readonly embedCachePath?: string
}

export type LMEDataManifest = {
  readonly bundle: LMEUpstreamBundleName
  readonly split: LMEUpstreamDatasetName
  readonly path: string
  readonly sha256: string
  readonly examples: number
  readonly categories: readonly string[]
  readonly sample?: LMESampleManifest
  readonly upstream?: LMEUpstreamFetchMetadata
}

export type LMEOutputManifest = {
  readonly runDir: string
  readonly reportPath?: string
  readonly manifestPath?: string
  readonly predictionsPath?: string
  readonly officialEvalLogPath?: string
}

export type LMEManifest = {
  readonly version: 1
  readonly runId: string
  readonly createdAt: string
  readonly dataset: LMEDataManifest
  readonly run: LMERunConfig
  readonly models: LMEModelManifest
  readonly cache: LMECacheManifest
  readonly officialRepo?: LMEOfficialRepoFetchResult
  readonly outputs: LMEOutputManifest
}

export type BuildLMEManifestArgs = {
  readonly runId: string
  readonly dataset: LoadedDataset
  readonly datasetPath: string
  readonly bundle: LMEUpstreamBundleName
  readonly split: LMEUpstreamDatasetName
  readonly run: LMERunConfig
  readonly models: LMEModelManifest
  readonly cache: Omit<LMECacheManifest, 'key'>
  readonly outputs: LMEOutputManifest
  readonly sample?: LMESampleManifest
  readonly upstream?: LMEUpstreamFetchMetadata
  readonly officialRepo?: LMEOfficialRepoFetchResult
  readonly createdAt?: Date
}

export const buildLMEManifest = (
  args: BuildLMEManifestArgs,
): LMEManifest => {
  const key = createLMECacheKey(manifestCacheKeyInput(args))
  return {
    version: 1,
    runId: args.runId,
    createdAt: (args.createdAt ?? new Date()).toISOString(),
    dataset: {
      bundle: args.bundle,
      split: args.split,
      path: args.datasetPath,
      sha256: args.dataset.sha256,
      examples: args.dataset.examples.length,
      categories: args.dataset.categories,
      ...(args.sample !== undefined ? { sample: args.sample } : {}),
      ...(args.upstream !== undefined ? { upstream: args.upstream } : {}),
    },
    run: args.run,
    models: args.models,
    cache: {
      key,
      brainRoot: args.cache.brainRoot,
      brainCacheHit: args.cache.brainCacheHit,
      ...(args.cache.embedCachePath !== undefined
        ? { embedCachePath: args.cache.embedCachePath }
        : {}),
    },
    ...(args.officialRepo !== undefined ? { officialRepo: args.officialRepo } : {}),
    outputs: args.outputs,
  }
}

export const manifestCacheKeyInput = (
  args: Pick<BuildLMEManifestArgs, 'bundle' | 'split' | 'dataset' | 'run' | 'models' | 'sample'>,
): LMECacheKeyInput => ({
  datasetName: args.split,
  datasetSha256: args.dataset.sha256,
  ...(args.sample !== undefined ? { sampleSignature: args.sample.signature } : {}),
  ingestMode: args.run.ingestMode,
  datasetBundle: args.bundle,
  retrievalMode: args.run.retrievalMode,
  rerank: args.run.rerank,
  topK: args.run.topK,
  candidateK: args.run.candidateK,
  rerankTopN: args.run.rerankTopN,
  ingestConcurrency: args.run.ingestConcurrency,
  judgeConcurrency: args.run.judgeConcurrency,
  embeddingBatchSize: args.run.embeddingBatchSize,
  readerBudgetChars: args.run.readerBudgetChars,
  ...(args.models.extractModel !== undefined
    ? { extractModel: args.models.extractModel }
    : {}),
  ...(args.models.readerModel !== undefined
    ? { readerModel: args.models.readerModel }
    : {}),
  ...(args.models.judgeModel !== undefined
    ? { judgeModel: args.models.judgeModel }
    : {}),
  ...(args.models.embedder !== undefined ? { embedder: args.models.embedder } : {}),
  ...(args.models.reranker !== undefined ? { reranker: args.models.reranker } : {}),
})
