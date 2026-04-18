// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import type { Embedder, Logger, Provider } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import {
  createRetrieval,
  type HybridMode,
  type Retrieval as HybridRetrieval,
} from '../retrieval/index.js'
import type { Reranker } from '../rerank/index.js'
import { createSearchIndex, type Chunk as SearchChunk } from '../search/index.js'
import { createFsStore } from '../store/index.js'
import { lastSegment, pathUnder, toPath } from '../store/index.js'
import type { FileInfo, Path } from '../store/index.js'
import {
  EXTRACTION_SYSTEM_PROMPT,
  createMemory,
  createStoreBackedCursorStore,
  mergeRecallHits,
  scopePrefix,
  type RecallHit,
  type Scope,
  type SearchIndex as MemorySearchIndex,
} from '../memory/index.js'
import { parseFrontmatter } from '../memory/frontmatter.js'
import { createProviderJudge } from './judge.js'
import { createProviderReader } from './read.js'
import {
  buildLMEManifest,
  type LMEManifest,
  type LMEModelManifest,
  type LMERunConfig,
  type LMESampleManifest,
} from './manifest.js'
import { compareReports, type LMEReportComparison } from './compare.js'
import { loadDataset, type LoadedDataset } from './dataset.js'
import { LMEEmbedCache } from './embed-cache.js'
import { sampleExamples } from './sample.js'
import {
  resultsToOfficialEvalLog,
  scoreOfficialEvalLog,
  serialiseOfficialEvalLog,
  serialiseOfficialHypotheses,
  type OfficialLMEScoreSummary,
} from './scorer.js'
import {
  augmentQueryWithTemporal,
  dateSearchTokens,
  parseQuestionDate,
  resolvedTemporalHintLine,
} from './temporal.js'
import type {
  IngestMode,
  IngestOutcome,
  LMEExample,
  LMEReport,
  LMEResult,
  RetrievalFn,
  RetrievedPassage,
} from './types.js'
import { createLMERunner, DEFAULT_OUT_DIR } from './index.js'
import type {
  LMEOfficialRepoFetchResult,
  LMEUpstreamBundleName,
  LMEUpstreamDatasetName,
  LMEUpstreamFetchMetadata,
} from './upstream.js'

const DEFAULT_TOP_K = 50
const DEFAULT_CANDIDATE_K = 50
const DEFAULT_RERANK_TOP_N = 20
const DEFAULT_INGEST_CONCURRENCY = 16
const DEFAULT_JUDGE_CONCURRENCY = 8
const DEFAULT_EMBEDDING_BATCH_SIZE = 8
const DEFAULT_READER_BUDGET_CHARS = 100_000
const DEFAULT_RETRIEVAL_MODE: HybridMode = 'bm25'
const REPLAY_PIPELINE_VERSION = 'replay-v12'
const SEARCH_FETCH_MULTIPLIER = 4
const MAX_SCOPE_FILTER_FETCH = 512
const REPLAY_RECALL_SCOPES: readonly Scope[] = ['project', 'global']

export type StandaloneLMERunArgs = {
  readonly datasetPath: string
  readonly bundle: LMEUpstreamBundleName
  readonly split: LMEUpstreamDatasetName
  readonly ingestMode?: IngestMode
  readonly retrievalMode?: 'bm25' | 'semantic' | 'hybrid'
  readonly rerank?: boolean
  readonly topK?: number
  readonly candidateK?: number
  readonly rerankTopN?: number
  readonly sampleSize?: number
  readonly seed?: number
  readonly ingestConcurrency?: number
  readonly judgeConcurrency?: number
  readonly embeddingBatchSize?: number
  readonly readerBudgetChars?: number
  readonly questionCategories?: readonly string[]
  readonly outDir?: string
  readonly cacheDir?: string
  readonly runId?: string
  readonly actorId?: string
  readonly provider?: Provider
  readonly extractProvider?: Provider
  readonly readerProvider?: Provider
  readonly judgeProvider?: Provider
  readonly readerModel?: string
  readonly judgeModel?: string
  readonly embedder?: Embedder
  readonly reranker?: Reranker
  readonly logger?: Logger
  readonly upstream?: LMEUpstreamFetchMetadata
  readonly officialRepo?: LMEOfficialRepoFetchResult
  readonly compareAgainst?: LMEReport
  readonly createdAt?: Date
  readonly now?: () => Date
}

export type StandaloneLMERunOutcome = {
  readonly manifest: LMEManifest
  readonly ingest: IngestOutcome
  readonly dataset: LoadedDataset
  readonly results: readonly LMEResult[]
  readonly report: LMEReport
  readonly reportPath: string
  readonly runDir: string
  readonly manifestPath: string
  readonly officialHypothesesPath: string
  readonly officialEvalLogPath: string
  readonly officialScore: OfficialLMEScoreSummary
  readonly officialHypothesesJsonl: string
  readonly officialEvalLogJsonl: string
  readonly comparison?: LMEReportComparison
}

type PreparedDataset = {
  readonly dataset: LoadedDataset
  readonly sample: LMESampleManifest | undefined
}

type EvalRetrieval = {
  readonly retrieval: RetrievalFn
  readonly embedCachePath?: string
  close(): Promise<void>
}

type EvalSearchPipeline = {
  readonly retrieval: HybridRetrieval
  readonly chunksById: ReadonlyMap<string, SearchChunk>
  readonly embedCachePath?: string
  close(): Promise<void>
}

type EvalDocument = {
  readonly path: string
  readonly title: string
  readonly summary: string
  readonly tags: readonly string[]
  readonly content: string
  readonly metadata: Readonly<Record<string, unknown>>
}

type EvalSessionDocument = {
  readonly path: string
  readonly sessionId: string
  readonly date?: string
  readonly raw: string
}

export const runStandaloneLMEEval = async (
  args: StandaloneLMERunArgs,
): Promise<StandaloneLMERunOutcome> => {
  const now = args.now ?? (() => new Date())
  const createdAt = args.createdAt ?? now()
  const outDir = resolve(args.outDir ?? DEFAULT_OUT_DIR)
  const cacheDir = resolve(args.cacheDir ?? join(outDir, 'cache'))
  const ingestMode = args.ingestMode ?? 'replay'
  const retrievalMode = args.retrievalMode ?? DEFAULT_RETRIEVAL_MODE
  const rerank = args.rerank ?? false
  const topK = args.topK ?? DEFAULT_TOP_K
  const candidateK = args.candidateK ?? DEFAULT_CANDIDATE_K
  const rerankTopN = args.rerankTopN ?? DEFAULT_RERANK_TOP_N
  const ingestConcurrency =
    args.ingestConcurrency ??
    (ingestMode === 'replay' ? 1 : DEFAULT_INGEST_CONCURRENCY)
  const judgeConcurrency = args.judgeConcurrency ?? DEFAULT_JUDGE_CONCURRENCY
  const embeddingBatchSize =
    args.embeddingBatchSize ?? DEFAULT_EMBEDDING_BATCH_SIZE
  const readerBudgetChars =
    args.readerBudgetChars ?? DEFAULT_READER_BUDGET_CHARS
  const runId = args.runId ?? defaultRunId(createdAt)
  const actorId = args.actorId ?? 'lme'
  const logger = args.logger ?? noopLogger
  const retrievalEmbedder =
    retrievalMode === 'bm25' ? undefined : args.embedder
  const activeReranker = rerank ? args.reranker : undefined

  const preparedDataset = await loadAndSampleDataset(
    args.datasetPath,
    args.sampleSize,
    args.seed,
  )
  const dataset = preparedDataset.dataset
  const evalExamples = filterExamplesByCategory(
    dataset.examples,
    args.questionCategories,
  )
  if (evalExamples.length === 0) {
    throw new Error('runStandaloneLMEEval: no examples matched questionCategories')
  }
  const extractProvider = args.extractProvider ?? args.provider
  const readerProvider = args.readerProvider ?? args.provider
  const judgeProvider = args.judgeProvider ?? args.provider
  const extractProviderName = extractProvider?.name()
  const extractModel = extractProvider?.modelName()
  const readerProviderName = readerProvider?.name()
  const resolvedReaderModel = args.readerModel ?? readerProvider?.modelName()
  const judgeProviderName = judgeProvider?.name()
  const resolvedJudgeModel = args.judgeModel ?? judgeProvider?.modelName()
  const modelManifest: LMEModelManifest = {
    ...(extractProviderName !== undefined
      ? { extractProvider: extractProviderName }
      : {}),
    ...(extractModel !== undefined ? { extractModel } : {}),
    ...(readerProviderName !== undefined ? { readerProvider: readerProviderName } : {}),
    ...(resolvedReaderModel !== undefined ? { readerModel: resolvedReaderModel } : {}),
    ...(judgeProviderName !== undefined ? { judgeProvider: judgeProviderName } : {}),
    ...(resolvedJudgeModel !== undefined ? { judgeModel: resolvedJudgeModel } : {}),
    ...(retrievalEmbedder !== undefined
      ? { embedder: retrievalEmbedder.model() }
      : {}),
    ...(activeReranker !== undefined ? { reranker: activeReranker.name() } : {}),
  }

  const runConfig: LMERunConfig = {
    ingestMode,
    retrievalMode,
    rerank,
    topK,
    candidateK,
    rerankTopN,
    judgeConcurrency,
    ingestConcurrency,
    embeddingBatchSize,
    readerBudgetChars,
    ...(args.questionCategories !== undefined && args.questionCategories.length > 0
      ? {
          questionCategories: dedupeStrings(args.questionCategories).sort((left, right) =>
            left.localeCompare(right),
          ),
        }
      : {}),
  }

  const brainRoot = join(cacheDir, 'brains', createBrainCacheKey({
    datasetSha256: dataset.sha256,
    ...(preparedDataset.sample !== undefined
      ? { sampleSignature: preparedDataset.sample.signature }
      : {}),
    ingestMode,
    actorId,
    extractModel: modelManifest.extractModel,
  }))
  const scratchRoot = join(cacheDir, 'scratch', runId)
  const storeRoot = ingestMode === 'bulk' ? scratchRoot : brainRoot
  const brainFingerprint = createBrainFingerprint({
    datasetSha256: dataset.sha256,
    ...(preparedDataset.sample !== undefined
      ? { sampleSignature: preparedDataset.sample.signature }
      : {}),
    ingestMode,
    actorId,
    extractModel: modelManifest.extractModel,
  })
  const fingerprintPath = join(brainRoot, '.lme-brain.fingerprint')
  let brainCacheHit = false

  if (ingestMode === 'bulk') {
    await rm(storeRoot, { recursive: true, force: true })
  } else {
    brainCacheHit = (await readFingerprint(fingerprintPath)) === brainFingerprint
    if (!brainCacheHit) {
      await rm(storeRoot, { recursive: true, force: true })
    }
  }

  await mkdir(outDir, { recursive: true })
  await mkdir(cacheDir, { recursive: true })
  const store = await createFsStore({ root: storeRoot })

  let retrievalBuild: EvalRetrieval | undefined
  let retrieval: RetrievalFn | undefined
  try {
    const ingest = await ingestDataset({
      ingestMode,
      brainCacheHit,
      dataset,
      actorId,
      ingestConcurrency,
      store,
      provider: extractProvider,
      embedder: args.embedder,
      logger,
    })

    if (ingestMode !== 'bulk' && !brainCacheHit) {
      await writeFile(fingerprintPath, `${brainFingerprint}\n`, 'utf8')
    }

    if (ingestMode === 'bulk' && args.split === 'oracle') {
      retrieval = await buildOracleBulkRetrieval(store)
    } else if (ingestMode === 'replay') {
      retrievalBuild = await buildReplayRecallRetrieval({
        store,
        provider: extractProvider ?? readerProvider ?? judgeProvider,
        actorId,
        retrievalMode,
        rerank,
        topK,
        candidateK,
        rerankTopN,
        embedder: retrievalEmbedder,
        reranker: activeReranker,
        cacheDir,
        runId,
        embeddingBatchSize,
        logger,
      })
      retrieval = retrievalBuild.retrieval
    } else {
      retrievalBuild = await buildEvalRetrieval({
        store,
        corpus: ingestMode === 'bulk' ? 'raw' : 'memory',
        retrievalMode,
        rerank,
        topK,
        candidateK,
        rerankTopN,
        embedder: retrievalEmbedder,
        reranker: activeReranker,
        cacheDir,
        runId,
        embeddingBatchSize,
        logger,
      })
      retrieval = retrievalBuild.retrieval
    }

    const provider = args.provider ?? args.readerProvider ?? args.judgeProvider
    const runner = createLMERunner({
      store,
      ...(provider !== undefined ? { provider } : {}),
      ...(readerProvider !== undefined
        ? {
            reader: createProviderReader(readerProvider, {
              ...(args.readerModel !== undefined ? { model: args.readerModel } : {}),
              budgetChars: readerBudgetChars,
            }),
          }
        : {}),
      ...(judgeProvider !== undefined
        ? {
          judge: createProviderJudge({
              provider: judgeProvider,
              ...(args.judgeModel !== undefined ? { model: args.judgeModel } : {}),
            }),
          }
        : {}),
      retrieval: retrieval ?? (async () => ({ passages: [], rendered: '' })),
      outDir,
      runId,
      readerBudgetChars,
      ...(args.readerModel !== undefined ? { readerModel: args.readerModel } : {}),
      ...(args.judgeModel !== undefined ? { judgeModel: args.judgeModel } : {}),
      logger,
    })

    const results = await runner.judge({
      examples: evalExamples,
      concurrency: judgeConcurrency,
    })
    const { report, written } = await runner.report({
      ingestMode,
      results,
      datasetSha256: dataset.sha256,
      ...(ingest.warnings.length > 0 ? { errors: ingest.warnings } : {}),
    })

    const officialScore = scoreOfficialEvalLog({
      references: evalExamples,
      entries: resultsToOfficialEvalLog(
        results,
        modelManifest.judgeModel ?? 'gpt-4o-2024-08-06',
      ),
    })
    const officialHypothesesJsonl = `${serialiseOfficialHypotheses(results)}\n`
    const officialEvalLogJsonl = `${serialiseOfficialEvalLog(
      results,
      modelManifest.judgeModel ?? 'gpt-4o-2024-08-06',
    )}\n`
    const officialHypothesesPath = join(written.runDir, 'predictions.jsonl')
    const officialEvalLogPath = join(written.runDir, 'official-eval-log.jsonl')
    const manifestPath = join(written.runDir, 'manifest.json')
    await writeFile(officialHypothesesPath, officialHypothesesJsonl, 'utf8')
    await writeFile(officialEvalLogPath, officialEvalLogJsonl, 'utf8')

    const manifest = buildLMEManifest({
      runId,
      dataset,
      datasetPath: resolve(args.datasetPath),
      bundle: args.bundle,
      split: args.split,
      ...(preparedDataset.sample !== undefined
        ? { sample: preparedDataset.sample }
        : {}),
      run: runConfig,
      models: modelManifest,
      cache: {
        brainRoot,
        brainCacheHit,
        ...(retrievalBuild?.embedCachePath !== undefined
          ? { embedCachePath: retrievalBuild.embedCachePath }
          : {}),
      },
      outputs: {
        runDir: written.runDir,
        reportPath: written.reportPath,
        manifestPath,
        predictionsPath: officialHypothesesPath,
        officialEvalLogPath,
      },
      ...(args.upstream !== undefined ? { upstream: args.upstream } : {}),
      ...(args.officialRepo !== undefined ? { officialRepo: args.officialRepo } : {}),
      createdAt,
    })
    await writeFile(
      manifestPath,
      `${JSON.stringify(manifest, null, 2)}\n`,
      'utf8',
    )

    const comparison =
      args.compareAgainst !== undefined
        ? compareReports(args.compareAgainst, report)
        : undefined

    return {
      manifest,
      ingest,
      dataset,
      results,
      report,
      reportPath: written.reportPath,
      runDir: written.runDir,
      manifestPath,
      officialHypothesesPath,
      officialEvalLogPath,
      officialScore,
      officialHypothesesJsonl,
      officialEvalLogJsonl,
      ...(comparison !== undefined ? { comparison } : {}),
    }
  } finally {
    try {
      await retrievalBuild?.close()
    } finally {
      await store.close()
    }
  }
}

const loadAndSampleDataset = async (
  datasetPath: string,
  sampleSize: number | undefined,
  seed: number | undefined,
): Promise<PreparedDataset> => {
  const loaded = await loadDataset(datasetPath)
  if (
    sampleSize === undefined ||
    sampleSize <= 0 ||
    sampleSize >= loaded.examples.length
  ) {
    return {
      dataset: loaded,
      sample: undefined,
    }
  }
  const resolvedSeed = seed ?? 0
  const examples = sampleExamples({
    examples: loaded.examples,
    size: sampleSize,
    seed: resolvedSeed,
  })
  const categories = [...new Set(examples.map((example) => example.category))].sort(
    (left, right) => left.localeCompare(right),
  )
  return {
    dataset: {
      examples,
      sha256: loaded.sha256,
      categories,
    },
    sample: {
      size: sampleSize,
      seed: resolvedSeed,
      signature: createSampleSignature(examples, resolvedSeed),
    },
  }
}

const filterExamplesByCategory = (
  examples: readonly LMEExample[],
  categories: readonly string[] | undefined,
): readonly LMEExample[] => {
  if (categories === undefined || categories.length === 0) return examples
  const wanted = new Set(
    categories
      .map((category) => category.trim())
      .filter((category) => category !== ''),
  )
  if (wanted.size === 0) return examples
  return examples.filter((example) => wanted.has(example.category))
}

const ingestDataset = async (args: {
  readonly ingestMode: IngestMode
  readonly brainCacheHit: boolean
  readonly dataset: LoadedDataset
  readonly actorId: string
  readonly ingestConcurrency: number
  readonly store: Awaited<ReturnType<typeof createFsStore>>
  readonly provider: Provider | undefined
  readonly embedder: Embedder | undefined
  readonly logger: Logger
}): Promise<IngestOutcome> => {
  if (args.brainCacheHit) {
    return {
      mode: args.ingestMode,
      sessionsWritten: 0,
      examplesIngested: args.dataset.examples.length,
      warnings: [],
    }
  }

  if (args.ingestMode === 'bulk') {
    const runner = createLMERunner({
      store: args.store,
      logger: args.logger,
    })
    return runner.runBulk({ examples: args.dataset.examples })
  }

  if (args.provider === undefined) {
    throw new Error(
      `runStandaloneLMEEval: ${args.ingestMode} ingest requires a provider`,
    )
  }

  const memory = createMemory({
    store: args.store,
    provider: args.provider,
    ...(args.embedder !== undefined ? { embedder: args.embedder } : {}),
    scope: args.ingestMode === 'replay' ? 'project' : 'global',
    actorId: args.actorId,
    cursorStore: createStoreBackedCursorStore(args.store),
    ...(args.ingestMode === 'replay' ? { extractMinMessages: 2 } : {}),
    logger: args.logger,
  })
  const runner = createLMERunner({
    store: args.store,
    memory,
    logger: args.logger,
  })
  if (args.ingestMode === 'replay') {
    return runner.runReplay({
      examples: args.dataset.examples,
      concurrency: args.ingestConcurrency,
    })
  }
  return runner.runAgentic({ examples: args.dataset.examples })
}

const buildEvalRetrieval = async (args: {
  readonly store: Awaited<ReturnType<typeof createFsStore>>
  readonly corpus: 'raw' | 'memory'
  readonly retrievalMode: 'bm25' | 'semantic' | 'hybrid'
  readonly rerank: boolean
  readonly topK: number
  readonly candidateK: number
  readonly rerankTopN: number
  readonly embedder: Embedder | undefined
  readonly reranker: Reranker | undefined
  readonly cacheDir: string
  readonly runId: string
  readonly embeddingBatchSize: number
  readonly logger: Logger
}): Promise<EvalRetrieval> => {
  const search = await buildEvalSearchPipeline(args)
  if (search === undefined) {
    return {
      retrieval: async () => ({ passages: [], rendered: '' }),
      close: async () => {},
    }
  }

  return {
    retrieval: async ({ question, questionDate }) => {
      const query = augmentQueryWithTemporal(question, questionDate)
      const response = await search.retrieval.searchRaw({
        query,
        topK: args.topK,
        candidateK: args.candidateK,
        mode: toHybridMode(args.retrievalMode),
        rerank: args.rerank,
        rerankTopN: args.rerankTopN,
      })
      const passages = response.results.map((result): RetrievedPassage => {
        const metadata = (search.chunksById.get(result.id)?.metadata ?? {}) as Record<
          string,
          unknown
        >
        const date = extractDate(metadata)
        return {
          path: result.path,
          score: result.score,
          body: result.content,
          ...(date !== undefined ? { date } : {}),
          ...(typeof metadata['sessionId'] === 'string'
            ? { sessionId: metadata['sessionId'] }
            : {}),
        }
      })
      const rendered = renderRetrievedPassages(passages, {
        question,
        ...(questionDate !== undefined ? { questionDate } : {}),
      })
      return { passages, rendered }
    },
    ...(search.embedCachePath !== undefined
      ? { embedCachePath: search.embedCachePath }
      : {}),
    close: search.close,
  }
}

const buildReplayRecallRetrieval = async (args: {
  readonly store: Awaited<ReturnType<typeof createFsStore>>
  readonly provider: Provider | undefined
  readonly actorId: string
  readonly retrievalMode: 'bm25' | 'semantic' | 'hybrid'
  readonly rerank: boolean
  readonly topK: number
  readonly candidateK: number
  readonly rerankTopN: number
  readonly embedder: Embedder | undefined
  readonly reranker: Reranker | undefined
  readonly cacheDir: string
  readonly runId: string
  readonly embeddingBatchSize: number
  readonly logger: Logger
}): Promise<EvalRetrieval> => {
  if (args.provider === undefined) {
    throw new Error('buildReplayRecallRetrieval: provider is required')
  }

  const search = await buildEvalSearchPipeline({
    store: args.store,
    corpus: 'memory',
    retrievalMode: args.retrievalMode,
    rerank: args.rerank,
    topK: args.topK,
    candidateK: args.candidateK,
    rerankTopN: args.rerankTopN,
    embedder: args.embedder,
    reranker: args.reranker,
    cacheDir: args.cacheDir,
    runId: args.runId,
    embeddingBatchSize: args.embeddingBatchSize,
    logger: args.logger,
  })
  if (search === undefined) {
    return {
      retrieval: async () => ({ passages: [], rendered: '' }),
      close: async () => {},
    }
  }

  const memory = createMemory({
    store: args.store,
    provider: args.provider,
    cursorStore: createStoreBackedCursorStore(args.store),
    searchIndex: createRetrievalBackedSearchIndex({
      retrieval: search.retrieval,
      actorId: args.actorId,
      retrievalMode: args.retrievalMode,
      rerank: args.rerank,
      candidateK: args.candidateK,
      rerankTopN: args.rerankTopN,
    }),
    scope: 'project',
    actorId: args.actorId,
    logger: args.logger,
  })

  return {
    retrieval: async ({ question, questionDate }) => {
      const query = augmentQueryWithTemporal(question, questionDate)
      let recallK = Math.max(1, Math.min(args.topK, MAX_SCOPE_FILTER_FETCH))
      let hydratedPassages: readonly RetrievedPassage[] = []

      while (true) {
        const hits = await recallAcrossScopes(memory, {
          query,
          k: recallK,
          actorId: args.actorId,
        })
        hydratedPassages = await recallHitsToPassages(args.store, hits)
        const eligibleCount = countReplayEligiblePassages(
          hydratedPassages,
          questionDate,
        )
        if (
          eligibleCount >= args.topK ||
          recallK >= MAX_SCOPE_FILTER_FETCH ||
          hits.length < recallK
        ) {
          break
        }
        recallK = Math.min(recallK * 2, MAX_SCOPE_FILTER_FETCH)
      }

      const passages = selectReplayPassagesForReader(
        hydratedPassages,
        question,
        questionDate,
        args.topK,
      )
      const rendered = renderRetrievedPassages(passages, {
        question,
        ...(questionDate !== undefined ? { questionDate } : {}),
      })
      return { passages, rendered }
    },
    ...(search.embedCachePath !== undefined
      ? { embedCachePath: search.embedCachePath }
      : {}),
    close: search.close,
  }
}

const buildEvalSearchPipeline = async (args: {
  readonly store: Awaited<ReturnType<typeof createFsStore>>
  readonly corpus: 'raw' | 'memory'
  readonly retrievalMode: 'bm25' | 'semantic' | 'hybrid'
  readonly rerank: boolean
  readonly topK: number
  readonly candidateK: number
  readonly rerankTopN: number
  readonly embedder: Embedder | undefined
  readonly reranker: Reranker | undefined
  readonly cacheDir: string
  readonly runId: string
  readonly embeddingBatchSize: number
  readonly logger: Logger
}): Promise<EvalSearchPipeline | undefined> => {
  const docs = await collectEvalDocuments(args.store, args.corpus)
  if (docs.length === 0) return undefined

  const embedCachePath =
    args.embedder !== undefined
      ? join(args.cacheDir, 'embeddings', `${sanitiseFileSegment(args.embedder.model())}.sqlite`)
      : undefined
  const cache =
    embedCachePath !== undefined
      ? await LMEEmbedCache.open({ path: embedCachePath })
      : undefined
  let index: Awaited<ReturnType<typeof createSearchIndex>> | undefined

  try {
    const chunks = await buildSearchChunks({
      docs,
      embedder: args.embedder,
      cache,
      embeddingBatchSize: args.embeddingBatchSize,
    })
    const vectorDim = inferVectorDim(chunks, args.embedder)
    const indexPath = join(args.cacheDir, 'indices', `${args.runId}.sqlite`)
    await rm(indexPath, { force: true })
    await mkdir(join(args.cacheDir, 'indices'), { recursive: true })
    index = await createSearchIndex({ dbPath: indexPath, vectorDim })
    index.upsertChunks(chunks)

    const retrieval = createRetrieval({
      index,
      ...(args.embedder !== undefined ? { embedder: args.embedder } : {}),
      ...(args.reranker !== undefined ? { reranker: args.reranker } : {}),
      trigramChunks: chunks.map((chunk) => ({
        id: chunk.id,
        path: chunk.path,
        title: chunk.title ?? '',
        summary: chunk.summary ?? '',
        content: chunk.content,
      })),
      logger: args.logger,
    })

    return {
      retrieval,
      chunksById: new Map(chunks.map((chunk) => [chunk.id, chunk])),
      ...(embedCachePath !== undefined ? { embedCachePath } : {}),
      close: async () => {
        cache?.close()
        await index?.close()
      },
    }
  } catch (err) {
    cache?.close()
    await index?.close()
    throw err
  }
}

const createRetrievalBackedSearchIndex = (args: {
  readonly retrieval: HybridRetrieval
  readonly actorId: string
  readonly retrievalMode: 'bm25' | 'semantic' | 'hybrid'
  readonly rerank: boolean
  readonly candidateK: number
  readonly rerankTopN: number
}): MemorySearchIndex => ({
  search: async (query, _embedding, opts) => {
    const scope = opts.scope ?? 'project'
    const actorId = opts.actorId ?? args.actorId
    const prefix = scopePrefix(scope, actorId)
    let limit = Math.max(opts.k * SEARCH_FETCH_MULTIPLIER, opts.k)

    while (true) {
      const response = await args.retrieval.searchRaw({
        query,
        topK: limit,
        candidateK: Math.max(args.candidateK, limit),
        mode: toHybridMode(args.retrievalMode),
        rerank: args.rerank,
        rerankTopN: Math.max(args.rerankTopN, limit),
      })
      const hits = dedupeSearchHits(
        response.results
          .filter((result) => pathUnder(result.path, prefix, true))
          .map((result) => ({ path: toPath(result.path), score: result.score })),
      )
      if (hits.length >= opts.k || response.results.length < limit || limit >= MAX_SCOPE_FILTER_FETCH) {
        return hits.slice(0, opts.k)
      }
      limit = Math.min(limit * 2, MAX_SCOPE_FILTER_FETCH)
    }
  },
})

const recallAcrossScopes = async (
  memory: ReturnType<typeof createMemory>,
  args: {
    readonly query: string
    readonly k: number
    readonly actorId: string
  },
): Promise<readonly RecallHit[]> => {
  const perScope = await Promise.all(
    REPLAY_RECALL_SCOPES.map((scope) =>
      memory.recall({
        query: args.query,
        k: args.k,
        scope,
        actorId: args.actorId,
      }),
    ),
  )
  return mergeRecallHits(perScope.flat(), { query: args.query }).slice(0, args.k)
}

const recallHitsToPassages = async (
  store: Awaited<ReturnType<typeof createFsStore>>,
  hits: readonly RecallHit[],
): Promise<readonly RetrievedPassage[]> =>
  Promise.all(
    hits.map(async (hit): Promise<RetrievedPassage> => {
      const raw = (await store.read(hit.path)).toString('utf8')
      const { frontmatter } = parseFrontmatter(raw)
      const date = extractDate({
        sessionDate: frontmatter.session_date,
        observedOn: frontmatter.observed_on,
        modified: frontmatter.modified,
      })
      return {
        path: hit.path,
        score: hit.score,
        body: hit.content,
        ...(date !== undefined ? { date } : {}),
        ...(typeof frontmatter.session_id === 'string' && frontmatter.session_id !== ''
          ? { sessionId: frontmatter.session_id }
          : {}),
      }
    }),
  )

export const filterPassagesByQuestionDate = (
  passages: readonly RetrievedPassage[],
  questionDate: string | undefined,
): readonly RetrievedPassage[] => {
  if (questionDate === undefined || questionDate.trim() === '') return passages
  const anchor = parseQuestionDate(questionDate)
  if (anchor === undefined) return passages
  return passages.filter((passage) => {
    const date = passage.date?.trim() ?? ''
    if (date === '') return true
    const parsed = parseQuestionDate(date)
    if (parsed === undefined) return true
    return parsed.getTime() <= anchor.getTime()
  })
}

export const rankReplayPassagesForQuestion = (
  passages: readonly RetrievedPassage[],
  question: string,
  questionDate: string | undefined,
): readonly RetrievedPassage[] => {
  const anchor =
    questionDate !== undefined && questionDate.trim() !== ''
      ? parseQuestionDate(questionDate)
      : undefined
  if (anchor === undefined || !shouldPreferRecentPassages(question)) {
    return [...passages].sort((left, right) => right.score - left.score)
  }

  return [...passages].sort((left, right) => {
    const scoreDelta =
      replayAdjustedScore(right, anchor) - replayAdjustedScore(left, anchor)
    if (scoreDelta !== 0) return scoreDelta
    return right.score - left.score
  })
}

const shouldPreferRecentPassages = (question: string): boolean => {
  const lower = question.toLowerCase()
  return !(
    lower.includes(' before ') ||
    lower.includes(' after ') ||
    lower.includes(' between ') ||
    lower.includes(' first') ||
    lower.includes(' earlier') ||
    lower.includes(' ago') ||
    lower.includes(' compared') ||
    lower.includes(' difference')
  )
}

const replayAdjustedScore = (
  passage: RetrievedPassage,
  anchor: Date,
): number => {
  const base = passage.score
  const parsed =
    passage.date !== undefined && passage.date.trim() !== ''
      ? parseQuestionDate(passage.date)
      : undefined
  if (parsed === undefined) return base * 0.75
  const deltaDays = Math.max(
    0,
    Math.floor((anchor.getTime() - parsed.getTime()) / 86_400_000),
  )
  return base / (1 + deltaDays / 14)
}

export const selectReplayPassagesForReader = (
  passages: readonly RetrievedPassage[],
  question: string,
  questionDate: string | undefined,
  limit: number,
): readonly RetrievedPassage[] => {
  if (limit <= 0) return []
  return clusterPassagesBySession(
    rankReplayPassagesForQuestion(
      filterPassagesByQuestionDate(passages, questionDate),
      question,
      questionDate,
    ),
  ).slice(0, limit)
}

const buildOracleBulkRetrieval = async (
  store: Awaited<ReturnType<typeof createFsStore>>,
): Promise<RetrievalFn> => {
  const sessions = await collectSessionDocuments(store)
  const sessionsById = new Map(sessions.map((session) => [session.sessionId, session]))

  return async ({ example, question }) => {
    const matched = example.sessionIds
      .map((sessionId) => sessionsById.get(sessionId))
      .filter((session): session is EvalSessionDocument => session !== undefined)
      .sort((left, right) => (left.date ?? '').localeCompare(right.date ?? ''))

    if (matched.length === 0) {
      return { passages: [], rendered: '' }
    }

    const passages = matched.map(
      (session): RetrievedPassage => ({
        path: session.path,
        score: 1,
        body: session.raw,
        ...(session.date !== undefined ? { date: session.date } : {}),
        sessionId: session.sessionId,
      }),
    )
    const renderedBody = processSessionContextForQuestion(
      matched.map((session) => session.raw).join('\n\n---\n\n'),
      question,
    )
    const temporalHint = resolvedTemporalHintLine(question, example.questionDate)
    const rendered =
      temporalHint !== undefined ? `${temporalHint}\n\n${renderedBody}` : renderedBody
    return { passages, rendered }
  }
}

const collectSessionDocuments = async (
  store: Awaited<ReturnType<typeof createFsStore>>,
): Promise<readonly EvalSessionDocument[]> => {
  let entries: readonly FileInfo[]
  try {
    entries = await store.list(toPath('raw/lme'), {
      recursive: true,
      includeGenerated: false,
    })
  } catch {
    return []
  }

  const out: EvalSessionDocument[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!entry.path.endsWith('.md')) continue
    const raw = (await store.read(entry.path)).toString('utf8')
    const { frontmatter } = parseFrontmatter(raw)
    if (typeof frontmatter.session_id !== 'string' || frontmatter.session_id === '') {
      continue
    }
    out.push({
      path: entry.path,
      sessionId: frontmatter.session_id,
      ...(typeof frontmatter.session_date === 'string' && frontmatter.session_date !== ''
        ? { date: frontmatter.session_date }
        : {}),
      raw,
    })
  }
  return out
}

const collectEvalDocuments = async (
  store: Awaited<ReturnType<typeof createFsStore>>,
  corpus: 'raw' | 'memory',
): Promise<readonly EvalDocument[]> => {
  const root = corpus === 'raw' ? 'raw/lme' : 'memory'
  let entries: readonly FileInfo[]
  try {
    entries = await store.list(toPath(root), {
      recursive: true,
      includeGenerated: false,
    })
  } catch {
    return []
  }

  const docs: EvalDocument[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!entry.path.endsWith('.md')) continue
    if (corpus === 'memory' && lastSegment(entry.path) === 'MEMORY.md') continue
    const raw = (await store.read(entry.path)).toString('utf8')
    const { frontmatter, body } = parseFrontmatter(raw)
    docs.push({
      path: entry.path,
      title: frontmatter.name ?? lastSegment(entry.path),
      summary: frontmatter.description ?? '',
      tags: dedupeStrings([
        ...(frontmatter.tags ?? []),
        ...dateSearchTokens(
          typeof frontmatter.session_date === 'string'
            ? frontmatter.session_date
            : undefined,
        ),
        ...dateSearchTokens(
          typeof frontmatter.observed_on === 'string'
            ? frontmatter.observed_on
            : undefined,
        ),
        ...dateSearchTokens(
          typeof frontmatter.modified === 'string'
            ? frontmatter.modified
            : undefined,
        ),
      ]),
      content: body === '' ? raw : body,
      metadata: {
        ...(frontmatter.scope !== undefined ? { scope: frontmatter.scope } : {}),
        ...(frontmatter.type !== undefined ? { type: frontmatter.type } : {}),
        ...(frontmatter.session_id !== undefined
          ? { sessionId: frontmatter.session_id }
          : {}),
        ...(frontmatter.session_date !== undefined
          ? { sessionDate: frontmatter.session_date }
          : {}),
        ...(frontmatter.observed_on !== undefined
          ? { observedOn: frontmatter.observed_on }
          : {}),
        ...(frontmatter.modified !== undefined ? { modified: frontmatter.modified } : {}),
      },
    })
  }
  return docs
}

const buildSearchChunks = async (args: {
  readonly docs: readonly EvalDocument[]
  readonly embedder: Embedder | undefined
  readonly cache: LMEEmbedCache | undefined
  readonly embeddingBatchSize: number
}): Promise<readonly SearchChunk[]> => {
  const base: SearchChunk[] = args.docs.map((doc, index) => ({
    id: `${doc.path}#${String(index)}`,
    path: doc.path,
    ordinal: index,
    title: doc.title,
    summary: doc.summary,
    tags: doc.tags,
    content: doc.content,
    metadata: doc.metadata,
  }))
  if (args.embedder === undefined) return base

  const enriched: SearchChunk[] = [...base]
  const misses: { readonly index: number; readonly text: string }[] = []
  for (let index = 0; index < enriched.length; index++) {
    const chunk = enriched[index]
    const doc = args.docs[index]
    if (chunk === undefined) continue
    if (doc === undefined) continue
    const embedText = buildEmbeddingText(doc)
    const cached = args.cache?.get(args.embedder.model(), embedText)
    if (cached !== undefined && cached.length > 0) {
      enriched[index] = { ...chunk, embedding: cached }
      continue
    }
    misses.push({ index, text: embedText })
  }

  const batchSize = Math.max(1, args.embeddingBatchSize)
  for (let start = 0; start < misses.length; start += batchSize) {
    const batch = misses.slice(start, start + batchSize)
    const vectors = await args.embedder.embed(batch.map((item) => item.text))
    for (let offset = 0; offset < batch.length; offset++) {
      const miss = batch[offset]
      const vector = vectors[offset]
      if (miss === undefined || vector === undefined || vector.length === 0) continue
      const chunk = enriched[miss.index]
      if (chunk === undefined) continue
      args.cache?.put(args.embedder.model(), miss.text, vector)
      enriched[miss.index] = { ...chunk, embedding: vector }
    }
  }
  return enriched
}

const inferVectorDim = (
  chunks: readonly SearchChunk[],
  embedder: Embedder | undefined,
): number => {
  for (const chunk of chunks) {
    if (chunk.embedding !== undefined && chunk.embedding.length > 0) {
      return chunk.embedding.length
    }
  }
  const dim = embedder?.dimension() ?? 0
  return dim > 0 ? dim : 1024
}

const renderRetrievedPassages = (
  passages: readonly RetrievedPassage[],
  args?: { readonly question: string; readonly questionDate?: string },
): string => {
  const ordered = clusterPassagesBySession(passages)
  if (ordered.length === 0) return ''
  const parts: string[] = []
  const temporalHint =
    args !== undefined
      ? resolvedTemporalHintLine(args.question, args.questionDate)
      : undefined
  if (temporalHint !== undefined) {
    parts.push(temporalHint)
    parts.push('')
  }
  parts.push(`Retrieved facts (${String(ordered.length)}):`)
  parts.push('')
  for (const [index, passage] of ordered.entries()) {
    const labels = [`[${passage.date ?? 'unknown'}]`]
    if (passage.sessionId !== undefined) {
      labels.push(`[session=${passage.sessionId}]`)
    }
    const source = sourceTagFromPath(passage.path)
    if (source !== '') labels.push(`[${source}]`)
    parts.push(`${String(index + 1).padStart(2, ' ')}. ${labels.join(' ')}`)
    parts.push(passage.body.trim())
    parts.push('')
  }
  return parts.join('\n').trim()
}

const processSessionContextForQuestion = (
  raw: string,
  question: string,
): string => {
  const blocks = parseSessionBlocksForQuestion(raw, question)
  if (blocks.length === 0) return raw

  blocks.sort((left, right) => left.date.localeCompare(right.date))

  const parts: string[] = []
  for (const block of blocks) {
    const lines: string[] = []
    if (block.date !== '') {
      lines.push(`=== Session Date: ${block.date} ===`)
    }
    lines.push(block.filtered)
    parts.push(lines.join('\n'))
  }
  return parts.join('\n\n---\n\n')
}

type SessionBlock = {
  readonly date: string
  readonly filtered: string
}

const parseSessionBlocksForQuestion = (
  content: string,
  question: string,
): SessionBlock[] =>
  splitOnSessionBoundary(content)
    .map((part) => part.trim())
    .filter((part) => part !== '')
    .map((part) => ({
      date: firstFrontmatterValue(part, 'session_date'),
      filtered: filterAssistantTurnsForQuestion(part, question),
    }))

const splitOnSessionBoundary = (content: string): readonly string[] => {
  const bySessionId = content.split('\n\n---\nsession_id:')
  if (bySessionId.length > 1) {
    return bySessionId.map((part, index) =>
      index === 0 ? part : `---\nsession_id:${part}`,
    )
  }
  const byFrontmatter = content.split('\n\n---\n')
  if (byFrontmatter.length > 1) {
    return byFrontmatter.map((part, index) => (index === 0 ? part : `---\n${part}`))
  }
  return [content]
}

const firstFrontmatterValue = (content: string, key: string): string => {
  const lines = content.split('\n')
  let inFrontmatter = false
  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === '---') {
      if (inFrontmatter) break
      inFrontmatter = true
      continue
    }
    if (!inFrontmatter) continue
    const prefix = `${key}:`
    if (trimmed.startsWith(prefix)) {
      return trimmed.slice(prefix.length).trim()
    }
  }
  return ''
}

const filterAssistantTurnsForQuestion = (
  content: string,
  question: string,
): string => {
  const lines = content.split('\n')
  const userLines: string[] = []
  const assistantChunks: string[] = []
  let inAssistant = false
  let currentAssistant: string[] = []

  const flushAssistant = (): void => {
    if (inAssistant && currentAssistant.length > 0) {
      assistantChunks.push(currentAssistant.join('\n'))
      currentAssistant = []
    }
  }

  for (const line of lines) {
    if (line.startsWith('[user]:')) {
      flushAssistant()
      inAssistant = false
      userLines.push(line)
      continue
    }
    if (line.startsWith('[assistant]:')) {
      flushAssistant()
      inAssistant = true
      currentAssistant = [line]
      continue
    }
    if (inAssistant) {
      currentAssistant.push(line)
      continue
    }
    userLines.push(line)
  }
  flushAssistant()

  const maxAssistantChunks = 5
  const keptAssistantChunks =
    assistantChunks.length > maxAssistantChunks
      ? [...assistantChunks]
          .sort((left, right) => {
            const leftScore = scoreChunkRelevance(left, questionTokens(question))
            const rightScore = scoreChunkRelevance(right, questionTokens(question))
            if (leftScore !== rightScore) return rightScore - leftScore
            return left.length - right.length
          })
          .slice(0, maxAssistantChunks)
      : assistantChunks

  const parts = [userLines.join('\n').trim()]
  if (keptAssistantChunks.length > 0) {
    parts.push(
      [
        '[Assistant context (summarised)]:',
        ...keptAssistantChunks.map((chunk) => chunk.trim()),
      ].join('\n'),
    )
  }
  return parts.filter((part) => part !== '').join('\n\n')
}

const questionTokens = (question: string): readonly string[] => {
  if (question === '') return []
  const stopWords = new Set([
    'the',
    'and',
    'for',
    'with',
    'what',
    'who',
    'when',
    'where',
    'why',
    'how',
    'did',
    'does',
    'was',
    'were',
    'are',
    'you',
    'your',
    'about',
    'this',
    'that',
    'have',
    'has',
    'had',
    'from',
    'into',
    'than',
    'then',
    'them',
    'they',
    'their',
  ])
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of question.toLowerCase().split(/\s+/)) {
    const token = raw.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '')
    if (token.length < 3 || stopWords.has(token) || seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

const scoreChunkRelevance = (
  chunk: string,
  tokens: readonly string[],
): number => {
  const lower = chunk.toLowerCase()
  let score = 0
  for (const token of tokens) {
    if (lower.includes(token)) score++
  }
  return score
}

const buildEmbeddingText = (doc: EvalDocument): string =>
  [doc.title, doc.summary, doc.content.slice(0, 2000)]
    .map((part) => part.trim())
    .filter((part) => part !== '')
    .join('\n\n')

const clusterPassagesBySession = (
  passages: readonly RetrievedPassage[],
): RetrievedPassage[] => {
  if (passages.length <= 1) return [...passages]
  const order: string[] = []
  const groups = new Map<string, RetrievedPassage[]>()
  for (const [index, passage] of passages.entries()) {
    const key = passage.sessionId ?? `__solo_${String(index)}__`
    if (!groups.has(key)) order.push(key)
    const group = groups.get(key) ?? []
    group.push(passage)
    groups.set(key, group)
  }
  return order.flatMap((key) => groups.get(key) ?? [])
}

const sourceTagFromPath = (value: string): string => {
  const base = value.split('/').filter(Boolean).pop() ?? value
  return base.replace(/\.md$/i, '')
}

const dedupeStrings = (values: readonly string[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const trimmed = value.trim()
    if (trimmed === '' || seen.has(trimmed)) continue
    seen.add(trimmed)
    out.push(trimmed)
  }
  return out
}

const countReplayEligiblePassages = (
  passages: readonly RetrievedPassage[],
  questionDate: string | undefined,
): number => filterPassagesByQuestionDate(passages, questionDate).length

const dedupeSearchHits = (
  hits: ReadonlyArray<{ readonly path: Path; readonly score: number }>,
): Array<{ readonly path: Path; readonly score: number }> => {
  const best = new Map<Path, number>()
  for (const hit of hits) {
    const current = best.get(hit.path)
    if (current === undefined || hit.score > current) {
      best.set(hit.path, hit.score)
    }
  }
  return [...best.entries()]
    .map(([path, score]) => ({ path, score }))
    .sort((left, right) => right.score - left.score)
}

const extractDate = (
  metadata: Readonly<Record<string, unknown>>,
): string | undefined => {
  const sessionDate = metadata['sessionDate']
  if (typeof sessionDate === 'string' && sessionDate !== '') return sessionDate
  const observedOn = metadata['observedOn']
  if (typeof observedOn === 'string' && observedOn !== '') return observedOn
  const modified = metadata['modified']
  if (typeof modified === 'string' && modified !== '') return modified
  return undefined
}

const toHybridMode = (
  mode: 'bm25' | 'semantic' | 'hybrid',
): HybridMode => {
  switch (mode) {
    case 'bm25':
      return 'bm25'
    case 'semantic':
      return 'semantic'
    case 'hybrid':
      return 'hybrid'
  }
}

const createBrainFingerprint = (args: {
  readonly datasetSha256: string
  readonly sampleSignature?: string
  readonly ingestMode: IngestMode
  readonly actorId: string
  readonly extractModel: string | undefined
}): string =>
  createHash('sha256')
    .update(args.datasetSha256)
    .update('\x1f')
    .update(args.sampleSignature ?? '')
    .update('\x1f')
    .update(args.ingestMode)
    .update('\x1f')
    .update(args.actorId)
    .update('\x1f')
    .update(args.extractModel ?? '')
    .update('\x1f')
    .update(EXTRACTION_SYSTEM_PROMPT)
    .update('\x1f')
    .update(REPLAY_PIPELINE_VERSION)
    .digest('hex')

const createBrainCacheKey = (args: {
  readonly datasetSha256: string
  readonly sampleSignature?: string
  readonly ingestMode: IngestMode
  readonly actorId: string
  readonly extractModel: string | undefined
}): string =>
  createBrainFingerprint(args).slice(0, 24)

const readFingerprint = async (filePath: string): Promise<string | undefined> => {
  try {
    return (await readFile(filePath, 'utf8')).trim()
  } catch {
    return undefined
  }
}

const defaultRunId = (date: Date): string => {
  const pad = (value: number): string => String(value).padStart(2, '0')
  return `lme-${date.getUTCFullYear()}${pad(date.getUTCMonth() + 1)}${pad(
    date.getUTCDate(),
  )}-${pad(date.getUTCHours())}${pad(date.getUTCMinutes())}${pad(
    date.getUTCSeconds(),
  )}`
}

const sanitiseFileSegment = (value: string): string =>
  value.replace(/[^A-Za-z0-9._-]/g, '_')

const createSampleSignature = (
  examples: readonly { readonly id: string }[],
  seed: number,
): string =>
  createHash('sha256')
    .update(String(seed))
    .update('\x1f')
    .update(examples.map((example) => example.id).join('\n'))
    .digest('hex')
