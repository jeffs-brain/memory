// SPDX-License-Identifier: Apache-2.0

import { access, mkdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { defineCommand } from 'citty'
import {
  DEFAULT_OUT_DIR,
  compareReports,
  fetchOfficialRepo,
  fetchUpstreamDatasets,
  loadDataset,
  LONG_MEM_EVAL_OFFICIAL_REPO_REF,
  officialScorerScriptPath,
  resolveUpstreamDatasetPath,
  runOfficialScorer,
  runStandaloneLMEEval,
  verifyOfficialScorer,
  type LMEManifest,
  type LMEReport,
  type LMEUpstreamBundleName,
  type LMEUpstreamDatasetName,
  type OfficialLMEScoreSummary,
} from '../../eval/index.js'
import {
  buildEmbedder,
  buildProvider,
  buildReranker,
  CliError,
  CliUsageError,
  embedderFromEnv,
  providerFromEnvOptional,
  rerankerFromEnv,
} from '../config.js'

const DEFAULT_CACHE_DIR = path.join(DEFAULT_OUT_DIR, 'cache')
const DEFAULT_SPLITS = 'oracle,s,m'
const DEFAULT_METRIC_MODEL = 'gpt-4o'

type ParsedManifest = LMEManifest

type ScoreSummary = {
  readonly report: {
    readonly path: string
    readonly runId: string
    readonly overallAccuracy: number
    readonly taskAvgAccuracy: number
    readonly examples: number
  }
  readonly official?: {
    readonly resultPath: string
    readonly summary: OfficialLMEScoreSummary
  }
}

type DoctorCheck = {
  readonly name: string
  readonly ok: boolean
  readonly detail: string
  readonly value?: unknown
}

const isString = (value: unknown): value is string => typeof value === 'string'

const getEnv = (name: string): string | undefined => {
  const raw = process.env[name]
  return raw !== undefined && raw !== '' ? raw : undefined
}

const resolveOptionalString = (
  value: unknown,
  envName: string,
): string | undefined => {
  if (isString(value) && value !== '') return value
  return getEnv(envName)
}

const resolveRequiredString = (
  value: unknown,
  envName: string,
  label: string,
): string => {
  const resolved = resolveOptionalString(value, envName)
  if (resolved !== undefined) return resolved
  throw new CliUsageError(`${label}: missing value and ${envName} is not set`)
}

const parsePositiveInt = (
  value: unknown,
  fallback: number,
  label: string,
): number => {
  if (!isString(value) || value === '') return fallback
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new CliUsageError(`${label}: invalid integer '${value}'`)
  }
  return parsed
}

const parseNonNegativeInt = (
  value: unknown,
  fallback: number,
  label: string,
): number => {
  if (!isString(value) || value === '') return fallback
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new CliUsageError(`${label}: invalid integer '${value}'`)
  }
  return parsed
}

const parseBoolean = (value: unknown, fallback: boolean): boolean => {
  if (typeof value === 'boolean') return value
  if (!isString(value) || value === '') return fallback
  if (value === 'true') return true
  if (value === 'false') return false
  throw new CliUsageError(`invalid boolean '${String(value)}'`)
}

const parseBundle = (value: unknown): LMEUpstreamBundleName => {
  const raw = isString(value) && value !== '' ? value : 'cleaned'
  if (raw === 'cleaned' || raw === 'legacy') return raw
  throw new CliUsageError(
    `invalid bundle '${raw}'; expected cleaned|legacy`,
  )
}

const parseSplit = (value: unknown): LMEUpstreamDatasetName => {
  const raw = isString(value) && value !== '' ? value : 'oracle'
  if (raw === 'oracle' || raw === 's' || raw === 'm') return raw
  throw new CliUsageError(`invalid split '${raw}'; expected oracle|s|m`)
}

const parseIngestMode = (value: unknown) => {
  const raw =
    resolveOptionalString(value, 'JB_LME_INGEST_MODE') ?? 'replay'
  if (raw === 'bulk' || raw === 'replay' || raw === 'agentic') return raw
  throw new CliUsageError(
    `invalid ingest mode '${raw}'; expected bulk|replay|agentic`,
  )
}

const parseRetrievalMode = (value: unknown) => {
  const raw =
    resolveOptionalString(value, 'JB_LME_RETRIEVAL_MODE') ?? 'bm25'
  if (raw === 'bm25' || raw === 'semantic' || raw === 'hybrid') return raw
  throw new CliUsageError(
    `invalid retrieval mode '${raw}'; expected bm25|semantic|hybrid`,
  )
}

const parseMetricModel = (
  value: unknown,
): 'gpt-4o' | 'gpt-4o-mini' | 'llama-3.1-70b-instruct' => {
  const raw =
    resolveOptionalString(value, 'JB_LME_METRIC_MODEL') ??
    DEFAULT_METRIC_MODEL
  if (
    raw === 'gpt-4o' ||
    raw === 'gpt-4o-mini' ||
    raw === 'llama-3.1-70b-instruct'
  ) {
    return raw
  }
  throw new CliUsageError(
    `invalid metric model '${raw}'; expected gpt-4o|gpt-4o-mini|llama-3.1-70b-instruct`,
  )
}

const parseSplits = (value: unknown): readonly LMEUpstreamDatasetName[] => {
  const raw = isString(value) && value !== '' ? value : DEFAULT_SPLITS
  const parts = raw.split(',').map((part) => part.trim()).filter(Boolean)
  const out: LMEUpstreamDatasetName[] = []
  for (const part of parts) {
    if (part !== 'oracle' && part !== 's' && part !== 'm') {
      throw new CliUsageError(`invalid split '${part}' in '${raw}'`)
    }
    if (!out.includes(part)) out.push(part)
  }
  if (out.length === 0) {
    throw new CliUsageError('at least one split is required')
  }
  return out
}

const parseCsv = (value: unknown): readonly string[] | undefined => {
  if (!isString(value) || value === '') return undefined
  const items = value.split(',').map((part) => part.trim()).filter(Boolean)
  return items.length > 0 ? items : undefined
}

const resolveCacheDir = (value: unknown): string =>
  path.resolve(resolveOptionalString(value, 'JB_LME_CACHE_DIR') ?? DEFAULT_CACHE_DIR)

const resolveOutDir = (value: unknown): string =>
  path.resolve(resolveOptionalString(value, 'JB_LME_OUT_DIR') ?? DEFAULT_OUT_DIR)

const defaultDatasetDir = (
  cacheDir: string,
  bundle: LMEUpstreamBundleName,
): string => path.join(cacheDir, 'datasets', bundle)

const defaultRepoDir = (cacheDir: string): string =>
  path.join(cacheDir, 'upstream', 'LongMemEval')

const readJsonFile = async <T>(filePath: string): Promise<T> => {
  const raw = await readFile(filePath, 'utf8')
  return JSON.parse(raw) as T
}

const readReport = async (filePath: string): Promise<LMEReport> =>
  readJsonFile<LMEReport>(filePath)

const readManifest = async (filePath: string): Promise<ParsedManifest> =>
  readJsonFile<ParsedManifest>(filePath)

const latestRunDir = async (outDir: string): Promise<string> => {
  const symlinkPath = path.join(outDir, 'latest')
  try {
    return path.resolve(outDir, (await readFile(`${symlinkPath}.txt`, 'utf8')).trim())
  } catch {
    return path.join(outDir, 'latest')
  }
}

const resolveReportPath = async (
  reportArg: unknown,
  outDirArg: unknown,
): Promise<string> => {
  const explicit = resolveOptionalString(reportArg, 'JB_LME_REPORT')
  if (explicit !== undefined) return path.resolve(explicit)
  const outDir = resolveOutDir(outDirArg)
  return path.join(await latestRunDir(outDir), 'report.json')
}

const resolveManifestPath = async (
  manifestArg: unknown,
  reportArg: unknown,
  outDirArg: unknown,
): Promise<string> => {
  const explicit = resolveOptionalString(manifestArg, 'JB_LME_MANIFEST')
  if (explicit !== undefined) return path.resolve(explicit)
  return path.join(path.dirname(await resolveReportPath(reportArg, outDirArg)), 'manifest.json')
}

const buildProviders = () => {
  const providerSettings = providerFromEnvOptional()
  const provider =
    providerSettings !== undefined ? buildProvider(providerSettings) : undefined
  const embedderSettings = embedderFromEnv()
  const embedder =
    embedderSettings !== undefined ? buildEmbedder(embedderSettings) : undefined
  const rerankerSettings = rerankerFromEnv()
  const reranker =
    rerankerSettings !== undefined ? buildReranker(rerankerSettings) : undefined
  return { provider, embedder, reranker }
}

const fetchCommand = defineCommand({
  meta: {
    name: 'fetch',
    description: 'Fetch the official LongMemEval repo and dataset bundle',
  },
  args: {
    bundle: {
      type: 'string',
      description: 'Dataset bundle: cleaned|legacy',
      default: 'cleaned',
    },
    splits: {
      type: 'string',
      description: 'Comma-separated splits: oracle,s,m',
      default: DEFAULT_SPLITS,
    },
    cacheDir: {
      type: 'string',
      description: 'Cache directory (overrides JB_LME_CACHE_DIR)',
    },
    datasetDir: {
      type: 'string',
      description: 'Dataset directory override',
    },
    repoDir: {
      type: 'string',
      description: 'Official repo directory override',
    },
    repoRef: {
      type: 'string',
      description: 'Official repo ref to fetch',
      default: LONG_MEM_EVAL_OFFICIAL_REPO_REF,
    },
    skipRepo: {
      type: 'boolean',
      description: 'Skip the official repo fetch',
      default: false,
    },
    skipDataset: {
      type: 'boolean',
      description: 'Skip the dataset fetch',
      default: false,
    },
  },
  run: async ({ args }) => {
    const bundle = parseBundle(args.bundle)
    const splits = parseSplits(args.splits)
    const cacheDir = resolveCacheDir(args.cacheDir)
    const datasetDir = path.resolve(
      resolveOptionalString(args.datasetDir, 'JB_LME_DATASET_DIR') ??
        defaultDatasetDir(cacheDir, bundle),
    )
    const repoDir = path.resolve(
      resolveOptionalString(args.repoDir, 'JB_LME_REPO_DIR') ??
        defaultRepoDir(cacheDir),
    )
    const skipRepo = parseBoolean(args.skipRepo, false)
    const skipDataset = parseBoolean(args.skipDataset, false)

    await mkdir(cacheDir, { recursive: true })
    const dataset =
      skipDataset === true
        ? undefined
        : await fetchUpstreamDatasets(bundle, {
            dir: datasetDir,
            splits,
          })
    const repo =
      skipRepo === true
        ? undefined
        : await fetchOfficialRepo({
            dir: repoDir,
            ref:
              resolveOptionalString(args.repoRef, 'JB_LME_REPO_REF') ??
              LONG_MEM_EVAL_OFFICIAL_REPO_REF,
          })

    process.stdout.write(
      `${JSON.stringify({
        cacheDir,
        ...(dataset !== undefined ? { dataset } : {}),
        ...(repo !== undefined ? { repo } : {}),
      })}\n`,
    )
  },
})

const runCommand = defineCommand({
  meta: {
    name: 'run',
    description: 'Run LongMemEval against the local memory package',
  },
  args: {
    dataset: {
      type: 'string',
      description: 'Dataset path override',
    },
    bundle: {
      type: 'string',
      description: 'Dataset bundle: cleaned|legacy',
      default: 'cleaned',
    },
    split: {
      type: 'string',
      description: 'Dataset split: oracle|s|m',
      default: 'oracle',
    },
    outDir: {
      type: 'string',
      description: 'Output directory',
    },
    cacheDir: {
      type: 'string',
      description: 'Cache directory',
    },
    runId: {
      type: 'string',
      description: 'Run id override',
    },
    ingestMode: {
      type: 'string',
      description: 'Ingest mode: bulk|replay|agentic',
      default: 'replay',
    },
    retrievalMode: {
      type: 'string',
      description: 'Retrieval mode: bm25|semantic|hybrid',
      default: 'bm25',
    },
    rerank: {
      type: 'boolean',
      description: 'Enable reranking',
      default: false,
    },
    topK: {
      type: 'string',
      description: 'Top-k passages',
      default: '50',
    },
    candidateK: {
      type: 'string',
      description: 'Per-retriever candidate count',
      default: '50',
    },
    rerankTopN: {
      type: 'string',
      description: 'Top-N candidates to rerank',
      default: '20',
    },
    sampleSize: {
      type: 'string',
      description: 'Optional sample size',
    },
    seed: {
      type: 'string',
      description: 'Sampling seed',
      default: '0',
    },
    ingestConcurrency: {
      type: 'string',
      description: 'Ingest worker concurrency',
    },
    judgeConcurrency: {
      type: 'string',
      description: 'Judge worker concurrency',
      default: '8',
    },
    embeddingBatchSize: {
      type: 'string',
      description: 'Embedding batch size',
      default: '8',
    },
    readerBudgetChars: {
      type: 'string',
      description: 'Reader context budget in characters',
      default: '100000',
    },
    categories: {
      type: 'string',
      description: 'Optional comma-separated question categories to evaluate',
    },
    readerModel: {
      type: 'string',
      description: 'Reader model override',
    },
    judgeModel: {
      type: 'string',
      description: 'Judge model override',
    },
    actorId: {
      type: 'string',
      description: 'Actor id for extracted notes',
      default: 'lme',
    },
    repoDir: {
      type: 'string',
      description: 'Official repo directory for manifest parity',
    },
    compareReport: {
      type: 'string',
      description: 'Optional baseline report path to compare against',
    },
  },
  run: async ({ args }) => {
    const bundle = parseBundle(args.bundle)
    const split = parseSplit(args.split)
    const outDir = resolveOutDir(args.outDir)
    const cacheDir = resolveCacheDir(args.cacheDir)
    const datasetPath = path.resolve(
      resolveOptionalString(args.dataset, 'JB_LME_DATASET') ??
        resolveUpstreamDatasetPath(
          bundle,
          split,
          resolveOptionalString(undefined, 'JB_LME_DATASET_DIR') ??
            defaultDatasetDir(cacheDir, bundle),
        ),
    )
    const repoDir = path.resolve(
      resolveOptionalString(args.repoDir, 'JB_LME_REPO_DIR') ??
        defaultRepoDir(cacheDir),
    )
    const { provider, embedder, reranker } = buildProviders()
    const ingestMode = parseIngestMode(args.ingestMode)
    if (ingestMode !== 'bulk' && provider === undefined) {
      throw new CliUsageError(
        'lme run: replay and agentic modes require JB_LLM_PROVIDER',
      )
    }
    if (!(await pathExists(datasetPath))) {
        throw new CliUsageError(
          `lme run: dataset not found at ${datasetPath}; run \`memory eval lme fetch\` first or pass --dataset`,
        )
    }

    const compareReportPath = resolveOptionalString(
      args.compareReport,
      'JB_LME_COMPARE_REPORT',
    )
    const questionCategories = parseCsv(
      resolveOptionalString(args.categories, 'JB_LME_CATEGORIES'),
    )
    const outcome = await runStandaloneLMEEval({
      datasetPath,
      bundle,
      split,
      ingestMode,
      retrievalMode: parseRetrievalMode(args.retrievalMode),
      rerank: parseBoolean(args.rerank, false),
      topK: parsePositiveInt(args.topK, 50, 'lme run'),
      candidateK: parsePositiveInt(args.candidateK, 50, 'lme run'),
      rerankTopN: parsePositiveInt(args.rerankTopN, 20, 'lme run'),
      ...(resolveOptionalString(args.sampleSize, 'JB_LME_SAMPLE_SIZE') !== undefined
        ? {
            sampleSize: parsePositiveInt(
              resolveOptionalString(args.sampleSize, 'JB_LME_SAMPLE_SIZE'),
              1,
              'lme run',
            ),
          }
        : {}),
      seed: parseNonNegativeInt(args.seed, 0, 'lme run'),
      ingestConcurrency: parsePositiveInt(
        args.ingestConcurrency,
        ingestMode === 'replay' ? 1 : 16,
        'lme run',
      ),
      judgeConcurrency: parsePositiveInt(args.judgeConcurrency, 8, 'lme run'),
      embeddingBatchSize: parsePositiveInt(
        args.embeddingBatchSize,
        8,
        'lme run',
      ),
      readerBudgetChars: parsePositiveInt(
        args.readerBudgetChars,
        100000,
        'lme run',
      ),
      ...(questionCategories !== undefined ? { questionCategories } : {}),
      outDir,
      cacheDir,
      ...(resolveOptionalString(args.runId, 'JB_LME_RUN_ID') !== undefined
        ? { runId: resolveRequiredString(args.runId, 'JB_LME_RUN_ID', 'lme run') }
        : {}),
      ...(resolveOptionalString(args.actorId, 'JB_LME_ACTOR_ID') !== undefined
        ? { actorId: resolveRequiredString(args.actorId, 'JB_LME_ACTOR_ID', 'lme run') }
        : {}),
      ...(provider !== undefined ? { provider } : {}),
      ...(embedder !== undefined ? { embedder } : {}),
      ...(reranker !== undefined ? { reranker } : {}),
      ...(resolveOptionalString(args.readerModel, 'JB_LME_READER_MODEL') !== undefined
        ? {
            readerModel: resolveRequiredString(
              args.readerModel,
              'JB_LME_READER_MODEL',
              'lme run',
            ),
          }
        : {}),
      ...(resolveOptionalString(args.judgeModel, 'JB_LME_JUDGE_MODEL') !== undefined
        ? {
            judgeModel: resolveRequiredString(
              args.judgeModel,
              'JB_LME_JUDGE_MODEL',
              'lme run',
            ),
          }
        : {}),
      ...(await pathExists(officialScorerScriptPath(repoDir))
        ? { officialRepo: await fetchOfficialRepo({ dir: repoDir }) }
        : {}),
      ...(compareReportPath !== undefined
        ? { compareAgainst: await readReport(path.resolve(compareReportPath)) }
        : {}),
    })

    process.stdout.write(
      `${JSON.stringify({
        runId: outcome.manifest.runId,
        runDir: outcome.runDir,
        reportPath: outcome.reportPath,
        manifestPath: outcome.manifestPath,
        officialHypothesesPath: outcome.officialHypothesesPath,
        officialEvalLogPath: outcome.officialEvalLogPath,
        overallAccuracy: outcome.report.overallAccuracy,
        taskAvgAccuracy: outcome.report.taskAvgAccuracy,
        officialScore: outcome.officialScore,
        ...(outcome.comparison !== undefined ? { comparison: outcome.comparison } : {}),
      })}\n`,
    )
  },
})

const scoreCommand = defineCommand({
  meta: {
    name: 'score',
    description: 'Summarise a run and optionally invoke the official scorer',
  },
  args: {
    report: {
      type: 'string',
      description: 'Report path override',
    },
    manifest: {
      type: 'string',
      description: 'Manifest path override',
    },
    outDir: {
      type: 'string',
      description: 'Output directory when resolving latest',
    },
    official: {
      type: 'boolean',
      description: 'Run the official Python scorer',
      default: false,
    },
    predictions: {
      type: 'string',
      description: 'Predictions JSONL path override',
    },
    dataset: {
      type: 'string',
      description: 'Dataset path override',
    },
    repoDir: {
      type: 'string',
      description: 'Official repo directory',
    },
    metricModel: {
      type: 'string',
      description: 'Official metric model',
      default: DEFAULT_METRIC_MODEL,
    },
    python: {
      type: 'string',
      description: 'Python binary',
      default: 'python3',
    },
  },
  run: async ({ args }) => {
    const reportPath = await resolveReportPath(args.report, args.outDir)
    const report = await readReport(reportPath)
    const reportSummary: ScoreSummary['report'] = {
      path: reportPath,
      runId: report.evalRunId,
      overallAccuracy: report.overallAccuracy,
      taskAvgAccuracy: report.taskAvgAccuracy,
      examples: report.examples,
    }
    let officialSummary: ScoreSummary['official'] | undefined
    if (parseBoolean(args.official, false)) {
      const manifestPath = await resolveManifestPath(args.manifest, args.report, args.outDir)
      const manifest = await readManifest(manifestPath)
      const repoDir = path.resolve(
        resolveOptionalString(args.repoDir, 'JB_LME_REPO_DIR') ??
          manifest.officialRepo?.dir ??
          defaultRepoDir(resolveCacheDir(undefined)),
      )
      const predictionsPath = path.resolve(
        resolveOptionalString(args.predictions, 'JB_LME_PREDICTIONS') ??
          manifest.outputs.predictionsPath ??
          path.join(path.dirname(reportPath), 'predictions.jsonl'),
      )
      const datasetPath = path.resolve(
        resolveOptionalString(args.dataset, 'JB_LME_DATASET') ??
          manifest.dataset.path,
      )
      const official = await runOfficialScorer({
        repoDir,
        datasetPath,
        predictionsPath,
        metricModel: parseMetricModel(args.metricModel),
        pythonBin: resolveOptionalString(args.python, 'JB_PYTHON_BIN') ?? 'python3',
      })
      officialSummary = {
        resultPath: official.resultPath,
        summary: official.summary,
      }
    }
    const summary: ScoreSummary = {
      report: {
        path: reportSummary.path,
        runId: reportSummary.runId,
        overallAccuracy: reportSummary.overallAccuracy,
        taskAvgAccuracy: reportSummary.taskAvgAccuracy,
        examples: reportSummary.examples,
      },
      ...(officialSummary !== undefined ? { official: officialSummary } : {}),
    }
    process.stdout.write(`${JSON.stringify(summary)}\n`)
  },
})

const compareCommand = defineCommand({
  meta: {
    name: 'compare',
    description: 'Compare two native LongMemEval reports',
  },
  args: {
    left: {
      type: 'string',
      description: 'Left report path',
    },
    right: {
      type: 'string',
      description: 'Right report path',
    },
  },
  run: async ({ args }) => {
    const leftPath = path.resolve(
      resolveRequiredString(args.left, 'JB_LME_LEFT_REPORT', 'lme compare'),
    )
    const rightPath = path.resolve(
      resolveRequiredString(args.right, 'JB_LME_RIGHT_REPORT', 'lme compare'),
    )
    const comparison = compareReports(
      await readReport(leftPath),
      await readReport(rightPath),
    )
    process.stdout.write(`${JSON.stringify(comparison)}\n`)
  },
})

const doctorCommand = defineCommand({
  meta: {
    name: 'doctor',
    description: 'Check the LongMemEval environment and local artefacts',
  },
  args: {
    bundle: {
      type: 'string',
      description: 'Dataset bundle: cleaned|legacy',
      default: 'cleaned',
    },
    split: {
      type: 'string',
      description: 'Dataset split: oracle|s|m',
      default: 'oracle',
    },
    dataset: {
      type: 'string',
      description: 'Dataset path override',
    },
    repoDir: {
      type: 'string',
      description: 'Official repo directory override',
    },
    outDir: {
      type: 'string',
      description: 'Output directory',
    },
    cacheDir: {
      type: 'string',
      description: 'Cache directory',
    },
    official: {
      type: 'boolean',
      description: 'Check the official scorer prerequisites as well',
      default: false,
    },
    python: {
      type: 'string',
      description: 'Python binary',
      default: 'python3',
    },
  },
  run: async ({ args }) => {
    const bundle = parseBundle(args.bundle)
    const split = parseSplit(args.split)
    const cacheDir = resolveCacheDir(args.cacheDir)
    const datasetPath = path.resolve(
      resolveOptionalString(args.dataset, 'JB_LME_DATASET') ??
        resolveUpstreamDatasetPath(
          bundle,
          split,
          defaultDatasetDir(cacheDir, bundle),
        ),
    )
    const repoDir = path.resolve(
      resolveOptionalString(args.repoDir, 'JB_LME_REPO_DIR') ??
        defaultRepoDir(cacheDir),
    )
    const outDir = resolveOutDir(args.outDir)
    const checks: DoctorCheck[] = []
    let ok = true

    try {
      const dataset = await loadDataset(datasetPath)
      checks.push({
        name: 'dataset',
        ok: true,
        detail: 'dataset is readable and valid',
        value: {
          path: datasetPath,
          sha256: dataset.sha256,
          examples: dataset.examples.length,
          categories: dataset.categories,
        },
      })
    } catch (err) {
      checks.push({
        name: 'dataset',
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
      })
      ok = false
    }

    try {
      const provider = providerFromEnvOptional()
      checks.push({
        name: 'provider',
        ok: true,
        detail:
          provider !== undefined
            ? 'provider configuration is valid'
            : 'JB_LLM_PROVIDER not set',
        ...(provider !== undefined
          ? { value: { kind: provider.kind, model: provider.model } }
          : {}),
      })
    } catch (err) {
      checks.push({
        name: 'provider',
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
      })
      ok = false
    }

    try {
      const embedder = embedderFromEnv()
      checks.push({
        name: 'embedder',
        ok: true,
        detail:
          embedder !== undefined
            ? 'embedder configuration is valid'
            : 'JB_EMBED_PROVIDER not set',
        ...(embedder !== undefined ? { value: embedder } : {}),
      })
    } catch (err) {
      checks.push({
        name: 'embedder',
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
      })
      ok = false
    }

    try {
      const reranker = rerankerFromEnv()
      checks.push({
        name: 'reranker',
        ok: true,
        detail:
          reranker !== undefined
            ? 'reranker configuration is valid'
            : 'JB_RERANK_PROVIDER not set',
        ...(reranker !== undefined ? { value: reranker } : {}),
      })
    } catch (err) {
      checks.push({
        name: 'reranker',
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
      })
      ok = false
    }

    try {
      await mkdir(outDir, { recursive: true })
      await mkdir(cacheDir, { recursive: true })
      checks.push({
        name: 'paths',
        ok: true,
        detail: 'output and cache directories are writable',
        value: { outDir, cacheDir },
      })
    } catch (err) {
      checks.push({
        name: 'paths',
        ok: false,
        detail: err instanceof Error ? err.message : String(err),
      })
      ok = false
    }

    if (parseBoolean(args.official, false)) {
      try {
        await verifyOfficialScorer(
          repoDir,
          resolveOptionalString(args.python, 'JB_PYTHON_BIN') ?? 'python3',
        )
        checks.push({
          name: 'official',
          ok: true,
          detail: 'official scorer prerequisites are available',
          value: {
            repoDir,
            script: officialScorerScriptPath(repoDir),
          },
        })
      } catch (err) {
        checks.push({
          name: 'official',
          ok: false,
          detail: err instanceof Error ? err.message : String(err),
        })
        ok = false
      }
    }

    process.stdout.write(`${JSON.stringify({ ok, checks })}\n`)
    if (!ok) {
      throw new CliError('lme doctor: one or more checks failed')
    }
  },
})

export const lmeCommand = defineCommand({
  meta: {
    name: 'lme',
    description: 'LongMemEval commands',
  },
  subCommands: {
    fetch: fetchCommand,
    run: runCommand,
    score: scoreCommand,
    compare: compareCommand,
    doctor: doctorCommand,
  },
})

const pathExists = async (filePath: string): Promise<boolean> => {
  try {
    await access(filePath)
    return true
  } catch {
    return false
  }
}
