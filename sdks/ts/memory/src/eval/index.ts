// SPDX-License-Identifier: Apache-2.0

/**
 * LongMemEval (LME) benchmark harness. `createLMERunner` wires the
 * ingest modes, read pipeline, judge, and reporter into a single object
 * so callers can drive a full benchmark in a handful of lines.
 *
 * The harness is deliberately injection-heavy: retrieval, reader, and
 * judge functions may all be stubbed for tests or swapped for alternate
 * providers. Nothing here talks to the network directly — every external
 * call goes through one of the injected dependencies.
 */

import { noopLogger } from '../llm/index.js'
import type { Logger } from '../llm/index.js'
import { ingestAgentic } from './ingest-agentic.js'
import { ingestBulk } from './ingest-bulk.js'
import { ingestReplay } from './ingest-replay.js'
import { createProviderJudge, exactMatchVerdict } from './judge.js'
import { createProviderReader, runRead } from './read.js'
import { buildReport, writeReport } from './report.js'
import type {
  CreateLMERunnerOpts,
  IngestMode,
  JudgeFn,
  LMEExample,
  LMEResult,
  LMERunner,
  ReaderFn,
  RetrievalFn,
  RetrievalResult,
} from './types.js'

export * from './types.js'
export { loadDataset, parseDatasetText, DatasetLoadError } from './dataset.js'
export type { LoadedDataset } from './dataset.js'
export { ingestBulk, deduplicateSessions } from './ingest-bulk.js'
export { ingestReplay, ingestReplayFromSessions, sessionTextToMessages } from './ingest-replay.js'
export { ingestAgentic } from './ingest-agentic.js'
export {
  LONG_MEM_EVAL_UPSTREAM_DATASETS,
  LONG_MEM_EVAL_OFFICIAL_REPO_URL,
  LONG_MEM_EVAL_OFFICIAL_REPO_REF,
  fetchOfficialRepo,
  fetchUpstreamDatasets,
  resolveUpstreamDatasetPath,
} from './upstream.js'
export type {
  LMEOfficialRepoFetchResult,
  LMEUpstreamBundleName,
  LMEUpstreamDatasetName,
  LMEUpstreamFetchResult,
  LMEUpstreamDatasetSpec,
  LMEUpstreamBundleSpec,
  LMEUpstreamFetchMetadata,
} from './upstream.js'
export {
  buildLMEManifest,
  manifestCacheKeyInput,
} from './manifest.js'
export type {
  LMEDataManifest,
  LMEManifest,
  LMEModelManifest,
  LMERunConfig,
  LMESampleManifest,
} from './manifest.js'
export {
  createLMECacheKey,
  lmeCacheKeySeed,
} from './cache-key.js'
export type { LMECacheKeyInput } from './cache-key.js'
export {
  compareResultSets,
  compareReports,
  compareResults,
} from './compare.js'
export type {
  LMECategoryComparison,
  LMEReportComparison,
  LMEResultComparison,
  LMEResultSetComparison,
} from './compare.js'
export {
  compareOfficialScoreSummaries,
  officialHypothesesFromResults,
  parseOfficialEvalLog,
  resultsToOfficialEvalLog,
  scoreOfficialEvalLog,
  serialiseOfficialEvalLog,
  serialiseOfficialHypotheses,
} from './scorer.js'
export type {
  OfficialLMECategoryScore,
  OfficialLMEEvalLabel,
  OfficialLMEEvalLogRow,
  OfficialLMEHypothesisRow,
  OfficialLMEScoreSummary,
} from './scorer.js'
export {
  OFFICIAL_SCORER_SCRIPT,
  officialScorerRequirements,
  officialScorerScriptPath,
  runOfficialScorer,
  verifyOfficialScorer,
} from './official.js'
export type { OfficialScorerOutcome, RunOfficialScorerArgs } from './official.js'
export { runStandaloneLMEEval } from './run.js'
export type { StandaloneLMERunArgs, StandaloneLMERunOutcome } from './run.js'
export { READER_USER_TEMPLATE } from '../augmented-reader/prompt.js'
export { runRead, createProviderReader, truncateSmartly } from './read.js'
export {
  createProviderJudge,
  createStaticJudge,
  exactMatchVerdict,
  formatJudgePrompt,
  judgePromptForCategory,
  isAbstention,
  parseYesNo,
  JUDGE_PROMPT_VERSION,
  JUDGE_PROMPT_STANDARD,
  JUDGE_PROMPT_TEMPORAL,
  JUDGE_PROMPT_KNOWLEDGE_UPDATE,
  JUDGE_PROMPT_PREFERENCE,
  JUDGE_PROMPT_ABSTENTION,
} from './judge.js'
export { buildReport, writeReport } from './report.js'

export const DEFAULT_OUT_DIR = '.eval/lme'

export const createLMERunner = (opts: CreateLMERunnerOpts): LMERunner => {
  const logger: Logger = opts.logger ?? noopLogger
  const outDir = opts.outDir ?? DEFAULT_OUT_DIR
  const now = opts.now ?? (() => new Date())

  const retrieval: RetrievalFn = opts.retrieval ?? defaultRetrieval(opts)
  const reader: ReaderFn = opts.reader ?? defaultReader(opts)
  const judge: JudgeFn = opts.judge ?? defaultJudge(opts)
  const runIdBase = opts.runId

  const resolveRunId = (): string => runIdBase ?? defaultRunId(now())

  const runBulk: LMERunner['runBulk'] = async ({ examples }) => {
    logger.info('lme runBulk: starting', { examples: examples.length })
    return ingestBulk(opts.store, examples)
  }

  const runReplay: LMERunner['runReplay'] = async ({ examples, concurrency }) => {
    if (!opts.memory) {
      throw new Error('lme runReplay: memory is required')
    }
    logger.info('lme runReplay: starting', { examples: examples.length })
    return ingestReplay(opts.memory, examples, {
      ...(concurrency !== undefined ? { concurrency } : {}),
      logger,
    })
  }

  const runAgentic: LMERunner['runAgentic'] = async ({ examples }) => {
    if (!opts.memory) {
      throw new Error('lme runAgentic: memory is required')
    }
    logger.info('lme runAgentic: starting', { examples: examples.length })
    return ingestAgentic(opts.memory, examples, { logger })
  }

  const judgeFn: LMERunner['judge'] = async ({ examples, concurrency }) => {
    const runOne = async (ex: LMEExample): Promise<LMEResult> => {
      const started = Date.now()
      const read = await runRead({ retrieval, reader }, ex)
      if (read.error !== undefined) {
        return {
          id: ex.id,
          category: ex.category,
          question: ex.question,
          ...(ex.questionDate !== undefined ? { questionDate: ex.questionDate } : {}),
          groundTruth: ex.answer,
          predicted: read.predicted,
          verdict: 'error',
          rationale: read.error,
          latencyMs: Date.now() - started,
          retrievalMs: read.retrievalMs,
          readMs: read.readMs,
          judgeMs: 0,
          error: read.error,
        }
      }

      const predicted = read.predicted
      const judgeStart = Date.now()
      let verdict: LMEResult['verdict'] = 'incorrect'
      let rationale = ''
      let judgeError: string | undefined
      let rawResponse: string | undefined
      if (predicted === '') {
        verdict = exactMatchVerdict(ex, predicted)
        rationale = 'empty prediction; exact-match fallback'
      } else {
        try {
          const j = await judge({ example: ex, predicted })
          verdict = j.verdict
          rationale = j.rationale
          rawResponse = j.rawResponse
        } catch (err) {
          judgeError = err instanceof Error ? err.message : String(err)
          verdict = exactMatchVerdict(ex, predicted)
          rationale = `judge failed: ${judgeError}`
        }
      }
      const judgeMs = Date.now() - judgeStart
      const result: LMEResult = {
        id: ex.id,
        category: ex.category,
        question: ex.question,
        ...(ex.questionDate !== undefined ? { questionDate: ex.questionDate } : {}),
        groundTruth: ex.answer,
        predicted,
        verdict,
        rationale,
        latencyMs: Date.now() - started,
        retrievalMs: read.retrievalMs,
        readMs: read.readMs,
        judgeMs,
        ...(judgeError !== undefined ? { error: judgeError } : {}),
      }
      // `rawResponse` is intentionally dropped from the persisted result to
      // keep the JSON report compact; use the JudgeFn callback directly
      // when a caller needs raw traces.
      void rawResponse
      return result
    }
    return mapConcurrent(examples, concurrency ?? 1, runOne)
  }

  const report: LMERunner['report'] = async ({ ingestMode, results, datasetSha256, errors }) => {
    const runId = resolveRunId()
    const report = buildReport({
      ingestMode,
      results,
      runId,
      timestamp: now(),
      ...(datasetSha256 !== undefined ? { datasetSha256 } : {}),
      ...(errors !== undefined ? { errors } : {}),
    })
    const written = await writeReport(report, outDir)
    logger.info('lme report: written', {
      reportPath: written.reportPath,
      runDir: written.runDir,
    })
    return { report, written }
  }

  return { runBulk, runReplay, runAgentic, judge: judgeFn, report }
}

const defaultRunId = (ts: Date): string => {
  const pad = (n: number, w = 2): string => String(n).padStart(w, '0')
  return `lme-${ts.getUTCFullYear()}${pad(ts.getUTCMonth() + 1)}${pad(ts.getUTCDate())}-${pad(ts.getUTCHours())}${pad(ts.getUTCMinutes())}${pad(ts.getUTCSeconds())}`
}

const defaultRetrieval = (opts: CreateLMERunnerOpts): RetrievalFn => {
  const fn: RetrievalFn = async ({ example }): Promise<RetrievalResult> => {
    // Minimal fallback: read any bulk-ingested session files belonging
    // to this question directly from the store. This keeps the harness
    // usable when no hybrid index is wired up; callers that want the
    // full retrieval pipeline should inject `retrieval` explicitly.
    if (example.sessionIds.length === 0) {
      return { passages: [], rendered: '' }
    }
    const passages: import('./types.js').RetrievedPassage[] = []
    const chunks: string[] = []
    for (const sid of example.sessionIds) {
      const pathStr = `raw/lme/session-${sid}.md`
      try {
        const buf = await opts.store.read(pathStr as never)
        const body = buf.toString('utf8')
        passages.push({ path: pathStr, score: 1, body })
        chunks.push(body)
      } catch {}
    }
    return { passages, rendered: chunks.join('\n\n') }
  }
  return fn
}

const defaultReader = (opts: CreateLMERunnerOpts): ReaderFn => {
  if (!opts.provider) {
    return async ({ context }) => context
  }
  return createProviderReader(opts.provider, {
    ...(opts.readerModel !== undefined ? { model: opts.readerModel } : {}),
    ...(opts.readerBudgetChars !== undefined ? { budgetChars: opts.readerBudgetChars } : {}),
  })
}

const defaultJudge = (opts: CreateLMERunnerOpts): JudgeFn => {
  if (!opts.provider) {
    // No provider: fall back to deterministic exact-match so the harness
    // still produces a verdict rather than panicking.
    return async ({ example, predicted }) => ({
      verdict: exactMatchVerdict(example, predicted),
      rationale: 'exact-match (no judge provider configured)',
    })
  }
  return createProviderJudge({
    provider: opts.provider,
    ...(opts.judgeModel !== undefined ? { model: opts.judgeModel } : {}),
  })
}

const mapConcurrent = async <TIn, TOut>(
  items: readonly TIn[],
  requestedWorkers: number,
  worker: (item: TIn) => Promise<TOut>,
): Promise<readonly TOut[]> => {
  if (items.length === 0) return []
  const results = new Array<TOut>(items.length)
  const workers = clampConcurrency(requestedWorkers, items.length)
  let nextIndex = 0

  const runWorker = async (): Promise<void> => {
    while (true) {
      const current = nextIndex
      nextIndex++
      if (current >= items.length) return
      const item = items[current]
      if (item === undefined) return
      results[current] = await worker(item)
    }
  }

  await Promise.all(Array.from({ length: workers }, () => runWorker()))
  return results
}

const clampConcurrency = (requestedWorkers: number, max: number): number => {
  if (!Number.isFinite(requestedWorkers) || requestedWorkers <= 1) return 1
  return Math.max(1, Math.min(Math.trunc(requestedWorkers), Math.max(max, 1)))
}

// Re-export a couple of helpers used by tests so they do not need to
// import from deeply-nested files.
export type { LMEExample, LMEResult, IngestMode }
