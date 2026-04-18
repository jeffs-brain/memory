// SPDX-License-Identifier: Apache-2.0

/**
 * Public types for the LME (LongMemEval) harness. Mirrors the Go surface
 * in apps/jeff/internal/knowledge/eval/lme trimmed to the primitives the
 * TypeScript port exposes: example loading, phase 0/1/2 orchestration,
 * judge verdicts, and aggregate reporting.
 */

import type { Embedder, Logger, Provider } from '../llm/index.js'
import type { Memory } from '../memory/types.js'
import type { SearchIndex } from '../search/index.js'
import type { Store } from '../store/index.js'

/** Ingest mode controlling how haystack data lands in the brain. */
export type IngestMode = 'bulk' | 'replay' | 'agentic'

/** One message within an LME haystack session. */
export type LMESessionMessage = {
  readonly role: 'user' | 'assistant'
  readonly content: string
  readonly hasAnswer?: boolean
}

/**
 * A single LongMemEval example. Field names mirror the oracle JSONL
 * schema (question_id, question_type, ...) but are exposed with camelCase
 * so downstream code stays idiomatic.
 */
export type LMEExample = {
  readonly id: string
  readonly category: string
  readonly question: string
  readonly answer: string
  readonly questionDate?: string
  readonly haystackDates?: readonly string[]
  readonly sessionIds: readonly string[]
  readonly answerSessionIds?: readonly string[]
  readonly haystackSessions?: readonly (readonly LMESessionMessage[])[]
}

/** Canonical judge verdict. Matches the Go score_judge.go aggregation. */
export type JudgeVerdict =
  | 'correct'
  | 'incorrect'
  | 'abstain_correct'
  | 'abstain_incorrect'
  | 'error'

export type JudgeTrace = {
  readonly questionId: string
  readonly verdict: JudgeVerdict
  readonly rationale: string
  readonly rawResponse?: string
  readonly latencyMs: number
  readonly error?: string
}

/** Per-example result produced by the read stage + judge. */
export type LMEResult = {
  readonly id: string
  readonly category: string
  readonly question: string
  readonly questionDate?: string
  readonly groundTruth: string
  readonly predicted: string
  readonly verdict: JudgeVerdict
  readonly rationale: string
  readonly latencyMs: number
  readonly retrievalMs: number
  readonly readMs: number
  readonly judgeMs: number
  readonly error?: string
}

/** Category-level aggregate for the report. */
export type LMECategoryBreakdown = {
  readonly run: number
  readonly correct: number
  readonly incorrect: number
  readonly abstainCorrect: number
  readonly abstainIncorrect: number
  readonly errors: number
  readonly accuracy: number
}

export type LatencyStats = {
  readonly count: number
  readonly p50Ms: number
  readonly p95Ms: number
  readonly meanMs: number
  readonly maxMs: number
}

/** Top-level aggregated report written to disk. */
export type LMEReport = {
  readonly evalRunId: string
  readonly timestamp: string
  readonly ingestMode: IngestMode
  readonly datasetSha256?: string
  readonly examples: number
  readonly overallAccuracy: number
  readonly taskAvgAccuracy: number
  readonly perCategory: Readonly<Record<string, LMECategoryBreakdown>>
  readonly latency: LatencyStats
  readonly retrievalLatency: LatencyStats
  readonly results: readonly LMEResult[]
  readonly errors: readonly string[]
}

export type IngestOutcome = {
  readonly mode: IngestMode
  readonly sessionsWritten: number
  readonly examplesIngested: number
  readonly warnings: readonly string[]
}

/**
 * Retrieval hook injected by the caller. Given a question, returns the
 * concatenated text that the reader should summarise into an answer. The
 * eval harness deliberately keeps this narrow: it does not own hybrid
 * search, rerank, or temporal augmentation — those all belong to the
 * retrieval pipeline constructed outside the harness.
 */
export type RetrievalFn = (args: {
  readonly question: string
  readonly questionDate?: string
  readonly example: LMEExample
}) => Promise<RetrievalResult>

export type RetrievalResult = {
  readonly passages: readonly RetrievedPassage[]
  /** Pre-rendered context string, ready for the reader prompt. */
  readonly rendered: string
}

export type RetrievedPassage = {
  readonly path: string
  readonly score: number
  readonly body: string
  readonly date?: string
  readonly sessionId?: string
}

/**
 * LLM judge. Given a question, ground truth, predicted answer, plus the
 * category flag, returns a binary verdict with rationale. The harness
 * supplies a default provider-backed judge that honours the official LME
 * prompts, but callers may swap in a stub (e.g. tests).
 */
export type JudgeFn = (args: {
  readonly example: LMEExample
  readonly predicted: string
}) => Promise<{ readonly verdict: JudgeVerdict; readonly rationale: string; readonly rawResponse?: string }>

/**
 * Reader LLM. Given the question and the rendered retrieval context,
 * produce a final answer string. The default implementation calls the
 * supplied Provider with the official LME CoT template.
 */
export type ReaderFn = (args: {
  readonly question: string
  readonly questionDate?: string
  readonly context: string
}) => Promise<string>

export type CreateLMERunnerOpts = {
  readonly store: Store
  readonly provider?: Provider
  readonly embedder?: Embedder
  readonly index?: SearchIndex
  readonly retrieval?: RetrievalFn
  readonly memory?: Memory
  readonly reader?: ReaderFn
  readonly judge?: JudgeFn
  readonly logger?: Logger
  /** Directory to write report artefacts into. Defaults to `.eval/lme`. */
  readonly outDir?: string
  /** Override the run timestamp for deterministic tests. */
  readonly now?: () => Date
  /** Override the run id. Defaults to `lme-<yyyymmdd>-<HHMMSS>`. */
  readonly runId?: string
  /** Readability budget for retrieved passages. Defaults to 16000 chars. */
  readonly readerBudgetChars?: number
  /** Optional reader model override for provider-backed readers. */
  readonly readerModel?: string
  /** Optional judge model override for provider-backed judges. */
  readonly judgeModel?: string
}

export type RunBulkArgs = {
  readonly examples: readonly LMEExample[]
}

export type RunReplayArgs = {
  readonly examples: readonly LMEExample[]
  /** Extraction concurrency. Defaults to 4. */
  readonly concurrency?: number
}

export type RunAgenticArgs = {
  readonly examples: readonly LMEExample[]
}

export type ReportWriteResult = {
  readonly reportPath: string
  readonly latestSymlink: string
  readonly runDir: string
}

export type LMERunner = {
  readonly runBulk: (args: RunBulkArgs) => Promise<IngestOutcome>
  readonly runReplay: (args: RunReplayArgs) => Promise<IngestOutcome>
  readonly runAgentic: (args: RunAgenticArgs) => Promise<IngestOutcome>
  readonly judge: (args: {
    readonly examples: readonly LMEExample[]
    /** Question concurrency. Defaults to 1. */
    readonly concurrency?: number
  }) => Promise<readonly LMEResult[]>
  readonly report: (args: {
    readonly ingestMode: IngestMode
    readonly results: readonly LMEResult[]
    readonly datasetSha256?: string
    readonly errors?: readonly string[]
  }) => Promise<{ readonly report: LMEReport; readonly written: ReportWriteResult }>
}
