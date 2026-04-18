/**
 * Comparison helpers for LongMemEval runs.
 */

import type { LMEReport, LMEResult } from './types.js'

export type LMEResultComparison = {
  readonly id: string
  readonly sameQuestion: boolean
  readonly sameCategory: boolean
  readonly samePrediction: boolean
  readonly sameVerdict: boolean
  readonly sameRationale: boolean
  readonly leftVerdict: LMEResult['verdict']
  readonly rightVerdict: LMEResult['verdict']
}

export type LMEResultSetComparison = {
  readonly baselineIds: readonly string[]
  readonly candidateIds: readonly string[]
  readonly sharedIds: readonly string[]
  readonly addedIds: readonly string[]
  readonly removedIds: readonly string[]
  readonly changed: readonly LMEResultComparison[]
}

export type LMECategoryComparison = {
  readonly category: string
  readonly baselineAccuracy: number
  readonly candidateAccuracy: number
  readonly deltaAccuracy: number
  readonly baselineCount: number
  readonly candidateCount: number
}

export type LMEReportComparison = {
  readonly baselineRunId: string
  readonly candidateRunId: string
  readonly overallAccuracyDelta: number
  readonly taskAvgAccuracyDelta: number
  readonly latencyDeltaMs: number
  readonly retrievalLatencyDeltaMs: number
  readonly categoryComparisons: Readonly<Record<string, LMECategoryComparison>>
  readonly resultSet: LMEResultSetComparison
}

export const compareResults = (
  baseline: LMEResult,
  candidate: LMEResult,
): LMEResultComparison => ({
  id: baseline.id,
  sameQuestion: baseline.question === candidate.question,
  sameCategory: baseline.category === candidate.category,
  samePrediction: baseline.predicted === candidate.predicted,
  sameVerdict: baseline.verdict === candidate.verdict,
  sameRationale: baseline.rationale === candidate.rationale,
  leftVerdict: baseline.verdict,
  rightVerdict: candidate.verdict,
})

export const compareResultSets = (
  baseline: readonly LMEResult[],
  candidate: readonly LMEResult[],
): LMEResultSetComparison => {
  const baselineById = new Map(baseline.map((r) => [r.id, r]))
  const candidateById = new Map(candidate.map((r) => [r.id, r]))
  const baselineIds = [...baselineById.keys()].sort()
  const candidateIds = [...candidateById.keys()].sort()
  const sharedIds = baselineIds.filter((id) => candidateById.has(id))
  const addedIds = candidateIds.filter((id) => !baselineById.has(id))
  const removedIds = baselineIds.filter((id) => !candidateById.has(id))

  const changed: LMEResultComparison[] = []
  for (const id of sharedIds) {
    const left = baselineById.get(id)
    const right = candidateById.get(id)
    if (!left || !right) continue
    const cmp = compareResults(left, right)
    if (
      !cmp.sameQuestion ||
      !cmp.sameCategory ||
      !cmp.samePrediction ||
      !cmp.sameVerdict ||
      !cmp.sameRationale
    ) {
      changed.push(cmp)
    }
  }

  return {
    baselineIds,
    candidateIds,
    sharedIds,
    addedIds,
    removedIds,
    changed,
  }
}

export const compareReports = (
  baseline: LMEReport,
  candidate: LMEReport,
): LMEReportComparison => {
  const categories = new Set<string>([
    ...Object.keys(baseline.perCategory),
    ...Object.keys(candidate.perCategory),
  ])
  const categoryComparisons: Record<string, LMECategoryComparison> = {}
  for (const category of [...categories].sort()) {
    const left = baseline.perCategory[category]
    const right = candidate.perCategory[category]
    categoryComparisons[category] = {
      category,
      baselineAccuracy: left?.accuracy ?? 0,
      candidateAccuracy: right?.accuracy ?? 0,
      deltaAccuracy: (right?.accuracy ?? 0) - (left?.accuracy ?? 0),
      baselineCount: left?.run ?? 0,
      candidateCount: right?.run ?? 0,
    }
  }

  return {
    baselineRunId: baseline.evalRunId,
    candidateRunId: candidate.evalRunId,
    overallAccuracyDelta: candidate.overallAccuracy - baseline.overallAccuracy,
    taskAvgAccuracyDelta: candidate.taskAvgAccuracy - baseline.taskAvgAccuracy,
    latencyDeltaMs: candidate.latency.meanMs - baseline.latency.meanMs,
    retrievalLatencyDeltaMs: candidate.retrievalLatency.meanMs - baseline.retrievalLatency.meanMs,
    categoryComparisons,
    resultSet: compareResultSets(baseline.results, candidate.results),
  }
}
