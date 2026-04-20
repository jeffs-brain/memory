// SPDX-License-Identifier: Apache-2.0

/**
 * Bridge primitives for the official LongMemEval scorer.
 *
 * The upstream Python scripts operate on JSONL rows with `question_id`,
 * `hypothesis`, and `autoeval_label`. These helpers keep that format
 * available from TypeScript without shelling out to Python.
 */

import type { LMEExample, LMEResult } from './types.js'

export type OfficialLMEHypothesisRow = {
  readonly questionId: string
  readonly hypothesis: string
}

export type OfficialLMEEvalLabel = {
  readonly model: string
  readonly label: boolean
}

export type OfficialLMEEvalLogRow = OfficialLMEHypothesisRow & {
  readonly autoevalLabel: OfficialLMEEvalLabel
}

export type OfficialLMECategoryScore = {
  readonly count: number
  readonly accuracy: number
}

export type OfficialLMEScoreSummary = {
  readonly perCategory: Readonly<Record<string, OfficialLMECategoryScore>>
  readonly taskAverageAccuracy: number
  readonly overallAccuracy: number
  readonly abstentionAccuracy: number
  readonly totalCount: number
  readonly abstentionCount: number
  readonly skippedQuestionIds: readonly string[]
}

export const officialHypothesesFromResults = (
  results: readonly LMEResult[],
): readonly OfficialLMEHypothesisRow[] =>
  results.map((result) => ({
    questionId: result.id,
    hypothesis: result.predicted,
  }))

export const serialiseOfficialHypotheses = (results: readonly LMEResult[]): string => {
  const rows = officialHypothesesFromResults(results)
  return rows
    .map((row) =>
      JSON.stringify({
        question_id: row.questionId,
        hypothesis: row.hypothesis,
      }),
    )
    .join('\n')
}

export const resultsToOfficialEvalLog = (
  results: readonly LMEResult[],
  model = 'gpt-4o-2024-08-06',
): readonly OfficialLMEEvalLogRow[] => {
  return results.map((result) => ({
    questionId: result.id,
    hypothesis: result.predicted,
    autoevalLabel: {
      model,
      label: isPositiveVerdict(result.verdict),
    },
  }))
}

export const serialiseOfficialEvalLog = (
  results: readonly LMEResult[],
  model = 'gpt-4o-2024-08-06',
): string =>
  resultsToOfficialEvalLog(results, model)
    .map((row) =>
      JSON.stringify({
        question_id: row.questionId,
        hypothesis: row.hypothesis,
        autoeval_label: row.autoevalLabel,
      }),
    )
    .join('\n')

type RawOfficialEvalLogRow = {
  readonly question_id?: unknown
  readonly hypothesis?: unknown
  readonly autoeval_label?: unknown
}

export const parseOfficialEvalLog = (text: string): readonly OfficialLMEEvalLogRow[] => {
  const rows: OfficialLMEEvalLogRow[] = []
  for (const [index, lineRaw] of text.split(/\r?\n/).entries()) {
    const line = lineRaw.trim()
    if (line === '') continue
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch (err) {
      throw new Error(`official scorer log line ${index + 1}: ${errorMessage(err)}`)
    }
    const row = parsed as RawOfficialEvalLogRow
    const questionId = toStringField(row.question_id)
    const hypothesis = toStringField(row.hypothesis)
    const label = toAutoEvalLabel(row.autoeval_label)
    if (questionId === '') {
      throw new Error(`official scorer log line ${index + 1}: empty question_id`)
    }
    rows.push({
      questionId,
      hypothesis,
      autoevalLabel: label,
    })
  }
  return rows
}

export type ScoreOfficialEvalLogArgs = {
  readonly references: readonly LMEExample[]
  readonly entries: readonly OfficialLMEEvalLogRow[]
}

export const scoreOfficialEvalLog = (args: ScoreOfficialEvalLogArgs): OfficialLMEScoreSummary => {
  const refById = new Map(args.references.map((example) => [example.id, example]))
  const categoryBuckets = new Map<string, number[]>()
  const skippedQuestionIds: string[] = []
  const allScores: number[] = []
  const abstentionScores: number[] = []

  for (const ref of args.references) {
    if (!categoryBuckets.has(ref.category)) {
      categoryBuckets.set(ref.category, [])
    }
  }

  for (const entry of args.entries) {
    const ref = refById.get(entry.questionId)
    if (!ref) {
      skippedQuestionIds.push(entry.questionId)
      continue
    }
    const label = entry.autoevalLabel.label ? 1 : 0
    categoryBuckets.get(ref.category)?.push(label)
    allScores.push(label)
    if (ref.id.includes('_abs')) {
      abstentionScores.push(label)
    }
  }

  const perCategory: Record<string, OfficialLMECategoryScore> = {}
  for (const [category, scores] of [...categoryBuckets.entries()].sort(([a], [b]) =>
    a.localeCompare(b),
  )) {
    perCategory[category] = {
      count: scores.length,
      accuracy: mean(scores),
    }
  }

  const taskAverageAccuracy = mean(Object.values(perCategory).map((v) => v.accuracy))

  return {
    perCategory,
    taskAverageAccuracy,
    overallAccuracy: mean(allScores),
    abstentionAccuracy: mean(abstentionScores),
    totalCount: allScores.length,
    abstentionCount: abstentionScores.length,
    skippedQuestionIds,
  }
}

export const compareOfficialScoreSummaries = (
  baseline: OfficialLMEScoreSummary,
  candidate: OfficialLMEScoreSummary,
): {
  readonly taskAverageAccuracyDelta: number
  readonly overallAccuracyDelta: number
  readonly abstentionAccuracyDelta: number
  readonly categoryDeltas: Readonly<Record<string, number>>
} => {
  const categories = new Set<string>([
    ...Object.keys(baseline.perCategory),
    ...Object.keys(candidate.perCategory),
  ])
  const categoryDeltas: Record<string, number> = {}
  for (const category of [...categories].sort()) {
    categoryDeltas[category] =
      (candidate.perCategory[category]?.accuracy ?? 0) -
      (baseline.perCategory[category]?.accuracy ?? 0)
  }
  return {
    taskAverageAccuracyDelta: candidate.taskAverageAccuracy - baseline.taskAverageAccuracy,
    overallAccuracyDelta: candidate.overallAccuracy - baseline.overallAccuracy,
    abstentionAccuracyDelta: candidate.abstentionAccuracy - baseline.abstentionAccuracy,
    categoryDeltas,
  }
}

const isPositiveVerdict = (verdict: LMEResult['verdict']): boolean =>
  verdict === 'correct' || verdict === 'abstain_correct'

const mean = (values: readonly number[]): number => {
  if (values.length === 0) return 0
  let sum = 0
  for (const value of values) sum += value
  return sum / values.length
}

const toStringField = (value: unknown): string => (typeof value === 'string' ? value : '')

const toAutoEvalLabel = (value: unknown): OfficialLMEEvalLabel => {
  if (value == null || typeof value !== 'object') {
    throw new Error('invalid autoeval_label')
  }
  const raw = value as { readonly model?: unknown; readonly label?: unknown }
  const model = toStringField(raw.model)
  if (model === '') {
    throw new Error('invalid autoeval_label.model')
  }
  return {
    model,
    label: raw.label === true,
  }
}

const errorMessage = (err: unknown): string => (err instanceof Error ? err.message : String(err))
