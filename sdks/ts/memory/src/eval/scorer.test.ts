// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { LMEExample, LMEResult } from './types.js'
import {
  parseOfficialEvalLog,
  resultsToOfficialEvalLog,
  scoreOfficialEvalLog,
  serialiseOfficialEvalLog,
  serialiseOfficialHypotheses,
} from './scorer.js'

const refs: readonly LMEExample[] = [
  {
    id: 'q1',
    category: 'single-session-user',
    question: 'Colour?',
    answer: 'blue',
    sessionIds: ['s1'],
  },
  {
    id: 'q2_abs',
    category: 'abstention',
    question: 'Missing?',
    answer: 'not answerable',
    sessionIds: ['s2'],
  },
]

const result = (
  id: string,
  category: string,
  predicted: string,
  verdict: LMEResult['verdict'],
): LMEResult => ({
  id,
  category,
  question: `question ${id}`,
  groundTruth: id === 'q1' ? 'blue' : 'not answerable',
  predicted,
  verdict,
  rationale: 'test',
  latencyMs: 10,
  retrievalMs: 4,
  readMs: 3,
  judgeMs: 2,
})

describe('official scorer bridge', () => {
  it('serialises and parses the official jsonl shape', () => {
    const results: LMEResult[] = [
      result('q1', 'single-session-user', 'blue', 'correct'),
      result('q2_abs', 'abstention', 'not answerable', 'abstain_correct'),
    ]

    const hypotheses = serialiseOfficialHypotheses(results)
    expect(hypotheses).toContain('"question_id":"q1"')

    const evalLog = serialiseOfficialEvalLog(results)
    const parsed = parseOfficialEvalLog(evalLog)
    expect(parsed).toHaveLength(2)
    expect(parsed[0]?.autoevalLabel.model).toBe('gpt-4o-2024-08-06')
    expect(resultsToOfficialEvalLog(results)).toHaveLength(2)
  })

  it('scores rows using the official task/overall/abstention semantics', () => {
    const rows = resultsToOfficialEvalLog([
      result('q1', 'single-session-user', 'blue', 'correct'),
      result('q2_abs', 'abstention', 'not answerable', 'abstain_correct'),
    ])
    const summary = scoreOfficialEvalLog({
      references: refs,
      entries: rows,
    })

    expect(summary.totalCount).toBe(2)
    expect(summary.abstentionCount).toBe(1)
    expect(summary.overallAccuracy).toBe(1)
    expect(summary.taskAverageAccuracy).toBe(1)
    expect(summary.abstentionAccuracy).toBe(1)
    expect(summary.perCategory['single-session-user']?.accuracy).toBe(1)
  })
})
