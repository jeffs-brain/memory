// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { buildReport } from './report.js'
import {
  compareReports,
  compareResultSets,
  compareResults,
} from './compare.js'
import type { LMEResult } from './types.js'

const result = (
  id: string,
  category: string,
  verdict: LMEResult['verdict'],
  predicted = verdict === 'correct' ? 'gold' : 'other',
): LMEResult => ({
  id,
  category,
  question: `q for ${id}`,
  groundTruth: 'gold',
  predicted,
  verdict,
  rationale: 'test',
  latencyMs: 10,
  retrievalMs: 5,
  readMs: 3,
  judgeMs: 2,
})

describe('compare helpers', () => {
  it('compares individual results and result sets', () => {
    const left = result('a', 'cat', 'correct', 'gold')
    const right = result('a', 'cat', 'incorrect', 'other')
    expect(compareResults(left, right)).toMatchObject({
      id: 'a',
      sameQuestion: true,
      sameCategory: true,
      samePrediction: false,
      sameVerdict: false,
    })

    const set = compareResultSets([left], [right, result('b', 'cat', 'correct')])
    expect(set.sharedIds).toEqual(['a'])
    expect(set.addedIds).toEqual(['b'])
    expect(set.removedIds).toEqual([])
    expect(set.changed).toHaveLength(1)
  })

  it('compares reports across shared categories', () => {
    const baseline = buildReport({
      ingestMode: 'bulk',
      results: [result('a', 'cat', 'incorrect')],
      runId: 'lme-1',
      timestamp: new Date('2026-04-17T12:00:00Z'),
    })
    const candidate = buildReport({
      ingestMode: 'bulk',
      results: [result('a', 'cat', 'correct')],
      runId: 'lme-2',
      timestamp: new Date('2026-04-17T13:00:00Z'),
    })

    const comparison = compareReports(baseline, candidate)
    expect(comparison.overallAccuracyDelta).toBe(1)
    expect(comparison.categoryComparisons.cat?.deltaAccuracy).toBe(1)
    expect(comparison.resultSet.changed).toHaveLength(1)
  })
})
