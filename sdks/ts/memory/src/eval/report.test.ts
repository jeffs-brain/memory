// SPDX-License-Identifier: Apache-2.0

import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildReport, writeReport } from './report.js'
import type { LMEResult } from './types.js'

const result = (
  id: string,
  category: string,
  verdict: LMEResult['verdict'],
  latencyMs = 100,
  retrievalMs = 40,
): LMEResult => ({
  id,
  category,
  question: `q for ${id}`,
  groundTruth: 'gold',
  predicted: verdict.startsWith('correct') ? 'gold' : 'other',
  verdict,
  rationale: 'test',
  latencyMs,
  retrievalMs,
  readMs: 20,
  judgeMs: 5,
})

describe('buildReport', () => {
  it('computes accuracy, per-category breakdown, and latency stats', () => {
    const results: LMEResult[] = [
      result('a1', 'single-session-user', 'correct', 100, 30),
      result('a2', 'single-session-user', 'incorrect', 200, 60),
      result('b1_abs', 'abstention', 'abstain_correct', 50, 10),
      result('c1', 'temporal-reasoning', 'error', 10, 5),
    ]
    const report = buildReport({
      ingestMode: 'bulk',
      results,
      runId: 'lme-test',
      timestamp: new Date('2026-04-17T12:00:00Z'),
      datasetSha256: 'cafe',
    })
    expect(report.examples).toBe(4)
    // 2 correct out of 4 = 0.5
    expect(report.overallAccuracy).toBeCloseTo(0.5, 5)
    expect(report.perCategory['single-session-user']).toMatchObject({
      run: 2,
      correct: 1,
      incorrect: 1,
      accuracy: 0.5,
    })
    expect(report.perCategory.abstention?.abstainCorrect).toBe(1)
    expect(report.perCategory['temporal-reasoning']?.errors).toBe(1)
    // Task-average: mean of (0.5, 1.0, 0.0) = 0.5
    expect(report.taskAvgAccuracy).toBeCloseTo(0.5, 5)
    expect(report.latency.count).toBe(4)
    expect(report.latency.maxMs).toBe(200)
    expect(report.retrievalLatency.meanMs).toBeGreaterThan(0)
    expect(report.ingestMode).toBe('bulk')
    expect(report.datasetSha256).toBe('cafe')
  })

  it('tolerates empty result sets', () => {
    const report = buildReport({
      ingestMode: 'bulk',
      results: [],
      runId: 'lme-empty',
      timestamp: new Date('2026-04-17T12:00:00Z'),
    })
    expect(report.overallAccuracy).toBe(0)
    expect(report.latency.count).toBe(0)
  })
})

describe('writeReport', () => {
  it('creates outDir, writes JSON, and updates the latest pointer', async () => {
    const outDir = await fs.mkdtemp(path.join(os.tmpdir(), 'lme-report-'))
    const report = buildReport({
      ingestMode: 'bulk',
      results: [result('a1', 'cat', 'correct')],
      runId: 'lme-run-1',
      timestamp: new Date('2026-04-17T12:00:00Z'),
    })
    const written = await writeReport(report, outDir)

    // Report file exists and parses back to the same shape.
    const raw = await fs.readFile(written.reportPath, 'utf8')
    const parsed = JSON.parse(raw) as typeof report
    expect(parsed.evalRunId).toBe('lme-run-1')
    expect(parsed.results).toHaveLength(1)

    // runDir sits under outDir with a timestamp-safe name.
    expect(written.runDir.startsWith(outDir)).toBe(true)
    const basename = path.basename(written.runDir)
    expect(basename).toBe('lme-run-1')

    // `latest` should resolve to the same directory.
    const st = await fs.lstat(written.latestSymlink)
    if (st.isSymbolicLink()) {
      const target = await fs.readlink(written.latestSymlink)
      expect(target).toBe('lme-run-1')
    } else {
      // Pointer file fallback.
      const txt = await fs.readFile(written.latestSymlink, 'utf8')
      expect(txt.trim()).toBe('lme-run-1')
    }
  })

  it('refreshes latest on subsequent writes', async () => {
    const outDir = await fs.mkdtemp(path.join(os.tmpdir(), 'lme-report-'))
    const r1 = buildReport({
      ingestMode: 'bulk',
      results: [],
      runId: 'lme-run-1',
      timestamp: new Date('2026-04-17T12:00:00Z'),
    })
    const r2 = buildReport({
      ingestMode: 'bulk',
      results: [],
      runId: 'lme-run-2',
      timestamp: new Date('2026-04-17T13:00:00Z'),
    })
    await writeReport(r1, outDir)
    const second = await writeReport(r2, outDir)
    const st = await fs.lstat(second.latestSymlink)
    if (st.isSymbolicLink()) {
      expect(await fs.readlink(second.latestSymlink)).toBe('lme-run-2')
    } else {
      expect((await fs.readFile(second.latestSymlink, 'utf8')).trim()).toBe('lme-run-2')
    }
  })
})
