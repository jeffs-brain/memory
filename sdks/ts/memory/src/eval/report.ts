/**
 * Report writer. Aggregates per-example results into the canonical
 * `LMEReport`, persists JSON under `<outDir>/<timestamp>/report.json`,
 * and refreshes the `<outDir>/latest` symlink to point at the newest run.
 */

import { promises as fs } from 'node:fs'
import path from 'node:path'
import type {
  IngestMode,
  LMECategoryBreakdown,
  LMEReport,
  LMEResult,
  LatencyStats,
  ReportWriteResult,
} from './types.js'

export type BuildReportArgs = {
  readonly ingestMode: IngestMode
  readonly results: readonly LMEResult[]
  readonly runId: string
  readonly timestamp: Date
  readonly datasetSha256?: string
  readonly errors?: readonly string[]
}

export const buildReport = (args: BuildReportArgs): LMEReport => {
  const perCategory: Record<string, { run: number; correct: number; incorrect: number; abstainCorrect: number; abstainIncorrect: number; errors: number }> = {}

  let overallCorrect = 0
  for (const r of args.results) {
    let slot = perCategory[r.category]
    if (!slot) {
      slot = { run: 0, correct: 0, incorrect: 0, abstainCorrect: 0, abstainIncorrect: 0, errors: 0 }
      perCategory[r.category] = slot
    }
    slot.run++
    switch (r.verdict) {
      case 'correct':
        slot.correct++
        overallCorrect++
        break
      case 'abstain_correct':
        slot.abstainCorrect++
        overallCorrect++
        break
      case 'incorrect':
        slot.incorrect++
        break
      case 'abstain_incorrect':
        slot.abstainIncorrect++
        break
      case 'error':
        slot.errors++
        break
    }
  }

  const breakdown: Record<string, LMECategoryBreakdown> = {}
  for (const [name, slot] of Object.entries(perCategory)) {
    const correct = slot.correct + slot.abstainCorrect
    breakdown[name] = {
      run: slot.run,
      correct: slot.correct,
      incorrect: slot.incorrect,
      abstainCorrect: slot.abstainCorrect,
      abstainIncorrect: slot.abstainIncorrect,
      errors: slot.errors,
      accuracy: slot.run === 0 ? 0 : correct / slot.run,
    }
  }

  const overallAccuracy = args.results.length === 0 ? 0 : overallCorrect / args.results.length
  const taskAvgAccuracy = computeTaskAvg(breakdown)

  const totalLatencies = args.results.map((r) => r.latencyMs)
  const retrievalLatencies = args.results.map((r) => r.retrievalMs)

  const out: LMEReport = {
    evalRunId: args.runId,
    timestamp: args.timestamp.toISOString(),
    ingestMode: args.ingestMode,
    ...(args.datasetSha256 !== undefined ? { datasetSha256: args.datasetSha256 } : {}),
    examples: args.results.length,
    overallAccuracy,
    taskAvgAccuracy,
    perCategory: breakdown,
    latency: latencyStats(totalLatencies),
    retrievalLatency: latencyStats(retrievalLatencies),
    results: args.results,
    errors: args.errors ?? [],
  }
  return out
}

const computeTaskAvg = (breakdown: Record<string, LMECategoryBreakdown>): number => {
  const values = Object.values(breakdown)
  if (values.length === 0) return 0
  let sum = 0
  for (const v of values) sum += v.accuracy
  return sum / values.length
}

const latencyStats = (values: readonly number[]): LatencyStats => {
  if (values.length === 0) {
    return { count: 0, p50Ms: 0, p95Ms: 0, meanMs: 0, maxMs: 0 }
  }
  const sorted = [...values].sort((a, b) => a - b)
  const percentile = (p: number): number => {
    if (sorted.length === 0) return 0
    const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))
    return sorted[idx] ?? 0
  }
  let sum = 0
  let max = 0
  for (const v of values) {
    sum += v
    if (v > max) max = v
  }
  return {
    count: values.length,
    p50Ms: percentile(50),
    p95Ms: percentile(95),
    meanMs: Math.round(sum / values.length),
    maxMs: max,
  }
}

/**
 * Write the report to `<outDir>/<timestamp>/report.json` and refresh the
 * `<outDir>/latest` symlink so downstream tooling can always resolve the
 * newest run without knowing the timestamp. Symlink creation falls back
 * to copying the directory path into a `latest.txt` pointer when the
 * platform rejects symlinks (Windows in particular).
 */
export const writeReport = async (
  report: LMEReport,
  outDir: string,
): Promise<ReportWriteResult> => {
  const timestampDir = sanitiseRunId(report.evalRunId)
  const runDir = path.join(outDir, timestampDir)
  await fs.mkdir(runDir, { recursive: true })
  const reportPath = path.join(runDir, 'report.json')
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`, 'utf8')

  const latestPath = path.join(outDir, 'latest')
  const result: ReportWriteResult = {
    reportPath,
    latestSymlink: latestPath,
    runDir,
  }

  try {
    // Always replace: a previous run's symlink / directory can block a
    // fresh `symlink` call, so we remove first and accept the small race.
    await removeIfExists(latestPath)
    await fs.symlink(timestampDir, latestPath, 'dir')
  } catch (err) {
    // Fall back to a plain pointer file so the caller still has a stable
    // "where is the latest run?" answer on platforms without symlink
    // permissions.
    const pointer = `${latestPath}.txt`
    await fs.writeFile(pointer, `${timestampDir}\n`, 'utf8')
    return {
      reportPath,
      latestSymlink: pointer,
      runDir,
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      ...(err ? {} : {}),
    }
  }
  return result
}

const sanitiseRunId = (id: string): string =>
  id.replace(/[^A-Za-z0-9._-]/g, '_')

const removeIfExists = async (p: string): Promise<void> => {
  try {
    const st = await fs.lstat(p)
    if (st.isDirectory() && !st.isSymbolicLink()) {
      await fs.rm(p, { recursive: true, force: true })
    } else {
      await fs.unlink(p)
    }
  } catch (err) {
    const code = (err as NodeJS.ErrnoException)?.code
    if (code === 'ENOENT') return
    throw err
  }
}
