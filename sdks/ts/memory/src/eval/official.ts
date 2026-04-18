import { access, readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { execFile as execFileCallback } from 'node:child_process'
import { promisify } from 'node:util'
import { parseOfficialEvalLog, scoreOfficialEvalLog, type OfficialLMEScoreSummary } from './scorer.js'
import { loadDataset } from './dataset.js'

const execFile = promisify(execFileCallback)

export const OFFICIAL_SCORER_SCRIPT = 'src/evaluation/evaluate_qa.py'

export type RunOfficialScorerArgs = {
  readonly repoDir: string
  readonly datasetPath: string
  readonly predictionsPath: string
  readonly metricModel: 'gpt-4o' | 'gpt-4o-mini' | 'llama-3.1-70b-instruct'
  readonly pythonBin?: string
}

export type OfficialScorerOutcome = {
  readonly resultPath: string
  readonly stdout: string
  readonly stderr: string
  readonly summary: OfficialLMEScoreSummary
}

export const verifyOfficialScorer = async (
  repoDir: string,
  pythonBin = 'python3',
): Promise<void> => {
  const scriptPath = join(resolve(repoDir), OFFICIAL_SCORER_SCRIPT)
  await access(scriptPath)
  await execFile(pythonBin, ['--version'])
}

export const runOfficialScorer = async (
  args: RunOfficialScorerArgs,
): Promise<OfficialScorerOutcome> => {
  const repoDir = resolve(args.repoDir)
  const scriptPath = join(repoDir, OFFICIAL_SCORER_SCRIPT)
  const pythonBin = args.pythonBin ?? 'python3'
  await verifyOfficialScorer(repoDir, pythonBin)

  const { stdout, stderr } = await execFile(
    pythonBin,
    [scriptPath, args.metricModel, resolve(args.predictionsPath), resolve(args.datasetPath)],
    {
      cwd: repoDir,
      env: process.env,
      maxBuffer: 16 * 1024 * 1024,
    },
  )
  const resultPath = `${resolve(args.predictionsPath)}.eval-results-${args.metricModel}`
  const dataset = await loadDataset(resolve(args.datasetPath))
  const logText = await readFile(resultPath, 'utf8')
  const entries = parseOfficialEvalLog(logText)
  const summary = scoreOfficialEvalLog({
    references: dataset.examples,
    entries,
  })
  return { resultPath, stdout, stderr, summary }
}

export const officialScorerScriptPath = (repoDir: string): string =>
  join(resolve(repoDir), OFFICIAL_SCORER_SCRIPT)

export const officialScorerRequirements = (repoDir: string, pythonBin = 'python3'): readonly string[] => [
  officialScorerScriptPath(repoDir),
  pythonBin,
  dirname(resolve(repoDir)),
]
