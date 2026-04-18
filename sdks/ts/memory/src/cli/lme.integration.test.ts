// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { buildReport, writeReport, type LMEResult } from '../eval/index.js'
import { rootCommand } from './main.js'

const createdDirs: string[] = []

type RunnableCommand = {
  readonly run?: (ctx: { readonly args: Record<string, unknown> }) => Promise<void> | void
  readonly subCommands?: Record<string, RunnableCommand> | ((...args: readonly unknown[]) => unknown)
}

const makeTempDir = async (): Promise<string> => {
  const dir = await mkdtemp(join(tmpdir(), 'memory-lme-cli-'))
  createdDirs.push(dir)
  return dir
}

const pickSub = (parent: RunnableCommand, name: string): RunnableCommand => {
  const subs = parent.subCommands
  if (subs === undefined || typeof subs === 'function') {
    throw new Error('expected subCommands')
  }
  const entry = subs[name]
  if (entry === undefined) throw new Error(`no subcommand ${name}`)
  return entry
}

const pickLMECommand = (name: string): RunnableCommand =>
  pickSub(pickSub(pickSub(rootCommand as unknown as RunnableCommand, 'eval'), 'lme'), name)

const captureLMECommand = async (
  subcommand: string,
  args: Record<string, unknown>,
): Promise<{
  readonly payload?: Record<string, unknown>
  readonly error?: unknown
}> => {
  const chunks: string[] = []
  const spy = vi
    .spyOn(process.stdout, 'write')
    .mockImplementation(((chunk: string | Uint8Array) => {
      chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
      return true
    }) as typeof process.stdout.write)

  let error: unknown
  try {
    await pickLMECommand(subcommand).run?.({ args })
  } catch (err) {
    error = err
  } finally {
    spy.mockRestore()
  }

  const stdout = chunks.join('').trim()
  return {
    ...(stdout !== '' ? { payload: JSON.parse(stdout) as Record<string, unknown> } : {}),
    ...(error !== undefined ? { error } : {}),
  }
}

const runLMECommand = async (
  subcommand: string,
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> => {
  const result = await captureLMECommand(subcommand, args)
  if (result.error !== undefined) throw result.error
  if (result.payload === undefined) {
    throw new Error(`expected JSON output from lme ${subcommand}`)
  }
  return result.payload
}

const result = (
  id: string,
  category: string,
  verdict: LMEResult['verdict'],
  predicted: string,
  overrides: Partial<LMEResult> = {},
): LMEResult => ({
  id,
  category,
  question: `Question ${id}`,
  groundTruth: `Ground truth ${id}`,
  predicted,
  verdict,
  rationale: `${verdict} rationale`,
  latencyMs: 12,
  retrievalMs: 5,
  readMs: 4,
  judgeMs: 3,
  ...overrides,
})

const writeJSON = async (filePath: string, value: unknown): Promise<void> => {
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8')
}

const writeDataset = async (filePath: string): Promise<void> => {
  await writeFile(
    filePath,
    `${JSON.stringify({
      question_id: 'q-1',
      question_type: 'single-session-preference',
      question: 'Which hotel should I choose?',
      answer: 'The one with the rooftop pool',
      haystack_session_ids: ['session-1'],
      haystack_dates: ['2026-04-01'],
      haystack_sessions: [
        [
          {
            role: 'user',
            content: 'I like hotels with rooftop pools.',
          },
        ],
      ],
    })}\n`,
    'utf8',
  )
}

afterEach(async () => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir !== undefined) await rm(dir, { recursive: true, force: true })
  }
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

describe('memory eval lme', () => {
  it('executes `compare` against two local reports', async () => {
    const root = await makeTempDir()
    const leftPath = join(root, 'left.json')
    const rightPath = join(root, 'right.json')

    await writeJSON(
      leftPath,
      buildReport({
        ingestMode: 'bulk',
        runId: 'baseline-run',
        timestamp: new Date('2026-04-18T10:00:00.000Z'),
        results: [
          result('q-1', 'single-session-preference', 'correct', 'Rooftop pool hotel'),
          result('q-2', 'multi-session', 'incorrect', 'Berlin'),
        ],
      }),
    )
    await writeJSON(
      rightPath,
      buildReport({
        ingestMode: 'bulk',
        runId: 'candidate-run',
        timestamp: new Date('2026-04-18T11:00:00.000Z'),
        results: [
          result('q-1', 'single-session-preference', 'correct', 'Rooftop pool hotel'),
          result('q-2', 'multi-session', 'correct', 'Amsterdam'),
        ],
      }),
    )

    const payload = await runLMECommand('compare', {
      left: leftPath,
      right: rightPath,
    })

    expect(payload).toMatchObject({
      baselineRunId: 'baseline-run',
      candidateRunId: 'candidate-run',
      overallAccuracyDelta: 0.5,
      taskAvgAccuracyDelta: 0.5,
      resultSet: {
        sharedIds: ['q-1', 'q-2'],
        addedIds: [],
        removedIds: [],
        changed: [
          expect.objectContaining({
            id: 'q-2',
            samePrediction: false,
            sameVerdict: false,
            leftVerdict: 'incorrect',
            rightVerdict: 'correct',
          }),
        ],
      },
    })
  })

  it('executes `score` using the latest pointer file', async () => {
    const root = await makeTempDir()
    const outDir = join(root, 'out')
    const report = buildReport({
      ingestMode: 'replay',
      runId: 'score-run',
      timestamp: new Date('2026-04-18T12:34:56.000Z'),
      results: [result('q-1', 'single-session-preference', 'correct', 'Rooftop pool hotel')],
    })

    const written = await writeReport(report, outDir)
    await rm(join(outDir, 'latest'), { recursive: true, force: true })
    await writeFile(join(outDir, 'latest.txt'), `${report.evalRunId}\n`, 'utf8')

    const payload = await runLMECommand('score', { outDir })

    expect(payload).toMatchObject({
      report: {
        path: written.reportPath,
        runId: 'score-run',
        overallAccuracy: 1,
        taskAvgAccuracy: 1,
        examples: 1,
      },
    })
  })

  it('executes `doctor` against local dataset and valid env configuration', async () => {
    const root = await makeTempDir()
    const datasetPath = join(root, 'dataset.jsonl')
    const outDir = join(root, 'out')
    const cacheDir = join(root, 'cache')
    await writeDataset(datasetPath)

    vi.stubEnv('JB_LLM_PROVIDER', 'ollama')
    vi.stubEnv('JB_LLM_MODEL', 'llama3.2')
    vi.stubEnv('JB_EMBED_PROVIDER', 'hash')
    vi.stubEnv('JB_RERANK_PROVIDER', 'tei')

    const payload = await runLMECommand('doctor', {
      dataset: datasetPath,
      outDir,
      cacheDir,
    })

    expect(payload).toMatchObject({
      ok: true,
      checks: expect.arrayContaining([
        expect.objectContaining({
          name: 'dataset',
          ok: true,
          value: expect.objectContaining({
            path: datasetPath,
            examples: 1,
            categories: ['single-session-preference'],
          }),
        }),
        expect.objectContaining({
          name: 'provider',
          ok: true,
          value: {
            kind: 'ollama',
            model: 'llama3.2',
          },
        }),
        expect.objectContaining({
          name: 'embedder',
          ok: true,
          value: expect.objectContaining({
            kind: 'hash',
          }),
        }),
        expect.objectContaining({
          name: 'reranker',
          ok: true,
          value: expect.objectContaining({
            kind: 'tei',
          }),
        }),
        expect.objectContaining({
          name: 'paths',
          ok: true,
          value: {
            outDir,
            cacheDir,
          },
        }),
      ]),
    })
  })

  it('executes `doctor --official` against a local scorer checkout', async () => {
    const root = await makeTempDir()
    const datasetPath = join(root, 'dataset.jsonl')
    const outDir = join(root, 'out')
    const cacheDir = join(root, 'cache')
    const repoDir = join(root, 'LongMemEval')
    await writeDataset(datasetPath)
    await mkdir(join(repoDir, 'src', 'evaluation'), { recursive: true })
    await writeFile(
      join(repoDir, 'src', 'evaluation', 'evaluate_qa.py'),
      'console.log("ok")\n',
      'utf8',
    )

    const payload = await runLMECommand('doctor', {
      dataset: datasetPath,
      outDir,
      cacheDir,
      repoDir,
      official: true,
      python: process.execPath,
    })

    expect(payload).toMatchObject({
      ok: true,
      checks: expect.arrayContaining([
        expect.objectContaining({
          name: 'official',
          ok: true,
          value: {
            repoDir,
            script: join(repoDir, 'src', 'evaluation', 'evaluate_qa.py'),
          },
        }),
      ]),
    })
  })

  it('emits structured failure output from `doctor` before raising', async () => {
    const root = await makeTempDir()
    const missingDatasetPath = join(root, 'missing.jsonl')
    const outDir = join(root, 'out')
    const cacheDir = join(root, 'cache')

    vi.stubEnv('JB_LLM_PROVIDER', 'broken')

    const { payload, error } = await captureLMECommand('doctor', {
      dataset: missingDatasetPath,
      outDir,
      cacheDir,
    })

    expect(error).toBeInstanceOf(Error)
    expect((error as Error).message).toBe('lme doctor: one or more checks failed')
    expect(payload).toMatchObject({
      ok: false,
      checks: expect.arrayContaining([
        expect.objectContaining({
          name: 'dataset',
          ok: false,
        }),
        expect.objectContaining({
          name: 'provider',
          ok: false,
          detail: "invalid JB_LLM_PROVIDER='broken'; expected anthropic|openai|ollama",
        }),
        expect.objectContaining({
          name: 'paths',
          ok: true,
          value: {
            outDir,
            cacheDir,
          },
        }),
      ]),
    })
  })
})
