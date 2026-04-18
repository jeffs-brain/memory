// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { createHashEmbedder } from '../llm/index.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
  StructuredRequest,
} from '../llm/index.js'
import {
  filterPassagesByQuestionDate,
  rankReplayPassagesForQuestion,
  selectReplayPassagesForReader,
  runStandaloneLMEEval,
} from './run.js'

const createdDirs: string[] = []

afterEach(async () => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir !== undefined) {
      await rm(dir, { recursive: true, force: true })
    }
  }
})

const makeTempDir = async (): Promise<string> => {
  const dir = await mkdtemp(path.join(tmpdir(), 'lme-run-'))
  createdDirs.push(dir)
  return dir
}

const createStubProvider = (): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  supportsStructuredDecoding: () => false,
  stream: async function* (_req: CompletionRequest): AsyncIterable<StreamEvent> {
    yield { type: 'done', stopReason: 'end_turn' }
  },
  complete: async (req: CompletionRequest): Promise<CompletionResponse> => {
    const prompt = req.messages.map((message) => message.content ?? '').join('\n')
    const content = prompt.includes('Is the model response correct?') ? 'yes' : 'blue'
    return {
      content,
      toolCalls: [],
      usage: { inputTokens: 1, outputTokens: 1 },
      stopReason: 'end_turn',
    }
  },
  structured: async (_req: StructuredRequest): Promise<string> => '{}',
})

describe('runStandaloneLMEEval', () => {
  it('runs a bulk benchmark end to end and writes artefacts', async () => {
    const root = await makeTempDir()
    const datasetPath = path.join(root, 'oracle.json')
    await writeFile(
      datasetPath,
      JSON.stringify([
        {
          question_id: 'q1',
          question_type: 'single-session-user',
          question: 'What colour did the user pick?',
          answer: 'blue',
          haystack_session_ids: ['s1'],
          haystack_dates: ['2024-02-15'],
          haystack_sessions: [
            [
              { role: 'user', content: 'I picked blue today.' },
              { role: 'assistant', content: 'Blue it is.' },
            ],
          ],
        },
      ]),
      'utf8',
    )

    const outDir = path.join(root, 'out')
    const provider = createStubProvider()
    const outcome = await runStandaloneLMEEval({
      datasetPath,
      bundle: 'cleaned',
      split: 'oracle',
      ingestMode: 'bulk',
      outDir,
      cacheDir: path.join(root, 'cache'),
      provider,
      createdAt: new Date('2026-04-17T12:00:00Z'),
    })

    expect(outcome.report.overallAccuracy).toBe(1)
    expect(outcome.officialScore.overallAccuracy).toBe(1)
    expect(outcome.manifest.dataset.path).toBe(datasetPath)
    expect(outcome.manifest.outputs.reportPath).toBe(outcome.reportPath)
    expect(outcome.runDir.startsWith(outDir)).toBe(true)

    const manifestRaw = JSON.parse(await readFile(outcome.manifestPath, 'utf8')) as {
      readonly version: number
    }
    expect(manifestRaw.version).toBe(1)
    expect(await readFile(outcome.officialHypothesesPath, 'utf8')).toContain(
      '"question_id":"q1"',
    )
    expect(await readFile(outcome.officialEvalLogPath, 'utf8')).toContain(
      '"autoeval_label"',
    )
  })

  it('keeps the default bm25 path free of vector side effects', async () => {
    const root = await makeTempDir()
    const datasetPath = path.join(root, 'oracle.json')
    await writeFile(
      datasetPath,
      JSON.stringify([
        {
          question_id: 'q1',
          question_type: 'single-session-user',
          question: 'What colour did the user pick?',
          answer: 'blue',
          haystack_session_ids: ['s1'],
          haystack_dates: ['2024-02-15'],
          haystack_sessions: [
            [
              { role: 'user', content: 'I picked blue today.' },
              { role: 'assistant', content: 'Blue it is.' },
            ],
          ],
        },
      ]),
      'utf8',
    )

    const provider = createStubProvider()
    const outcome = await runStandaloneLMEEval({
      datasetPath,
      bundle: 'cleaned',
      split: 'oracle',
      ingestMode: 'bulk',
      outDir: path.join(root, 'out'),
      cacheDir: path.join(root, 'cache'),
      provider,
      embedder: createHashEmbedder(),
      createdAt: new Date('2026-04-17T12:00:00Z'),
    })

    expect(outcome.manifest.run.retrievalMode).toBe('bm25')
    expect(outcome.manifest.models.embedder).toBeUndefined()
    expect(outcome.manifest.cache.embedCachePath).toBeUndefined()
  })

  it('includes a sample signature in sampled run manifests', async () => {
    const root = await makeTempDir()
    const datasetPath = path.join(root, 'oracle.json')
    await writeFile(
      datasetPath,
      JSON.stringify([
        {
          question_id: 'q1',
          question_type: 'single-session-user',
          question: 'What colour did the user pick?',
          answer: 'blue',
          haystack_session_ids: ['s1'],
          haystack_dates: ['2024-02-15'],
          haystack_sessions: [
            [
              { role: 'user', content: 'I picked blue today.' },
              { role: 'assistant', content: 'Blue it is.' },
            ],
          ],
        },
        {
          question_id: 'q2',
          question_type: 'single-session-user',
          question: 'What drink did the user pick?',
          answer: 'tea',
          haystack_session_ids: ['s2'],
          haystack_dates: ['2024-02-16'],
          haystack_sessions: [
            [
              { role: 'user', content: 'I picked tea today.' },
              { role: 'assistant', content: 'Tea it is.' },
            ],
          ],
        },
      ]),
      'utf8',
    )

    const provider = createStubProvider()
    const outcome = await runStandaloneLMEEval({
      datasetPath,
      bundle: 'cleaned',
      split: 'oracle',
      ingestMode: 'bulk',
      outDir: path.join(root, 'out'),
      cacheDir: path.join(root, 'cache'),
      provider,
      sampleSize: 1,
      seed: 7,
      createdAt: new Date('2026-04-17T12:00:00Z'),
    })

    expect(outcome.manifest.dataset.sample).toBeDefined()
    expect(outcome.manifest.dataset.sample?.size).toBe(1)
    expect(outcome.manifest.dataset.sample?.seed).toBe(7)
    expect(outcome.manifest.dataset.sample?.signature).toMatch(/^[0-9a-f]{64}$/)
  })

  it('filters judged questions by category while keeping the full dataset manifest', async () => {
    const root = await makeTempDir()
    const datasetPath = path.join(root, 'oracle.json')
    await writeFile(
      datasetPath,
      JSON.stringify([
        {
          question_id: 'q1',
          question_type: 'single-session-user',
          question: 'What colour did the user pick?',
          answer: 'blue',
          haystack_session_ids: ['s1'],
          haystack_dates: ['2024-02-15'],
          haystack_sessions: [
            [
              { role: 'user', content: 'I picked blue today.' },
              { role: 'assistant', content: 'Blue it is.' },
            ],
          ],
        },
        {
          question_id: 'q2',
          question_type: 'multi-session',
          question: 'How many plants did I buy?',
          answer: '2',
          haystack_session_ids: ['s2'],
          haystack_dates: ['2024-02-16'],
          haystack_sessions: [
            [
              { role: 'user', content: 'I bought two plants today.' },
              { role: 'assistant', content: 'Nice.' },
            ],
          ],
        },
      ]),
      'utf8',
    )

    const provider = createStubProvider()
    const outcome = await runStandaloneLMEEval({
      datasetPath,
      bundle: 'cleaned',
      split: 'oracle',
      ingestMode: 'bulk',
      outDir: path.join(root, 'out'),
      cacheDir: path.join(root, 'cache'),
      provider,
      questionCategories: ['multi-session'],
      createdAt: new Date('2026-04-17T12:00:00Z'),
    })

    expect(outcome.results).toHaveLength(1)
    expect(outcome.results[0]?.category).toBe('multi-session')
    expect(outcome.manifest.dataset.examples).toBe(2)
    expect(outcome.manifest.run.questionCategories).toEqual(['multi-session'])
    expect(outcome.officialScore.totalCount).toBe(1)
  })

  it('drops future-dated replay passages relative to the question date', () => {
    expect(
      filterPassagesByQuestionDate(
        [
          {
            path: 'memory/global/past.md',
            score: 1,
            body: 'past',
            date: '2023-05-31',
          },
          {
            path: 'memory/global/future.md',
            score: 1,
            body: 'future',
            date: '2023-09-30',
          },
          {
            path: 'memory/global/unknown.md',
            score: 1,
            body: 'unknown',
          },
        ],
        '2023/06/01 (Thu) 21:36',
      ),
    ).toEqual([
      {
        path: 'memory/global/past.md',
        score: 1,
        body: 'past',
        date: '2023-05-31',
      },
      {
        path: 'memory/global/unknown.md',
        score: 1,
        body: 'unknown',
      },
    ])
  })

  it('prefers recent replay passages for non-temporal reader questions', () => {
    expect(
      rankReplayPassagesForQuestion(
        [
          {
            path: 'memory/global/older.md',
            score: 2,
            body: 'older',
            date: '2023-04-11',
          },
          {
            path: 'memory/global/recent.md',
            score: 1,
            body: 'recent',
            date: '2023-06-01',
          },
        ],
        'What is the total amount of money I earned from selling my products at the markets?',
        '2023/06/01 (Thu) 21:36',
      ),
    ).toEqual([
      {
        path: 'memory/global/recent.md',
        score: 1,
        body: 'recent',
        date: '2023-06-01',
      },
      {
        path: 'memory/global/older.md',
        score: 2,
        body: 'older',
        date: '2023-04-11',
      },
    ])
  })

  it('clusters same-session replay passages before applying the reader limit', () => {
    expect(
      selectReplayPassagesForReader(
        [
          {
            path: 'memory/project/lme/a-1.md',
            score: 5,
            body: 'a1',
            date: '2023-05-20',
            sessionId: 'session-a',
          },
          {
            path: 'memory/project/lme/b-1.md',
            score: 4,
            body: 'b1',
            date: '2023-05-20',
            sessionId: 'session-b',
          },
          {
            path: 'memory/project/lme/a-2.md',
            score: 3,
            body: 'a2',
            date: '2023-05-20',
            sessionId: 'session-a',
          },
        ],
        'Can you suggest some accessories that would complement my current photography setup?',
        '2023/05/28 (Sun) 12:58',
        2,
      ),
    ).toEqual([
      {
        path: 'memory/project/lme/a-1.md',
        score: 5,
        body: 'a1',
        date: '2023-05-20',
        sessionId: 'session-a',
      },
      {
        path: 'memory/project/lme/a-2.md',
        score: 3,
        body: 'a2',
        date: '2023-05-20',
        sessionId: 'session-a',
      },
    ])
  })

  it('drops future replay passages before applying the reader limit', () => {
    expect(
      selectReplayPassagesForReader(
        [
          {
            path: 'memory/project/lme/future.md',
            score: 9,
            body: 'future',
            date: '2023-06-10',
            sessionId: 'future-session',
          },
          {
            path: 'memory/project/lme/past.md',
            score: 2,
            body: 'past',
            date: '2023-05-22',
            sessionId: 'past-session',
          },
        ],
        "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?",
        '2023/05/26 (Fri) 02:34',
        1,
      ),
    ).toEqual([
      {
        path: 'memory/project/lme/past.md',
        score: 2,
        body: 'past',
        date: '2023-05-22',
        sessionId: 'past-session',
      },
    ])
  })
})
