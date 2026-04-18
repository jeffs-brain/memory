import { describe, expect, it } from 'vitest'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createMemStore } from '../store/memstore.js'
import { createLMERunner } from './index.js'
import { ingestBulk } from './ingest-bulk.js'
import type {
  JudgeFn,
  JudgeVerdict,
  LMEExample,
  ReaderFn,
  RetrievalFn,
} from './types.js'

const sessionOne = [
  { role: 'user' as const, content: 'I picked blue today.' },
  { role: 'assistant' as const, content: 'Blue it is.' },
]
const sessionTwo = [
  { role: 'user' as const, content: 'Remind me of my favourite number: 7.' },
  { role: 'assistant' as const, content: 'Noted, 7.' },
]
const sessionThree = [
  { role: 'user' as const, content: 'When did we fly to Rome?' },
  { role: 'assistant' as const, content: 'You said July 2023.' },
]

const examples: LMEExample[] = [
  {
    id: 'q-colour',
    category: 'single-session-user',
    question: 'What colour did the user pick?',
    answer: 'blue',
    questionDate: '2024-03-01',
    sessionIds: ['s-colour'],
    haystackDates: ['2024-02-15'],
    haystackSessions: [sessionOne],
  },
  {
    id: 'q-number',
    category: 'single-session-preference',
    question: 'What is the user’s favourite number?',
    answer: '7',
    questionDate: '2024-03-05',
    sessionIds: ['s-number'],
    haystackDates: ['2024-02-20'],
    haystackSessions: [sessionTwo],
  },
  {
    id: 'q-trip_abs',
    category: 'abstention',
    question: 'When did the user fly to Paris?',
    answer: 'You did not mention Paris.',
    questionDate: '2024-03-10',
    sessionIds: ['s-trip'],
    haystackDates: ['2024-02-25'],
    haystackSessions: [sessionThree],
  },
]

// Deterministic stub provider / embedder. Neither is wired up in tests
// because we inject `retrieval`, `reader`, and `judge` directly.
const stubRetrieval: RetrievalFn = async ({ example }) => {
  const session = example.haystackSessions?.[0] ?? []
  const body = session.map((m) => `[${m.role}]: ${m.content}`).join('\n')
  return {
    passages: [{ path: `raw/lme/session-${example.sessionIds[0] ?? 'none'}.md`, score: 1, body }],
    rendered: body,
  }
}

const stubReader: ReaderFn = async ({ question }) => {
  // Echo back a deterministic answer based on the question. Covers the
  // two correct cases + one wrong case so the judge stub has something
  // to verify.
  if (question.includes('colour')) return 'blue'
  if (question.includes('favourite number')) return '7'
  if (question.includes('Paris')) return "I'm not sure, you never mentioned Paris."
  return ''
}

const stubJudge =
  (overrides: Record<string, JudgeVerdict> = {}): JudgeFn =>
  async ({ example, predicted }) => {
    const forced = overrides[example.id]
    if (forced !== undefined) return { verdict: forced, rationale: 'forced' }
    if (example.id === 'q-colour' && predicted.includes('blue')) {
      return { verdict: 'correct', rationale: 'matches blue' }
    }
    if (example.id === 'q-number' && predicted.includes('7')) {
      return { verdict: 'correct', rationale: 'matches 7' }
    }
    if (example.id === 'q-trip_abs' && predicted.toLowerCase().includes('not sure')) {
      return { verdict: 'abstain_correct', rationale: 'correctly abstained' }
    }
    return { verdict: 'incorrect', rationale: 'mismatch' }
  }

const withTempDir = async (prefix: string): Promise<string> =>
  fs.mkdtemp(path.join(os.tmpdir(), prefix))

describe('createLMERunner', () => {
  it('runs bulk ingest + judge + report end-to-end with stubs', async () => {
    const store = createMemStore()
    const outDir = await withTempDir('lme-runner-')
    const runner = createLMERunner({
      store,
      retrieval: stubRetrieval,
      reader: stubReader,
      judge: stubJudge(),
      outDir,
      runId: 'lme-test-1',
      now: () => new Date('2026-04-17T10:00:00Z'),
    })

    const ingest = await runner.runBulk({ examples })
    expect(ingest.mode).toBe('bulk')
    expect(ingest.sessionsWritten).toBe(3)
    expect(ingest.warnings).toEqual([])

    const results = await runner.judge({ examples })
    expect(results).toHaveLength(3)
    const verdicts = new Map(results.map((r) => [r.id, r.verdict]))
    expect(verdicts.get('q-colour')).toBe('correct')
    expect(verdicts.get('q-number')).toBe('correct')
    expect(verdicts.get('q-trip_abs')).toBe('abstain_correct')

    const { report, written } = await runner.report({
      ingestMode: ingest.mode,
      results,
      datasetSha256: 'deadbeef',
    })
    expect(report.overallAccuracy).toBeCloseTo(1, 5)
    expect(report.perCategory['single-session-user']?.accuracy).toBe(1)
    expect(report.perCategory['abstention']?.abstainCorrect).toBe(1)
    expect(report.ingestMode).toBe('bulk')
    expect(report.datasetSha256).toBe('deadbeef')

    // On-disk artefacts exist at the expected location.
    const raw = JSON.parse(await fs.readFile(written.reportPath, 'utf8'))
    expect(raw.examples).toBe(3)
    expect(raw.results).toHaveLength(3)
  })

  it('reflects stubbed judge verdicts in per-category breakdown', async () => {
    const store = createMemStore()
    const outDir = await withTempDir('lme-runner-')
    const runner = createLMERunner({
      store,
      retrieval: stubRetrieval,
      reader: stubReader,
      judge: stubJudge({ 'q-number': 'incorrect' }),
      outDir,
      runId: 'lme-test-2',
      now: () => new Date('2026-04-17T10:30:00Z'),
    })
    await runner.runBulk({ examples })
    const results = await runner.judge({ examples })
    const { report } = await runner.report({ ingestMode: 'bulk', results })
    expect(report.perCategory['single-session-preference']?.accuracy).toBe(0)
    // 2 of 3 correct -> 2/3.
    expect(report.overallAccuracy).toBeCloseTo(2 / 3, 5)
  })

  it('falls back to exact-match when the reader yields empty output', async () => {
    const store = createMemStore()
    const outDir = await withTempDir('lme-runner-')
    const runner = createLMERunner({
      store,
      retrieval: stubRetrieval,
      reader: async () => '',
      judge: async () => ({ verdict: 'correct', rationale: 'should not be called' }),
      outDir,
      runId: 'lme-test-empty',
      now: () => new Date('2026-04-17T11:00:00Z'),
    })
    const results = await runner.judge({ examples })
    for (const r of results) {
      // No predicted text -> exact-match fallback fires instead of judge.
      expect(r.predicted).toBe('')
      expect(r.rationale).toMatch(/exact-match/)
    }
  })
})

describe('ingestBulk', () => {
  it('deduplicates shared sessions across examples', async () => {
    const store = createMemStore()
    const shared: LMEExample = {
      id: 'qA',
      category: 'single-session-user',
      question: 'q?',
      answer: 'a',
      sessionIds: ['shared-1'],
      haystackDates: ['2024-01-01'],
      haystackSessions: [sessionOne],
    }
    const duplicate: LMEExample = {
      ...shared,
      id: 'qB',
      question: 'different',
    }
    await ingestBulk(store, [shared, duplicate])
    const files = await store.list('' as never, { recursive: true })
    const sessionFiles = files.filter((f) => f.path.startsWith('raw/lme/session-'))
    expect(sessionFiles).toHaveLength(1)
    const raw = (await store.read(sessionFiles[0]?.path as never)).toString('utf8')
    expect(raw).toMatch(/session_id: shared-1/)
    expect(raw).toMatch(/question_ids: \[qA, qB\]/)
  })
})
