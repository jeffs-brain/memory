// SPDX-License-Identifier: Apache-2.0

import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import { DatasetLoadError, loadDataset, parseDatasetText } from './dataset.js'

const sampleExample = {
  question_id: 'q1',
  question_type: 'single-session-user',
  question: 'What colour did the user pick?',
  answer: 'blue',
  question_date: '2024-03-01',
  haystack_session_ids: ['s1'],
  haystack_dates: ['2024-02-15'],
  haystack_sessions: [
    [
      { role: 'user', content: 'I picked blue today.' },
      { role: 'assistant', content: 'Nice, blue it is.' },
    ],
  ],
}

describe('loadDataset', () => {
  it('parses a JSON array dataset', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lme-ds-'))
    const p = path.join(dir, 'oracle.json')
    await fs.writeFile(p, JSON.stringify([sampleExample]), 'utf8')
    const ds = await loadDataset(p)
    expect(ds.examples).toHaveLength(1)
    const ex = ds.examples[0]
    if (!ex) throw new Error('missing example')
    expect(ex.id).toBe('q1')
    expect(ex.answer).toBe('blue')
    expect(ex.sessionIds).toEqual(['s1'])
    expect(ds.categories).toEqual(['single-session-user'])
    expect(ds.sha256).toMatch(/^[0-9a-f]{64}$/)
  })

  it('parses a JSONL dataset', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lme-ds-'))
    const p = path.join(dir, 'oracle.jsonl')
    await fs.writeFile(
      p,
      `${JSON.stringify(sampleExample)}\n${JSON.stringify({ ...sampleExample, question_id: 'q2' })}\n`,
      'utf8',
    )
    const ds = await loadDataset(p)
    expect(ds.examples.map((e) => e.id)).toEqual(['q1', 'q2'])
  })

  it('accepts numeric answers', () => {
    const ds = parseDatasetText(JSON.stringify([{ ...sampleExample, answer: 42 }]), 'inline', 'sha')
    expect(ds.examples[0]?.answer).toBe('42')
  })

  it('reports malformed JSONL with 1-based line numbers', async () => {
    const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'lme-ds-'))
    const p = path.join(dir, 'bad.jsonl')
    await fs.writeFile(
      p,
      `${JSON.stringify(sampleExample)}\nnot-json\n${JSON.stringify({ ...sampleExample, question_id: 'q3' })}\n`,
      'utf8',
    )
    await expect(loadDataset(p)).rejects.toMatchObject({
      name: 'DatasetLoadError',
      line: 2,
    })
    // Message should name the line number explicitly.
    try {
      await loadDataset(p)
    } catch (err) {
      if (!(err instanceof DatasetLoadError)) throw err
      expect(err.message).toMatch(/line 2/)
    }
  })

  it('rejects empty datasets', () => {
    expect(() => parseDatasetText('[]', 'inline', 'sha')).toThrow(/no questions/)
  })

  it('rejects examples missing required fields', () => {
    expect(() =>
      parseDatasetText(JSON.stringify([{ ...sampleExample, answer: '' }]), 'inline', 'sha'),
    ).toThrow(/empty answer/)
  })
})
