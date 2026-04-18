// SPDX-License-Identifier: Apache-2.0

/**
 * Dataset loader for LongMemEval oracle JSONL files.
 *
 * The oracle dataset ships as either a JSON array (legacy) or a
 * newline-delimited JSONL stream (one question per line). Both shapes are
 * accepted here; malformed JSON lines are reported with 1-based line
 * numbers so a broken file is easy to debug.
 */

import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import type { LMEExample, LMESessionMessage } from './types.js'

export type LoadedDataset = {
  readonly examples: readonly LMEExample[]
  readonly sha256: string
  readonly categories: readonly string[]
}

type RawExample = {
  question_id?: unknown
  question_type?: unknown
  question?: unknown
  answer?: unknown
  question_date?: unknown
  haystack_dates?: unknown
  haystack_session_ids?: unknown
  answer_session_ids?: unknown
  haystack_sessions?: unknown
}

export class DatasetLoadError extends Error {
  constructor(
    message: string,
    readonly path: string,
    readonly line?: number,
  ) {
    super(message)
    this.name = 'DatasetLoadError'
  }
}

/**
 * Read + parse an LME oracle dataset from disk. Returns the parsed
 * examples along with a SHA-256 fingerprint of the raw bytes (used
 * downstream as a brain-cache key and the report's `dataset_sha256`).
 */
export const loadDataset = async (path: string): Promise<LoadedDataset> => {
  let raw: Buffer
  try {
    raw = await fs.readFile(path)
  } catch (err) {
    throw new DatasetLoadError(`lme: read dataset: ${errorMessage(err)}`, path)
  }

  const sha256 = createHash('sha256').update(raw).digest('hex')
  const text = raw.toString('utf8').replace(/^\uFEFF/, '')
  return parseDatasetText(text, path, sha256)
}

/** Parse a raw dataset string. Exposed for tests. */
export const parseDatasetText = (
  text: string,
  path: string,
  sha256: string,
): LoadedDataset => {
  const trimmed = text.trimStart()
  const examples: LMEExample[] =
    trimmed.startsWith('[') ? parseArray(trimmed, path) : parseJSONL(text, path)

  if (examples.length === 0) {
    throw new DatasetLoadError('lme: dataset contains no questions', path)
  }

  const catSet = new Set<string>()
  for (const ex of examples) catSet.add(ex.category)
  const categories = [...catSet].sort()

  return { examples, sha256, categories }
}

const parseArray = (text: string, path: string): LMEExample[] => {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch (err) {
    throw new DatasetLoadError(`lme: parse dataset: ${errorMessage(err)}`, path)
  }
  if (!Array.isArray(parsed)) {
    throw new DatasetLoadError('lme: parse dataset: expected JSON array', path)
  }
  return parsed.map((raw, idx) => {
    try {
      return toExample(raw as RawExample, idx)
    } catch (err) {
      throw new DatasetLoadError(
        `lme: question ${idx}: ${errorMessage(err)}`,
        path,
      )
    }
  })
}

const parseJSONL = (text: string, path: string): LMEExample[] => {
  const lines = text.split(/\r?\n/)
  const out: LMEExample[] = []
  for (let i = 0; i < lines.length; i++) {
    const line = (lines[i] ?? '').trim()
    if (line === '') continue
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch (err) {
      throw new DatasetLoadError(
        `lme: parse dataset line ${i + 1}: ${errorMessage(err)}`,
        path,
        i + 1,
      )
    }
    try {
      out.push(toExample(parsed as RawExample, i))
    } catch (err) {
      throw new DatasetLoadError(
        `lme: line ${i + 1}: ${errorMessage(err)}`,
        path,
        i + 1,
      )
    }
  }
  return out
}

const toExample = (raw: RawExample, idx: number): LMEExample => {
  const id = toStringField(raw.question_id)
  if (id === '') throw new Error(`question ${idx} has empty question_id`)
  const category = toStringField(raw.question_type)
  if (category === '') throw new Error(`question ${id} has empty question_type`)
  const question = toStringField(raw.question)
  if (question === '') throw new Error(`question ${id} has empty question text`)

  // The official oracle answer field is a string or number (temporal
  // questions sometimes return "5" as an int). Stringify either way.
  let answer = ''
  if (typeof raw.answer === 'string') {
    answer = raw.answer
  } else if (typeof raw.answer === 'number' || typeof raw.answer === 'boolean') {
    answer = String(raw.answer)
  } else if (raw.answer != null) {
    answer = JSON.stringify(raw.answer)
  }
  if (answer === '') throw new Error(`question ${id} has empty answer`)

  const sessionIds = toStringArray(raw.haystack_session_ids)
  const haystackDates = toStringArray(raw.haystack_dates)
  const answerSessionIds = toStringArray(raw.answer_session_ids)
  const haystackSessions = toHaystackSessions(raw.haystack_sessions)

  const out: LMEExample = {
    id,
    category,
    question,
    answer,
    ...(typeof raw.question_date === 'string' && raw.question_date !== ''
      ? { questionDate: raw.question_date }
      : {}),
    ...(haystackDates.length > 0 ? { haystackDates } : {}),
    sessionIds,
    ...(answerSessionIds.length > 0 ? { answerSessionIds } : {}),
    ...(haystackSessions.length > 0 ? { haystackSessions } : {}),
  }
  return out
}

const toStringField = (v: unknown): string => (typeof v === 'string' ? v : '')

const toStringArray = (v: unknown): readonly string[] => {
  if (!Array.isArray(v)) return []
  const out: string[] = []
  for (const item of v) {
    if (typeof item === 'string' && item !== '') out.push(item)
  }
  return out
}

const toHaystackSessions = (v: unknown): readonly (readonly LMESessionMessage[])[] => {
  if (!Array.isArray(v)) return []
  const out: LMESessionMessage[][] = []
  for (const session of v) {
    if (!Array.isArray(session)) continue
    const msgs: LMESessionMessage[] = []
    for (const m of session) {
      if (m == null || typeof m !== 'object') continue
      const mo = m as { role?: unknown; content?: unknown; has_answer?: unknown }
      const role = mo.role === 'user' || mo.role === 'assistant' ? mo.role : undefined
      const content = typeof mo.content === 'string' ? mo.content : ''
      if (role === undefined || content === '') continue
      const msg: LMESessionMessage = {
        role,
        content,
        ...(mo.has_answer === true ? { hasAnswer: true } : {}),
      }
      msgs.push(msg)
    }
    out.push(msgs)
  }
  return out
}

const errorMessage = (err: unknown): string =>
  err instanceof Error ? err.message : String(err)
