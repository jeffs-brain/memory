// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 0 replay ingest. Re-renders each haystack session as a message
 * transcript, then feeds it through the caller-supplied `Memory.extract`
 * pipeline so extraction is exercised end-to-end. Extraction runs in a
 * bounded worker pool; write-ordering is handled by the underlying
 * memory implementation (it persists per-call).
 */

import type { Logger, Message } from '../llm/index.js'
import type { Memory } from '../memory/types.js'
import type { IngestOutcome, LMEExample, LMESessionMessage } from './types.js'
import { deduplicateSessions } from './ingest-bulk.js'

export type ReplayOpts = {
  readonly concurrency?: number
  readonly logger?: Logger
}

export const ingestReplay = async (
  memory: Memory,
  examples: readonly LMEExample[],
  opts: ReplayOpts = {},
): Promise<IngestOutcome> => {
  const sessions = deduplicateSessions(examples)
  if (sessions.length === 0) {
    return { mode: 'replay', sessionsWritten: 0, examplesIngested: examples.length, warnings: [] }
  }

  const concurrency = clamp(opts.concurrency ?? 1, 1, 32)
  const warnings: string[] = []
  let processed = 0

  // Rebuild the role-tagged text back into Message shape. We lean on the
  // same format the bulk ingest wrote (`[role]: content`), so a single
  // renderer keeps the two modes aligned.
  const jobs = sessions
    .map((s) => ({
      id: s.id,
      date: s.date,
      messages: sessionTextToMessages(s.text, s.date),
    }))
    .sort((left, right) => compareReplayDates(left.date, right.date))

  const queue = [...jobs]
  const workers: Promise<void>[] = []
  for (let w = 0; w < concurrency; w++) {
    workers.push(
      (async () => {
        while (true) {
          const job = queue.shift()
          if (!job) return
          try {
            await memory.extract({
              messages: job.messages,
              sessionId: job.id,
              ...(job.date !== undefined && job.date !== ''
                ? { sessionDate: job.date }
                : {}),
            })
            processed++
          } catch (err) {
            warnings.push(`session ${job.id}: extract failed: ${errText(err)}`)
            opts.logger?.warn('lme replay: extract failed', {
              session: job.id,
              err: errText(err),
            })
          }
        }
      })(),
    )
  }
  await Promise.all(workers)

  return {
    mode: 'replay',
    sessionsWritten: processed,
    examplesIngested: examples.length,
    warnings,
  }
}

/**
 * Alternative replay API that accepts pre-parsed session messages so
 * callers can skip the text-to-message round-trip.
 */
export const ingestReplayFromSessions = async (
  memory: Memory,
  sessions: readonly {
    readonly id: string
    readonly date?: string
    readonly messages: readonly LMESessionMessage[]
  }[],
  opts: ReplayOpts = {},
): Promise<IngestOutcome> => {
  const concurrency = clamp(opts.concurrency ?? 1, 1, 32)
  const warnings: string[] = []
  let processed = 0
  const queue = [...sessions].sort((left, right) => compareReplayDates(left.date, right.date))
  const workers: Promise<void>[] = []
  for (let w = 0; w < concurrency; w++) {
    workers.push(
      (async () => {
        while (true) {
          const job = queue.shift()
          if (!job) return
          const messages: Message[] = job.messages.map((m) => ({
            role: m.role,
            content: m.content,
          }))
          try {
            await memory.extract({
              messages,
              sessionId: job.id,
              ...(job.date !== undefined && job.date !== ''
                ? { sessionDate: job.date }
                : {}),
            })
            processed++
          } catch (err) {
            warnings.push(`session ${job.id}: extract failed: ${errText(err)}`)
          }
        }
      })(),
    )
  }
  await Promise.all(workers)
  return {
    mode: 'replay',
    sessionsWritten: processed,
    examplesIngested: sessions.length,
    warnings,
  }
}

const ROLE_RE = /^\[(user|assistant)\]:\s?(.*)$/

export const sessionTextToMessages = (text: string, date?: string): Message[] => {
  const messages: Message[] = []
  if (date !== undefined && date !== '') {
    messages.push({ role: 'system', content: `This conversation took place on ${date}.` })
  }

  let currentRole: 'user' | 'assistant' | null = null
  let buffer: string[] = []
  const flush = (): void => {
    if (currentRole !== null && buffer.length > 0) {
      messages.push({ role: currentRole, content: buffer.join('\n').trim() })
    }
    buffer = []
  }

  for (const lineRaw of text.split('\n')) {
    const line = lineRaw.trim()
    if (line === '') {
      if (buffer.length > 0) buffer.push('')
      continue
    }
    const m = ROLE_RE.exec(line)
    if (m) {
      flush()
      currentRole = m[1] as 'user' | 'assistant'
      buffer = [m[2] ?? '']
      continue
    }
    buffer.push(line)
  }
  flush()

  if (
    messages.length === 0 ||
    (messages.length === 1 && messages[0]?.role === 'system')
  ) {
    // Session lacked role markers; fall back to a single user message so
    // extraction still receives the raw transcript.
    messages.push({ role: 'user', content: text.trim() })
  }
  return messages
}

const clamp = (n: number, min: number, max: number): number =>
  n < min ? min : n > max ? max : n

const compareReplayDates = (left?: string, right?: string): number =>
  normaliseReplayDate(left).localeCompare(normaliseReplayDate(right))

const normaliseReplayDate = (value?: string): string => {
  const trimmed = value?.trim() ?? ''
  if (trimmed === '') return ''
  const matched =
    /^(\d{4})[/-](\d{2})[/-](\d{2})(?:\s+\([A-Za-z]{3}\))?(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?$/.exec(
      trimmed,
    )
  if (matched !== null) {
    const year = matched[1] ?? '0000'
    const month = matched[2] ?? '01'
    const day = matched[3] ?? '01'
    const hour = matched[4] ?? '00'
    const minute = matched[5] ?? '00'
    const second = matched[6] ?? '00'
    return `${year}-${month}-${day}T${hour}:${minute}:${second}Z`
  }
  const parsed = new Date(trimmed)
  return Number.isNaN(parsed.getTime()) ? trimmed : parsed.toISOString()
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))
