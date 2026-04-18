// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 0 bulk ingest. Writes each LME session into the eval brain as a
 * single `raw/lme/session-<id>.md` file with frontmatter, skipping any
 * extraction. Mirrors the Go `IngestBulk` flow but simplified: no
 * question-index / dates-index sidecars (those are easy to add later when
 * downstream consumers need them).
 */

import type { Store } from '../store/index.js'
import { toPath } from '../store/path.js'
import type { IngestOutcome, LMEExample } from './types.js'

type SessionPayload = {
  readonly id: string
  readonly text: string
  readonly date?: string
  readonly questionIds: readonly string[]
}

export const ingestBulk = async (
  store: Store,
  examples: readonly LMEExample[],
): Promise<IngestOutcome> => {
  const sessions = deduplicateSessions(examples)
  const warnings: string[] = []
  let written = 0

  await store.batch(
    { reason: 'lme-ingest-bulk', message: `Bulk ingest ${sessions.length} LME sessions` },
    async (batch) => {
      for (const s of sessions) {
        const path = toPath(`raw/lme/session-${s.id}.md`)
        const content = buildSessionContent(s)
        try {
          await batch.write(path, Buffer.from(content, 'utf8'))
          written++
        } catch (err) {
          warnings.push(`failed to write session ${s.id}: ${errText(err)}`)
        }
      }
    },
  )

  return {
    mode: 'bulk',
    sessionsWritten: written,
    examplesIngested: examples.length,
    warnings,
  }
}

/**
 * Group per-session text across every example that references it. When
 * an example lacks `haystackSessions` we fall back to the joined haystack
 * render so ingest never silently drops content.
 */
export const deduplicateSessions = (
  examples: readonly LMEExample[],
): readonly SessionPayload[] => {
  const seen = new Map<string, { text: string; date?: string; questionIds: string[] }>()
  const order: string[] = []

  const ensure = (id: string, init: () => { text: string; date?: string }): { text: string; date?: string; questionIds: string[] } => {
    const existing = seen.get(id)
    if (existing) return existing
    const { text, date } = init()
    const fresh = date !== undefined ? { text, date, questionIds: [] } : { text, questionIds: [] }
    seen.set(id, fresh)
    order.push(id)
    return fresh
  }

  for (const ex of examples) {
    const sessions = ex.haystackSessions ?? []
    const dates = ex.haystackDates ?? []
    if (ex.sessionIds.length === 0) {
      if (sessions.length === 0) continue
      const fakeId = `q-${ex.id}`
      const slot = ensure(fakeId, () => ({ text: renderHaystack(sessions) }))
      slot.questionIds.push(ex.id)
      continue
    }
    for (let i = 0; i < ex.sessionIds.length; i++) {
      const sid = ex.sessionIds[i] ?? ''
      if (sid === '') continue
      const slot = ensure(sid, () => {
        const msgs = sessions[i]
        const date = dates[i]
        const text = msgs !== undefined ? renderSession(msgs) : renderHaystack(sessions)
        return date !== undefined && date !== '' ? { text, date } : { text }
      })
      slot.questionIds.push(ex.id)
    }
  }

  const out: SessionPayload[] = []
  for (const id of order) {
    const slot = seen.get(id)
    if (!slot || slot.text === '') continue
    const payload: SessionPayload =
      slot.date !== undefined
        ? { id, text: slot.text, date: slot.date, questionIds: slot.questionIds }
        : { id, text: slot.text, questionIds: slot.questionIds }
    out.push(payload)
  }
  return out
}

const renderSession = (msgs: readonly { role: string; content: string }[]): string => {
  const parts: string[] = []
  for (const m of msgs) parts.push(`[${m.role}]: ${m.content}`)
  return parts.join('\n\n')
}

const renderHaystack = (
  sessions: readonly (readonly { role: string; content: string }[])[],
): string => sessions.map(renderSession).join('\n---\n\n')

const ISO_DATE_RE = /\b\d{4}-\d{2}-\d{2}\b/

const buildSessionContent = (s: SessionPayload): string => {
  let date = s.date ?? ''
  if (date === '') {
    const match = s.text.slice(0, 4096).match(ISO_DATE_RE)
    if (match) date = match[0]
  }
  const lines: string[] = ['---', `session_id: ${s.id}`]
  if (date !== '') lines.push(`session_date: ${date}`)
  if (s.questionIds.length > 0) lines.push(`question_ids: [${s.questionIds.join(', ')}]`)
  lines.push('source: lme-bulk-ingest', '---', '')
  if (date !== '') lines.push(`[This conversation took place on ${date}]`, '')
  lines.push(s.text.endsWith('\n') ? s.text.slice(0, -1) : s.text)
  if (date !== '') lines.push('', `[Question date reference: ${date}]`)
  return `${lines.join('\n')}\n`
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))
