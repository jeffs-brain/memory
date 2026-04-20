// SPDX-License-Identifier: Apache-2.0

/**
 * Operations-log append helper. Writes Markdown entries to `_log.md`
 * at the root of the Store. The log is append-only; we never rewrite
 * in place. Ported from apps/jeff/internal/knowledge/log.go.
 */

import { type Batch, type Path, type Store, toPath } from '../store/index.js'
import type { Operation } from './types.js'

export const LOG_PATH: Path = toPath('_log.md')
const LOG_HEADER = '# Operations Log\n\n'

export const formatEntry = (op: Operation): string => {
  const date = op.when.slice(0, 10)
  return `## [${date}] ${op.kind} | ${op.title}\n\n${op.detail}\n\n`
}

/** Append a single entry using an already-running Batch. */
export const appendLogInBatch = async (batch: Batch, op: Operation): Promise<void> => {
  const entry = formatEntry(op)
  const exists = await batch.exists(LOG_PATH)
  if (!exists) {
    await batch.write(LOG_PATH, Buffer.from(`${LOG_HEADER}${entry}`, 'utf8'))
    return
  }
  await batch.append(LOG_PATH, Buffer.from(entry, 'utf8'))
}

/** Append a standalone entry in its own one-op batch. */
export const appendLog = async (store: Store, op: Operation): Promise<void> => {
  await store.batch({ reason: 'log' }, async (batch) => {
    await appendLogInBatch(batch, op)
  })
}

/** Read the raw log contents, or '' when absent. */
export const readLog = async (store: Store): Promise<string> => {
  const exists = await store.exists(LOG_PATH)
  if (!exists) return ''
  const buf = await store.read(LOG_PATH)
  return buf.toString('utf8')
}

/** Parse log contents into structured entries. Best-effort. */
export const parseLog = (raw: string): readonly Operation[] => {
  if (raw === '') return []
  const parts = raw.split(/\n## \[/)
  const out: Operation[] = []
  for (let i = 0; i < parts.length; i++) {
    const part = i === 0 ? (parts[i]?.replace(/^# Operations Log\n+/, '') ?? '') : (parts[i] ?? '')
    if (part === '' || !part.includes(']')) continue
    const body = i === 0 && part.startsWith('## [') ? part.slice(4) : part
    const bracketEnd = body.indexOf(']')
    if (bracketEnd < 0) continue
    const date = body.slice(0, bracketEnd)
    const rest = body.slice(bracketEnd + 1).trim()
    const pipeIdx = rest.indexOf(' | ')
    if (pipeIdx < 0) continue
    const kind = rest.slice(0, pipeIdx).trim()
    const after = rest.slice(pipeIdx + 3)
    const nlIdx = after.indexOf('\n')
    const title = nlIdx < 0 ? after.trim() : after.slice(0, nlIdx).trim()
    const detail = nlIdx < 0 ? '' : after.slice(nlIdx + 1).trim()
    out.push({
      kind: kind as Operation['kind'],
      title,
      detail,
      when: `${date}T00:00:00Z`,
    })
  }
  return out
}
