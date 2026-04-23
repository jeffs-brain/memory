// SPDX-License-Identifier: Apache-2.0

import { type Batch, type Path, type Store, toPath } from '../store/index.js'
import type { Operation } from './types.js'

export const LOG_PATH: Path = toPath('_log.md')
const LOG_HEADER = '# Operations Log\n\n'

export const formatEntry = (op: Operation): string => {
  const date = op.when.slice(0, 10)
  return `## [${date}] ${op.kind} | ${op.title}\n\n${op.detail}\n\n`
}

export const appendLogInBatch = async (batch: Batch, op: Operation): Promise<void> => {
  const entry = formatEntry(op)
  const exists = await batch.exists(LOG_PATH)
  if (!exists) {
    await batch.write(LOG_PATH, `${LOG_HEADER}${entry}`)
    return
  }
  await batch.append(LOG_PATH, entry)
}

export const appendLog = async (store: Store, op: Operation): Promise<void> => {
  await store.batch({ reason: 'log' }, async (batch) => {
    await appendLogInBatch(batch, op)
  })
}

export const readLog = async (store: Store): Promise<string> => {
  const exists = await store.exists(LOG_PATH)
  if (!exists) return ''
  return await store.read(LOG_PATH)
}

export const parseLog = (raw: string): readonly Operation[] => {
  if (raw === '') return []
  const parts = raw.split(/\n## \[/)
  const out: Operation[] = []
  for (let index = 0; index < parts.length; index += 1) {
    const part =
      index === 0 ? (parts[index]?.replace(/^# Operations Log\n+/, '') ?? '') : (parts[index] ?? '')
    if (part === '' || !part.includes(']')) continue
    const body = index === 0 && part.startsWith('## [') ? part.slice(4) : part
    const bracketEnd = body.indexOf(']')
    if (bracketEnd < 0) continue
    const date = body.slice(0, bracketEnd)
    const rest = body.slice(bracketEnd + 1).trim()
    const pipeIndex = rest.indexOf(' | ')
    if (pipeIndex < 0) continue
    const kind = rest.slice(0, pipeIndex).trim()
    const after = rest.slice(pipeIndex + 3)
    const newlineIndex = after.indexOf('\n')
    const title = newlineIndex < 0 ? after.trim() : after.slice(0, newlineIndex).trim()
    const detail = newlineIndex < 0 ? '' : after.slice(newlineIndex + 1).trim()
    out.push({
      kind: kind as Operation['kind'],
      title,
      detail,
      when: `${date}T00:00:00Z`,
    })
  }
  return out
}
