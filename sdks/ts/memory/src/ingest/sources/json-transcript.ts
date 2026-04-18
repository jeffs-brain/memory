// SPDX-License-Identifier: Apache-2.0

/**
 * JSON transcript adapter. Accepts { messages: [{ role, content, timestamp? }] }
 * and lowers it to a markdown document with per-turn headings so the
 * chunker picks up turn boundaries naturally.
 */

import type { LoadedSource, SourceLoadOptions } from './types.js'

export type TranscriptMessage = {
  readonly role: string
  readonly content: string
  readonly timestamp?: string
  readonly name?: string
}

export type TranscriptPayload = {
  readonly title?: string
  readonly messages: readonly TranscriptMessage[]
}

const ROLE_TITLE: Record<string, string> = {
  user: 'User',
  human: 'User',
  assistant: 'Assistant',
  agent: 'Assistant',
  system: 'System',
  tool: 'Tool',
  function: 'Tool',
}

const renderRole = (role: string): string => ROLE_TITLE[role.toLowerCase()] ?? role

export const loadJsonTranscript = async (
  bytes: Buffer,
  opts: SourceLoadOptions = {},
): Promise<LoadedSource> => {
  const text = bytes.toString('utf8')
  let payload: unknown
  try {
    payload = JSON.parse(text)
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    throw new Error(`loadJsonTranscript: invalid JSON (${msg})`)
  }
  if (
    payload === null ||
    typeof payload !== 'object' ||
    !Array.isArray((payload as { messages?: unknown }).messages)
  ) {
    throw new Error('loadJsonTranscript: expected object with messages array')
  }
  const { messages, title } = payload as TranscriptPayload
  const headerTitle = opts.title ?? title ?? 'Conversation Transcript'
  const lines: string[] = [`# ${headerTitle}`, '']
  for (const msg of messages) {
    if (msg === null || typeof msg !== 'object') continue
    const role = typeof msg.role === 'string' ? msg.role : 'unknown'
    const content = typeof msg.content === 'string' ? msg.content : ''
    const stamp = typeof msg.timestamp === 'string' ? ` · ${msg.timestamp}` : ''
    lines.push(`## ${renderRole(role)}${stamp}`, '', content.trim(), '')
  }
  return {
    content: Buffer.from(lines.join('\n').trim(), 'utf8'),
    mime: 'text/markdown',
    title: headerTitle,
    meta: { turns: messages.length },
  }
}
