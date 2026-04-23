import { toText } from '../../knowledge/hash.js'
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
  bytes: Uint8Array | ArrayBuffer,
  opts: SourceLoadOptions = {},
): Promise<LoadedSource> => {
  let payload: unknown
  try {
    payload = JSON.parse(toText(bytes))
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`loadJsonTranscript: invalid JSON (${message})`)
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
  for (const message of messages) {
    if (message === null || typeof message !== 'object') continue
    const role = typeof message.role === 'string' ? message.role : 'unknown'
    const content = typeof message.content === 'string' ? message.content : ''
    const stamp = typeof message.timestamp === 'string' ? ` · ${message.timestamp}` : ''
    lines.push(`## ${renderRole(role)}${stamp}`, '', content.trim(), '')
  }

  return {
    content: lines.join('\n').trim(),
    mime: 'text/markdown',
    title: headerTitle,
    meta: { turns: messages.length },
  }
}
