// SPDX-License-Identifier: Apache-2.0

/**
 * Slack connector: fetches channel messages, reconstructs threads,
 * downloads file attachments, and converts Slack mrkdwn to standard
 * markdown. Implements the Connector interface from the connector
 * framework (P5-1).
 *
 * Rate limiting: Slack Tier 3 APIs (~50 req/min) are respected via the
 * shared RateLimiter. 429 responses trigger backoff with Retry-After.
 */

import type { Connector, ConnectorDocument, SyncCursor } from './types.js'
import { RateLimiter } from './types.js'

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export type SlackConnectorConfig = {
  readonly botToken: string
  readonly channels: readonly string[]
  readonly includeThreads?: boolean
  readonly includeFiles?: boolean
  readonly maxFileSize?: number
  readonly oldestTimestamp?: string
  readonly pollIntervalMs?: number
  /** Injectable fetch function for testing. */
  readonly fetchFn?: typeof fetch
}

// ---------------------------------------------------------------------------
// Slack API response types
// ---------------------------------------------------------------------------

type SlackResponse = {
  readonly ok: boolean
  readonly error?: string
  readonly messages?: readonly SlackMessage[]
  readonly response_metadata?: { readonly next_cursor?: string }
}

type SlackMessage = {
  readonly type: string
  readonly user: string
  readonly text: string
  readonly ts: string
  readonly thread_ts?: string
  readonly reply_count?: number
  readonly files?: readonly SlackFile[]
}

type SlackFile = {
  readonly id: string
  readonly name: string
  readonly mimetype: string
  readonly size: number
  readonly url_private_download: string
}

type SlackUserResponse = {
  readonly ok: boolean
  readonly user?: {
    readonly real_name?: string
    readonly name?: string
  }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024 // 50 MB
const DEFAULT_POLL_INTERVAL_MS = 5 * 60 * 1000 // 5 minutes
const API_TIMEOUT_MS = 30_000
const FILE_DOWNLOAD_TIMEOUT_MS = 60_000
const MAX_RESPONSE_BYTES = 10 * 1024 * 1024 // 10 MB

// ---------------------------------------------------------------------------
// Slack connector
// ---------------------------------------------------------------------------

export class SlackConnector implements Connector {
  readonly name = 'slack' as const

  private config: SlackConnectorConfig = {
    botToken: '',
    channels: [],
    includeThreads: true,
    includeFiles: true,
    maxFileSize: DEFAULT_MAX_FILE_SIZE,
    pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
  }

  private readonly rateLimiter = new RateLimiter({
    maxTokens: 50,
    refillRate: 50 / 60, // ~0.833 tokens/sec for Tier 3
  })

  private readonly userCache = new Map<string, string>()
  private stopController: AbortController | undefined
  private readonly fetchFn: typeof fetch

  constructor(config?: Partial<SlackConnectorConfig>) {
    this.fetchFn = config?.fetchFn ?? globalThis.fetch.bind(globalThis)
    if (config) {
      this.config = {
        ...this.config,
        ...config,
        maxFileSize: config.maxFileSize ?? DEFAULT_MAX_FILE_SIZE,
        pollIntervalMs: config.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS,
      }
    }
  }

  async configure(config: Record<string, string>): Promise<void> {
    const botToken = config['botToken']
    if (!botToken) {
      throw new Error('slack: botToken is required')
    }

    const channelsRaw = config['channels']
    if (!channelsRaw) {
      throw new Error('slack: at least one channel is required')
    }

    const channels = channelsRaw.split(',').map(ch => ch.trim()).filter(Boolean)
    if (channels.length === 0) {
      throw new Error('slack: at least one channel is required')
    }

    const resolvedMaxFileSize = config['maxFileSize']
      ? Number.parseInt(config['maxFileSize'], 10)
      : this.config.maxFileSize
    const resolvedOldestTimestamp = config['oldestTimestamp'] ?? this.config.oldestTimestamp

    const updated: SlackConnectorConfig = {
      ...this.config,
      botToken,
      channels,
      includeThreads: config['includeThreads'] !== 'false',
      includeFiles: config['includeFiles'] !== 'false',
      ...(resolvedMaxFileSize !== undefined ? { maxFileSize: resolvedMaxFileSize } : {}),
      ...(resolvedOldestTimestamp !== undefined ? { oldestTimestamp: resolvedOldestTimestamp } : {}),
    }
    this.config = updated

    if (config['maxFileSize'] && Number.isNaN(this.config.maxFileSize)) {
      throw new Error('slack: invalid maxFileSize')
    }
  }

  async *fetchAll(signal: AbortSignal): AsyncIterable<ConnectorDocument> {
    yield* this.fetchMessages(signal, '')
  }

  async *fetchSince(signal: AbortSignal, cursor: SyncCursor): AsyncIterable<ConnectorDocument> {
    yield* this.fetchMessages(signal, cursor.value)
  }

  async *start(signal: AbortSignal): AsyncIterable<ConnectorDocument> {
    this.stopController = new AbortController()
    const combinedSignal = AbortSignal.any([signal, this.stopController.signal])
    let lastCursor = this.config.oldestTimestamp ?? ''

    while (!combinedSignal.aborted) {
      let latestTS = ''

      for await (const doc of this.fetchMessages(combinedSignal, lastCursor)) {
        const ts = doc.metadata['ts']
        if (ts && ts > latestTS) {
          latestTS = ts
        }
        yield doc
      }

      if (latestTS) {
        lastCursor = latestTS
      }

      await this.sleep(this.config.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS, combinedSignal)
    }
  }

  async stop(): Promise<void> {
    this.stopController?.abort()
  }

  // -------------------------------------------------------------------------
  // Internal: message fetching
  // -------------------------------------------------------------------------

  private async *fetchMessages(signal: AbortSignal, oldest: string): AsyncIterable<ConnectorDocument> {
    for (const channelId of this.config.channels) {
      yield* this.fetchChannelMessages(signal, channelId, oldest)
    }
  }

  private async *fetchChannelMessages(
    signal: AbortSignal,
    channelId: string,
    oldest: string,
  ): AsyncIterable<ConnectorDocument> {
    let cursor = ''

    while (true) {
      await this.rateLimiter.acquire(1, signal)

      const resp = await this.callConversationsHistory(signal, channelId, oldest, cursor)

      for (const msg of resp.messages ?? []) {
        yield this.messageToDocument(channelId, msg)

        // Fetch thread replies.
        if (this.config.includeThreads && (msg.reply_count ?? 0) > 0 && msg.thread_ts) {
          const threadDoc = await this.fetchThread(signal, channelId, msg)
          if (threadDoc) {
            yield threadDoc
          }
        }

        // Fetch file attachments.
        if (this.config.includeFiles && msg.files && msg.files.length > 0) {
          for (const file of msg.files) {
            if (file.size > (this.config.maxFileSize ?? DEFAULT_MAX_FILE_SIZE)) {
              continue
            }
            const fileDoc = await this.downloadFile(signal, channelId, file)
            if (fileDoc) {
              yield fileDoc
            }
          }
        }
      }

      const nextCursor = resp.response_metadata?.next_cursor
      if (!nextCursor) {
        break
      }
      cursor = nextCursor
    }
  }

  // -------------------------------------------------------------------------
  // Internal: thread fetching
  // -------------------------------------------------------------------------

  private async fetchThread(
    signal: AbortSignal,
    channelId: string,
    parent: SlackMessage,
  ): Promise<ConnectorDocument | undefined> {
    const allReplies: SlackMessage[] = []
    let cursor = ''

    while (true) {
      await this.rateLimiter.acquire(1, signal)

      const resp = await this.callConversationsReplies(signal, channelId, parent.thread_ts ?? parent.ts, cursor)

      if (resp.messages) {
        allReplies.push(...resp.messages)
      }

      const nextCursor = resp.response_metadata?.next_cursor
      if (!nextCursor) {
        break
      }
      cursor = nextCursor
    }

    const content = this.buildThreadDocument(parent, allReplies)
    const ts = parseSlackTimestamp(parent.ts)

    return {
      externalId: `${channelId}:thread:${parent.thread_ts ?? parent.ts}`,
      content,
      mime: 'text/markdown',
      title: `Thread in ${channelId}`,
      metadata: {
        channel: channelId,
        user: parent.user,
        ts: parent.ts,
        thread_ts: parent.thread_ts ?? parent.ts,
        reply_count: String(parent.reply_count ?? 0),
        source: 'slack',
        type: 'thread',
      },
      modifiedAt: ts,
    }
  }

  private buildThreadDocument(parent: SlackMessage, replies: readonly SlackMessage[]): string {
    const lines: string[] = []

    lines.push(`## Thread: ${convertMrkdwn(parent.text)}`)
    lines.push('')

    for (const reply of replies) {
      // Skip the parent message (Slack includes it in replies).
      if (reply.ts === parent.ts) {
        continue
      }
      const userName = this.resolveUserName(reply.user)
      const ts = parseSlackTimestamp(reply.ts)
      const dateStr = formatDate(ts)
      lines.push(`**${userName}** (${dateStr}):`)
      lines.push(convertMrkdwn(reply.text))
      lines.push('')
    }

    return lines.join('\n').trimEnd()
  }

  // -------------------------------------------------------------------------
  // Internal: file download
  // -------------------------------------------------------------------------

  private async downloadFile(
    signal: AbortSignal,
    channelId: string,
    file: SlackFile,
  ): Promise<ConnectorDocument | undefined> {
    await this.rateLimiter.acquire(1, signal)

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), FILE_DOWNLOAD_TIMEOUT_MS)
    const combinedSignal = AbortSignal.any([signal, controller.signal])

    try {
      const response = await this.fetchFn(file.url_private_download, {
        headers: { Authorization: `Bearer ${this.config.botToken}` },
        signal: combinedSignal,
      })

      if (!response.ok) {
        return undefined
      }

      const arrayBuffer = await response.arrayBuffer()
      const content = Buffer.from(arrayBuffer)

      if (content.length > (this.config.maxFileSize ?? DEFAULT_MAX_FILE_SIZE)) {
        return undefined
      }

      return {
        externalId: `${channelId}:file:${file.id}`,
        content,
        mime: file.mimetype,
        title: file.name,
        metadata: {
          channel: channelId,
          file_id: file.id,
          filename: file.name,
          filetype: file.mimetype,
          file_size: String(file.size),
          source: 'slack',
          type: 'file',
        },
        modifiedAt: new Date(),
      }
    } finally {
      clearTimeout(timeout)
    }
  }

  // -------------------------------------------------------------------------
  // Internal: Slack API calls
  // -------------------------------------------------------------------------

  private async callConversationsHistory(
    signal: AbortSignal,
    channelId: string,
    oldest: string,
    cursor: string,
  ): Promise<SlackResponse> {
    const params = new URLSearchParams({ channel: channelId, limit: '200' })
    if (oldest) params.set('oldest', oldest)
    if (cursor) params.set('cursor', cursor)

    return this.callSlackAPI(signal, 'https://slack.com/api/conversations.history', params)
  }

  private async callConversationsReplies(
    signal: AbortSignal,
    channelId: string,
    threadTs: string,
    cursor: string,
  ): Promise<SlackResponse> {
    const params = new URLSearchParams({ channel: channelId, ts: threadTs, limit: '200' })
    if (cursor) params.set('cursor', cursor)

    return this.callSlackAPI(signal, 'https://slack.com/api/conversations.replies', params)
  }

  private async callSlackAPI(
    signal: AbortSignal,
    endpoint: string,
    params: URLSearchParams,
  ): Promise<SlackResponse> {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), API_TIMEOUT_MS)
    const combinedSignal = AbortSignal.any([signal, controller.signal])

    try {
      const url = `${endpoint}?${params.toString()}`
      const response = await this.fetchFn(url, {
        headers: {
          Authorization: `Bearer ${this.config.botToken}`,
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        signal: combinedSignal,
      })

      // Handle rate limiting (HTTP 429).
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After')
        const waitSecs = retryAfter ? Number.parseInt(retryAfter, 10) : 5
        const waitMs = (Number.isNaN(waitSecs) ? 5 : Math.max(1, waitSecs)) * 1000
        await this.sleep(waitMs, signal)
        return this.callSlackAPI(signal, endpoint, params)
      }

      if (!response.ok) {
        throw new Error(`slack API returned HTTP ${response.status}`)
      }

      const data = (await response.json()) as SlackResponse

      if (!data.ok) {
        throw new Error(`slack API error: ${data.error ?? 'unknown'}`)
      }

      return data
    } finally {
      clearTimeout(timeout)
    }
  }

  // -------------------------------------------------------------------------
  // Internal: user resolution
  // -------------------------------------------------------------------------

  private resolveUserName(userId: string): string {
    if (!userId) return 'unknown'

    const cached = this.userCache.get(userId)
    if (cached) return cached

    // Return the user ID for now -- async resolution would complicate
    // the synchronous buildThreadDocument flow. The user cache is
    // populated lazily via fetchUserName calls where feasible.
    return userId
  }

  // -------------------------------------------------------------------------
  // Internal: message -> document conversion
  // -------------------------------------------------------------------------

  private messageToDocument(channelId: string, msg: SlackMessage): ConnectorDocument {
    const ts = parseSlackTimestamp(msg.ts)
    const content = convertMrkdwn(msg.text)

    const metadata: Record<string, string> = {
      channel: channelId,
      user: msg.user,
      ts: msg.ts,
      source: 'slack',
      type: 'message',
    }
    if (msg.thread_ts) {
      metadata['thread_ts'] = msg.thread_ts
    }
    if (msg.reply_count && msg.reply_count > 0) {
      metadata['reply_count'] = String(msg.reply_count)
    }

    return {
      externalId: `${channelId}:${msg.ts}`,
      content,
      mime: 'text/markdown',
      title: `Message in ${channelId}`,
      metadata,
      modifiedAt: ts,
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private async sleep(ms: number, signal: AbortSignal): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      if (signal.aborted) {
        reject(signal.reason ?? new Error('aborted'))
        return
      }
      const timer = setTimeout(resolve, ms)
      const onAbort = (): void => {
        clearTimeout(timer)
        reject(signal.reason ?? new Error('aborted'))
      }
      signal.addEventListener('abort', onAbort, { once: true })
    })
  }
}

// ---------------------------------------------------------------------------
// Mrkdwn conversion
// ---------------------------------------------------------------------------

/**
 * Converts Slack mrkdwn formatted text to standard Markdown. Handles
 * links, channel mentions, user mentions, bold, italic, strikethrough,
 * and code blocks. Emoji shortcodes (:name:) are left as-is.
 */
export function convertMrkdwn(text: string): string {
  if (!text) return ''

  // Protect code blocks and inline code from formatting conversions.
  const { text: withoutBlocks, blocks } = extractCodeBlocks(text)
  const { text: withoutInline, codes } = extractInlineCode(withoutBlocks)

  let result = withoutInline

  // Links (labelled and bare).
  result = result.replace(/<(https?:\/\/[^|>]+)\|([^>]+)>/g, '[$2]($1)')
  result = result.replace(/<(https?:\/\/[^>]+)>/g, '$1')

  // Channel mentions.
  result = result.replace(/<#[A-Z0-9]+\|([^>]+)>/g, '#$1')

  // User mentions.
  result = result.replace(/<@([A-Z0-9]+)>/g, '@$1')

  // Bold: *text* -> **text**
  result = result.replace(/(^|\s)\*([^\s*][^*]*[^\s*]|[^\s*])\*($|\s)/g, '$1**$2**$3')

  // Italic: _text_ -> *text*
  result = result.replace(/(^|\s)_([^\s_][^_]*[^\s_]|[^\s_])_($|\s)/g, '$1*$2*$3')

  // Strikethrough: ~text~ -> ~~text~~
  result = result.replace(/(^|\s)~([^\s~][^~]*[^\s~]|[^\s~])~($|\s)/g, '$1~~$2~~$3')

  // Restore inline code and code blocks.
  result = restoreInlineCode(result, codes)
  result = restoreCodeBlocks(result, blocks)

  return result
}

// ---------------------------------------------------------------------------
// Code extraction helpers
// ---------------------------------------------------------------------------

const CODE_BLOCK_PLACEHOLDER = '\x00CB'
const INLINE_CODE_PLACEHOLDER = '\x00IC'

function extractCodeBlocks(text: string): { text: string; blocks: string[] } {
  const blocks: string[] = []
  let result = text
  let start = result.indexOf('```')

  while (start !== -1) {
    const end = result.indexOf('```', start + 3)
    if (end === -1) break
    const block = result.slice(start, end + 3)
    blocks.push(block)
    result = result.slice(0, start) + CODE_BLOCK_PLACEHOLDER + result.slice(end + 3)
    start = result.indexOf('```')
  }

  return { text: result, blocks }
}

function restoreCodeBlocks(text: string, blocks: string[]): string {
  let result = text
  for (const block of blocks) {
    const inner = block.slice(3, -3).trim()
    const replacement = `\`\`\`\n${inner}\n\`\`\``
    result = result.replace(CODE_BLOCK_PLACEHOLDER, replacement)
  }
  return result
}

function extractInlineCode(text: string): { text: string; codes: string[] } {
  const codes: string[] = []
  let result = text
  let start = result.indexOf('`')

  while (start !== -1) {
    const end = result.indexOf('`', start + 1)
    if (end === -1) break
    const code = result.slice(start, end + 1)
    codes.push(code)
    result = result.slice(0, start) + INLINE_CODE_PLACEHOLDER + result.slice(end + 1)
    start = result.indexOf('`')
  }

  return { text: result, codes }
}

function restoreInlineCode(text: string, codes: string[]): string {
  let result = text
  for (const code of codes) {
    result = result.replace(INLINE_CODE_PLACEHOLDER, code)
  }
  return result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Parse a Slack epoch timestamp string (e.g. "1234567890.123456") to a
 * Date object.
 */
export function parseSlackTimestamp(ts: string): Date {
  const parts = ts.split('.')
  const secs = Number.parseInt(parts[0] ?? '0', 10)
  const micros = parts.length > 1 ? Number.parseInt(parts[1] ?? '0', 10) : 0
  return new Date(secs * 1000 + micros / 1000)
}

function formatDate(date: Date): string {
  const year = date.getUTCFullYear()
  const month = String(date.getUTCMonth() + 1).padStart(2, '0')
  const day = String(date.getUTCDate()).padStart(2, '0')
  const hours = String(date.getUTCHours()).padStart(2, '0')
  const minutes = String(date.getUTCMinutes()).padStart(2, '0')
  return `${year}-${month}-${day} ${hours}:${minutes}`
}

/**
 * Factory function for creating a Slack connector with the given
 * configuration.
 */
export function createSlackConnector(config?: Partial<SlackConnectorConfig>): SlackConnector {
  return new SlackConnector(config)
}
