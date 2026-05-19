// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { SlackConnector, convertMrkdwn, createSlackConnector, parseSlackTimestamp } from './slack.js'
import type { ConnectorDocument } from './types.js'

// ---------------------------------------------------------------------------
// Mock fetch helper
// ---------------------------------------------------------------------------

type MockHandler = {
  readonly pattern: string
  readonly handler: (url: string, init?: RequestInit) => Promise<Response>
}

function createMockFetch(handlers: readonly MockHandler[]): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input.toString()
    for (const h of handlers) {
      if (url.includes(h.pattern)) {
        return h.handler(url, init)
      }
    }
    return new Response('Not Found', { status: 404 })
  }
}

function jsonResponse(data: Record<string, unknown>): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })
}

async function collectDocs(iter: AsyncIterable<ConnectorDocument>): Promise<ConnectorDocument[]> {
  const docs: ConnectorDocument[] = []
  for await (const doc of iter) {
    docs.push(doc)
  }
  return docs
}

// ---------------------------------------------------------------------------
// Tests: Name and Configure
// ---------------------------------------------------------------------------

describe('SlackConnector', () => {
  describe('name', () => {
    it('returns "slack"', () => {
      const connector = createSlackConnector()
      expect(connector.name).toBe('slack')
    })
  })

  describe('configure', () => {
    it('throws when botToken is missing', async () => {
      const connector = createSlackConnector()
      await expect(connector.configure({ channels: 'C123' })).rejects.toThrow('botToken is required')
    })

    it('throws when channels is missing', async () => {
      const connector = createSlackConnector()
      await expect(connector.configure({ botToken: 'xoxb-test' })).rejects.toThrow('at least one channel')
    })

    it('throws when channels is empty after split', async () => {
      const connector = createSlackConnector()
      await expect(
        connector.configure({ botToken: 'xoxb-test', channels: '  ,  ,  ' }),
      ).rejects.toThrow('at least one channel')
    })

    it('accepts valid configuration', async () => {
      const connector = createSlackConnector()
      await expect(
        connector.configure({ botToken: 'xoxb-test', channels: 'C123,C456' }),
      ).resolves.toBeUndefined()
    })

    it('throws for invalid maxFileSize', async () => {
      const connector = createSlackConnector()
      await expect(
        connector.configure({ botToken: 'xoxb-test', channels: 'C123', maxFileSize: 'abc' }),
      ).rejects.toThrow('invalid maxFileSize')
    })
  })

  // -------------------------------------------------------------------------
  // Tests: FetchAll
  // -------------------------------------------------------------------------

  describe('fetchAll', () => {
    it('fetches single channel messages', async () => {
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () =>
            Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  { type: 'message', user: 'U001', text: 'Hello world', ts: '1700000001.000000' },
                  { type: 'message', user: 'U002', text: 'Good morning', ts: '1700000002.000000' },
                  { type: 'message', user: 'U003', text: 'How are you?', ts: '1700000003.000000' },
                ],
              }),
            ),
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      expect(docs).toHaveLength(3)
      expect(docs[0]?.externalId).toBe('C123ABC:1700000001.000000')
      expect(docs[0]?.content).toBe('Hello world')
      expect(docs[0]?.metadata['source']).toBe('slack')
    })

    it('handles paginated messages', async () => {
      let callCount = 0
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: (url) => {
            callCount++
            const hasNextCursor = !url.includes('cursor=')

            return Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  { type: 'message', user: 'U001', text: `Page ${callCount} msg 1`, ts: `${1700000000 + callCount * 2 - 1}.000000` },
                  { type: 'message', user: 'U001', text: `Page ${callCount} msg 2`, ts: `${1700000000 + callCount * 2}.000000` },
                ],
                ...(hasNextCursor
                  ? { response_metadata: { next_cursor: 'cursor_page2' } }
                  : {}),
              }),
            )
          },
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      expect(docs).toHaveLength(4)
    })

    it('reconstructs threads', async () => {
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () =>
            Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  {
                    type: 'message',
                    user: 'U001',
                    text: 'Thread parent',
                    ts: '1700000001.000000',
                    thread_ts: '1700000001.000000',
                    reply_count: 2,
                  },
                ],
              }),
            ),
        },
        {
          pattern: 'conversations.replies',
          handler: () =>
            Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  { type: 'message', user: 'U001', text: 'Thread parent', ts: '1700000001.000000' },
                  { type: 'message', user: 'U002', text: 'Reply one', ts: '1700000002.000000' },
                  { type: 'message', user: 'U003', text: 'Reply two', ts: '1700000003.000000' },
                ],
              }),
            ),
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        includeThreads: true,
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      // 1 message + 1 thread = 2
      expect(docs).toHaveLength(2)

      const threadDoc = docs[1]
      expect(threadDoc?.metadata['type']).toBe('thread')
      const content = typeof threadDoc?.content === 'string' ? threadDoc.content : threadDoc?.content.toString()
      expect(content).toContain('## Thread: Thread parent')
      expect(content).toContain('Reply one')
      expect(content).toContain('Reply two')
    })

    it('downloads file attachments', async () => {
      const fileContent = 'file content bytes'

      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () =>
            Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  {
                    type: 'message',
                    user: 'U001',
                    text: 'Check this file',
                    ts: '1700000001.000000',
                    files: [
                      {
                        id: 'F001',
                        name: 'report.pdf',
                        mimetype: 'application/pdf',
                        size: fileContent.length,
                        url_private_download: 'https://files.slack.com/F001',
                      },
                    ],
                  },
                ],
              }),
            ),
        },
        {
          pattern: 'files.slack.com',
          handler: (_url, init) => {
            const auth = (init?.headers as Record<string, string>)?.['Authorization']
            if (auth !== 'Bearer xoxb-test') {
              return Promise.resolve(new Response('Unauthorized', { status: 401 }))
            }
            return Promise.resolve(new Response(fileContent, { status: 200 }))
          },
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        includeFiles: true,
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      // 1 message + 1 file = 2
      expect(docs).toHaveLength(2)

      const fileDoc = docs[1]
      expect(fileDoc?.externalId).toBe('C123ABC:file:F001')
      expect(fileDoc?.mime).toBe('application/pdf')
      expect(fileDoc?.title).toBe('report.pdf')
    })

    it('skips files exceeding size limit', async () => {
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () =>
            Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  {
                    type: 'message',
                    user: 'U001',
                    text: 'Big file',
                    ts: '1700000001.000000',
                    files: [
                      {
                        id: 'F002',
                        name: 'huge.zip',
                        mimetype: 'application/zip',
                        size: 100 * 1024 * 1024, // 100 MB > 50 MB limit
                        url_private_download: 'https://files.slack.com/F002',
                      },
                    ],
                  },
                ],
              }),
            ),
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        includeFiles: true,
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      // Only message, file skipped.
      expect(docs).toHaveLength(1)
    })

    it('handles empty channel', async () => {
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () =>
            Promise.resolve(jsonResponse({ ok: true, messages: [] })),
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      expect(docs).toHaveLength(0)
    })

    it('throws on Slack API error', async () => {
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () =>
            Promise.resolve(jsonResponse({ ok: false, error: 'channel_not_found' })),
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      await expect(collectDocs(connector.fetchAll(controller.signal))).rejects.toThrow(
        'channel_not_found',
      )
    })

    it('handles multiple channels', async () => {
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: (url) => {
            const params = new URL(url).searchParams
            const channel = params.get('channel')
            return Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  { type: 'message', user: 'U001', text: `Message in ${channel}`, ts: '1700000001.000000' },
                ],
              }),
            )
          },
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C001', 'C002', 'C003'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      expect(docs).toHaveLength(3)
    })
  })

  // -------------------------------------------------------------------------
  // Tests: FetchSince (incremental sync)
  // -------------------------------------------------------------------------

  describe('fetchSince', () => {
    it('passes cursor value as oldest parameter', async () => {
      let capturedOldest = ''
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: (url) => {
            const params = new URL(url).searchParams
            capturedOldest = params.get('oldest') ?? ''
            return Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  { type: 'message', user: 'U001', text: 'New message', ts: '1700000003.000000' },
                ],
              }),
            )
          },
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const cursor = { value: '1700000002.000000', updatedAt: new Date() }
      const docs = await collectDocs(connector.fetchSince(controller.signal, cursor))

      expect(capturedOldest).toBe('1700000002.000000')
      expect(docs).toHaveLength(1)
    })
  })

  // -------------------------------------------------------------------------
  // Tests: Rate limit 429 handling
  // -------------------------------------------------------------------------

  describe('rate limiting', () => {
    it('retries on 429 with Retry-After header', async () => {
      let callCount = 0
      const mockFetch = createMockFetch([
        {
          pattern: 'conversations.history',
          handler: () => {
            callCount++
            if (callCount === 1) {
              return Promise.resolve(
                new Response('Rate limited', {
                  status: 429,
                  headers: { 'Retry-After': '1' },
                }),
              )
            }
            return Promise.resolve(
              jsonResponse({
                ok: true,
                messages: [
                  { type: 'message', user: 'U001', text: 'After retry', ts: '1700000001.000000' },
                ],
              }),
            )
          },
        },
      ])

      const connector = new SlackConnector({
        botToken: 'xoxb-test',
        channels: ['C123ABC'],
        fetchFn: mockFetch,
      })

      const controller = new AbortController()
      const docs = await collectDocs(connector.fetchAll(controller.signal))

      expect(docs).toHaveLength(1)
      expect(callCount).toBeGreaterThanOrEqual(2)
    })
  })

  // -------------------------------------------------------------------------
  // Tests: Stop
  // -------------------------------------------------------------------------

  describe('stop', () => {
    it('can be called multiple times safely', async () => {
      const connector = createSlackConnector({ botToken: 'xoxb-test' })
      await expect(connector.stop()).resolves.toBeUndefined()
      await expect(connector.stop()).resolves.toBeUndefined()
    })
  })
})

// ---------------------------------------------------------------------------
// Tests: mrkdwn conversion
// ---------------------------------------------------------------------------

describe('convertMrkdwn', () => {
  it('returns empty string for empty input', () => {
    expect(convertMrkdwn('')).toBe('')
  })

  it('leaves plain text unchanged', () => {
    expect(convertMrkdwn('Hello world')).toBe('Hello world')
  })

  it('converts bold', () => {
    expect(convertMrkdwn('this is *bold text* here')).toBe('this is **bold text** here')
  })

  it('converts italic', () => {
    expect(convertMrkdwn('this is _italic text_ here')).toBe('this is *italic text* here')
  })

  it('converts strikethrough', () => {
    expect(convertMrkdwn('this is ~struck text~ here')).toBe('this is ~~struck text~~ here')
  })

  it('converts labelled links', () => {
    expect(convertMrkdwn('<https://example.com|Example Site>')).toBe(
      '[Example Site](https://example.com)',
    )
  })

  it('converts bare links', () => {
    expect(convertMrkdwn('<https://example.com>')).toBe('https://example.com')
  })

  it('converts channel mentions', () => {
    expect(convertMrkdwn('Check <#C123ABC|general> for updates')).toBe(
      'Check #general for updates',
    )
  })

  it('converts user mentions', () => {
    expect(convertMrkdwn('Hey <@U123ABC> check this')).toBe('Hey @U123ABC check this')
  })

  it('preserves inline code', () => {
    expect(convertMrkdwn('Run `go test` to verify')).toBe('Run `go test` to verify')
  })

  it('converts code blocks', () => {
    const input = 'Here is code:\n```func main() {}```'
    const result = convertMrkdwn(input)
    expect(result).toContain('```\nfunc main() {}\n```')
  })

  it('preserves emoji shortcodes', () => {
    expect(convertMrkdwn(':wave: Hello :smile:')).toBe(':wave: Hello :smile:')
  })

  it('handles multiple transformations', () => {
    const input = 'Hey <@U001>, check <https://example.com|this link> and _remember_ to *review*'
    const result = convertMrkdwn(input)
    expect(result).toContain('@U001')
    expect(result).toContain('[this link](https://example.com)')
    expect(result).toContain('*remember*')
    expect(result).toContain('**review**')
  })

  it('preserves blockquotes', () => {
    expect(convertMrkdwn('> This is a quote')).toBe('> This is a quote')
  })
})

// ---------------------------------------------------------------------------
// Tests: parseSlackTimestamp
// ---------------------------------------------------------------------------

describe('parseSlackTimestamp', () => {
  it('parses standard Slack timestamps', () => {
    const result = parseSlackTimestamp('1700000001.123456')
    expect(result.getTime()).toBeGreaterThan(0)
    expect(Math.floor(result.getTime() / 1000)).toBe(1700000001)
  })

  it('handles zero timestamp', () => {
    const result = parseSlackTimestamp('0.0')
    expect(result.getTime()).toBe(0)
  })
})

// ---------------------------------------------------------------------------
// Tests: Connector interface compliance
// ---------------------------------------------------------------------------

describe('type compliance', () => {
  it('SlackConnector satisfies Connector type', () => {
    const connector = createSlackConnector({ botToken: 'xoxb-test' })
    // Verify all Connector interface methods exist.
    expect(typeof connector.name).toBe('string')
    expect(typeof connector.configure).toBe('function')
    expect(typeof connector.fetchAll).toBe('function')
    expect(typeof connector.fetchSince).toBe('function')
    expect(typeof connector.start).toBe('function')
    expect(typeof connector.stop).toBe('function')
  })
})
