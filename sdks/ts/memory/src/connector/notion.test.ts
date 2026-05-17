// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  createNotionConnector,
  type NotionHTTPFetcher,
} from './notion.js'
import type { ConnectorConfig, ConnectorDocument } from './types.js'

// ---------- Test helpers ----------

type MockResponse = {
  readonly status: number
  readonly body: string
}

/**
 * Creates a mock fetcher that returns canned responses in sequence.
 */
const createMockFetcher = (
  responses: readonly MockResponse[],
): {
  readonly fetcher: NotionHTTPFetcher
  readonly calls: { url: string; method: string; body: string }[]
} => {
  let callIndex = 0
  const calls: { url: string; method: string; body: string }[] = []

  const fetcher: NotionHTTPFetcher = async (url, options) => {
    const bodyText =
      options.body !== undefined && options.body !== null
        ? String(options.body)
        : ''
    calls.push({ url, method: options.method ?? 'GET', body: bodyText })

    const idx = callIndex
    callIndex++

    const resp =
      idx < responses.length
        ? responses[idx]
        : { status: 200, body: '{}' }

    return new Response(resp.body, {
      status: resp.status,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  return { fetcher, calls }
}

/**
 * Collects all documents from an async iterable.
 */
const collectDocs = async (
  iterable: AsyncIterable<ConnectorDocument>,
): Promise<readonly ConnectorDocument[]> => {
  const docs: ConnectorDocument[] = []
  for await (const doc of iterable) {
    docs.push(doc)
  }
  return docs
}

/**
 * Creates a minimal ConnectorConfig for tests.
 */
const testDeps = (): ConnectorConfig =>
  ({
    name: 'notion',
    brainId: 'test-brain',
    store: {} as ConnectorConfig['store'],
  }) as ConnectorConfig

// ---------- Tests ----------

describe('NotionConnector', () => {
  describe('name', () => {
    it('returns notion', () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      expect(conn.name).toBe('notion')
    })
  })

  describe('configure', () => {
    it('rejects missing apiToken', async () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      await expect(conn.configure({})).rejects.toThrow('apiToken is required')
    })

    it('rejects empty apiToken', async () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      await expect(conn.configure({ apiToken: '' })).rejects.toThrow(
        'apiToken is required',
      )
    })

    it('accepts valid configuration', async () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_abc123',
        rootPageIds: ['page-1', 'page-2'],
        databaseIds: ['db-1'],
        includeChildPages: false,
        maxDepth: 5,
      })
      // No error thrown.
    })
  })

  describe('fetchAll', () => {
    it('errors when not configured', async () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      const signal = AbortSignal.timeout(5000)
      await expect(collectDocs(conn.fetchAll(signal))).rejects.toThrow(
        'not configured',
      )
    })

    it('fetches page via markdown API', async () => {
      const pageResp = JSON.stringify({
        id: 'page-123',
        url: 'https://notion.so/page-123',
        last_edited_time: '2026-05-10T14:30:00.000Z',
        properties: {
          Name: {
            type: 'title',
            title: [{ plain_text: 'Test Page' }],
          },
        },
        parent: { type: 'workspace' },
      })
      const markdownResp = JSON.stringify({
        markdown: '# Test Page\n\nThis is test content.',
      })
      const childBlocksResp = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: pageResp },
        { status: 200, body: markdownResp },
        { status: 200, body: childBlocksResp },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['page-123'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))

      expect(docs).toHaveLength(1)
      expect(docs[0].externalId).toBe('page-123')
      expect(docs[0].title).toBe('Test Page')
      expect(docs[0].mime).toBe('text/markdown')
      expect(String(docs[0].content)).toContain('This is test content')
      expect(docs[0].metadata.source).toBe('notion')
    })

    it('falls back to block retrieval when markdown API returns 404', async () => {
      const pageResp = JSON.stringify({
        id: 'page-456',
        url: 'https://notion.so/page-456',
        last_edited_time: '2026-05-10T14:30:00.000Z',
        properties: {
          Title: {
            type: 'title',
            title: [{ plain_text: 'Block Page' }],
          },
        },
        parent: { type: 'workspace' },
      })
      const blocksResp = JSON.stringify({
        results: [
          {
            id: 'block-1',
            type: 'heading_1',
            has_children: false,
            heading_1: {
              rich_text: [{ type: 'text', plain_text: 'Heading One' }],
            },
          },
          {
            id: 'block-2',
            type: 'paragraph',
            has_children: false,
            paragraph: {
              rich_text: [
                { type: 'text', plain_text: 'Paragraph content here.' },
              ],
            },
          },
        ],
        has_more: false,
      })
      const childBlocksResp = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: pageResp },
        { status: 404, body: '{"message": "not found"}' },
        { status: 200, body: blocksResp },
        { status: 200, body: childBlocksResp },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['page-456'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))

      expect(docs).toHaveLength(1)
      const content = String(docs[0].content)
      expect(content).toContain('# Heading One')
      expect(content).toContain('Paragraph content here.')
    })

    it('queries database with property listings', async () => {
      const dbQueryResp = JSON.stringify({
        results: [
          {
            id: 'entry-1',
            url: 'https://notion.so/entry-1',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Entry One' }],
              },
              Status: {
                type: 'select',
                select: { name: 'Active' },
              },
              Priority: {
                type: 'number',
                number: 3,
              },
            },
          },
          {
            id: 'entry-2',
            url: 'https://notion.so/entry-2',
            last_edited_time: '2026-05-10T11:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Entry Two' }],
              },
              Status: {
                type: 'select',
                select: { name: 'Done' },
              },
            },
          },
        ],
        has_more: false,
      })
      const entryMD = JSON.stringify({ markdown: 'Details.' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbQueryResp },
        { status: 200, body: entryMD },
        { status: 200, body: entryMD },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-test'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))

      expect(docs).toHaveLength(2)
      expect(docs[0].title).toBe('Entry One')
      expect(docs[0].metadata.type).toBe('database_entry')
      expect(String(docs[0].content)).toContain('## Entry One')
      expect(String(docs[0].content)).toContain('Active')
    })

    it('handles paginated database query', async () => {
      const page1 = JSON.stringify({
        results: [
          {
            id: 'entry-a',
            url: 'https://notion.so/entry-a',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'A' }],
              },
            },
          },
        ],
        has_more: true,
        next_cursor: 'cursor-page-2',
      })
      const page2 = JSON.stringify({
        results: [
          {
            id: 'entry-b',
            url: 'https://notion.so/entry-b',
            last_edited_time: '2026-05-10T11:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'B' }],
              },
            },
          },
        ],
        has_more: false,
      })
      const entryMD = JSON.stringify({ markdown: 'content' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: page1 },
        { status: 200, body: entryMD },
        { status: 200, body: page2 },
        { status: 200, body: entryMD },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-paginated'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(2)
    })

    it('recursively fetches child pages', async () => {
      const rootPage = JSON.stringify({
        id: 'root',
        url: 'https://notion.so/root',
        last_edited_time: '2026-05-10T14:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'Root' }] },
        },
        parent: { type: 'workspace' },
      })
      const rootMD = JSON.stringify({ markdown: 'Root content' })
      const rootChildren = JSON.stringify({
        results: [
          { id: 'child-1', type: 'child_page', has_children: false },
        ],
        has_more: false,
      })
      const childPage = JSON.stringify({
        id: 'child-1',
        url: 'https://notion.so/child-1',
        last_edited_time: '2026-05-10T15:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'Child' }] },
        },
        parent: { type: 'page_id', page_id: 'root' },
      })
      const childMD = JSON.stringify({ markdown: 'Child content' })
      const childChildren = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: rootPage },
        { status: 200, body: rootMD },
        { status: 200, body: rootChildren },
        { status: 200, body: childPage },
        { status: 200, body: childMD },
        { status: 200, body: childChildren },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['root'],
        includeChildPages: true,
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(2)
    })

    it('enforces max depth', async () => {
      const rootPage = JSON.stringify({
        id: 'root',
        url: 'https://notion.so/root',
        last_edited_time: '2026-05-10T14:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'Root' }] },
        },
        parent: { type: 'workspace' },
      })
      const md = JSON.stringify({ markdown: 'content' })
      const children = JSON.stringify({
        results: [
          { id: 'child-1', type: 'child_page', has_children: false },
        ],
        has_more: false,
      })
      const childPage = JSON.stringify({
        id: 'child-1',
        url: 'https://notion.so/child-1',
        last_edited_time: '2026-05-10T15:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'Child' }] },
        },
        parent: { type: 'page_id', page_id: 'root' },
      })
      const childChildren = JSON.stringify({
        results: [
          {
            id: 'grandchild-1',
            type: 'child_page',
            has_children: false,
          },
        ],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: rootPage },
        { status: 200, body: md },
        { status: 200, body: children },
        { status: 200, body: childPage },
        { status: 200, body: md },
        { status: 200, body: childChildren },
        // grandchild would be depth 2, maxDepth=1 stops it.
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['root'],
        maxDepth: 1,
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(2) // root + child, no grandchild
    })

    it('detects cycles and prevents infinite loops', async () => {
      const pageA = JSON.stringify({
        id: 'page-a',
        url: 'https://notion.so/page-a',
        last_edited_time: '2026-05-10T14:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'A' }] },
        },
        parent: { type: 'workspace' },
      })
      const md = JSON.stringify({ markdown: 'content' })
      const childrenA = JSON.stringify({
        results: [
          { id: 'page-b', type: 'child_page', has_children: false },
        ],
        has_more: false,
      })
      const pageB = JSON.stringify({
        id: 'page-b',
        url: 'https://notion.so/page-b',
        last_edited_time: '2026-05-10T15:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'B' }] },
        },
        parent: { type: 'page_id', page_id: 'page-a' },
      })
      const childrenB = JSON.stringify({
        results: [
          { id: 'page-a', type: 'child_page', has_children: false },
        ],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: pageA },
        { status: 200, body: md },
        { status: 200, body: childrenA },
        { status: 200, body: pageB },
        { status: 200, body: md },
        { status: 200, body: childrenB },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['page-a'],
        includeChildPages: true,
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(2) // A and B, no cycle
    })

    it('handles rate limit 429 responses', async () => {
      const { fetcher } = createMockFetcher([
        { status: 429, body: '{"message": "rate limited"}' },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['page-rl'],
      })

      const signal = AbortSignal.timeout(5000)
      // The fetch should not throw but skip the page due to the error.
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(0) // Page skipped due to error
    })

    it('returns no documents for empty workspace', async () => {
      const searchResp = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: searchResp },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({ apiToken: 'secret_test' })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(0)
    })

    it('discovers pages via workspace search', async () => {
      const searchResp = JSON.stringify({
        results: [
          {
            id: 'ws-page-1',
            object: 'page',
            last_edited_time: '2026-05-10T14:00:00.000Z',
          },
        ],
        has_more: false,
      })
      const pageResp = JSON.stringify({
        id: 'ws-page-1',
        url: 'https://notion.so/ws-page-1',
        last_edited_time: '2026-05-10T14:00:00.000Z',
        properties: {
          Title: {
            type: 'title',
            title: [{ plain_text: 'WS Page' }],
          },
        },
        parent: { type: 'workspace' },
      })
      const md = JSON.stringify({ markdown: 'workspace content' })
      const childBlocks = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: searchResp },
        { status: 200, body: pageResp },
        { status: 200, body: md },
        { status: 200, body: childBlocks },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({ apiToken: 'secret_test' })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs).toHaveLength(1)
      expect(docs[0].title).toBe('WS Page')
    })
  })

  describe('fetchSince', () => {
    it('passes filter for incremental sync', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'recent',
            url: 'https://notion.so/recent',
            last_edited_time: '2026-05-15T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Recent' }],
              },
            },
          },
        ],
        has_more: false,
      })
      const entryMD = JSON.stringify({ markdown: 'updated' })

      const { fetcher, calls } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: entryMD },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-inc'],
      })

      const cursor = {
        value: '2026-05-01T00:00:00.000Z',
        updatedAt: new Date(),
      }
      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchSince(signal, cursor))

      expect(docs).toHaveLength(1)
      // Verify the filter was sent.
      const dbCall = calls[0]
      expect(dbCall.body).toContain('on_or_after')
    })

    it('tracks modified time for cursor updates', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'e1',
            url: 'https://notion.so/e1',
            last_edited_time: '2026-05-15T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'E1' }],
              },
            },
          },
        ],
        has_more: false,
      })
      const md = JSON.stringify({ markdown: 'content' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: md },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-cursor'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(docs[0].modifiedAt).toEqual(
        new Date('2026-05-15T10:00:00.000Z'),
      )
    })
  })

  describe('stop', () => {
    it('does not throw when called before start', async () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      await expect(conn.stop()).resolves.toBeUndefined()
    })

    it('can be called multiple times safely', async () => {
      const { fetcher } = createMockFetcher([])
      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.stop()
      await conn.stop()
    })
  })

  describe('property extraction', () => {
    it('handles select properties in database entries', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'sel-1',
            url: 'https://notion.so/sel-1',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Select Test' }],
              },
              Priority: {
                type: 'select',
                select: { name: 'High' },
              },
            },
          },
        ],
        has_more: false,
      })
      const md = JSON.stringify({ markdown: '' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: md },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-sel'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(String(docs[0].content)).toContain('High')
    })

    it('handles multi_select properties', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'ms-1',
            url: 'https://notion.so/ms-1',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Tags Test' }],
              },
              Tags: {
                type: 'multi_select',
                multi_select: [{ name: 'Tag1' }, { name: 'Tag2' }],
              },
            },
          },
        ],
        has_more: false,
      })
      const md = JSON.stringify({ markdown: '' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: md },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-ms'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(String(docs[0].content)).toContain('Tag1, Tag2')
    })

    it('handles date range properties', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'dt-1',
            url: 'https://notion.so/dt-1',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Date Test' }],
              },
              Dates: {
                type: 'date',
                date: { start: '2026-05-01', end: '2026-05-15' },
              },
            },
          },
        ],
        has_more: false,
      })
      const md = JSON.stringify({ markdown: '' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: md },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-dt'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(String(docs[0].content)).toContain('2026-05-01 to 2026-05-15')
    })

    it('handles checkbox properties', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'cb-1',
            url: 'https://notion.so/cb-1',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Checkbox Test' }],
              },
              Done: {
                type: 'checkbox',
                checkbox: true,
              },
            },
          },
        ],
        has_more: false,
      })
      const md = JSON.stringify({ markdown: '' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: md },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-cb'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      expect(String(docs[0].content)).toContain('true')
    })

    it('handles null select property', async () => {
      const dbResp = JSON.stringify({
        results: [
          {
            id: 'ns-1',
            url: 'https://notion.so/ns-1',
            last_edited_time: '2026-05-10T10:00:00.000Z',
            properties: {
              Name: {
                type: 'title',
                title: [{ plain_text: 'Null Select' }],
              },
              Category: {
                type: 'select',
                select: null,
              },
            },
          },
        ],
        has_more: false,
      })
      const md = JSON.stringify({ markdown: '' })

      const { fetcher } = createMockFetcher([
        { status: 200, body: dbResp },
        { status: 200, body: md },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        databaseIds: ['db-ns'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      // Null select should not appear in property listing.
      expect(String(docs[0].content)).not.toContain('Category')
    })
  })

  describe('block rendering', () => {
    it('renders code blocks with language', async () => {
      const pageResp = JSON.stringify({
        id: 'code-page',
        url: 'https://notion.so/code-page',
        last_edited_time: '2026-05-10T14:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'Code' }] },
        },
        parent: { type: 'workspace' },
      })
      const blocksResp = JSON.stringify({
        results: [
          {
            id: 'code-1',
            type: 'code',
            has_children: false,
            code: {
              rich_text: [
                { type: 'text', plain_text: 'const x = 1' },
              ],
              language: 'typescript',
            },
          },
        ],
        has_more: false,
      })
      const childBlocks = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: pageResp },
        { status: 404, body: '{}' }, // markdown API not available
        { status: 200, body: blocksResp },
        { status: 200, body: childBlocks },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['code-page'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      const content = String(docs[0].content)
      expect(content).toContain('```typescript')
      expect(content).toContain('const x = 1')
    })

    it('renders rich text with formatting', async () => {
      const pageResp = JSON.stringify({
        id: 'rt-page',
        url: 'https://notion.so/rt-page',
        last_edited_time: '2026-05-10T14:00:00.000Z',
        properties: {
          Title: { type: 'title', title: [{ plain_text: 'Rich' }] },
        },
        parent: { type: 'workspace' },
      })
      const blocksResp = JSON.stringify({
        results: [
          {
            id: 'rt-1',
            type: 'paragraph',
            has_children: false,
            paragraph: {
              rich_text: [
                {
                  type: 'text',
                  plain_text: 'bold',
                  annotations: { bold: true },
                },
                { type: 'text', plain_text: ' and ' },
                {
                  type: 'text',
                  plain_text: 'italic',
                  annotations: { italic: true },
                },
              ],
            },
          },
        ],
        has_more: false,
      })
      const childBlocks = JSON.stringify({
        results: [],
        has_more: false,
      })

      const { fetcher } = createMockFetcher([
        { status: 200, body: pageResp },
        { status: 404, body: '{}' },
        { status: 200, body: blocksResp },
        { status: 200, body: childBlocks },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['rt-page'],
      })

      const signal = AbortSignal.timeout(5000)
      const docs = await collectDocs(conn.fetchAll(signal))
      const content = String(docs[0].content)
      expect(content).toContain('**bold**')
      expect(content).toContain('*italic*')
    })
  })

  describe('context cancellation', () => {
    it('does not hang when signal is already aborted', async () => {
      const { fetcher } = createMockFetcher([
        { status: 200, body: JSON.stringify({
          id: 'p1',
          url: 'https://notion.so/p1',
          last_edited_time: '2026-05-10T14:00:00.000Z',
          properties: {
            Title: { type: 'title', title: [{ plain_text: 'T' }] },
          },
          parent: { type: 'workspace' },
        }) },
        { status: 200, body: JSON.stringify({ markdown: 'content' }) },
        { status: 200, body: JSON.stringify({ results: [], has_more: false }) },
      ])

      const conn = createNotionConnector(testDeps(), { fetcher })
      await conn.configure({
        apiToken: 'secret_test',
        rootPageIds: ['p1'],
      })

      const controller = new AbortController()
      controller.abort()

      // Should not hang; may throw or return empty.
      try {
        const docs = await collectDocs(conn.fetchAll(controller.signal))
        expect(docs.length).toBeLessThanOrEqual(1)
      } catch {
        // Expected when signal is aborted.
      }
    })
  })
})
