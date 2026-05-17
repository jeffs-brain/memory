// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, vi, beforeEach } from 'vitest'
import { GDriveConnector } from './gdrive.js'
import type { ConnectorConfig, ConnectorDocument, HTTPClient } from './types.js'
import { noopLogger } from '../llm/types.js'

// -- Test helpers -------------------------------------------------------------

type MockResponse = {
  status: number
  body: string
}

/**
 * Mock HTTP client that returns pre-configured responses based on URL
 * pattern matching. Tracks all requests for assertion.
 */
const createMockHTTPClient = (): HTTPClient & {
  requests: { url: string; init: RequestInit }[]
  responses: Map<string, MockResponse>
  addResponse: (pattern: string, status: number, body: string) => void
  addJSONResponse: (pattern: string, status: number, payload: unknown) => void
} => {
  const responses = new Map<string, MockResponse>()
  const requests: { url: string; init: RequestInit }[] = []

  return {
    requests,
    responses,
    addResponse(pattern: string, status: number, body: string) {
      responses.set(pattern, { status, body })
    },
    addJSONResponse(pattern: string, status: number, payload: unknown) {
      responses.set(pattern, { status, body: JSON.stringify(payload) })
    },
    async fetch(url: string, init: RequestInit): Promise<Response> {
      requests.push({ url, init })

      for (const [pattern, resp] of responses) {
        if (url.includes(pattern)) {
          return new Response(resp.body, {
            status: resp.status,
            headers: { 'Content-Type': 'application/json' },
          })
        }
      }

      return new Response('{"error":{"message":"not found"}}', { status: 404 })
    },
  }
}

const createTestDeps = (): ConnectorConfig => ({
  name: 'gdrive',
  brainId: 'test-brain',
  logger: noopLogger,
  store: {} as ConnectorConfig['store'],
})

const defaultConfig = (): Record<string, string> => ({
  oauth2_client_id: 'test-client-id',
  oauth2_client_secret: 'test-client-secret',
  access_token: 'test-access-token',
})

const collectDocuments = async (
  iterable: AsyncIterable<ConnectorDocument>,
): Promise<ConnectorDocument[]> => {
  const docs: ConnectorDocument[] = []
  for await (const doc of iterable) {
    docs.push(doc)
  }
  return docs
}

const makeAbortSignal = (): AbortSignal => new AbortController().signal

// -- Tests --------------------------------------------------------------------

describe('GDriveConnector', () => {
  let client: ReturnType<typeof createMockHTTPClient>
  let connector: GDriveConnector

  beforeEach(() => {
    client = createMockHTTPClient()
    connector = new GDriveConnector(createTestDeps(), client)
  })

  describe('name', () => {
    it('returns gdrive', () => {
      expect(connector.name).toBe('gdrive')
    })
  })

  describe('configure', () => {
    it('accepts valid configuration', async () => {
      await expect(connector.configure(defaultConfig())).resolves.toBeUndefined()
    })

    it('rejects missing client ID', async () => {
      await expect(
        connector.configure({
          oauth2_client_secret: 'secret',
        }),
      ).rejects.toThrow('oauth2_client_id is required')
    })

    it('rejects empty client ID', async () => {
      await expect(
        connector.configure({
          oauth2_client_id: '',
          oauth2_client_secret: 'secret',
        }),
      ).rejects.toThrow('oauth2_client_id is required')
    })

    it('rejects missing client secret', async () => {
      await expect(
        connector.configure({
          oauth2_client_id: 'id',
        }),
      ).rejects.toThrow('oauth2_client_secret is required')
    })

    it('accepts folder ID', async () => {
      await expect(
        connector.configure({
          ...defaultConfig(),
          folder_id: 'folder-123',
        }),
      ).resolves.toBeUndefined()
    })

    it('accepts MIME type filter', async () => {
      await expect(
        connector.configure({
          ...defaultConfig(),
          mime_type_filter: 'application/pdf,text/plain',
        }),
      ).resolves.toBeUndefined()
    })

    it('accepts shared drives flag', async () => {
      await expect(
        connector.configure({
          ...defaultConfig(),
          include_shared_drives: 'true',
        }),
      ).resolves.toBeUndefined()
    })

    it('rejects invalid max file size', async () => {
      await expect(
        connector.configure({
          ...defaultConfig(),
          max_file_size: 'not-a-number',
        }),
      ).rejects.toThrow('invalid max_file_size')
    })

    it('accepts custom max file size', async () => {
      await expect(
        connector.configure({
          ...defaultConfig(),
          max_file_size: '1048576',
        }),
      ).resolves.toBeUndefined()
    })
  })

  describe('fetchAll', () => {
    it('lists files in folder and yields documents', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/files?', 200, {
        files: [
          { id: 'f1', name: 'doc1.txt', mimeType: 'text/plain', modifiedTime: '2026-01-15T10:00:00Z', size: '100' },
          { id: 'f2', name: 'doc2.pdf', mimeType: 'application/pdf', modifiedTime: '2026-01-16T10:00:00Z', size: '200' },
          { id: 'f3', name: 'image.png', mimeType: 'image/png', modifiedTime: '2026-01-17T10:00:00Z', size: '300' },
        ],
      })
      client.addResponse('f1?alt=media', 200, 'file one content')
      client.addResponse('f2?alt=media', 200, 'file two content')
      client.addResponse('f3?alt=media', 200, 'file three content')

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(3)
      expect(docs[0].externalId).toBe('f1')
      expect(docs[0].title).toBe('doc1.txt')
      expect(docs[1].externalId).toBe('f2')
      expect(docs[2].externalId).toBe('f3')
    })

    it('handles paginated file listing', async () => {
      await connector.configure(defaultConfig())

      let callCount = 0
      const originalFetch = client.fetch.bind(client)
      client.fetch = async (url: string, init: RequestInit): Promise<Response> => {
        client.requests.push({ url, init })
        if (url.includes('drive/v3/files?')) {
          callCount++
          const body =
            callCount === 1
              ? JSON.stringify({
                  nextPageToken: 'page2token',
                  files: [
                    { id: 'f1', name: 'doc1.txt', mimeType: 'text/plain', modifiedTime: '2026-01-15T10:00:00Z', size: '100' },
                  ],
                })
              : JSON.stringify({
                  files: [
                    { id: 'f2', name: 'doc2.txt', mimeType: 'text/plain', modifiedTime: '2026-01-16T10:00:00Z', size: '200' },
                  ],
                })
          return new Response(body, { status: 200 })
        }
        return new Response('content', { status: 200 })
      }

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(2)
      expect(docs[0].externalId).toBe('f1')
      expect(docs[1].externalId).toBe('f2')
    })

    it('exports Google Doc as markdown', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/files?', 200, {
        files: [
          { id: 'doc1', name: 'My Document', mimeType: 'application/vnd.google-apps.document', modifiedTime: '2026-01-15T10:00:00Z' },
        ],
      })
      client.addResponse('doc1/export', 200, '# My Document\n\nHello world')

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(1)
      expect(docs[0].mime).toBe('text/markdown')
      expect(docs[0].content).toBe('# My Document\n\nHello world')
    })

    it('exports Google Sheet as CSV', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/files?', 200, {
        files: [
          { id: 'sheet1', name: 'Budget', mimeType: 'application/vnd.google-apps.spreadsheet', modifiedTime: '2026-01-15T10:00:00Z' },
        ],
      })
      client.addResponse('sheet1/export', 200, 'Name,Amount\nRent,1500')

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(1)
      expect(docs[0].mime).toBe('text/csv')
    })

    it('directly downloads PDF files', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/files?', 200, {
        files: [
          { id: 'pdf1', name: 'report.pdf', mimeType: 'application/pdf', modifiedTime: '2026-01-15T10:00:00Z', size: '5000' },
        ],
      })
      client.addResponse('pdf1?alt=media', 200, '%PDF-1.4 binary content')

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(1)
      expect(docs[0].mime).toBe('application/pdf')
      expect(docs[0].content).toContain('%PDF')
    })

    it('skips files exceeding size limit', async () => {
      await connector.configure({
        ...defaultConfig(),
        max_file_size: '100',
      })

      client.addJSONResponse('drive/v3/files?', 200, {
        files: [
          { id: 'small', name: 'small.txt', mimeType: 'text/plain', modifiedTime: '2026-01-15T10:00:00Z', size: '50' },
          { id: 'large', name: 'large.bin', mimeType: 'application/octet-stream', modifiedTime: '2026-01-15T10:00:00Z', size: '500' },
        ],
      })
      client.addResponse('small?alt=media', 200, 'small content')

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(1)
      expect(docs[0].externalId).toBe('small')
    })

    it('throws on rate limit 403', async () => {
      await connector.configure(defaultConfig())

      client.addResponse(
        'drive/v3/files?',
        403,
        '{"error":{"message":"Rate Limit Exceeded","errors":[{"reason":"rateLimitExceeded"}]}}',
      )

      await expect(
        collectDocuments(connector.fetchAll(makeAbortSignal())),
      ).rejects.toThrow('rate limit exceeded')
    })

    it('includes folder ID in query parameter', async () => {
      await connector.configure({
        ...defaultConfig(),
        folder_id: 'folder-abc',
      })

      client.addJSONResponse('drive/v3/files?', 200, { files: [] })

      await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(client.requests.length).toBeGreaterThan(0)
      const reqUrl = client.requests[0].url
      expect(reqUrl).toContain('folder-abc')
    })

    it('includes MIME type filter in query', async () => {
      await connector.configure({
        ...defaultConfig(),
        mime_type_filter: 'application/pdf',
      })

      client.addJSONResponse('drive/v3/files?', 200, { files: [] })

      await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(client.requests.length).toBeGreaterThan(0)
      const reqUrl = client.requests[0].url
      expect(reqUrl).toContain('application%2Fpdf')
    })

    it('sets supportsAllDrives when shared drives enabled', async () => {
      await connector.configure({
        ...defaultConfig(),
        include_shared_drives: 'true',
      })

      client.addJSONResponse('drive/v3/files?', 200, { files: [] })

      await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(client.requests.length).toBeGreaterThan(0)
      const reqUrl = client.requests[0].url
      expect(reqUrl).toContain('supportsAllDrives=true')
    })

    it('throws when not configured', async () => {
      await expect(
        collectDocuments(connector.fetchAll(makeAbortSignal())),
      ).rejects.toThrow('not configured')
    })
  })

  describe('fetchSince', () => {
    it('processes incremental changes with modified and deleted files', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/changes?', 200, {
        newStartPageToken: 'new-token-abc',
        changes: [
          {
            fileId: 'f1',
            file: { id: 'f1', name: 'updated.txt', mimeType: 'text/plain', modifiedTime: '2026-01-20T10:00:00Z', size: '150' },
            removed: false,
          },
          {
            fileId: 'f2',
            removed: true,
          },
        ],
      })
      client.addResponse('f1?alt=media', 200, 'updated content')

      const cursor: SyncCursor = { value: 'old-token-xyz', updatedAt: new Date() }
      const docs = await collectDocuments(connector.fetchSince(makeAbortSignal(), cursor))

      // Modified file + deleted file + cursor update = 3
      expect(docs).toHaveLength(3)

      // First: modified file.
      expect(docs[0].externalId).toBe('f1')
      expect(docs[0].deleted).toBeFalsy()

      // Second: deleted file.
      expect(docs[1].externalId).toBe('f2')
      expect(docs[1].deleted).toBe(true)

      // Third: cursor update.
      expect(docs[2].externalId).toBe('__cursor_update__')
      expect(docs[2].metadata['new_start_page_token']).toBe('new-token-abc')
    })

    it('updates cursor after sync with no changes', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/changes?', 200, {
        newStartPageToken: 'updated-cursor-token',
        changes: [],
      })

      const cursor: SyncCursor = { value: 'initial-token', updatedAt: new Date() }
      const docs = await collectDocuments(connector.fetchSince(makeAbortSignal(), cursor))

      expect(docs).toHaveLength(1)
      expect(docs[0].metadata['new_start_page_token']).toBe('updated-cursor-token')
    })
  })

  describe('getStartPageToken', () => {
    it('retrieves the start page token', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('changes/startPageToken', 200, {
        startPageToken: 'token-12345',
      })

      const token = await connector.getStartPageToken(makeAbortSignal())
      expect(token).toBe('token-12345')
    })
  })

  describe('start', () => {
    it('throws not supported error', async () => {
      await connector.configure(defaultConfig())
      await expect(connector.start(makeAbortSignal())).rejects.toThrow(
        'continuous sync not yet supported',
      )
    })
  })

  describe('stop', () => {
    it('resolves without error', async () => {
      await expect(connector.stop()).resolves.toBeUndefined()
    })
  })

  describe('setAccessToken', () => {
    it('updates the access token used for requests', async () => {
      await connector.configure(defaultConfig())
      connector.setAccessToken('new-token-value')

      client.addJSONResponse('drive/v3/files?', 200, { files: [] })

      await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(client.requests.length).toBeGreaterThan(0)
      const authHeader = (client.requests[0].init.headers as Record<string, string>)['Authorization']
      expect(authHeader).toBe('Bearer new-token-value')
    })
  })

  describe('metadata', () => {
    it('includes source, file_id, mime_type, parent_id, and size', async () => {
      await connector.configure(defaultConfig())

      client.addJSONResponse('drive/v3/files?', 200, {
        files: [
          {
            id: 'file-123',
            name: 'test.txt',
            mimeType: 'text/plain',
            modifiedTime: '2026-01-15T10:00:00Z',
            size: '1024',
            parents: ['parent-456'],
          },
        ],
      })
      client.addResponse('file-123?alt=media', 200, 'content')

      const docs = await collectDocuments(connector.fetchAll(makeAbortSignal()))

      expect(docs).toHaveLength(1)
      expect(docs[0].metadata['source']).toBe('gdrive')
      expect(docs[0].metadata['file_id']).toBe('file-123')
      expect(docs[0].metadata['mime_type']).toBe('text/plain')
      expect(docs[0].metadata['parent_id']).toBe('parent-456')
      expect(docs[0].metadata['size']).toBe('1024')
    })
  })
})

// Type import for the test file scope.
type SyncCursor = { value: string; updatedAt: Date }
