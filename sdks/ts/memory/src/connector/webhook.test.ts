// SPDX-License-Identifier: Apache-2.0

import { createHmac } from 'node:crypto'

import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { ConnectorDocument, DocumentDispatcher } from './types.js'
import {
  WebhookReceiver,
  type WebhookAuthConfig,
  type WebhookDocument,
  type WebhookPayload,
  type WebhookReceiverConfig,
  type WebhookResponse,
} from './webhook.js'

// ---- Mock dispatcher ----

type MockDispatcher = DocumentDispatcher & {
  readonly dispatched: ConnectorDocument[]
  readonly failIds: Set<string>
}

const createMockDispatcher = (): MockDispatcher => {
  const dispatched: ConnectorDocument[] = []
  const failIds = new Set<string>()
  let nextId = 0
  return {
    dispatched,
    failIds,
    async dispatch(doc) {
      if (failIds.has(doc.externalId)) {
        throw new Error(`dispatch error for ${doc.externalId}`)
      }
      dispatched.push(doc)
      nextId++
      return { documentId: `doc-${String(nextId)}` }
    },
  }
}

// ---- Helpers ----

const bearerAuth = (): WebhookAuthConfig => ({
  method: 'bearer',
  secret: 'test-secret-token',
})

const hmacAuth = (): WebhookAuthConfig => ({
  method: 'hmac',
  secret: 'hmac-secret-key',
})

const makePayload = (docs: WebhookDocument[]): string =>
  JSON.stringify({ documents: docs } satisfies WebhookPayload)

const singleDoc = (): WebhookDocument[] => [
  { externalId: 'ext-1', content: 'hello world' },
]

const doRequest = async (
  receiver: WebhookReceiver,
  method: string,
  body: string | null,
  headers: Record<string, string> = {},
): Promise<Response> => {
  const handler = receiver.handler()
  const init: RequestInit = {
    method,
    headers: {
      'content-type': 'application/json',
      ...headers,
    },
  }
  if (body !== null) {
    init.body = body
  }
  const req = new Request('http://localhost/webhook/ingest', init)
  return handler(req)
}

const computeHMAC = (
  secret: string,
  timestamp: number,
  body: string,
): string => {
  const mac = createHmac('sha256', secret)
  mac.update(String(timestamp))
  mac.update('.')
  mac.update(body)
  return `sha256=${mac.digest('hex')}`
}

const hmacHeaders = (
  secret: string,
  body: string,
): Record<string, string> => {
  const ts = Math.floor(Date.now() / 1000)
  return {
    'x-webhook-signature': computeHMAC(secret, ts, body),
    'x-webhook-timestamp': String(ts),
  }
}

const decodeResponse = async (res: Response): Promise<WebhookResponse> =>
  (await res.json()) as WebhookResponse

const createTestReceiver = (
  auth: WebhookAuthConfig,
  dispatcher: DocumentDispatcher,
  overrides?: Partial<WebhookReceiverConfig>,
): WebhookReceiver =>
  new WebhookReceiver({
    brainId: 'test-brain',
    auth,
    enableIdempotency: true,
    dispatcher,
    ...overrides,
  })

// ---- Tests ----

describe('WebhookReceiver', () => {
  let receiver: WebhookReceiver
  let dispatcher: MockDispatcher

  beforeEach(() => {
    dispatcher = createMockDispatcher()
  })

  afterEach(() => {
    receiver?.close()
  })

  describe('valid payloads', () => {
    it('accepts a single document', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(1)
      expect(data.rejected).toBe(0)
      expect(dispatcher.dispatched).toHaveLength(1)
    })

    it('accepts multiple documents', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const docs: WebhookDocument[] = [
        { externalId: 'ext-1', content: 'doc one' },
        { externalId: 'ext-2', content: 'doc two' },
        { externalId: 'ext-3', content: 'doc three' },
      ]
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(3)
    })

    it('reports partial failure with per-document status', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const docs: WebhookDocument[] = [
        { externalId: 'ext-1', content: 'valid' },
        { externalId: '', content: 'missing id' },
        { externalId: 'ext-3', content: 'also valid' },
      ]
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(2)
      expect(data.rejected).toBe(1)
      expect(data.results).toHaveLength(3)
    })
  })

  describe('bearer authentication', () => {
    it('succeeds with correct token', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })
      expect(res.status).toBe(200)
    })

    it('fails with wrong token', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer wrong-token',
      })
      expect(res.status).toBe(401)
      // Body must be empty to prevent information leakage.
      const text = await res.text()
      expect(text).toBe('')
    })

    it('fails with missing auth header', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {})
      expect(res.status).toBe(401)
    })
  })

  describe('HMAC authentication', () => {
    it('succeeds with correct signature', async () => {
      receiver = createTestReceiver(hmacAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const headers = hmacHeaders('hmac-secret-key', body)
      const res = await doRequest(receiver, 'POST', body, headers)
      expect(res.status).toBe(200)
    })

    it('fails with wrong signature', async () => {
      receiver = createTestReceiver(hmacAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const headers = hmacHeaders('wrong-secret', body)
      const res = await doRequest(receiver, 'POST', body, headers)
      expect(res.status).toBe(401)
    })

    it('fails with missing signature header', async () => {
      receiver = createTestReceiver(hmacAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        'x-webhook-timestamp': String(Math.floor(Date.now() / 1000)),
      })
      expect(res.status).toBe(401)
    })

    it('fails with expired timestamp', async () => {
      receiver = createTestReceiver(hmacAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const oldTs = Math.floor(Date.now() / 1000) - 600 // 10 minutes ago
      const sig = computeHMAC('hmac-secret-key', oldTs, body)
      const res = await doRequest(receiver, 'POST', body, {
        'x-webhook-signature': sig,
        'x-webhook-timestamp': String(oldTs),
      })
      expect(res.status).toBe(401)
    })

    it('fails with missing timestamp', async () => {
      receiver = createTestReceiver(hmacAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        'x-webhook-signature': 'sha256=deadbeef',
      })
      expect(res.status).toBe(401)
    })
  })

  describe('payload validation', () => {
    it('rejects empty documents array', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload([])
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })
      expect(res.status).toBe(400)
    })

    it('rejects too many documents', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher, {
        maxDocuments: 3,
      })
      const docs: WebhookDocument[] = Array.from({ length: 4 }, (_, i) => ({
        externalId: `ext-${String(i)}`,
        content: 'content',
      }))
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })
      expect(res.status).toBe(400)
    })

    it('rejects payload too large', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher, {
        maxPayloadBytes: 100,
      })
      const bigContent = 'x'.repeat(200)
      const body = makePayload([
        { externalId: 'ext-1', content: bigContent },
      ])
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })
      expect(res.status).toBe(413)
    })

    it('rejects individual document too large', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher, {
        maxDocumentBytes: 50,
      })
      const docs: WebhookDocument[] = [
        { externalId: 'small', content: 'ok' },
        { externalId: 'big', content: 'x'.repeat(100) },
      ]
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(1)
      expect(data.rejected).toBe(1)
    })

    it('handles base64 encoding', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const originalContent = 'binary content here'
      const encoded = Buffer.from(originalContent).toString('base64')
      const docs: WebhookDocument[] = [
        {
          externalId: 'ext-b64',
          content: encoded,
          encoding: 'base64',
          mime: 'application/octet-stream',
        },
      ]
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(1)
      expect(dispatcher.dispatched).toHaveLength(1)
      expect(dispatcher.dispatched[0]!.content.toString()).toBe(originalContent)
    })

    it('rejects invalid base64', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const docs: WebhookDocument[] = [
        {
          externalId: 'ext-bad64',
          content: 'not-valid-base64!!!',
          encoding: 'base64',
        },
      ]
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.rejected).toBe(1)
    })

    it('rejects wrong content type', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
        'content-type': 'text/plain',
      })
      expect(res.status).toBe(400)
    })

    it('rejects non-POST method', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const res = await doRequest(receiver, 'GET', null, {
        authorization: 'Bearer test-secret-token',
      })
      expect(res.status).toBe(405)
    })

    it('rejects malformed JSON', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const res = await doRequest(receiver, 'POST', '{invalid', {
        authorization: 'Bearer test-secret-token',
      })
      expect(res.status).toBe(400)
    })
  })

  describe('idempotency', () => {
    it('processes first request normally', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
        'x-idempotency-key': 'unique-key-123',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(1)
    })

    it('returns cached response for duplicate key', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload(singleDoc())
      const headers = {
        authorization: 'Bearer test-secret-token',
        'x-idempotency-key': 'dup-key-456',
      }

      // First request
      const res1 = await doRequest(receiver, 'POST', body, headers)
      expect(res1.status).toBe(200)

      // Second request with same key — no re-processing.
      const res2 = await doRequest(receiver, 'POST', body, headers)
      expect(res2.status).toBe(200)

      // Dispatcher should only have received the document once.
      expect(dispatcher.dispatched).toHaveLength(1)
    })
  })

  describe('document validation', () => {
    it('rejects document missing externalId', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = JSON.stringify({
        documents: [{ content: 'content but no id' }],
      })
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.rejected).toBe(1)
    })

    it('rejects document missing content', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = JSON.stringify({
        documents: [{ externalId: 'ext-1' }],
      })
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.rejected).toBe(1)
    })

    it('rejects unsupported encoding', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = JSON.stringify({
        documents: [
          { externalId: 'ext-1', content: 'data', encoding: 'gzip' },
        ],
      })
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.rejected).toBe(1)
    })
  })

  describe('dispatch', () => {
    it('handles dispatcher failure gracefully', async () => {
      dispatcher.failIds.add('ext-fail')
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const docs: WebhookDocument[] = [
        { externalId: 'ext-ok', content: 'good' },
        { externalId: 'ext-fail', content: 'will fail' },
      ]
      const body = makePayload(docs)
      const res = await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(res.status).toBe(200)
      const data = await decodeResponse(res)
      expect(data.accepted).toBe(1)
      expect(data.rejected).toBe(1)
    })

    it('sets default MIME to text/plain', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload([
        { externalId: 'ext-1', content: 'plain text' },
      ])
      await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(dispatcher.dispatched).toHaveLength(1)
      expect(dispatcher.dispatched[0]!.mime).toBe('text/plain')
    })

    it('preserves custom MIME', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload([
        { externalId: 'ext-1', content: '<html></html>', mime: 'text/html' },
      ])
      await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(dispatcher.dispatched).toHaveLength(1)
      expect(dispatcher.dispatched[0]!.mime).toBe('text/html')
    })

    it('passes metadata through to connector document', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      const body = makePayload([
        {
          externalId: 'ext-meta',
          content: 'content',
          title: 'My Document',
          url: 'https://example.com/doc',
          metadata: { author: 'test', source: 'webhook' },
        },
      ])
      await doRequest(receiver, 'POST', body, {
        authorization: 'Bearer test-secret-token',
      })

      expect(dispatcher.dispatched).toHaveLength(1)
      const doc = dispatcher.dispatched[0]!
      expect(doc.title).toBe('My Document')
      expect(doc.url).toBe('https://example.com/doc')
      expect(doc.metadata).toEqual({ author: 'test', source: 'webhook' })
    })
  })

  describe('rate limiting', () => {
    it('returns 429 when rate limit exceeded', async () => {
      receiver = createTestReceiver(bearerAuth(), dispatcher)
      // Exhaust the token bucket.
      const handler = receiver.handler()
      const promises: Promise<Response>[] = []
      for (let i = 0; i < 65; i++) {
        const body = makePayload(singleDoc())
        const req = new Request('http://localhost/webhook/ingest', {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: 'Bearer test-secret-token',
          },
          body,
        })
        promises.push(handler(req))
      }
      const responses = await Promise.all(promises)
      const rateLimited = responses.filter((r) => r.status === 429)
      expect(rateLimited.length).toBeGreaterThan(0)
    })
  })
})
