// SPDX-License-Identifier: Apache-2.0

/**
 * Generic webhook receiver for push-based document ingestion. Accepts
 * POST requests with a JSON payload of documents, validates authentication
 * and schema, then dispatches accepted documents to the ingestion pipeline.
 *
 * Authentication supports bearer token or HMAC-SHA256 signature verification.
 * HMAC signatures bind the timestamp and body together to provide replay
 * protection with a configurable expiry window (default: 5 minutes).
 */

import { createHmac, timingSafeEqual, createHash } from 'node:crypto'

import type { ConnectorDocument, DocumentDispatcher } from './types.js'

// ---- Configuration types ----

export type WebhookAuthMethod = 'bearer' | 'hmac'

export type WebhookAuthConfig = {
  readonly method: WebhookAuthMethod
  readonly secret: string
}

export type WebhookReceiverConfig = {
  readonly brainId: string
  readonly auth: WebhookAuthConfig
  readonly maxDocuments?: number
  readonly maxPayloadBytes?: number
  readonly maxDocumentBytes?: number
  readonly enableIdempotency?: boolean
  readonly timestampExpiryMs?: number
  readonly dispatcher: DocumentDispatcher
}

// ---- Payload types ----

export type WebhookPayload = {
  readonly documents: readonly WebhookDocument[]
}

export type WebhookDocument = {
  readonly externalId: string
  readonly content: string
  readonly encoding?: 'utf8' | 'base64'
  readonly mime?: string
  readonly title?: string
  readonly url?: string
  readonly metadata?: Readonly<Record<string, unknown>>
}

export type WebhookResponse = {
  readonly accepted: number
  readonly rejected: number
  readonly results: readonly WebhookDocumentResult[]
}

export type WebhookDocumentResult = {
  readonly externalId: string
  readonly status: 'accepted' | 'rejected'
  readonly documentId?: string
  readonly error?: string
}

// ---- Rate limiter ----

class TokenBucket {
  private tokens: number
  private readonly maxTokens: number
  private readonly refillRate: number
  private lastRefill: number

  constructor(maxTokens: number, refillRate: number) {
    this.tokens = maxTokens
    this.maxTokens = maxTokens
    this.refillRate = refillRate
    this.lastRefill = Date.now()
  }

  tryAcquire(): boolean {
    this.refill()
    if (this.tokens < 1) {
      return false
    }
    this.tokens--
    return true
  }

  private refill(): void {
    const now = Date.now()
    const elapsed = (now - this.lastRefill) / 1000
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsed * this.refillRate)
    this.lastRefill = now
  }
}

// ---- Idempotency store ----

type IdempotencyEntry = {
  readonly response: WebhookResponse
  readonly expiresAt: number
}

class IdempotencyStore {
  private readonly entries = new Map<string, IdempotencyEntry>()
  private sweepTimer: ReturnType<typeof setInterval> | undefined

  constructor() {
    this.sweepTimer = setInterval(() => this.sweep(), 60 * 60 * 1000)
  }

  get(key: string): WebhookResponse | undefined {
    const entry = this.entries.get(key)
    if (entry === undefined) {
      return undefined
    }
    if (Date.now() > entry.expiresAt) {
      this.entries.delete(key)
      return undefined
    }
    return entry.response
  }

  set(key: string, response: WebhookResponse, ttlMs: number): void {
    this.entries.set(key, {
      response,
      expiresAt: Date.now() + ttlMs,
    })
  }

  sweep(): void {
    const now = Date.now()
    for (const [key, entry] of this.entries) {
      if (now > entry.expiresAt) {
        this.entries.delete(key)
      }
    }
  }

  close(): void {
    if (this.sweepTimer !== undefined) {
      clearInterval(this.sweepTimer)
      this.sweepTimer = undefined
    }
  }

  /** Visible for testing. */
  get size(): number {
    return this.entries.size
  }
}

// ---- Constants ----

const DEFAULT_MAX_DOCUMENTS = 50
const DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024 // 8 MiB
const DEFAULT_MAX_DOCUMENT_BYTES = 10 * 1024 * 1024 // 10 MiB
const DEFAULT_TIMESTAMP_EXPIRY_MS = 5 * 60 * 1000 // 5 minutes
const DEFAULT_RATE_LIMIT_TOKENS = 60
const DEFAULT_RATE_LIMIT_RATE = 10.0 // tokens per second
const IDEMPOTENCY_TTL_MS = 24 * 60 * 60 * 1000 // 24 hours

// ---- Authenticators ----

type Authenticator = (
  headers: Headers,
  body: Uint8Array,
  secret: string,
  timestampExpiryMs: number,
) => boolean

const authenticateBearer: Authenticator = (headers, _body, secret): boolean => {
  const header = headers.get('authorization') ?? ''
  if (header === '') {
    return false
  }
  const lowerHeader = header.toLowerCase()
  if (!lowerHeader.startsWith('bearer ')) {
    return false
  }
  const actual = header.slice(7).trim()
  if (actual === '') {
    return false
  }
  // Timing-safe comparison via SHA-256 hash to prevent length-based leaks.
  const actualHash = createHash('sha256').update(actual).digest()
  const expectedHash = createHash('sha256').update(secret).digest()
  return timingSafeEqual(actualHash, expectedHash)
}

const authenticateHMAC: Authenticator = (
  headers,
  body,
  secret,
  timestampExpiryMs,
): boolean => {
  const sigHeader = headers.get('x-webhook-signature') ?? ''
  if (sigHeader === '') {
    return false
  }

  // Verify timestamp expiry for replay protection.
  const tsHeader = headers.get('x-webhook-timestamp') ?? ''
  if (tsHeader === '') {
    return false
  }
  const ts = parseInt(tsHeader, 10)
  if (Number.isNaN(ts)) {
    return false
  }
  const requestTime = ts * 1000
  if (Math.abs(Date.now() - requestTime) > timestampExpiryMs) {
    return false
  }

  // Parse signature: expect "sha256=<hex>"
  const sigPrefix = 'sha256='
  if (!sigHeader.startsWith(sigPrefix)) {
    return false
  }
  const providedSigHex = sigHeader.slice(sigPrefix.length)
  let providedSig: Buffer
  try {
    providedSig = Buffer.from(providedSigHex, 'hex')
  } catch {
    return false
  }
  if (providedSig.length === 0) {
    return false
  }

  // Compute expected HMAC-SHA256 over timestamp + "." + body to bind
  // the signature to both the timestamp and the payload.
  const mac = createHmac('sha256', secret)
  mac.update(tsHeader)
  mac.update('.')
  mac.update(body)
  const expectedSig = mac.digest()

  if (providedSig.length !== expectedSig.length) {
    return false
  }
  return timingSafeEqual(providedSig, expectedSig)
}

const authenticators: Record<WebhookAuthMethod, Authenticator> = {
  bearer: authenticateBearer,
  hmac: authenticateHMAC,
}

// ---- Content decoders ----

type ContentDecoder = (raw: string) => Buffer

const decodeUtf8: ContentDecoder = (raw) => Buffer.from(raw, 'utf8')

const decodeBase64: ContentDecoder = (raw) => {
  const decoded = Buffer.from(raw, 'base64')
  // Verify round-trip to detect invalid base64.
  if (decoded.toString('base64') !== raw) {
    throw new Error('invalid base64 content')
  }
  return decoded
}

const contentDecoders: Record<string, ContentDecoder> = {
  '': decodeUtf8,
  utf8: decodeUtf8,
  base64: decodeBase64,
}

// ---- WebhookReceiver ----

/**
 * HTTP handler that accepts POST requests with document payloads for
 * ingestion. Compatible with any server that uses the fetch-style
 * `Request => Response` pattern.
 */
export class WebhookReceiver {
  private readonly brainId: string
  private readonly auth: WebhookAuthConfig
  private readonly maxDocuments: number
  private readonly maxPayloadBytes: number
  private readonly maxDocumentBytes: number
  private readonly enableIdempotency: boolean
  private readonly timestampExpiryMs: number
  private readonly dispatcher: DocumentDispatcher
  private readonly limiter: TokenBucket
  private readonly idempotency: IdempotencyStore

  constructor(config: WebhookReceiverConfig) {
    this.brainId = config.brainId
    this.auth = config.auth
    this.maxDocuments = config.maxDocuments ?? DEFAULT_MAX_DOCUMENTS
    this.maxPayloadBytes = config.maxPayloadBytes ?? DEFAULT_MAX_PAYLOAD_BYTES
    this.maxDocumentBytes = config.maxDocumentBytes ?? DEFAULT_MAX_DOCUMENT_BYTES
    this.enableIdempotency = config.enableIdempotency ?? true
    this.timestampExpiryMs = config.timestampExpiryMs ?? DEFAULT_TIMESTAMP_EXPIRY_MS
    this.dispatcher = config.dispatcher
    this.limiter = new TokenBucket(DEFAULT_RATE_LIMIT_TOKENS, DEFAULT_RATE_LIMIT_RATE)
    this.idempotency = new IdempotencyStore()
  }

  /** Release background resources (sweep timer). */
  close(): void {
    this.idempotency.close()
  }

  /**
   * Returns a fetch-style request handler. The handler is stateless and
   * safe for concurrent use.
   */
  handler(): (req: Request) => Promise<Response> {
    return (req: Request) => this.handleRequest(req)
  }

  private async handleRequest(req: Request): Promise<Response> {
    // Method check
    if (req.method !== 'POST') {
      return this.errorResponse(405, 'method not allowed')
    }

    // Rate limiting
    if (!this.limiter.tryAcquire()) {
      return this.errorResponse(429, 'rate limit exceeded')
    }

    // Content-Type check
    const ct = (req.headers.get('content-type') ?? '').toLowerCase()
    if (!ct.startsWith('application/json')) {
      return this.errorResponse(400, 'content-type must be application/json')
    }

    // Read body with size limit
    const bodyResult = await this.readLimitedBody(req)
    if (bodyResult instanceof Response) {
      return bodyResult
    }
    const body = bodyResult

    // Authentication
    const authenticator = authenticators[this.auth.method]
    if (authenticator === undefined) {
      return new Response(null, { status: 401 })
    }
    if (!authenticator(req.headers, body, this.auth.secret, this.timestampExpiryMs)) {
      // Empty body on auth failure to prevent information leakage.
      return new Response(null, { status: 401 })
    }

    // Idempotency check
    const idempotencyKey = req.headers.get('x-idempotency-key') ?? ''
    if (this.enableIdempotency && idempotencyKey !== '') {
      const cached = this.idempotency.get(idempotencyKey)
      if (cached !== undefined) {
        return this.jsonResponse(200, cached)
      }
    }

    // Parse payload
    let payload: WebhookPayload
    try {
      payload = JSON.parse(new TextDecoder().decode(body)) as WebhookPayload
    } catch {
      return this.errorResponse(400, 'invalid json')
    }

    // Validate document count
    if (!Array.isArray(payload.documents) || payload.documents.length === 0) {
      return this.errorResponse(400, 'documents array is empty')
    }
    if (payload.documents.length > this.maxDocuments) {
      return this.errorResponse(
        400,
        `exceeds maximum of ${String(this.maxDocuments)} documents`,
      )
    }

    // Process each document
    const resp = await this.processDocuments(payload.documents)

    // Cache response for idempotency
    if (this.enableIdempotency && idempotencyKey !== '') {
      this.idempotency.set(idempotencyKey, resp, IDEMPOTENCY_TTL_MS)
    }

    return this.jsonResponse(200, resp)
  }

  private async readLimitedBody(req: Request): Promise<Uint8Array | Response> {
    const raw = await req.arrayBuffer()
    if (raw.byteLength > this.maxPayloadBytes) {
      return this.errorResponse(413, 'payload too large')
    }
    return new Uint8Array(raw)
  }

  private async processDocuments(
    docs: readonly WebhookDocument[],
  ): Promise<WebhookResponse> {
    const results: WebhookDocumentResult[] = []
    let accepted = 0
    let rejected = 0

    for (const doc of docs) {
      const result = await this.processDocument(doc)
      const statusUpdaters: Record<'accepted' | 'rejected', () => void> = {
        accepted: () => { accepted++ },
        rejected: () => { rejected++ },
      }
      statusUpdaters[result.status]()
      results.push(result)
    }

    return { accepted, rejected, results }
  }

  private async processDocument(
    doc: WebhookDocument,
  ): Promise<WebhookDocumentResult> {
    // Validate required fields.
    if (typeof doc.externalId !== 'string' || doc.externalId === '') {
      return {
        externalId: doc.externalId ?? '',
        status: 'rejected',
        error: 'externalId is required',
      }
    }
    if (typeof doc.content !== 'string' || doc.content === '') {
      return {
        externalId: doc.externalId,
        status: 'rejected',
        error: 'content is required',
      }
    }

    // Decode content.
    let content: Buffer
    const encoding = doc.encoding ?? ''
    const decoder = contentDecoders[encoding]
    if (decoder === undefined) {
      return {
        externalId: doc.externalId,
        status: 'rejected',
        error: `unsupported encoding: ${encoding}`,
      }
    }
    try {
      content = decoder(doc.content)
    } catch (err) {
      return {
        externalId: doc.externalId,
        status: 'rejected',
        error: err instanceof Error ? err.message : 'content decode failed',
      }
    }

    // Check individual document size.
    if (content.length > this.maxDocumentBytes) {
      return {
        externalId: doc.externalId,
        status: 'rejected',
        error: `document exceeds maximum size of ${String(this.maxDocumentBytes)} bytes`,
      }
    }

    // Build connector document.
    const connDoc: ConnectorDocument = {
      externalId: doc.externalId,
      content,
      mime: doc.mime ?? 'text/plain',
      title: doc.title ?? '',
      ...(doc.url !== undefined ? { url: doc.url } : {}),
      metadata: doc.metadata ?? {},
      modifiedAt: new Date(),
    }

    // Dispatch to pipeline.
    try {
      const { documentId } = await this.dispatcher.dispatch(connDoc)
      return {
        externalId: doc.externalId,
        status: 'accepted',
        documentId,
      }
    } catch {
      return {
        externalId: doc.externalId,
        status: 'rejected',
        error: 'dispatch failed',
      }
    }
  }

  private jsonResponse(status: number, body: unknown): Response {
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    })
  }

  private errorResponse(status: number, detail: string): Response {
    return new Response(
      JSON.stringify({ status, title: statusText(status), detail }),
      {
        status,
        headers: { 'content-type': 'application/problem+json' },
      },
    )
  }
}

const STATUS_TEXT: Record<number, string> = {
  400: 'Bad Request',
  401: 'Unauthorized',
  405: 'Method Not Allowed',
  413: 'Payload Too Large',
  429: 'Too Many Requests',
}

const statusText = (code: number): string => STATUS_TEXT[code] ?? 'Error'
