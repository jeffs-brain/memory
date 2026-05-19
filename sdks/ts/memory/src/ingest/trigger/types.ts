// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for the ingestion trigger subsystem. The event bus, Redis
 * bridge, and PostgreSQL bridge all operate on these types.
 */

import type { Logger } from '../../llm/types.js'

export type IngestTriggerSource = 'event-bus' | 'redis' | 'postgres' | 'hook' | 'schedule'

export type IngestTriggerPayload =
  | { readonly kind: 'file'; readonly path: string; readonly mime?: string }
  | { readonly kind: 'url'; readonly url: string }
  | { readonly kind: 'raw'; readonly content: string; readonly title?: string }

export type IngestTriggerEvent = {
  readonly id: string
  readonly brainId: string
  readonly source: IngestTriggerSource
  readonly payload: IngestTriggerPayload
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly timestamp: Date
}

export type TriggerHandler = (event: IngestTriggerEvent) => Promise<void> | void

export type Unsubscribe = () => void

/** Options for subscribe. When filter is provided, only events for which
 *  the filter returns true are delivered to the handler. */
export type SubscribeOptions = {
  readonly filter?: (event: IngestTriggerEvent) => boolean
}

export type TriggerBus = {
  /** Publish an event to all subscribers. Throws for malformed events. */
  publish(event: IngestTriggerEvent): void
  /** Register a handler. When opts.filter is provided, only matching events are delivered. */
  subscribe(handler: TriggerHandler, opts?: SubscribeOptions): Unsubscribe
  /** Drain pending events and release resources. */
  close(): Promise<void>
}

export type TriggerBusOptions = {
  /** Backpressure threshold. Oldest events are dropped when exceeded. Defaults to 1000. */
  readonly maxQueueDepth?: number
  readonly logger?: Logger
}

/**
 * Extension point for custom event delivery mechanisms. Bridges (Redis,
 * Postgres) implement this interface to forward events between processes.
 */
export type EventTransport = {
  start(bus: TriggerBus): Promise<void>
  close(): Promise<void>
}

/** Validate an IngestTriggerEvent, throwing a descriptive error for malformed inputs. */
export const validateTriggerEvent = (event: IngestTriggerEvent): void => {
  if (!event.id) throw new Error('trigger: event id is required')
  if (!event.brainId) throw new Error('trigger: event brainId is required')
  if (!event.timestamp) throw new Error('trigger: event timestamp is required')

  const { payload } = event
  switch (payload.kind) {
    case 'file':
      if (!payload.path) throw new Error('trigger: file payload requires a path')
      break
    case 'url':
      if (!payload.url) throw new Error('trigger: url payload requires a URL')
      break
    case 'raw':
      if (!payload.content) throw new Error('trigger: raw payload requires content')
      break
    default:
      throw new Error(`trigger: invalid payload kind: ${(payload as { kind: string }).kind}`)
  }
}

/** Result of parsing a raw JSON string into an IngestTriggerEvent. */
export type ParseResult =
  | { readonly ok: true; readonly event: IngestTriggerEvent }
  | { readonly ok: false; readonly error: string }

const VALID_PAYLOAD_KINDS: ReadonlySet<string> = new Set(['file', 'url', 'raw'])

/**
 * Parse a raw JSON string into an IngestTriggerEvent with validation.
 * Returns a discriminated union instead of throwing so callers can handle
 * failures without try/catch.
 *
 * @param raw    JSON string from the transport layer
 * @param source Default source when the payload omits it
 */
/**
 * Shape expected from JSON.parse of a trigger event wire format.
 * Uses optional unknown fields for safe narrowing without
 * Record<string, unknown>.
 */
type RawTriggerEvent = {
  id?: unknown
  brainId?: unknown
  source?: unknown
  payload?: unknown
  timestamp?: unknown
  metadata?: unknown
}

type RawPayload = {
  kind?: unknown
  path?: unknown
  url?: unknown
  content?: unknown
  title?: unknown
  mime?: unknown
}

export const parseIngestTriggerEvent = (
  raw: string,
  source: IngestTriggerSource,
): ParseResult => {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    return { ok: false, error: `invalid JSON: ${String(err)}` }
  }

  if (typeof parsed !== 'object' || parsed === null) {
    return { ok: false, error: 'parsed value is not an object' }
  }

  const obj = parsed as RawTriggerEvent

  if (typeof obj.id !== 'string' || obj.id === '') {
    return { ok: false, error: 'missing or empty id' }
  }

  if (typeof obj.brainId !== 'string' || obj.brainId === '') {
    return { ok: false, error: 'missing or empty brainId' }
  }

  if (typeof obj.payload !== 'object' || obj.payload === null) {
    return { ok: false, error: 'missing or invalid payload' }
  }
  const payloadObj = obj.payload as RawPayload
  if (typeof payloadObj.kind !== 'string' || !VALID_PAYLOAD_KINDS.has(payloadObj.kind)) {
    return { ok: false, error: `invalid payload kind: ${String(payloadObj.kind)}` }
  }

  const payload = obj.payload as IngestTriggerPayload

  const rawTimestamp = obj.timestamp
  const timestamp = rawTimestamp instanceof Date
    ? rawTimestamp
    : typeof rawTimestamp === 'string'
      ? new Date(rawTimestamp)
      : new Date()

  const eventSource = (typeof obj.source === 'string' && obj.source !== '')
    ? obj.source as IngestTriggerSource
    : source

  // metadata is intentionally Readonly<Record<string, unknown>> -- user-provided
  // key-value pairs with truly unknown keys at compile time.
  const metadata = (typeof obj.metadata === 'object' && obj.metadata !== null)
    ? obj.metadata as Readonly<Record<string, unknown>>
    : undefined

  const event: IngestTriggerEvent = {
    id: obj.id,
    brainId: obj.brainId,
    source: eventSource,
    payload,
    timestamp,
    ...(metadata !== undefined ? { metadata } : {}),
  }

  return { ok: true, event }
}
