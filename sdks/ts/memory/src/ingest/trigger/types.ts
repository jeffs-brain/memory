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

export type TriggerBus = {
  /** Publish an event to all subscribers. Throws for malformed events. */
  publish(event: IngestTriggerEvent): void
  /** Register a handler. Returns a function to unsubscribe. */
  subscribe(handler: TriggerHandler): Unsubscribe
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
