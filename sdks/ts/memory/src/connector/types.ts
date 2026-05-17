// SPDX-License-Identifier: Apache-2.0

/**
 * Core connector framework types. Defines the Connector contract that
 * all external-service connectors implement, along with shared types
 * for documents, configuration, and sync state.
 */

import type { Store } from '../store/index.js'

/** Shared configuration injected into every connector instance. */
export type ConnectorConfig = {
  /** Connector type identifier (e.g. "slack", "gdrive"). */
  readonly name: string
  /** Brain ID scoping all stored state (tokens, cursors). */
  readonly brainId: string
  /** Delay between sync iterations in continuous mode (ms). Defaults to 300000 (5 min). */
  readonly pollInterval?: number
  /** Brain Store for persisting tokens and sync state. */
  readonly store: Store
  /** Optional rate limiter for API calls. */
  readonly rateLimiter?: RateLimiter
}

/** Default poll interval: 5 minutes. */
export const DEFAULT_POLL_INTERVAL = 300_000

/** A document fetched from an external service, ready for ingestion. */
export type ConnectorDocument = {
  /** Unique identifier in the source system. */
  readonly externalId: string
  /** Raw document content. */
  readonly content: Buffer | string
  /** Content type (e.g. "text/markdown"). */
  readonly mime: string
  /** Document title. */
  readonly title: string
  /** Optional link back to source. */
  readonly url?: string
  /** Source-specific metadata. */
  readonly metadata: Readonly<Record<string, unknown>>
  /** Last modification time in source system. */
  readonly modifiedAt: Date
  /** Optional content hash from source for change detection. */
  readonly checksum?: string
}

/**
 * Sync cursor representing a position in an external service's change
 * stream. The value is opaque to the framework.
 */
export type SyncCursor = {
  /** Opaque cursor value. */
  readonly value: string
  /** When the cursor was last persisted. */
  readonly updatedAt: Date
  /** Optional connector-specific context. */
  readonly metadata?: Readonly<Record<string, unknown>>
}

/**
 * The Connector interface that all external-service connectors
 * implement. Handles service-specific API calls, authentication, and
 * data transformation.
 */
export type Connector = {
  /** Connector type identifier. */
  readonly name: string

  /** Validate and store connector-specific configuration. */
  configure(config: Record<string, unknown>): Promise<void>

  /** Full sync: yield all available documents. */
  fetchAll(signal: AbortSignal): AsyncIterable<ConnectorDocument>

  /** Incremental sync: yield documents modified since the cursor. */
  fetchSince(signal: AbortSignal, cursor: SyncCursor): AsyncIterable<ConnectorDocument>

  /** Start a continuous sync loop polling at the configured interval. */
  start(signal: AbortSignal): Promise<void>

  /** Gracefully stop the continuous sync loop. */
  stop(): Promise<void>
}

/** Factory function that creates a new Connector with shared config. */
export type ConnectorFactory = (cfg: ConnectorConfig) => Connector

// ---- Rate Limiter types (re-exported from rate-limiter module) ----

/** Rate limiter configuration for token-bucket limiting. */
export type RateLimiterConfig = {
  /** Bucket capacity. */
  readonly maxTokens: number
  /** Tokens added per second. */
  readonly refillRate: number
  /** Refill tick interval (ms). Defaults to 1000. */
  readonly refillInterval?: number
  /** Maximum retry attempts. Defaults to 5. */
  readonly maxRetries?: number
  /** Base backoff delay (ms). Defaults to 1000. */
  readonly baseBackoff?: number
  /** Maximum backoff delay (ms). Defaults to 60000. */
  readonly maxBackoff?: number
}

/** Rate limit headers for adaptive adjustment. */
export type RateLimitHeaders = {
  readonly remaining?: string
  readonly limit?: string
  readonly retryAfter?: string
}

/** Rate limiter interface for token-bucket with exponential backoff. */
export type RateLimiter = {
  /** Block until count tokens are available, then consume them. */
  acquire(count?: number): Promise<void>
  /** Non-blocking: return true if tokens consumed, false otherwise. */
  tryAcquire(count?: number): boolean
  /** Refill bucket to maximum capacity. */
  reset(): void
  /** Adjust bucket from rate limit response headers. */
  adjustFromHeaders(headers: RateLimitHeaders): void
  /** Sleep with exponential backoff. */
  backoff(attempt: number): Promise<void>
  /** Return current available tokens. */
  tokens(): number
  /** Release resources. */
  close(): void
}

/** Paginator page result. */
export type PageResult<T> = {
  readonly items: readonly T[]
  readonly nextCursor?: string
  readonly total?: number
}

/** Function that fetches a single page. Empty cursor = first page. */
export type FetchPageFn<T> = (cursor?: string) => Promise<PageResult<T>>

/** Paginator configuration. */
export type PaginatorConfig<T> = {
  /** Function to fetch a single page. */
  readonly fetchPage: FetchPageFn<T>
  /** Page size hint. Defaults to 100. */
  readonly pageSize?: number
  /** Maximum pages to fetch. Defaults to 10000. */
  readonly maxPages?: number
  /** Optional rate limiter. */
  readonly rateLimiter?: RateLimiter
}
