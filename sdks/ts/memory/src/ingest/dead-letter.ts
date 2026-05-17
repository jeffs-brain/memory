// SPDX-License-Identifier: Apache-2.0

/**
 * Dead letter queue for permanent failure isolation. Jobs that exhaust
 * their retry budget are moved here with full error context so that
 * operators can inspect, retry, or purge them.
 *
 * Provides an in-memory adapter for testing/local use and a type-safe
 * interface that PostgreSQL and SQLite adapters implement.
 */

import { randomUUID } from 'node:crypto'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Serialised job data preserved when a job is moved to the DLQ. */
export type JobPayload = {
  readonly documentHash: string
  readonly brainId: string
  readonly source?: string
  readonly contentType?: string
}

/** A job that has been moved to the dead letter queue. */
export type DeadLetterEntry = {
  readonly id: string
  readonly originalJobId: string
  readonly brainId: string
  readonly payload: JobPayload
  readonly failureReason: string
  readonly lastError: string
  readonly errorHistory?: readonly string[]
  readonly retryCount: number
  readonly metadata?: Readonly<Record<string, string>>
  readonly groupId?: string
  readonly movedAt: Date
  readonly resolvedAt?: Date
  readonly resolvedBy?: string
}

/** Options for listing dead letter entries. */
export type DeadLetterListOptions = {
  readonly brainId?: string
  readonly limit?: number
  readonly offset?: number
  readonly includeResolved?: boolean
}

/** Paginated result from a list operation. */
export type DeadLetterListResult = {
  readonly entries: readonly DeadLetterEntry[]
  readonly total: number
}

/** Discriminated union for purge operations. */
export type PurgeOptions =
  | { readonly kind: 'by-id'; readonly id: string }
  | { readonly kind: 'by-brain'; readonly brainId: string }
  | { readonly kind: 'older-than'; readonly days: number }
  | { readonly kind: 'all-resolved' }

/**
 * Callback that creates a new queue job in the main queue from a dead
 * letter entry. The implementation is responsible for resetting the
 * retry count and assigning a new job ID. Returning a rejected promise
 * prevents the DLQ entry from being marked as resolved.
 */
export type ReEnqueueFn = (entry: DeadLetterEntry) => Promise<void>

/** Interface for dead letter queue operations. */
export type DeadLetterAdapter = {
  /** Move a failed job to the dead letter queue. */
  readonly move: (entry: DeadLetterEntry) => Promise<DeadLetterEntry>

  /** List dead letter entries with pagination and filtering. */
  readonly list: (opts?: DeadLetterListOptions) => Promise<DeadLetterListResult>

  /** Get a single entry by ID. Returns undefined when not found. */
  readonly get: (id: string) => Promise<DeadLetterEntry | undefined>

  /**
   * Mark an entry as resolved for retry. If reEnqueue is provided, it
   * is called with the entry to create a new queue job before the
   * entry is committed as resolved. If reEnqueue throws, the entry
   * remains unresolved.
   */
  readonly retry: (id: string, resolvedBy?: string, reEnqueue?: ReEnqueueFn) => Promise<DeadLetterEntry>

  /** Purge entries matching the given options. Returns count removed. */
  readonly purge: (opts: PurgeOptions) => Promise<number>

  /** Count unresolved entries, optionally filtered by brain ID. */
  readonly count: (brainId?: string) => Promise<number>
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** Dead letter entry not found. */
export class DeadLetterNotFoundError extends Error {
  constructor(id: string) {
    super(`Dead letter entry not found: ${id}`)
    this.name = 'DeadLetterNotFoundError'
  }
}

/** Dead letter entry already resolved (double retry). */
export class DeadLetterAlreadyResolvedError extends Error {
  constructor(id: string) {
    super(`Dead letter entry already resolved: ${id}`)
    this.name = 'DeadLetterAlreadyResolvedError'
  }
}

// ---------------------------------------------------------------------------
// Default configuration
// ---------------------------------------------------------------------------

/** Default maximum failed attempts before a job is moved to the DLQ. */
export const DEFAULT_MAX_ATTEMPTS = 3

/** Default retention period in days for dead letter entries. */
export const DEFAULT_RETENTION_DAYS = 30

/** Default page size for list operations. */
const DEFAULT_LIST_LIMIT = 50

// ---------------------------------------------------------------------------
// In-memory adapter (testing / local use)
// ---------------------------------------------------------------------------

/**
 * Creates an in-memory dead letter adapter. Useful for testing and
 * local/embedded deployments that do not need persistent storage.
 */
export const createInMemoryDeadLetterAdapter = (): DeadLetterAdapter => {
  const store = new Map<string, DeadLetterEntry>()

  const move = async (entry: DeadLetterEntry): Promise<DeadLetterEntry> => {
    const resolved: DeadLetterEntry = {
      ...entry,
      id: entry.id || randomUUID(),
      movedAt: entry.movedAt ?? new Date(),
    }
    store.set(resolved.id, resolved)
    return resolved
  }

  const list = async (opts?: DeadLetterListOptions): Promise<DeadLetterListResult> => {
    const limit = opts?.limit ?? DEFAULT_LIST_LIMIT
    const offset = opts?.offset ?? 0
    const includeResolved = opts?.includeResolved ?? false
    const brainId = opts?.brainId

    const filtered = Array.from(store.values()).filter((entry) => {
      if (!includeResolved && entry.resolvedAt !== undefined) return false
      if (brainId !== undefined && entry.brainId !== brainId) return false
      return true
    })

    // Sort by movedAt descending (most recent first).
    filtered.sort((a, b) => b.movedAt.getTime() - a.movedAt.getTime())

    const total = filtered.length
    const entries = filtered.slice(offset, offset + limit)
    return { entries, total }
  }

  const get = async (id: string): Promise<DeadLetterEntry | undefined> => {
    return store.get(id)
  }

  const retry = async (id: string, resolvedBy?: string, reEnqueue?: ReEnqueueFn): Promise<DeadLetterEntry> => {
    const entry = store.get(id)
    if (entry === undefined) {
      throw new DeadLetterNotFoundError(id)
    }
    if (entry.resolvedAt !== undefined) {
      throw new DeadLetterAlreadyResolvedError(id)
    }

    const resolved: DeadLetterEntry = {
      ...entry,
      resolvedAt: new Date(),
      resolvedBy: resolvedBy ?? 'system',
    }

    if (reEnqueue !== undefined) {
      await reEnqueue(resolved)
    }

    store.set(id, resolved)
    return resolved
  }

  const purge = async (opts: PurgeOptions): Promise<number> => {
    const purgeStrategies: Record<PurgeOptions['kind'], () => number> = {
      'by-id': () => {
        const optsById = opts as { readonly kind: 'by-id'; readonly id: string }
        return store.delete(optsById.id) ? 1 : 0
      },
      'by-brain': () => {
        const optsByBrain = opts as { readonly kind: 'by-brain'; readonly brainId: string }
        let removed = 0
        for (const [id, entry] of store) {
          if (entry.brainId === optsByBrain.brainId) {
            store.delete(id)
            removed++
          }
        }
        return removed
      },
      'older-than': () => {
        const optsOlder = opts as { readonly kind: 'older-than'; readonly days: number }
        const cutoff = new Date()
        cutoff.setDate(cutoff.getDate() - optsOlder.days)
        let removed = 0
        for (const [id, entry] of store) {
          if (entry.movedAt < cutoff) {
            store.delete(id)
            removed++
          }
        }
        return removed
      },
      'all-resolved': () => {
        let removed = 0
        for (const [id, entry] of store) {
          if (entry.resolvedAt !== undefined) {
            store.delete(id)
            removed++
          }
        }
        return removed
      },
    }

    return purgeStrategies[opts.kind]()
  }

  const count = async (brainId?: string): Promise<number> => {
    let total = 0
    for (const entry of store.values()) {
      if (entry.resolvedAt !== undefined) continue
      if (brainId !== undefined && entry.brainId !== brainId) continue
      total++
    }
    return total
  }

  return { move, list, get, retry, purge, count } as const
}
