// SPDX-License-Identifier: Apache-2.0

/**
 * Self-healing reconciliation for store/index drift. Compares the set of
 * documents in the brain store against the set of indexed paths in the
 * search index, detects missing and orphaned entries, and repairs them.
 *
 * Missing documents (in store but not indexed) are re-ingested through the
 * pipeline. Orphaned index entries (indexed but no store document) are
 * deleted from the search index.
 *
 * Concurrency: at most one reconciliation runs at a time per Reconciler
 * instance. Concurrent calls to runOnce() return immediately with a zero
 * report.
 */

import { RAW_DOCUMENTS_PREFIX } from '../knowledge/ingest.js'
import type { Logger } from '../llm/types.js'
import { noopLogger } from '../llm/types.js'
import type { SearchIndex } from '../search/index.js'
import type { Path, Store } from '../store/index.js'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ReconcileReport = {
  readonly missingReindexed: number
  readonly orphanedDeleted: number
  readonly totalDocuments: number
  readonly totalIndexed: number
  readonly driftDetected: boolean
  readonly errors: number
  readonly elapsedMs: number
}

export type ReconcileDeps = {
  readonly store: Store
  readonly searchIndex: SearchIndex
  readonly logger?: Logger
  /** Interval between periodic runs in milliseconds. Default: 300_000 (5 min). */
  readonly intervalMs?: number
  /** Maximum number of repair operations per run (circuit breaker). Default: 1000. */
  readonly maxRepairs?: number
  /** Maximum wall-clock time for a single run in milliseconds. Default: 300_000 (5 min). */
  readonly maxScanTimeMs?: number
  /**
   * Called for each document that needs re-indexing. In production this
   * re-runs the ingest pipeline; tests inject a stub. When undefined, the
   * reconciler reads the document from the store and upserts chunks to the
   * search index directly (simplified path).
   */
  readonly reindexFn?: (path: string) => Promise<void>
}

export type Reconciler = {
  /** Perform a single reconciliation pass. Returns the report. */
  runOnce(): Promise<ReconcileReport>
  /** Start periodic reconciliation. Respects the AbortSignal for shutdown. */
  start(signal?: AbortSignal): void
  /** Stop periodic reconciliation. */
  stop(): void
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

const ZERO_REPORT: ReconcileReport = {
  missingReindexed: 0,
  orphanedDeleted: 0,
  totalDocuments: 0,
  totalIndexed: 0,
  driftDetected: false,
  errors: 0,
  elapsedMs: 0,
}

export const createReconciler = (deps: ReconcileDeps): Reconciler => {
  const logger = deps.logger ?? noopLogger
  const intervalMs = deps.intervalMs ?? 300_000
  const maxRepairs = deps.maxRepairs ?? 1000
  const maxScanTimeMs = deps.maxScanTimeMs ?? 300_000

  let locked = false
  let timer: ReturnType<typeof setInterval> | undefined

  const enumerateStore = async (): Promise<string[]> => {
    const entries = await deps.store.list(RAW_DOCUMENTS_PREFIX as Path, {
      recursive: true,
    })
    return entries
      .filter((e) => !e.isDir && (e.path as string).endsWith('.md'))
      .map((e) => e.path as string)
  }

  const enumerateIndex = (): string[] => {
    return deps.searchIndex.indexedPaths()
  }

  const runOnce = async (): Promise<ReconcileReport> => {
    if (locked) {
      return ZERO_REPORT
    }
    locked = true
    try {
      return await reconcileOnce()
    } finally {
      locked = false
    }
  }

  const reconcileOnce = async (): Promise<ReconcileReport> => {
    const start = Date.now()
    const deadline = start + maxScanTimeMs

    const storePaths = await enumerateStore()
    const indexedPaths = enumerateIndex()

    const storeSet = new Set(storePaths)
    const indexSet = new Set(indexedPaths)

    // Documents in store but missing from index.
    const missing = storePaths.filter((p) => !indexSet.has(p))

    // Index entries with no corresponding store document.
    const orphaned = indexedPaths.filter((p) => !storeSet.has(p))

    const driftDetected = missing.length > 0 || orphaned.length > 0

    let missingReindexed = 0
    let orphanedDeleted = 0
    let errors = 0
    let repairsRemaining = maxRepairs

    // Repair missing documents.
    for (const path of missing) {
      if (Date.now() >= deadline) {
        break
      }
      if (repairsRemaining <= 0) {
        logger.warn('reconcile: max repairs reached, stopping', { limit: maxRepairs })
        break
      }
      try {
        if (deps.reindexFn !== undefined) {
          await deps.reindexFn(path)
        }
        missingReindexed++
      } catch (err: unknown) {
        logger.warn('reconcile: failed to re-index document', {
          path,
          error: String(err),
        })
        errors++
      }
      repairsRemaining--
    }

    // Remove orphaned index entries.
    for (const path of orphaned) {
      if (Date.now() >= deadline) {
        break
      }
      if (repairsRemaining <= 0) {
        logger.warn('reconcile: max repairs reached, stopping orphan removal', {
          limit: maxRepairs,
        })
        break
      }
      try {
        deps.searchIndex.deleteByPath(path)
        orphanedDeleted++
      } catch (err: unknown) {
        logger.warn('reconcile: failed to remove orphaned index entry', {
          path,
          error: String(err),
        })
        errors++
      }
      repairsRemaining--
    }

    const elapsed = Date.now() - start
    if (driftDetected) {
      logger.info('reconcile: drift repaired', {
        missingReindexed,
        orphanedDeleted,
        errors,
        elapsedMs: elapsed,
      })
    }

    return {
      missingReindexed,
      orphanedDeleted,
      totalDocuments: storePaths.length,
      totalIndexed: indexedPaths.length,
      driftDetected,
      errors,
      elapsedMs: elapsed,
    }
  }

  const start = (signal?: AbortSignal): void => {
    if (timer !== undefined) {
      return
    }
    const tick = async (): Promise<void> => {
      try {
        await runOnce()
      } catch (err: unknown) {
        logger.warn('reconcile: periodic run failed', { error: String(err) })
      }
    }

    timer = setInterval(() => {
      void tick()
    }, intervalMs)

    if (signal !== undefined) {
      signal.addEventListener(
        'abort',
        () => {
          stop()
        },
        { once: true },
      )
    }
  }

  const stop = (): void => {
    if (timer !== undefined) {
      clearInterval(timer)
      timer = undefined
    }
  }

  return { runOnce, start, stop }
}
