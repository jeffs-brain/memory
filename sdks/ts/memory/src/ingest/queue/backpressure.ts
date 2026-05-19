// SPDX-License-Identifier: Apache-2.0

/**
 * Backpressure detection for the ingestion queue. Monitors pending job
 * count via countByStatus and signals when producers should stop
 * enqueuing new jobs. Follows the RabbitMQ/SQS pattern of per-tenant
 * depth thresholds.
 */

import type { QueueAdapter } from './adapter.js'

/**
 * Default backpressure threshold per tenant. 1000 pending items
 * signals the system should pause accepting new work.
 */
const DEFAULT_MAX_QUEUE_DEPTH = 1000

/**
 * Evaluates whether the queue has exceeded its capacity threshold.
 * Producers should call isBackpressured before enqueuing and skip
 * the enqueue when it returns true.
 *
 * Time: O(1) for cached reads, O(1) for check (single adapter call).
 * Space: O(1).
 */
export type BackpressureChecker = {
  /**
   * Queries the adapter for current pending job count and updates the
   * cached state. The brainId scopes the depth check; pass an empty
   * string for global depth.
   */
  check(brainId: string): Promise<boolean>

  /**
   * Returns the last known backpressure state without querying the
   * adapter. Safe to call from hot paths where a stale value is
   * acceptable.
   */
  isBackpressured(): boolean

  /**
   * Returns the configured threshold.
   */
  maxDepth(): number

  /**
   * Returns the last observed queue depth from the most recent check
   * call. Returns 0 before the first check.
   */
  lastDepth(): number
}

/**
 * Creates a backpressure checker bound to the given adapter.
 * A maxQueueDepth of zero or negative falls back to
 * DEFAULT_MAX_QUEUE_DEPTH (1000).
 */
export const createBackpressureChecker = (
  adapter: QueueAdapter,
  maxQueueDepth: number,
): BackpressureChecker => {
  const safeMax = maxQueueDepth > 0 ? maxQueueDepth : DEFAULT_MAX_QUEUE_DEPTH
  let pressured = false
  let observedDepth = 0

  return {
    async check(brainId: string): Promise<boolean> {
      const counts = await adapter.countByStatus(brainId === '' ? undefined : brainId)
      const depth = counts.pending ?? 0
      observedDepth = depth
      pressured = depth >= safeMax
      return pressured
    },

    isBackpressured(): boolean {
      return pressured
    },

    maxDepth(): number {
      return safeMax
    },

    lastDepth(): number {
      return observedDepth
    },
  }
}
