// SPDX-License-Identifier: Apache-2.0

/**
 * Store mutation hook types. The mutation hook subscribes to store
 * ChangeEvents and dispatches ingestion requests for matching paths,
 * with debouncing and opt-out support.
 */

import type { Logger } from '../../llm/types.js'
import type { ChangeEvent, Path } from '../../store/index.js'

/**
 * Callback invoked when a matching change event should trigger ingestion.
 *
 * The function is called after the debounce timer fires. Implementations
 * that perform network I/O or long-running work should enforce their own
 * timeout (e.g. AbortSignal) to avoid hanging the event loop. Thrown
 * errors and rejected promises are logged but do not propagate.
 */
export type DispatchFn = (brainId: string, path: string) => Promise<void> | void

/** Determines whether a store path should trigger ingestion. */
export type PathMatcher = (path: string) => boolean

export type MutationHookOptions = {
  /** Brain that owns the store being watched. */
  readonly brainId: string
  /** Called when a debounced matching event fires. */
  readonly dispatch: DispatchFn
  /** Path matchers. Defaults to raw/documents/** when empty. */
  readonly pathMatchers?: readonly PathMatcher[]
  /** Batch reasons that suppress ingestion. Events with a matching reason are ignored. */
  readonly optOutReasons?: ReadonlySet<string>
  /** Minimum delay between dispatches for the same path. Defaults to 1000ms. */
  readonly debounceIntervalMs?: number
  readonly logger?: Logger
}

export type MutationHook = {
  /** EventSink to pass to store.subscribe(). */
  readonly sink: (event: ChangeEvent) => void
  /** Stop all pending debounce timers. */
  close(): void
}
