// SPDX-License-Identifier: Apache-2.0

/**
 * Store mutation hook implementation. Subscribes to ChangeEvent from a
 * Store and dispatches ingestion requests for file writes matching
 * configurable path patterns. Rapid writes to the same path are
 * debounced. Batch writes with opt-out reasons are silently skipped.
 */

import type { ChangeEvent } from '../../store/index.js'
import type { DispatchFn, MutationHook, MutationHookOptions, PathMatcher } from './types.js'

const DEFAULT_DEBOUNCE_MS = 1000
const DEFAULT_PATH_PREFIX = 'raw/documents/'

/** Default path matcher: files under raw/documents/. */
export const defaultPathMatcher: PathMatcher = (path: string): boolean =>
  path.startsWith(DEFAULT_PATH_PREFIX) && path.length > DEFAULT_PATH_PREFIX.length

/** Prefix-based path matcher. */
export const prefixPathMatcher = (prefix: string): PathMatcher =>
  (path: string): boolean => path.startsWith(prefix)

/** Glob-based path matcher supporting * and ** patterns. */
export const globPathMatcher = (pattern: string): PathMatcher =>
  (path: string): boolean => globMatch(pattern, path)

export const createMutationHook = (opts: MutationHookOptions): MutationHook => {
  const debounceMs = opts.debounceIntervalMs ?? DEFAULT_DEBOUNCE_MS
  const matchers: readonly PathMatcher[] =
    opts.pathMatchers && opts.pathMatchers.length > 0
      ? opts.pathMatchers
      : [defaultPathMatcher]
  const optOutReasons = opts.optOutReasons ?? new Set<string>()
  const logger = opts.logger

  const timers = new Map<string, ReturnType<typeof setTimeout>>()
  let closed = false

  const matchesAny = (path: string): boolean => {
    for (const matcher of matchers) {
      if (matcher(path)) return true
    }
    return false
  }

  const debounceDispatch = (path: string): void => {
    const existing = timers.get(path)
    if (existing !== undefined) {
      clearTimeout(existing)
    }

    timers.set(
      path,
      setTimeout(() => {
        timers.delete(path)
        try {
          const result = opts.dispatch(opts.brainId, path)
          if (result && typeof (result as Promise<void>).catch === 'function') {
            ;(result as Promise<void>).catch((err) => {
              logger?.error('hooks: dispatch failed', { path, error: String(err) })
            })
          }
        } catch (err) {
          logger?.error('hooks: dispatch failed', { path, error: String(err) })
        }
      }, debounceMs),
    )
  }

  const sink = (event: ChangeEvent): void => {
    if (closed) return

    // Only react to created and updated events.
    if (event.kind !== 'created' && event.kind !== 'updated') return

    const path = event.path

    // Opt-out by batch reason.
    if (event.reason && optOutReasons.has(event.reason)) {
      logger?.debug('hooks: skipping event due to opt-out reason', {
        path,
        reason: event.reason,
      })
      return
    }

    if (!matchesAny(path)) return

    debounceDispatch(path)
  }

  const close = (): void => {
    if (closed) return
    closed = true
    for (const [, timer] of timers) {
      clearTimeout(timer)
    }
    timers.clear()
  }

  return { sink, close }
}

// Glob matching implementation.
const globMatch = (pattern: string, name: string): boolean => matchAt(pattern, 0, name, 0)

const matchAt = (pattern: string, pi: number, name: string, ni: number): boolean => {
  while (pi < pattern.length && ni < name.length) {
    if (pi + 1 < pattern.length && pattern[pi] === '*' && pattern[pi + 1] === '*') {
      let nextPi = pi + 2
      if (nextPi < pattern.length && pattern[nextPi] === '/') nextPi++
      for (let k = ni; k <= name.length; k++) {
        if (matchAt(pattern, nextPi, name, k)) return true
      }
      return false
    }
    if (pattern[pi] === '*') {
      const nextPi = pi + 1
      for (let k = ni; k <= name.length; k++) {
        if (k > ni && name[k - 1] === '/') break
        if (matchAt(pattern, nextPi, name, k)) return true
      }
      return false
    }
    if (pattern[pi] === '?') {
      if (name[ni] === '/') return false
      pi++
      ni++
      continue
    }
    if (pattern[pi] !== name[ni]) return false
    pi++
    ni++
  }
  while (pi < pattern.length && pattern[pi] === '*') pi++
  return pi === pattern.length && ni === name.length
}
