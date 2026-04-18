// SPDX-License-Identifier: Apache-2.0

/**
 * Plugin invocation helpers. Plugins fire in registration order for Start
 * events and in reverse order for End events so "last registered = first
 * torn down" mirrors the Go implementation. Exceptions from a plugin are
 * logged and swallowed.
 */

import type { Logger } from '../llm/index.js'
import type {
  ConsolidationPayload,
  ExtractionPayload,
  Plugin,
  ReflectionPayload,
} from './types.js'

type HookPayload = ExtractionPayload | ReflectionPayload | ConsolidationPayload

type HookFn<T extends HookPayload> = ((ctx: T) => Promise<void> | void) | undefined

const runHook = async <T extends HookPayload>(
  plugins: readonly Plugin[],
  select: (p: Plugin) => HookFn<T>,
  payload: T,
  logger: Logger,
  reverse: boolean,
): Promise<void> => {
  const ordered = reverse ? [...plugins].reverse() : plugins
  for (const plugin of ordered) {
    const fn = select(plugin)
    if (!fn) continue
    try {
      await fn(payload)
    } catch (err) {
      logger.warn('memory: plugin hook threw', {
        plugin: plugin.name,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
}

export const fireExtractionStart = (
  plugins: readonly Plugin[],
  payload: ExtractionPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (p) => p.onExtractionStart, payload, logger, false)

export const fireExtractionEnd = (
  plugins: readonly Plugin[],
  payload: ExtractionPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (p) => p.onExtractionEnd, payload, logger, true)

export const fireReflectionStart = (
  plugins: readonly Plugin[],
  payload: ReflectionPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (p) => p.onReflectionStart, payload, logger, false)

export const fireReflectionEnd = (
  plugins: readonly Plugin[],
  payload: ReflectionPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (p) => p.onReflectionEnd, payload, logger, true)

export const fireConsolidationStart = (
  plugins: readonly Plugin[],
  payload: ConsolidationPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (p) => p.onConsolidationStart, payload, logger, false)

export const fireConsolidationEnd = (
  plugins: readonly Plugin[],
  payload: ConsolidationPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (p) => p.onConsolidationEnd, payload, logger, true)
