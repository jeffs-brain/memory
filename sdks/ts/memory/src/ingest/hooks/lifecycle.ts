// SPDX-License-Identifier: Apache-2.0

/**
 * Ingestion lifecycle hook helpers. These follow the same error-swallowing
 * pattern as the existing plugin hooks in `memory/plugins.ts`. Start hooks
 * fire in registration order; End hooks fire in reverse.
 */

import type { Logger } from '../../llm/types.js'
import type {
  DocumentDetectedEvent,
  IngestHookEvent,
  Plugin,
} from '../../memory/types.js'

/**
 * Fire onDocumentDetected on all plugins. If any plugin returns `false`,
 * the result `cancelled` is `true`. Errors are logged and treated as
 * non-cancelling (fail open).
 */
export const fireDocumentDetected = async (
  plugins: readonly Plugin[],
  event: DocumentDetectedEvent,
  logger: Logger,
): Promise<{ readonly cancelled: boolean }> => {
  for (const plugin of plugins) {
    const fn = plugin.onDocumentDetected
    if (!fn) continue
    try {
      const result = await fn(event)
      if (result === false) {
        logger.info('ingest: plugin cancelled document detection', {
          plugin: plugin.name,
        })
        return { cancelled: true }
      }
    } catch (err) {
      logger.warn('ingest: onDocumentDetected hook threw', {
        plugin: plugin.name,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
  return { cancelled: false }
}

/**
 * Fire onIngestStart on all plugins in registration order. If any plugin
 * returns `false`, ingestion is cancelled. Errors are logged and swallowed.
 */
export const fireIngestStart = async (
  plugins: readonly Plugin[],
  event: IngestHookEvent,
  logger: Logger,
): Promise<{ readonly cancelled: boolean }> => {
  for (const plugin of plugins) {
    const fn = plugin.onIngestStart
    if (!fn) continue
    try {
      const result = await fn(event)
      if (result === false) {
        logger.info('ingest: plugin cancelled ingest start', {
          plugin: plugin.name,
        })
        return { cancelled: true }
      }
    } catch (err) {
      logger.warn('ingest: onIngestStart hook threw', {
        plugin: plugin.name,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
  return { cancelled: false }
}

/**
 * Fire onIngestEnd on all plugins in reverse registration order.
 * Errors are logged and swallowed.
 */
export const fireIngestEnd = async (
  plugins: readonly Plugin[],
  event: IngestHookEvent,
  logger: Logger,
): Promise<void> => {
  const reversed = [...plugins].reverse()
  for (const plugin of reversed) {
    const fn = plugin.onIngestEnd
    if (!fn) continue
    try {
      await fn(event)
    } catch (err) {
      logger.warn('ingest: onIngestEnd hook threw', {
        plugin: plugin.name,
        err: err instanceof Error ? err.message : String(err),
      })
    }
  }
}
