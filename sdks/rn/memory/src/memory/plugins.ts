import type { Logger } from '../llm/types.js'
import type { ConsolidationPayload, ExtractionPayload, Plugin, ReflectionPayload } from './types.js'

type HookPayload = ExtractionPayload | ReflectionPayload | ConsolidationPayload

type HookFn<T extends HookPayload> = ((ctx: T) => Promise<void> | void) | undefined

const runHook = async <T extends HookPayload>(
  plugins: readonly Plugin[],
  select: (plugin: Plugin) => HookFn<T>,
  payload: T,
  logger: Logger,
  reverse: boolean,
): Promise<void> => {
  const ordered = reverse ? [...plugins].reverse() : plugins
  for (const plugin of ordered) {
    const fn = select(plugin)
    if (fn === undefined) continue
    try {
      await fn(payload)
    } catch (error) {
      logger.warn('memory client: plugin hook threw', {
        plugin: plugin.name,
        error: error instanceof Error ? error.message : String(error),
      })
    }
  }
}

export const fireExtractionStart = (
  plugins: readonly Plugin[],
  payload: ExtractionPayload,
  logger: Logger,
): Promise<void> => runHook(plugins, (plugin) => plugin.onExtractionStart, payload, logger, false)

export const fireExtractionEnd = (
  plugins: readonly Plugin[],
  payload: ExtractionPayload,
  logger: Logger,
): Promise<void> => runHook(plugins, (plugin) => plugin.onExtractionEnd, payload, logger, true)

export const fireReflectionStart = (
  plugins: readonly Plugin[],
  payload: ReflectionPayload,
  logger: Logger,
): Promise<void> => runHook(plugins, (plugin) => plugin.onReflectionStart, payload, logger, false)

export const fireReflectionEnd = (
  plugins: readonly Plugin[],
  payload: ReflectionPayload,
  logger: Logger,
): Promise<void> => runHook(plugins, (plugin) => plugin.onReflectionEnd, payload, logger, true)

export const fireConsolidationStart = (
  plugins: readonly Plugin[],
  payload: ConsolidationPayload,
  logger: Logger,
): Promise<void> =>
  runHook(plugins, (plugin) => plugin.onConsolidationStart, payload, logger, false)

export const fireConsolidationEnd = (
  plugins: readonly Plugin[],
  payload: ConsolidationPayload,
  logger: Logger,
): Promise<void> => runHook(plugins, (plugin) => plugin.onConsolidationEnd, payload, logger, true)
