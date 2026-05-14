// SPDX-License-Identifier: Apache-2.0

/**
 * Public entrypoint for `@jeffs-brain/memory-pi`. Exposes a single
 * factory, `createMemoryExtension`, that wires the eleven memory tools
 * and four lifecycle hooks into the pi runtime, plus a default export
 * that conforms to the pi extension contract (default function that
 * accepts an `ExtensionAPI`).
 *
 * Heavy initialisation (Store, SearchIndex, Embedder, Provider, Memory)
 * runs asynchronously via the returned `ready` promise so the pi loader
 * can await it before firing `session_start`. The synchronous handle
 * returned from `createMemoryExtension` is what gets passed back to
 * callers that need to flush / close the runtime explicitly.
 */

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { type MemoryExtensionConfig, MemoryExtensionConfigSchema } from './config.js'
import { type MemoryHookHandle, registerMemoryHooks } from './hooks.js'
import { type MemoryRuntime, type RuntimeLogger, createMemoryRuntime } from './runtime.js'
import { registerMemoryTools } from './tools.js'

export type MemoryExtension = {
  /** Wait for the runtime to be fully initialised. */
  readonly ready: Promise<MemoryRuntime>
  /**
   * The handle returned by `registerMemoryHooks`. Useful for tests that
   * want to drive lifecycle events directly without round-tripping
   * through `pi.on(...)` dispatch.
   */
  readonly hooks: Promise<MemoryHookHandle>
  /** Drain the extract queue without closing the runtime. */
  flush(): Promise<void>
  /** Flush + close the underlying store and search index. Idempotent. */
  close(): Promise<void>
}

export type CreateMemoryExtensionOptions = {
  readonly logger?: RuntimeLogger
}

/**
 * Build a memory-pi extension instance and register its tools / hooks
 * on the supplied pi runtime. Returns synchronously so callers can hold
 * a handle, but most consumers should await `ext.ready` to make sure
 * tools and hooks are wired before driving the agent.
 */
export const createMemoryExtension = (
  pi: ExtensionAPI,
  config?: MemoryExtensionConfig,
  options: CreateMemoryExtensionOptions = {},
): MemoryExtension => {
  const parsed = config !== undefined ? MemoryExtensionConfigSchema.parse(config) : {}
  let runtime: MemoryRuntime | undefined
  let hookHandle: MemoryHookHandle | undefined

  const init = (async (): Promise<MemoryRuntime> => {
    const built = await createMemoryRuntime(
      parsed,
      options.logger !== undefined ? { logger: options.logger } : {},
    )
    runtime = built
    registerMemoryTools(pi, built)
    hookHandle = registerMemoryHooks(pi, built)
    return built
  })()

  const hooks = (async (): Promise<MemoryHookHandle> => {
    await init
    if (hookHandle === undefined) {
      throw new Error('memory-pi: hooks were not initialised')
    }
    return hookHandle
  })()

  return {
    ready: init,
    hooks,
    async flush() {
      const r = runtime ?? (await init)
      await r.extractQueue.flush()
    },
    async close() {
      const r = runtime ?? (await init)
      await r.close()
    },
  }
}

/**
 * Default export that conforms to the pi extension contract. Reads
 * configuration from the `MEMORY_PI_CONFIG` env var (JSON-encoded) so
 * `pi -e ./dist/index.js` works without recompiling the extension.
 */
const defaultExport = async (pi: ExtensionAPI): Promise<void> => {
  const raw = process.env.MEMORY_PI_CONFIG
  const config: MemoryExtensionConfig = raw !== undefined && raw !== '' ? JSON.parse(raw) : {}
  const ext = createMemoryExtension(pi, config)
  await ext.ready
}

export default defaultExport

export { MemoryExtensionConfigSchema, MEMORY_TOOL_NAMES } from './config.js'
export type { MemoryExtensionConfig, MemoryToolName } from './config.js'
export type {
  MemoryRuntime,
  ResolvedRuntimeConfig,
  RuntimeLogger,
  ExtractJob,
  RecallSnapshot,
} from './runtime.js'
export { createMemoryRuntime, resolveRuntimeConfig, renderRecallBlock } from './runtime.js'
export { registerMemoryHooks, type MemoryHookHandle } from './hooks.js'
export { registerMemoryTools, buildTools } from './tools.js'
export {
  detectProvider,
  detectEmbedder,
  ollamaReachable,
  resolveBrainPaths,
  DEFAULT_BRAIN_ROOT,
  DEFAULT_BRAIN_ID,
  DEFAULT_OLLAMA_BASE_URL,
} from './defaults.js'
