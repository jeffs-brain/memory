// SPDX-License-Identifier: Apache-2.0

/**
 * Runtime wiring shared by the tools and hooks. Builds the memory
 * pipeline (Store + Embedder + Provider + SearchIndex + Retrieval +
 * Memory) once per extension instance and exposes a small set of
 * coordinator primitives (extract queue, recall cache).
 *
 * The runtime is intentionally agnostic of the pi `ExtensionAPI`. Both
 * the tools and the hook handlers receive a runtime via dependency
 * injection so unit tests can drive the runtime without spinning up
 * the pi loader.
 */

import { mkdir } from 'node:fs/promises'
import {
  AnthropicProvider,
  type Embedder,
  type ExtractedMemory,
  type Memory,
  type MemoryNote,
  OllamaEmbedder,
  OllamaProvider,
  OpenAIEmbedder,
  OpenAIProvider,
  type Provider,
  type RecallHit,
  type Scope,
  type SqliteSearchIndex,
  type Store,
  autodetectStore,
  createMemory,
  createSearchIndex,
  createStoreBackedCursorStore,
} from '@jeffs-brain/memory'
import type { Message } from '@jeffs-brain/memory'
import type { Retrieval } from '@jeffs-brain/memory/retrieval'
import { createRetrieval } from '@jeffs-brain/memory/retrieval'
import type { MemoryExtensionConfig } from './config.js'
import {
  DEFAULT_BRAIN_ID,
  DEFAULT_BRAIN_ROOT,
  DEFAULT_OLLAMA_BASE_URL,
  detectEmbedder,
  detectProvider,
  resolveBrainPaths,
} from './defaults.js'

export type RuntimeLogger = {
  info(message: string, ctx?: Record<string, unknown>): void
  warn(message: string, ctx?: Record<string, unknown>): void
  error(message: string, ctx?: Record<string, unknown>): void
}

export const noopRuntimeLogger: RuntimeLogger = {
  info: () => {},
  warn: () => {},
  error: () => {},
}

export type ExtractJob = {
  readonly messages: readonly Message[]
  readonly actorId?: string
  readonly sessionId?: string
  readonly scope?: Scope
}

export type ExtractQueueState = {
  enqueue(job: ExtractJob): void
  flush(): Promise<void>
  size(): number
}

export type RecallSnapshot = {
  readonly query: string
  readonly paths: readonly string[]
  readonly block: string
  readonly hits: readonly RecallHit[]
}

export type MemoryRuntime = {
  readonly config: ResolvedRuntimeConfig
  readonly store: Store
  readonly searchIndex: SqliteSearchIndex
  readonly embedder: Embedder | undefined
  readonly provider: Provider | undefined
  readonly memory: Memory
  readonly retrieval: Retrieval
  readonly extractQueue: ExtractQueueState
  /** Most recent recall snapshot keyed by hashed user-message. */
  lastRecall?: RecallSnapshot
  /** Last extracted batch, exposed for tests and observability. */
  lastExtracted?: readonly ExtractedMemory[]
  close(): Promise<void>
}

export type ResolvedRuntimeConfig = {
  readonly brainRoot: string
  readonly brainId: string
  readonly scope: Scope
  readonly fallbackScopes: readonly Scope[]
  readonly actorId: string
  readonly extractMinMessages: number
  readonly recallTopK: number
  readonly recallMinScore: number
  readonly extractOnTurnEnd: boolean
  readonly reflectOnSessionEnd: boolean
  readonly consolidateSchedule: 'manual' | 'session' | '@daily'
  readonly recallOnPrompt: boolean
  readonly recallCacheFriendly: boolean
  readonly exposedTools: readonly string[] | undefined
}

/**
 * Resolve the user-supplied configuration into the dense runtime form.
 * Missing fields fall back to documented defaults so callers can pass
 * `{}` and get a working extension.
 */
export const resolveRuntimeConfig = (
  config: MemoryExtensionConfig | undefined,
  env: NodeJS.ProcessEnv = process.env,
): ResolvedRuntimeConfig => {
  const brainRoot = config?.brainRoot ?? env.MEMORY_PI_BRAIN_ROOT ?? DEFAULT_BRAIN_ROOT
  const brainId = config?.brainId ?? env.MEMORY_PI_BRAIN_ID ?? DEFAULT_BRAIN_ID
  const scope = (config?.recall?.scope as Scope | undefined) ?? 'global'
  const fallbackScopes = (config?.recall?.fallbackScopes as readonly Scope[] | undefined) ?? []
  const actorId = config?.acl?.actorId ?? env.MEMORY_PI_ACTOR_ID ?? 'pi-user'
  return {
    brainRoot,
    brainId,
    scope,
    fallbackScopes,
    actorId,
    extractMinMessages: config?.extract?.minMessages ?? 6,
    recallTopK: config?.recall?.topK ?? 5,
    recallMinScore: config?.recall?.minScore ?? 0,
    extractOnTurnEnd: config?.extract?.onTurnEnd ?? true,
    reflectOnSessionEnd: config?.reflect?.onSessionEnd ?? true,
    consolidateSchedule: config?.consolidate?.schedule ?? 'manual',
    recallOnPrompt: config?.recall?.onPrompt ?? true,
    recallCacheFriendly: config?.recall?.cacheFriendly ?? true,
    exposedTools: config?.tools?.expose,
  }
}

/**
 * Build a Store from a `StoreConfig`. `auto` defers to `autodetectStore`
 * which picks fs or git based on whether `.git` is present.
 */
const buildStore = async (
  config: MemoryExtensionConfig | undefined,
  brainRoot: string,
): Promise<Store> => {
  const storeConfig = config?.store
  const kind = storeConfig?.kind ?? 'auto'
  await mkdir(brainRoot, { recursive: true })
  if (kind === 'auto' || kind === 'fs' || kind === 'git') {
    return autodetectStore({ root: brainRoot })
  }
  if (kind === 'http') {
    // The http store lives behind the same Store interface but the
    // pi extension does not own its lifecycle today; surface a clear
    // error so callers know to plumb it themselves until W5.
    throw new Error('memory-pi: http store kind is not wired yet; pass store: { kind: "auto" }')
  }
  throw new Error(`memory-pi: unknown store kind: ${JSON.stringify(storeConfig)}`)
}

const buildEmbedder = async (
  config: MemoryExtensionConfig | undefined,
): Promise<Embedder | undefined> => {
  const embedderConfig = config?.embedder
  const kind = embedderConfig?.kind ?? 'auto'
  if (kind === 'off') return undefined
  if (kind === 'auto') {
    const detected = await detectEmbedder()
    if (detected === undefined) return undefined
    if (detected.kind === 'ollama') {
      return new OllamaEmbedder({ baseURL: detected.baseUrl, model: detected.model })
    }
    return new OpenAIEmbedder({ apiKey: detected.apiKey, model: detected.model })
  }
  if (embedderConfig !== undefined && embedderConfig.kind === 'ollama') {
    return new OllamaEmbedder({
      baseURL: embedderConfig.baseUrl ?? DEFAULT_OLLAMA_BASE_URL,
      model: embedderConfig.model ?? 'bge-m3',
    })
  }
  if (embedderConfig !== undefined && embedderConfig.kind === 'openai') {
    return new OpenAIEmbedder({
      apiKey: embedderConfig.apiKey,
      model: embedderConfig.model ?? 'text-embedding-3-small',
    })
  }
  if (embedderConfig !== undefined && embedderConfig.kind === 'tei') {
    // TEI embedder needs custom URL plumbing not yet stabilised; punt
    // to an explicit error so the user does not silently get no
    // embeddings.
    throw new Error('memory-pi: tei embedder is not wired yet; pass embedder: { kind: "auto" }')
  }
  throw new Error(`memory-pi: unknown embedder kind: ${JSON.stringify(embedderConfig)}`)
}

const buildProvider = async (
  config: MemoryExtensionConfig | undefined,
): Promise<Provider | undefined> => {
  const providerConfig = config?.provider
  const kind = providerConfig?.kind ?? 'auto'
  if (kind === 'auto') {
    const detected = await detectProvider()
    if (detected === undefined) return undefined
    if (detected.kind === 'openai') {
      return new OpenAIProvider({ apiKey: detected.apiKey, model: detected.model })
    }
    if (detected.kind === 'anthropic') {
      return new AnthropicProvider({ apiKey: detected.apiKey, model: detected.model })
    }
    return new OllamaProvider({ baseURL: detected.baseUrl, model: detected.model })
  }
  if (providerConfig !== undefined && providerConfig.kind === 'openai') {
    return new OpenAIProvider({
      apiKey: providerConfig.apiKey,
      model: providerConfig.model ?? 'gpt-4o-mini',
    })
  }
  if (providerConfig !== undefined && providerConfig.kind === 'anthropic') {
    return new AnthropicProvider({
      apiKey: providerConfig.apiKey,
      model: providerConfig.model ?? 'claude-opus-4-5',
    })
  }
  if (providerConfig !== undefined && providerConfig.kind === 'ollama') {
    return new OllamaProvider({
      baseURL: providerConfig.baseUrl ?? DEFAULT_OLLAMA_BASE_URL,
      model: providerConfig.model ?? 'gemma3',
    })
  }
  throw new Error(`memory-pi: unknown provider kind: ${JSON.stringify(providerConfig)}`)
}

/**
 * Build a `MemoryRuntime` from user config. Heavy initialisation runs
 * once per extension instance; subsequent recall / extract calls share
 * the same store and search index handles.
 */
export const createMemoryRuntime = async (
  config: MemoryExtensionConfig | undefined,
  options: { readonly logger?: RuntimeLogger } = {},
): Promise<MemoryRuntime> => {
  const logger = options.logger ?? noopRuntimeLogger
  const resolved = resolveRuntimeConfig(config)
  const paths = resolveBrainPaths(resolved.brainRoot, resolved.brainId)
  await mkdir(paths.root, { recursive: true })
  const store = await buildStore(config, paths.root)
  const searchIndex = await createSearchIndex({ dbPath: paths.searchIndexPath })
  const embedder = await buildEmbedder(config)
  const provider = await buildProvider(config)

  if (provider === undefined) {
    logger.warn(
      'memory-pi: no LLM provider detected; extract / reflect / consolidate will throw on call',
    )
  }
  if (embedder === undefined) {
    logger.warn('memory-pi: no embedder detected; vector recall disabled')
  }

  const retrieval = createRetrieval({
    index: searchIndex,
    ...(embedder !== undefined ? { embedder } : {}),
  })

  // The Memory pipeline requires a non-undefined Provider. When the
  // user has no chat provider we fall back to a "no-op" lazy provider
  // that throws on use; this keeps the runtime constructable so recall
  // still works.
  const memoryProvider = provider ?? createLazyProvider()
  const memory = createMemory({
    store,
    provider: memoryProvider,
    ...(embedder !== undefined ? { embedder } : {}),
    scope: resolved.scope,
    actorId: resolved.actorId,
    cursorStore: createStoreBackedCursorStore(store),
    extractMinMessages: resolved.extractMinMessages,
  })

  const extractQueue = createExtractQueue(memory, logger)

  const runtime: MemoryRuntime = {
    config: resolved,
    store,
    searchIndex,
    embedder,
    provider,
    memory,
    retrieval,
    extractQueue,
    async close() {
      await extractQueue.flush()
      await searchIndex.close().catch(() => undefined)
      await store.close().catch(() => undefined)
    },
  }
  return runtime
}

/**
 * Provider stub that throws when called. We install this so callers
 * that only use recall + remember keep working when no chat provider
 * is configured.
 */
const createLazyProvider = (): Provider => {
  const fail = (): never => {
    throw new Error(
      'memory-pi: no LLM provider configured; set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally',
    )
  }
  return {
    name: () => 'memory-pi/none',
    modelName: () => 'none',
    supportsStructuredDecoding: () => false,
    stream: () => fail(),
    complete: () => fail(),
    structured: () => fail(),
  }
}

const createExtractQueue = (memory: Memory, logger: RuntimeLogger): ExtractQueueState => {
  const queue: ExtractJob[] = []
  let running = false
  let pending: Promise<void> = Promise.resolve()

  const drain = async (): Promise<void> => {
    if (running) return pending
    running = true
    pending = (async () => {
      try {
        while (queue.length > 0) {
          const job = queue.shift()
          if (job === undefined) continue
          try {
            await memory.extract({
              messages: job.messages,
              ...(job.actorId !== undefined ? { actorId: job.actorId } : {}),
              ...(job.scope !== undefined ? { scope: job.scope } : {}),
              ...(job.sessionId !== undefined ? { sessionId: job.sessionId } : {}),
            })
          } catch (err) {
            logger.error('memory-pi: extract job failed', {
              error: err instanceof Error ? err.message : String(err),
            })
          }
        }
      } finally {
        running = false
      }
    })()
    return pending
  }

  return {
    enqueue(job) {
      queue.push(job)
      // Fire and forget; flush() awaits the same promise chain.
      void drain()
    },
    async flush() {
      await drain()
    },
    size() {
      return queue.length
    },
  }
}

/**
 * Render a recall block suitable for system-prompt injection. We use
 * an opening + closing tag so cache-friendly diffing in pi remains
 * trivial.
 */
export const renderRecallBlock = (
  hits: readonly RecallHit[],
  emptyLabel = 'no recalled memory',
): string => {
  if (hits.length === 0) {
    return `<recalled-memory>${emptyLabel}</recalled-memory>`
  }
  const body = hits
    .map((hit, idx) => {
      const note: MemoryNote = hit.note
      const title = note.name !== '' ? note.name : note.path
      const preview = hit.content.length > 1200 ? `${hit.content.slice(0, 1200)}…` : hit.content
      return `### [${idx + 1}] ${title} (score ${hit.score.toFixed(3)})\n${preview}`
    })
    .join('\n\n')
  return `<recalled-memory>\n${body}\n</recalled-memory>`
}
