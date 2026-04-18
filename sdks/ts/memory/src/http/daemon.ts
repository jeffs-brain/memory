// SPDX-License-Identifier: Apache-2.0

/**
 * Daemon + BrainManager for the memory HTTP server.
 *
 * Mirrors the Go `sdks/go/cmd/memory/daemon.go` design: the Daemon owns
 * shared LLM/embedder references plus a root directory, and defers
 * per-brain resource construction to {@link BrainManager}. A brain's
 * `BrainResources` bundle is built lazily on first access and cached
 * for the daemon lifetime.
 *
 * Concurrency: TypeScript is single-threaded so a `Map<string, Promise<...>>`
 * is enough to dedup concurrent open attempts on the same brain.
 */

import { access, mkdir, readdir, rm, stat as fsStat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { resolve as resolvePath, join as joinFs } from 'node:path'

import type { Embedder, Logger, Provider } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import {
  type ConsolidationReport,
  type ExtractedMemory,
  type Memory,
  type RecallHit,
  type ReflectionResult,
  type Scope,
  createMemory,
  createStoreBackedCursorStore,
} from '../memory/index.js'
import { createRetrieval, type Retrieval } from '../retrieval/index.js'
import type { SearchIndex as SqliteSearchIndex } from '../search/index.js'
import { createSearchIndex, type Chunk } from '../search/index.js'
import {
  type ChangeEvent,
  type FileInfo,
  type ListOpts,
  type Path,
  type Store,
  createFsStore,
  joinPath,
} from '../store/index.js'

export type DaemonConfig = {
  readonly root: string
  readonly authToken?: string
  readonly provider?: Provider
  readonly embedder?: Embedder
  readonly logger?: Logger
}

export type BrainResources = {
  readonly id: string
  readonly root: string
  readonly store: Store
  readonly memory: Memory
  readonly index: SqliteSearchIndex | undefined
  readonly retrieval: Retrieval | undefined
  /** Embedder shared from the daemon. `undefined` when no embedder was
   *  configured; callers must guard before running the ingest pipeline. */
  readonly embedder: Embedder | undefined
  /** Shared provider resolved from env (or the caller) at daemon build
   *  time. Handlers reach here rather than through the Daemon so the
   *  per-brain bundle stays self-sufficient. */
  readonly provider: Provider | undefined
  /** Promise tracking the background refresh used to keep the index
   *  warm. Awaited once per search/ask round so freshly committed
   *  writes are searchable. */
  refresh: () => Promise<void>
  close: () => Promise<void>
}

export class BrainNotFoundError extends Error {
  override readonly name = 'BrainNotFoundError'
  constructor(public readonly brainId: string) {
    super(`brain manager: not found: ${brainId}`)
  }
}

export class BrainConflictError extends Error {
  override readonly name = 'BrainConflictError'
  constructor(public readonly brainId: string) {
    super(`brain manager: already exists: ${brainId}`)
  }
}

export class Daemon {
  readonly root: string
  readonly authToken: string | undefined
  readonly provider: Provider | undefined
  readonly embedder: Embedder | undefined
  readonly logger: Logger
  readonly brains: BrainManager

  constructor(config: DaemonConfig) {
    this.root = resolvePath(config.root)
    this.authToken = config.authToken
    this.provider = config.provider
    this.embedder = config.embedder
    this.logger = config.logger ?? noopLogger
    this.brains = new BrainManager(this)
  }

  /** Initialise the on-disk root, creating it if missing. */
  async start(): Promise<void> {
    await mkdir(this.root, { recursive: true })
  }

  /** Close every per-brain resource currently cached. */
  async close(): Promise<void> {
    await this.brains.closeAll()
  }

  brainRoot(brainId: string): string {
    return joinFs(this.root, 'brains', brainId)
  }

  async brainExists(brainId: string): Promise<boolean> {
    try {
      const info = await fsStat(this.brainRoot(brainId))
      return info.isDirectory()
    } catch {
      return false
    }
  }
}

const INDEXABLE_EXT = /\.(md|markdown|txt)$/i

export class BrainManager {
  private readonly entries = new Map<string, Promise<BrainResources>>()

  constructor(private readonly daemon: Daemon) {}

  async list(): Promise<string[]> {
    const dir = joinFs(this.daemon.root, 'brains')
    try {
      const entries = await readdir(dir, { withFileTypes: true })
      const out = entries.filter((e) => e.isDirectory()).map((e) => e.name)
      out.sort()
      return out
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') return []
      throw err
    }
  }

  async create(brainId: string): Promise<BrainResources> {
    if (brainId === '') {
      throw new Error('brain manager: brainId required')
    }
    const root = this.daemon.brainRoot(brainId)
    try {
      await access(root)
      throw new BrainConflictError(brainId)
    } catch (err) {
      if (err instanceof BrainConflictError) throw err
      if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
    }
    await mkdir(root, { recursive: true })
    return this.get(brainId)
  }

  async get(brainId: string): Promise<BrainResources> {
    const cached = this.entries.get(brainId)
    if (cached !== undefined) return cached
    const pending = this.build(brainId)
    this.entries.set(brainId, pending)
    try {
      return await pending
    } catch (err) {
      this.entries.delete(brainId)
      throw err
    }
  }

  async delete(brainId: string): Promise<void> {
    const cached = this.entries.get(brainId)
    if (cached !== undefined) {
      try {
        const res = await cached
        await res.close()
      } catch {
        /* ignore teardown errors */
      }
      this.entries.delete(brainId)
    }
    const root = this.daemon.brainRoot(brainId)
    try {
      const info = await fsStat(root)
      if (!info.isDirectory()) throw new BrainNotFoundError(brainId)
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        throw new BrainNotFoundError(brainId)
      }
      throw err
    }
    await rm(root, { recursive: true, force: true })
    // Best-effort cleanup of the sidecar search index.
    await rm(joinFs(this.daemon.root, 'indices', brainId), {
      recursive: true,
      force: true,
    })
  }

  async closeAll(): Promise<void> {
    const pending = Array.from(this.entries.values())
    this.entries.clear()
    for (const p of pending) {
      try {
        const res = await p
        await res.close()
      } catch {
        /* ignore */
      }
    }
  }

  private async build(brainId: string): Promise<BrainResources> {
    if (!(await this.daemon.brainExists(brainId))) {
      throw new BrainNotFoundError(brainId)
    }
    const root = this.daemon.brainRoot(brainId)
    const store = await createFsStore({ root })

    // Build the best-effort sqlite search index. Failures here are
    // non-fatal and leave the retriever unset; search falls back to a
    // store-walk below. The db path sits outside the brain store root
    // so sqlite's auxiliary `-shm` / `-wal` files do not show up in
    // store listings.
    const indexRoot = joinFs(this.daemon.root, 'indices', brainId)
    let index: SqliteSearchIndex | undefined
    let retrieval: Retrieval | undefined
    try {
      const { mkdir } = await import('node:fs/promises')
      await mkdir(indexRoot, { recursive: true })
      index = await createSearchIndex({ dbPath: joinFs(indexRoot, 'search.db') })
      retrieval = createRetrieval({
        index,
        ...(this.daemon.embedder !== undefined ? { embedder: this.daemon.embedder } : {}),
      })
    } catch (err) {
      this.daemon.logger.debug('daemon: search index unavailable', {
        brainId,
        err: err instanceof Error ? err.message : String(err),
      })
      index = undefined
      retrieval = undefined
    }

    const memory = createMemory({
      store,
      provider: this.daemon.provider ?? fallbackProvider,
      ...(this.daemon.embedder !== undefined ? { embedder: this.daemon.embedder } : {}),
      cursorStore: createStoreBackedCursorStore(store),
      scope: 'global',
      actorId: 'daemon',
      extractMinMessages: 1,
    })

    // Subscribe the index to store mutations so writes land in FTS
    // without a poll. Index rows use the relative path as id to match
    // the Go daemon's simple classifier.
    let unsubscribe: (() => void) | undefined
    if (index !== undefined) {
      unsubscribe = store.subscribe((evt) => {
        void this.handleStoreEvent(index!, store, evt)
      })
    }

    // Fire an initial scan so prior contents surface on the first
    // search. Kick it off in the background; the public refresh()
    // awaits the in-flight promise.
    let refreshInFlight: Promise<void> | undefined
    let disposed = false
    const refresh = async (): Promise<void> => {
      if (index === undefined || disposed) return
      if (refreshInFlight !== undefined) {
        await refreshInFlight
        return
      }
      refreshInFlight = this.scanBrain(index, store)
        .catch(() => undefined)
        .finally(() => {
          refreshInFlight = undefined
        })
      await refreshInFlight
    }
    void refresh()

    return {
      id: brainId,
      root,
      store,
      memory,
      index,
      retrieval,
      embedder: this.daemon.embedder,
      provider: this.daemon.provider,
      refresh,
      close: async () => {
        disposed = true
        unsubscribe?.()
        // Let any in-flight scan finish so it is not observed as an
        // unhandled rejection against a closed store.
        if (refreshInFlight !== undefined) {
          try {
            await refreshInFlight
          } catch {
            /* ignore */
          }
        }
        if (index !== undefined) {
          try {
            await index.close()
          } catch {
            /* ignore */
          }
        }
        try {
          await store.close()
        } catch {
          /* ignore */
        }
      },
    }
  }

  /**
   * Apply a single change event to the SQLite index. Deletions drop
   * the row, writes re-upsert it with the latest bytes.
   */
  private async handleStoreEvent(
    index: SqliteSearchIndex,
    store: Store,
    evt: ChangeEvent,
  ): Promise<void> {
    try {
      if (evt.kind === 'deleted') {
        index.deleteChunk(evt.path)
        return
      }
      if (evt.kind === 'renamed' && evt.oldPath !== undefined) {
        index.deleteChunk(evt.oldPath)
      }
      if (!INDEXABLE_EXT.test(evt.path)) return
      const buf = await store.read(evt.path)
      const chunk: Chunk = {
        id: evt.path,
        path: evt.path,
        ordinal: 0,
        title: evt.path,
        content: buf.toString('utf8').slice(0, 32_000),
      }
      index.upsertChunk(chunk)
    } catch {
      /* index maintenance is best-effort */
    }
  }

  /**
   * Full-brain scan used on open + on manual refresh so a brain that
   * already has content surfaces it on the first search.
   */
  private async scanBrain(index: SqliteSearchIndex, store: Store): Promise<void> {
    let entries
    try {
      entries = await store.list('', { recursive: true, includeGenerated: true })
    } catch (err) {
      // Brain was torn down underneath us (test cleanup, delete, etc.).
      if ((err as NodeJS.ErrnoException | undefined)?.code === 'ENOENT') return
      throw err
    }
    for (const entry of entries) {
      if (entry.isDir) continue
      if (!INDEXABLE_EXT.test(entry.path)) continue
      try {
        const buf = await store.read(entry.path)
        const chunk: Chunk = {
          id: entry.path,
          path: entry.path,
          ordinal: 0,
          title: entry.path,
          content: buf.toString('utf8').slice(0, 32_000),
        }
        index.upsertChunk(chunk)
      } catch {
        /* ignore unreadable files */
      }
    }
  }
}

/**
 * Fallback provider used when no explicit LLM is configured. Returns a
 * deterministic stub so CLI surfaces that require a provider (like the
 * memory factory) can be constructed without aborting the daemon.
 */
const fallbackProvider: Provider = {
  name: () => 'noop',
  modelName: () => 'noop',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  complete: async () => ({
    content: '',
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn' as const,
  }),
  supportsStructuredDecoding: () => false,
  structured: async () => '',
}

export const defaultRoot = (): string => {
  const fromEnv = process.env['JB_HOME']
  if (fromEnv !== undefined && fromEnv !== '') return fromEnv
  return joinFs(homedir(), '.jeffs-brain')
}

export type { Retrieval, SqliteSearchIndex }
export type DaemonTypes = {
  Memory: Memory
  Scope: Scope
  Path: Path
  ListOpts: ListOpts
  FileInfo: FileInfo
  ChangeEvent: ChangeEvent
  ExtractedMemory: ExtractedMemory
  ReflectionResult: ReflectionResult
  ConsolidationReport: ConsolidationReport
  RecallHit: RecallHit
}

/** Re-exports for handler modules. */
export { joinPath }
