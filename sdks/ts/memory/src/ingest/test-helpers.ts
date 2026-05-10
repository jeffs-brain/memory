// SPDX-License-Identifier: Apache-2.0

/**
 * Shared test doubles for the ingest pipeline. Provides configurable
 * in-memory implementations of Store, Embedder, and SearchIndex that
 * support error injection at arbitrary call boundaries.
 *
 * Imported by pipeline-state.test.ts, pipeline.test.ts, and all future
 * ingest-related test suites (P0-2, P0-3, P1-*).
 */

import type { Embedder, Logger } from '../llm/types.js'
import type { Chunk, SearchIndex } from '../search/index.js'
import type {
  Batch,
  BatchOptions,
  ChangeEvent,
  EventSink,
  FileInfo,
  ListOpts,
  Path,
  Store,
  Unsubscribe,
} from '../store/index.js'
import { ErrNotFound, isGenerated, lastSegment, matchGlob, validatePath } from '../store/index.js'

// ---------------------------------------------------------------------------
// ErrorAfterN: wraps a function so it throws after N successful invocations.
// ---------------------------------------------------------------------------

export class ErrorAfterN<TArgs extends readonly unknown[], TReturn> {
  private callCount = 0

  constructor(
    private readonly successCount: number,
    private readonly delegate: (...args: TArgs) => TReturn,
    private readonly errorMessage: string = 'injected failure',
  ) {}

  call = (...args: TArgs): TReturn => {
    if (this.callCount >= this.successCount) {
      throw new Error(this.errorMessage)
    }
    this.callCount++
    return this.delegate(...args)
  }

  get invocations(): number {
    return this.callCount
  }

  reset(): void {
    this.callCount = 0
  }
}

// ---------------------------------------------------------------------------
// MockEmbedder: returns deterministic fixed-dimension vectors.
// ---------------------------------------------------------------------------

export type MockEmbedderOptions = {
  readonly dim?: number
  readonly embedFn?: (texts: readonly string[]) => Promise<number[][]>
}

const DEFAULT_MOCK_DIM = 32

export const createMockEmbedder = (opts: MockEmbedderOptions = {}): MockEmbedder => {
  return new MockEmbedder(opts)
}

export class MockEmbedder implements Embedder {
  private readonly dim: number
  private readonly embedFn: ((texts: readonly string[]) => Promise<number[][]>) | undefined
  private embedCalls: ReadonlyArray<readonly string[]> = []

  constructor(opts: MockEmbedderOptions = {}) {
    this.dim = opts.dim ?? DEFAULT_MOCK_DIM
    this.embedFn = opts.embedFn
  }

  name(): string {
    return 'mock-embedder'
  }

  model(): string {
    return 'mock-model'
  }

  dimension(): number {
    return this.dim
  }

  async embed(texts: readonly string[], _signal?: AbortSignal): Promise<number[][]> {
    this.embedCalls = [...this.embedCalls, texts]
    if (this.embedFn !== undefined) {
      return this.embedFn(texts)
    }
    return texts.map((text) => {
      const vec = new Array<number>(this.dim).fill(0)
      for (let i = 0; i < text.length && i < this.dim; i++) {
        vec[i] = (text.charCodeAt(i) % 100) / 100
      }
      return vec
    })
  }

  get calls(): ReadonlyArray<readonly string[]> {
    return this.embedCalls
  }
}

// ---------------------------------------------------------------------------
// MockSearchIndex: records upsert/delete operations in memory.
// ---------------------------------------------------------------------------

export type MockSearchIndexOptions = {
  readonly upsertFn?: (chunks: readonly Chunk[]) => void
}

export const createMockSearchIndex = (opts: MockSearchIndexOptions = {}): MockSearchIndex => {
  return new MockSearchIndex(opts)
}

export class MockSearchIndex implements Pick<
  SearchIndex,
  'upsertChunk' | 'upsertChunks' | 'deleteChunk' | 'deleteByPath' | 'getChunk'
> {
  private readonly storedChunks = new Map<string, Chunk>()
  private readonly upsertFn: ((chunks: readonly Chunk[]) => void) | undefined
  private upsertCalls = 0

  constructor(opts: MockSearchIndexOptions = {}) {
    this.upsertFn = opts.upsertFn
  }

  upsertChunk(chunk: Chunk): void {
    this.upsertCalls++
    if (this.upsertFn !== undefined) {
      this.upsertFn([chunk])
      return
    }
    this.storedChunks.set(chunk.id, chunk)
  }

  upsertChunks(chunks: readonly Chunk[]): void {
    this.upsertCalls++
    if (this.upsertFn !== undefined) {
      this.upsertFn(chunks)
      return
    }
    for (const chunk of chunks) {
      this.storedChunks.set(chunk.id, chunk)
    }
  }

  deleteChunk(id: string): void {
    this.storedChunks.delete(id)
  }

  deleteByPath(path: string): void {
    for (const [id, chunk] of this.storedChunks) {
      if (chunk.path === path) {
        this.storedChunks.delete(id)
      }
    }
  }

  getChunk(id: string): Chunk | undefined {
    return this.storedChunks.get(id)
  }

  get chunks(): ReadonlyMap<string, Chunk> {
    return this.storedChunks
  }

  get upsertCallCount(): number {
    return this.upsertCalls
  }

  hasChunksForDocument(documentId: string): boolean {
    for (const chunk of this.storedChunks.values()) {
      const meta = chunk.metadata as Record<string, unknown> | undefined
      if (meta?.['documentId'] === documentId) return true
    }
    return false
  }
}

// ---------------------------------------------------------------------------
// MockStore: in-memory Store with optional error injection.
// ---------------------------------------------------------------------------

type MockEntry = {
  content: Buffer
  modTime: Date
}

export const createMockStore = (): MockStore => new MockStore()

export class MockStore implements Store {
  private files = new Map<string, MockEntry>()
  private readonly sinks: EventSink[] = []
  private closed = false

  async read(p: Path): Promise<Buffer> {
    this.guardClosed()
    validatePath(p)
    const entry = this.files.get(p)
    if (entry === undefined) throw new ErrNotFound(p)
    return Buffer.from(entry.content)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    this.guardClosed()
    validatePath(p)
    this.files.set(p, { content: Buffer.from(content), modTime: new Date() })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    this.guardClosed()
    validatePath(p)
    const existing = this.files.get(p)
    const merged = existing
      ? Buffer.concat([existing.content, content])
      : Buffer.from(content)
    this.files.set(p, { content: merged, modTime: new Date() })
  }

  async delete(p: Path): Promise<void> {
    this.guardClosed()
    validatePath(p)
    if (!this.files.has(p)) throw new ErrNotFound(p)
    this.files.delete(p)
  }

  async rename(src: Path, dst: Path): Promise<void> {
    this.guardClosed()
    validatePath(src)
    validatePath(dst)
    const entry = this.files.get(src)
    if (entry === undefined) throw new ErrNotFound(src)
    this.files.delete(src)
    this.files.set(dst, { content: Buffer.from(entry.content), modTime: new Date() })
  }

  async exists(p: Path): Promise<boolean> {
    this.guardClosed()
    validatePath(p)
    if (this.files.has(p)) return true
    const prefix = `${p}/`
    for (const key of this.files.keys()) {
      if (key.startsWith(prefix)) return true
    }
    return false
  }

  async stat(p: Path): Promise<FileInfo> {
    this.guardClosed()
    validatePath(p)
    const entry = this.files.get(p)
    if (entry !== undefined) {
      return { path: p, size: entry.content.length, modTime: entry.modTime, isDir: false }
    }
    const prefix = `${p}/`
    for (const key of this.files.keys()) {
      if (key.startsWith(prefix)) {
        return { path: p, size: 0, modTime: new Date(0), isDir: true }
      }
    }
    throw new ErrNotFound(p)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.guardClosed()
    const prefix = dir === '' ? '' : dir.endsWith('/') ? dir : `${dir}/`
    const recursive = opts.recursive === true
    const includeGenerated = opts.includeGenerated === true
    const glob = opts.glob ?? ''
    const seenDirs = new Set<string>()
    const entries: FileInfo[] = []

    for (const [path, entry] of this.files) {
      if (prefix !== '' && !path.startsWith(prefix)) continue
      const rest = path.slice(prefix.length)
      if (rest === '') continue

      if (recursive) {
        if (!includeGenerated && isGenerated(path as Path)) continue
        if (glob !== '' && !matchGlob(glob, lastSegment(rest))) continue
        entries.push({
          path: path as Path,
          size: entry.content.length,
          modTime: entry.modTime,
          isDir: false,
        })
        continue
      }

      const slashIdx = rest.indexOf('/')
      if (slashIdx === -1) {
        if (!includeGenerated && isGenerated(path as Path)) continue
        if (glob !== '' && !matchGlob(glob, rest)) continue
        entries.push({
          path: path as Path,
          size: entry.content.length,
          modTime: entry.modTime,
          isDir: false,
        })
      } else {
        const childDir = prefix + rest.slice(0, slashIdx)
        if (!seenDirs.has(childDir)) {
          seenDirs.add(childDir)
          entries.push({
            path: childDir as Path,
            size: 0,
            modTime: new Date(0),
            isDir: true,
          })
        }
      }
    }

    entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return entries
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    this.guardClosed()
    const snapshot = new Map<string, MockEntry>()
    for (const [k, v] of this.files) {
      snapshot.set(k, { content: Buffer.from(v.content), modTime: v.modTime })
    }
    const working = new Map<string, MockEntry>()
    for (const [k, v] of this.files) {
      working.set(k, { content: Buffer.from(v.content), modTime: v.modTime })
    }
    const batch = new MockBatch(working)
    await fn(batch)
    this.files = working as Map<string, MockEntry>
    this.emitDiff(snapshot, working, opts.reason)
  }

  subscribe(sink: EventSink): Unsubscribe {
    this.sinks.push(sink)
    return () => {
      const idx = this.sinks.indexOf(sink)
      if (idx >= 0) this.sinks.splice(idx, 1)
    }
  }

  localPath(_p: Path): string | undefined {
    return undefined
  }

  async close(): Promise<void> {
    this.closed = true
  }

  /** Expose file map for test assertions. */
  get fileCount(): number {
    return this.files.size
  }

  hasFile(path: string): boolean {
    return this.files.has(path)
  }

  private guardClosed(): void {
    if (this.closed) throw new Error('store is closed')
  }

  private emitDiff(
    oldMap: ReadonlyMap<string, MockEntry>,
    newMap: ReadonlyMap<string, MockEntry>,
    reason: string,
  ): void {
    const now = new Date()
    for (const [path] of newMap) {
      const prev = oldMap.get(path)
      if (prev === undefined) {
        this.emit({ kind: 'created', path: path as Path, reason, when: now })
      }
    }
    for (const [path] of oldMap) {
      if (!newMap.has(path)) {
        this.emit({ kind: 'deleted', path: path as Path, reason, when: now })
      }
    }
  }

  private emit(event: ChangeEvent): void {
    for (const sink of this.sinks) sink(event)
  }
}

class MockBatch implements Batch {
  constructor(private readonly files: Map<string, MockEntry>) {}

  async read(p: Path): Promise<Buffer> {
    validatePath(p)
    const entry = this.files.get(p)
    if (entry === undefined) throw new ErrNotFound(p)
    return Buffer.from(entry.content)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    this.files.set(p, { content: Buffer.from(content), modTime: new Date() })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    const existing = this.files.get(p)
    const merged = existing
      ? Buffer.concat([existing.content, content])
      : Buffer.from(content)
    this.files.set(p, { content: merged, modTime: new Date() })
  }

  async delete(p: Path): Promise<void> {
    validatePath(p)
    if (!this.files.has(p)) throw new ErrNotFound(p)
    this.files.delete(p)
  }

  async rename(src: Path, dst: Path): Promise<void> {
    validatePath(src)
    validatePath(dst)
    const entry = this.files.get(src)
    if (entry === undefined) throw new ErrNotFound(src)
    this.files.delete(src)
    this.files.set(dst, { content: Buffer.from(entry.content), modTime: new Date() })
  }

  async exists(p: Path): Promise<boolean> {
    validatePath(p)
    if (this.files.has(p)) return true
    const prefix = `${p}/`
    for (const key of this.files.keys()) {
      if (key.startsWith(prefix)) return true
    }
    return false
  }

  async stat(p: Path): Promise<FileInfo> {
    validatePath(p)
    const entry = this.files.get(p)
    if (entry === undefined) throw new ErrNotFound(p)
    return { path: p, size: entry.content.length, modTime: entry.modTime, isDir: false }
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    const prefix = dir === '' ? '' : dir.endsWith('/') ? dir : `${dir}/`
    const entries: FileInfo[] = []
    for (const [path, entry] of this.files) {
      if (prefix !== '' && !path.startsWith(prefix)) continue
      const rest = path.slice(prefix.length)
      if (rest === '') continue
      if (opts.recursive === true || !rest.includes('/')) {
        entries.push({
          path: path as Path,
          size: entry.content.length,
          modTime: entry.modTime,
          isDir: false,
        })
      }
    }
    entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return entries
  }
}

// ---------------------------------------------------------------------------
// noopLogger re-export for test convenience.
// ---------------------------------------------------------------------------

export const testLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
}
