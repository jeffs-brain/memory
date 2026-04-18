import { ErrNotFound, ErrReadOnly } from './errors.js'
import {
  isGenerated as pathIsGenerated,
  lastSegment,
  matchGlob,
  validatePath,
  type Path,
} from './path.js'
import type {
  Batch,
  BatchOptions,
  ChangeEvent,
  EventSink,
  FileInfo,
  ListOpts,
  Store,
  Unsubscribe,
} from './index.js'

type Entry = {
  content: Buffer
  modTime: Date
}

export const createMemStore = (): MemStore => new MemStore()

export class MemStore implements Store {
  private files = new Map<Path, Entry>()
  private readonly sinks = new Map<number, EventSink>()
  private nextSinkId = 0
  private closed = false

  async read(p: Path): Promise<Buffer> {
    this.ensureOpen()
    validatePath(p)
    const e = this.files.get(p)
    if (!e) throw new ErrNotFound(p)
    return Buffer.from(e.content)
  }

  async exists(p: Path): Promise<boolean> {
    this.ensureOpen()
    validatePath(p)
    if (this.files.has(p)) return true
    const prefix = `${p}/`
    for (const k of this.files.keys()) if (k.startsWith(prefix)) return true
    return false
  }

  async stat(p: Path): Promise<FileInfo> {
    this.ensureOpen()
    validatePath(p)
    const e = this.files.get(p)
    if (e) {
      return { path: p, size: e.content.length, modTime: e.modTime, isDir: false }
    }
    const prefix = `${p}/`
    for (const k of this.files.keys()) {
      if (k.startsWith(prefix)) {
        return { path: p, size: 0, modTime: new Date(0), isDir: true }
      }
    }
    throw new ErrNotFound(p)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    return listFrom(this.files, dir, opts)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const existed = this.files.has(p)
    this.files.set(p, { content: Buffer.from(content), modTime: new Date() })
    this.emit({
      kind: existed ? 'updated' : 'created',
      path: p,
      when: new Date(),
    })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const existing = this.files.get(p)
    const existed = existing !== undefined
    const merged = existing
      ? Buffer.concat([existing.content, content])
      : Buffer.from(content)
    this.files.set(p, { content: merged, modTime: new Date() })
    this.emit({
      kind: existed ? 'updated' : 'created',
      path: p,
      when: new Date(),
    })
  }

  async delete(p: Path): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    if (!this.files.has(p)) throw new ErrNotFound(p)
    this.files.delete(p)
    this.emit({ kind: 'deleted', path: p, when: new Date() })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    this.ensureOpen()
    validatePath(src)
    validatePath(dst)
    const e = this.files.get(src)
    if (!e) throw new ErrNotFound(src)
    this.files.delete(src)
    this.files.set(dst, { content: Buffer.from(e.content), modTime: new Date() })
    this.emit({ kind: 'renamed', path: dst, oldPath: src, when: new Date() })
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    this.ensureOpen()
    const snapshot = new Map<Path, Entry>()
    for (const [k, v] of this.files) {
      snapshot.set(k, { content: Buffer.from(v.content), modTime: v.modTime })
    }
    const working = new Map<Path, Entry>()
    for (const [k, v] of this.files) {
      working.set(k, { content: Buffer.from(v.content), modTime: v.modTime })
    }
    const batch = new MemBatch(working)
    try {
      await fn(batch)
    } catch (err) {
      // working copy is discarded; live map untouched
      throw err
    }
    const events = diffEvents(snapshot, working, opts.reason)
    this.files = working
    for (const e of events) this.emit(e)
  }

  subscribe(sink: EventSink): Unsubscribe {
    const id = this.nextSinkId++
    this.sinks.set(id, sink)
    return () => {
      this.sinks.delete(id)
    }
  }

  localPath(_p: Path): string | undefined {
    return undefined
  }

  async close(): Promise<void> {
    this.closed = true
    this.sinks.clear()
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }

  private emit(event: ChangeEvent): void {
    for (const sink of this.sinks.values()) sink(event)
  }
}

class MemBatch implements Batch {
  constructor(private readonly files: Map<Path, Entry>) {}

  async read(p: Path): Promise<Buffer> {
    validatePath(p)
    const e = this.files.get(p)
    if (!e) throw new ErrNotFound(p)
    return Buffer.from(e.content)
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
    const e = this.files.get(src)
    if (!e) throw new ErrNotFound(src)
    this.files.delete(src)
    this.files.set(dst, { content: Buffer.from(e.content), modTime: new Date() })
  }

  async exists(p: Path): Promise<boolean> {
    validatePath(p)
    if (this.files.has(p)) return true
    const prefix = `${p}/`
    for (const k of this.files.keys()) if (k.startsWith(prefix)) return true
    return false
  }

  async stat(p: Path): Promise<FileInfo> {
    validatePath(p)
    const e = this.files.get(p)
    if (!e) throw new ErrNotFound(p)
    return { path: p, size: e.content.length, modTime: e.modTime, isDir: false }
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    return listFrom(this.files, dir, opts)
  }
}

const listFrom = (
  files: ReadonlyMap<Path, Entry>,
  dir: Path | '',
  opts: ListOpts,
): FileInfo[] => {
  const prefix = dir === '' ? '' : dir.endsWith('/') ? dir : `${dir}/`
  const recursive = opts.recursive === true
  const includeGenerated = opts.includeGenerated === true
  const glob = opts.glob ?? ''
  const seenDirs = new Set<Path>()
  const result: FileInfo[] = []

  for (const [p, e] of files) {
    const ps = p as string
    if (prefix !== '' && !ps.startsWith(prefix)) continue
    const rest = ps.slice(prefix.length)
    if (rest === '') continue

    if (recursive) {
      if (!includeGenerated && pathIsGenerated(p)) continue
      if (glob !== '' && !matchGlob(glob, lastSegment(rest))) continue
      result.push({ path: p, size: e.content.length, modTime: e.modTime, isDir: false })
      continue
    }

    const slash = rest.indexOf('/')
    if (slash === -1) {
      if (!includeGenerated && pathIsGenerated(p)) continue
      if (glob !== '' && !matchGlob(glob, rest)) continue
      result.push({ path: p, size: e.content.length, modTime: e.modTime, isDir: false })
    } else {
      const childDir = (prefix + rest.slice(0, slash)) as Path
      if (!seenDirs.has(childDir)) {
        seenDirs.add(childDir)
        if (glob !== '' && !matchGlob(glob, rest.slice(0, slash))) continue
        result.push({
          path: childDir,
          size: 0,
          modTime: new Date(0),
          isDir: true,
        })
      }
    }
  }

  result.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
  return result
}

const diffEvents = (
  oldMap: ReadonlyMap<Path, Entry>,
  newMap: ReadonlyMap<Path, Entry>,
  reason: string,
): ChangeEvent[] => {
  const events: ChangeEvent[] = []
  const now = new Date()
  for (const [p, e] of newMap) {
    const prev = oldMap.get(p)
    if (!prev) {
      events.push({ kind: 'created', path: p, reason, when: now })
      continue
    }
    if (!prev.content.equals(e.content)) {
      events.push({ kind: 'updated', path: p, reason, when: now })
    }
  }
  for (const p of oldMap.keys()) {
    if (!newMap.has(p)) events.push({ kind: 'deleted', path: p, reason, when: now })
  }
  return events
}
