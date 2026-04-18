import { randomBytes } from 'node:crypto'
import {
  mkdir,
  open,
  readFile,
  readdir,
  rename as fsRename,
  rm,
  stat as fsStat,
  writeFile,
} from 'node:fs/promises'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { ErrNotFound, ErrReadOnly, StoreError } from './errors.js'
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

export type FsStoreOptions = {
  readonly root: string
}

export const createFsStore = async (opts: FsStoreOptions): Promise<FsStore> => {
  const abs = isAbsolute(opts.root) ? opts.root : resolve(opts.root)
  await mkdir(abs, { recursive: true })
  return new FsStore(abs)
}

export class FsStore implements Store {
  private closed = false
  private readonly sinks = new Map<number, EventSink>()
  private nextSinkId = 0
  // serialises mutations so batch atomicity is not undermined by interleaved
  // single-call writes; reads stay lock-free
  private writeChain: Promise<unknown> = Promise.resolve()

  constructor(readonly root: string) {}

  async read(p: Path): Promise<Buffer> {
    this.ensureOpen()
    validatePath(p)
    const abs = this.resolve(p)
    try {
      return await readFile(abs)
    } catch (err) {
      throw wrapNotFound(p, err)
    }
  }

  async exists(p: Path): Promise<boolean> {
    this.ensureOpen()
    validatePath(p)
    const abs = this.resolve(p)
    try {
      await fsStat(abs)
      return true
    } catch (err) {
      if (isEnoent(err)) return false
      throw err
    }
  }

  async stat(p: Path): Promise<FileInfo> {
    this.ensureOpen()
    validatePath(p)
    const abs = this.resolve(p)
    try {
      const info = await fsStat(abs)
      return {
        path: p,
        size: info.size,
        modTime: info.mtime,
        isDir: info.isDirectory(),
      }
    } catch (err) {
      throw wrapNotFound(p, err)
    }
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    const absDir = dir === '' ? this.root : this.resolve(dir as Path)
    let rootStat
    try {
      rootStat = await fsStat(absDir)
    } catch (err) {
      if (isEnoent(err)) return []
      throw err
    }
    if (!rootStat.isDirectory()) {
      throw new StoreError(`fsstore: list ${dir}: not a directory`)
    }
    const out: FileInfo[] = []
    await this.walk(absDir, dir === '' ? '' : `${dir}/`, opts, out)
    out.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return out
  }

  async write(p: Path, content: Buffer): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      validatePath(p)
      const abs = this.resolve(p)
      let existedBefore = true
      try {
        await fsStat(abs)
      } catch (err) {
        if (isEnoent(err)) existedBefore = false
        else throw err
      }
      await atomicWrite(abs, content)
      this.emit({
        kind: existedBefore ? 'updated' : 'created',
        path: p,
        when: new Date(),
      })
    })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      validatePath(p)
      const abs = this.resolve(p)
      await mkdir(dirname(abs), { recursive: true })
      let existedBefore = true
      try {
        await fsStat(abs)
      } catch (err) {
        if (isEnoent(err)) existedBefore = false
        else throw err
      }
      const handle = await open(abs, 'a')
      try {
        await handle.writeFile(content)
        await handle.sync()
      } finally {
        await handle.close()
      }
      this.emit({
        kind: existedBefore ? 'updated' : 'created',
        path: p,
        when: new Date(),
      })
    })
  }

  async delete(p: Path): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      validatePath(p)
      const abs = this.resolve(p)
      try {
        await rm(abs)
      } catch (err) {
        if (isEnoent(err)) throw new ErrNotFound(p, err)
        throw err
      }
      this.emit({ kind: 'deleted', path: p, when: new Date() })
    })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      validatePath(src)
      validatePath(dst)
      const srcAbs = this.resolve(src)
      const dstAbs = this.resolve(dst)
      try {
        await fsStat(srcAbs)
      } catch (err) {
        if (isEnoent(err)) throw new ErrNotFound(src, err)
        throw err
      }
      await mkdir(dirname(dstAbs), { recursive: true })
      await fsRename(srcAbs, dstAbs)
      this.emit({
        kind: 'renamed',
        path: dst,
        oldPath: src,
        when: new Date(),
      })
    })
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      const journal: JournalOp[] = []
      const batch = new FsBatch(this, journal)
      try {
        await fn(batch)
      } catch (err) {
        throw err
      }
      await this.commitJournal(journal, opts.reason)
    })
  }

  subscribe(sink: EventSink): Unsubscribe {
    const id = this.nextSinkId++
    this.sinks.set(id, sink)
    return () => {
      this.sinks.delete(id)
    }
  }

  localPath(p: Path): string | undefined {
    try {
      return this.resolve(p)
    } catch {
      return undefined
    }
  }

  async close(): Promise<void> {
    this.closed = true
    this.sinks.clear()
  }

  // internal helpers used by batch

  resolve(p: Path): string {
    return join(this.root, p)
  }

  emit(event: ChangeEvent): void {
    for (const sink of this.sinks.values()) sink(event)
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }

  private async serialise<T>(fn: () => Promise<T>): Promise<T> {
    const prior = this.writeChain.catch(() => undefined)
    const next = prior.then(fn)
    this.writeChain = next.catch(() => undefined)
    return next
  }

  private async walk(
    absDir: string,
    logicalPrefix: string,
    opts: ListOpts,
    out: FileInfo[],
  ): Promise<void> {
    const entries = await readdir(absDir, { withFileTypes: true })
    const includeGenerated = opts.includeGenerated === true
    const glob = opts.glob ?? ''
    const recursive = opts.recursive === true
    for (const entry of entries) {
      const name = entry.name
      if (entry.isDirectory()) {
        if (shouldSkipDir(name)) continue
        const childLogical = `${logicalPrefix}${name}` as Path
        if (recursive) {
          await this.walk(join(absDir, name), `${logicalPrefix}${name}/`, opts, out)
          continue
        }
        if (glob !== '' && !matchGlob(glob, name)) continue
        out.push({
          path: childLogical,
          size: 0,
          modTime: new Date(0),
          isDir: true,
        })
        continue
      }
      if (!entry.isFile()) continue
      if (shouldSkipFile(name)) continue
      const childLogical = `${logicalPrefix}${name}` as Path
      if (!includeGenerated && pathIsGenerated(childLogical)) continue
      if (glob !== '' && !matchGlob(glob, name)) continue
      const info = await fsStat(join(absDir, name))
      out.push({
        path: childLogical,
        size: info.size,
        modTime: info.mtime,
        isDir: false,
      })
    }
  }

  private async commitJournal(journal: JournalOp[], reason: string): Promise<void> {
    if (journal.length === 0) return
    const { plan, events } = this.planJournal(journal, reason)
    // snapshot prior on-disk state so a mid-commit failure can roll back
    const rollback: RollbackStep[] = []
    try {
      for (const step of plan) {
        if (step.kind === 'write') {
          const abs = this.resolve(step.path)
          const existed = await readFileOrUndefined(abs)
          rollback.push(
            existed === undefined
              ? { kind: 'unlink', abs }
              : { kind: 'restore', abs, content: existed },
          )
          await atomicWrite(abs, step.content)
        } else if (step.kind === 'delete') {
          const abs = this.resolve(step.path)
          const existed = await readFileOrUndefined(abs)
          if (existed === undefined) continue
          rollback.push({ kind: 'restore', abs, content: existed })
          await rm(abs)
        }
      }
    } catch (err) {
      await applyRollback(rollback)
      throw err
    }
    for (const evt of events) this.emit(evt)
  }

  private planJournal(
    journal: JournalOp[],
    reason: string,
  ): { plan: PlanStep[]; events: ChangeEvent[] } {
    // Build ordered list of touched paths (first-mention order) and compute
    // effective state per path by replaying the journal.
    const order: Path[] = []
    const touched = new Set<Path>()
    for (const op of journal) {
      const paths: Path[] = []
      if (op.kind === 'rename') paths.push(op.src, op.dst)
      else paths.push(op.path)
      for (const p of paths) {
        if (!touched.has(p)) {
          touched.add(p)
          order.push(p)
        }
      }
    }

    const plan: PlanStep[] = []
    const events: ChangeEvent[] = []
    const now = new Date()
    for (const p of order) {
      const state = effectiveState(journal, p)
      if (state.kind === 'present') {
        plan.push({ kind: 'write', path: p, content: state.content })
        events.push({
          kind: state.wasPresentBefore ? 'updated' : 'created',
          path: p,
          reason,
          when: now,
        })
      } else if (state.kind === 'absent-was-present') {
        plan.push({ kind: 'delete', path: p })
        events.push({ kind: 'deleted', path: p, reason, when: now })
      }
      // absent-and-was-absent: nothing to do
    }
    return { plan, events }
  }
}

type JournalOp =
  | { kind: 'write'; path: Path; content: Buffer }
  | { kind: 'append'; path: Path; content: Buffer }
  | { kind: 'delete'; path: Path }
  | { kind: 'rename'; src: Path; dst: Path }

type PlanStep =
  | { kind: 'write'; path: Path; content: Buffer }
  | { kind: 'delete'; path: Path }

type RollbackStep =
  | { kind: 'restore'; abs: string; content: Buffer }
  | { kind: 'unlink'; abs: string }

class FsBatch implements Batch {
  constructor(
    private readonly store: FsStore,
    private readonly journal: JournalOp[],
  ) {}

  async read(p: Path): Promise<Buffer> {
    validatePath(p)
    const state = effectiveState(this.journal, p)
    if (state.kind === 'present') return Buffer.from(state.content)
    if (state.kind === 'absent-was-present') throw new ErrNotFound(p)
    // untouched by journal: fall through to store
    try {
      return await this.store.read(p)
    } catch (err) {
      if (err instanceof ErrNotFound) throw err
      throw err
    }
  }

  async write(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    this.journal.push({ kind: 'write', path: p, content: Buffer.from(content) })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    // materialise append against current effective state so the journal plan
    // can be executed as a simple series of writes/deletes.
    const state = effectiveState(this.journal, p)
    let base: Buffer
    if (state.kind === 'present') {
      base = state.content
    } else if (state.kind === 'absent-was-present') {
      base = Buffer.alloc(0)
    } else {
      try {
        base = await this.store.read(p)
      } catch (err) {
        if (err instanceof ErrNotFound) base = Buffer.alloc(0)
        else throw err
      }
    }
    this.journal.push({
      kind: 'write',
      path: p,
      content: Buffer.concat([base, content]),
    })
  }

  async delete(p: Path): Promise<void> {
    validatePath(p)
    const state = effectiveState(this.journal, p)
    if (state.kind === 'present') {
      this.journal.push({ kind: 'delete', path: p })
      return
    }
    if (state.kind === 'absent-was-present') {
      throw new ErrNotFound(p)
    }
    const exists = await this.store.exists(p)
    if (!exists) throw new ErrNotFound(p)
    this.journal.push({ kind: 'delete', path: p })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    validatePath(src)
    validatePath(dst)
    const srcState = await this.loadEffective(src)
    if (srcState === undefined) throw new ErrNotFound(src)
    this.journal.push({ kind: 'write', path: dst, content: srcState })
    this.journal.push({ kind: 'delete', path: src })
  }

  async exists(p: Path): Promise<boolean> {
    validatePath(p)
    const state = effectiveState(this.journal, p)
    if (state.kind === 'present') return true
    if (state.kind === 'absent-was-present') return false
    return this.store.exists(p)
  }

  async stat(p: Path): Promise<FileInfo> {
    validatePath(p)
    const state = effectiveState(this.journal, p)
    if (state.kind === 'present') {
      return {
        path: p,
        size: state.content.length,
        modTime: new Date(),
        isDir: false,
      }
    }
    if (state.kind === 'absent-was-present') throw new ErrNotFound(p)
    return this.store.stat(p)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    const base = await this.store.list(dir, opts)
    const byPath = new Map<Path, FileInfo>()
    for (const fi of base) byPath.set(fi.path, fi)
    const touched = new Set<Path>()
    for (const op of this.journal) {
      if (op.kind === 'rename') {
        touched.add(op.src)
        touched.add(op.dst)
      } else {
        touched.add(op.path)
      }
    }
    const recursive = opts.recursive === true
    const includeGenerated = opts.includeGenerated === true
    const glob = opts.glob ?? ''
    for (const p of touched) {
      if (!pathUnderLocal(p, dir, recursive)) continue
      const state = effectiveState(this.journal, p)
      if (state.kind === 'present') {
        if (!includeGenerated && pathIsGenerated(p)) {
          byPath.delete(p)
          continue
        }
        if (glob !== '' && !matchGlob(glob, lastSegment(p))) continue
        byPath.set(p, {
          path: p,
          size: state.content.length,
          modTime: new Date(),
          isDir: false,
        })
      } else if (state.kind === 'absent-was-present') {
        byPath.delete(p)
      }
    }
    const result = Array.from(byPath.values())
    result.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return result
  }

  private async loadEffective(p: Path): Promise<Buffer | undefined> {
    const state = effectiveState(this.journal, p)
    if (state.kind === 'present') return state.content
    if (state.kind === 'absent-was-present') return undefined
    try {
      return await this.store.read(p)
    } catch (err) {
      if (err instanceof ErrNotFound) return undefined
      throw err
    }
  }
}

type EffectiveState =
  | { kind: 'present'; content: Buffer; wasPresentBefore: boolean }
  | { kind: 'absent-was-present' }
  | { kind: 'untouched' }

// effectiveState replays the journal top-to-bottom for a single path and
// returns its final state. Ops that target other paths are ignored. The
// only op kinds in the journal at this layer are 'write' and 'delete';
// append and rename are materialised at push time.
const effectiveState = (journal: readonly JournalOp[], p: Path): EffectiveState => {
  let present = false
  let content: Buffer = Buffer.alloc(0)
  let touched = false
  for (const op of journal) {
    if (op.kind === 'write' && op.path === p) {
      present = true
      content = op.content
      touched = true
    } else if (op.kind === 'delete' && op.path === p) {
      present = false
      content = Buffer.alloc(0)
      touched = true
    }
  }
  if (!touched) return { kind: 'untouched' }
  if (present) {
    // We do not know wasPresentBefore from the journal alone; plan caller
    // determines via the on-disk snapshot. This flag is filled in later.
    return { kind: 'present', content, wasPresentBefore: false }
  }
  return { kind: 'absent-was-present' }
}

const pathUnderLocal = (p: Path, dir: Path | '', recursive: boolean): boolean => {
  if (dir === '') return true
  const ps = p as string
  const ds = dir as string
  if (ps === ds) return true
  if (!(ps.length > ds.length && ps.startsWith(ds) && ps[ds.length] === '/')) return false
  if (recursive) return true
  const rest = ps.slice(ds.length + 1)
  return !rest.includes('/')
}

const isEnoent = (err: unknown): boolean =>
  typeof err === 'object' && err !== null && (err as { code?: string }).code === 'ENOENT'

const wrapNotFound = (p: Path, err: unknown): Error => {
  if (isEnoent(err)) return new ErrNotFound(p, err)
  return err instanceof Error ? err : new StoreError(String(err))
}

const readFileOrUndefined = async (abs: string): Promise<Buffer | undefined> => {
  try {
    return await readFile(abs)
  } catch (err) {
    if (isEnoent(err)) return undefined
    throw err
  }
}

const applyRollback = async (steps: readonly RollbackStep[]): Promise<void> => {
  // walk in reverse so nested writes restore consistently
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i]
    if (!step) continue
    try {
      if (step.kind === 'restore') {
        await atomicWrite(step.abs, step.content)
      } else {
        await rm(step.abs).catch((err) => {
          if (!isEnoent(err)) throw err
        })
      }
    } catch {
      // best-effort rollback; first failure already threw
    }
  }
}

// atomicWrite writes content to abs via tmp + fsync + rename. Readers see
// either the old bytes or the new bytes, never a torn write.
export const atomicWrite = async (abs: string, content: Buffer): Promise<void> => {
  const dir = dirname(abs)
  await mkdir(dir, { recursive: true })
  const tmpName = `.brain-tmp-${randomBytes(8).toString('hex')}`
  const tmpAbs = join(dir, tmpName)
  let wrote = false
  try {
    const handle = await open(tmpAbs, 'wx', 0o644)
    try {
      await handle.writeFile(content)
      await handle.sync()
    } finally {
      await handle.close()
    }
    wrote = true
    await fsRename(tmpAbs, abs)
    // fsync the parent dir so the rename itself is durable; best-effort
    try {
      const dirHandle = await open(dir, 'r')
      try {
        await dirHandle.sync()
      } finally {
        await dirHandle.close()
      }
    } catch {
      // directory fsync unsupported on some filesystems; ignore
    }
  } catch (err) {
    if (wrote) {
      await rm(tmpAbs).catch(() => undefined)
    } else {
      await rm(tmpAbs).catch(() => undefined)
    }
    throw err
  }
}

const shouldSkipDir = (name: string): boolean => {
  if (name === '.git') return true
  if (name.startsWith('.brain-staging')) return true
  return false
}

const shouldSkipFile = (name: string): boolean => {
  if (name === '.git') return true
  if (name.startsWith('.brain-tmp-')) return true
  return false
}

// fallback writeFile export kept so tests can patch behaviour if needed
export const _unsafe = { writeFile, rm }
