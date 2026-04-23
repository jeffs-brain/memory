export type Path = string & { readonly __brand: 'BrainPath' }

export type FileInfo = {
  readonly path: Path
  readonly size: number
  readonly modTime: Date
  readonly isDir: boolean
}

export type ListOpts = {
  readonly recursive?: boolean
  readonly glob?: string
  readonly includeGenerated?: boolean
}

export type BatchOptions = {
  readonly reason: string
}

export type ChangeKind = 'created' | 'updated' | 'deleted' | 'renamed'

export type ChangeEvent = {
  readonly kind: ChangeKind
  readonly path: Path
  readonly oldPath?: Path
  readonly reason?: string
  readonly when: Date
}

export type EventSink = (event: ChangeEvent) => void

export type Unsubscribe = () => void

export type PortableFileInfo = {
  readonly uri: string
  readonly size: number
  readonly modTime: Date
  readonly isDir: boolean
}

export type FileAdapter = {
  readonly kind: string
  join(root: string, path: string): string
  ensureDirectory(uri: string): Promise<void>
  readText(uri: string): Promise<string>
  writeText(uri: string, content: string): Promise<void>
  appendText(uri: string, content: string): Promise<void>
  delete(uri: string): Promise<void>
  rename(from: string, to: string): Promise<void>
  exists(uri: string): Promise<boolean>
  stat(uri: string): Promise<PortableFileInfo>
  list(uri: string): Promise<readonly PortableFileInfo[]>
  basename(uri: string): string
}

export type Batch = {
  read(path: Path): Promise<string>
  write(path: Path, content: string): Promise<void>
  append(path: Path, content: string): Promise<void>
  delete(path: Path): Promise<void>
  rename(src: Path, dst: Path): Promise<void>
  exists(path: Path): Promise<boolean>
  stat(path: Path): Promise<FileInfo>
  list(dir: Path | '', opts?: ListOpts): Promise<FileInfo[]>
}

export type Store = {
  readonly root: string
  read(path: Path): Promise<string>
  write(path: Path, content: string): Promise<void>
  append(path: Path, content: string): Promise<void>
  delete(path: Path): Promise<void>
  rename(src: Path, dst: Path): Promise<void>
  exists(path: Path): Promise<boolean>
  stat(path: Path): Promise<FileInfo>
  list(dir: Path | '', opts?: ListOpts): Promise<FileInfo[]>
  batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void>
  subscribe(sink: EventSink): Unsubscribe
  localPath(path: Path): string | undefined
  close(): Promise<void>
}

export class StoreError extends Error {
  override readonly name: string = 'StoreError'

  constructor(
    message: string,
    override readonly cause?: unknown,
  ) {
    super(message)
  }
}

export class ErrNotFound extends StoreError {
  override readonly name = 'ErrNotFound'

  constructor(path: string, cause?: unknown) {
    super(`brain: not found: ${path}`, cause)
  }
}

export class ErrReadOnly extends StoreError {
  override readonly name = 'ErrReadOnly'

  constructor() {
    super('brain: read-only')
  }
}

export class ErrInvalidPath extends StoreError {
  override readonly name = 'ErrInvalidPath'

  constructor(reason: string) {
    super(`brain: invalid path: ${reason}`)
  }
}

export class ErrPayloadTooLarge extends StoreError {
  override readonly name = 'ErrPayloadTooLarge'

  constructor(reason: string, cause?: unknown) {
    super(`brain: payload too large: ${reason}`, cause)
  }
}

export const isNotFound = (err: unknown): err is ErrNotFound => err instanceof ErrNotFound
export const isInvalidPath = (err: unknown): err is ErrInvalidPath => err instanceof ErrInvalidPath

export const validatePath = (p: string): void => {
  if (p === '') throw new ErrInvalidPath('empty path')
  if (p.includes('\0')) throw new ErrInvalidPath(`contains null byte: ${JSON.stringify(p)}`)
  if (p.includes('\\')) throw new ErrInvalidPath(`contains backslash: ${p}`)
  if (p.startsWith('/')) throw new ErrInvalidPath(`leading slash: ${p}`)
  if (p.endsWith('/')) throw new ErrInvalidPath(`trailing slash: ${p}`)
  const cleaned = cleanPosix(p)
  if (cleaned !== p) throw new ErrInvalidPath(`non-canonical: ${p}`)
  for (const part of p.split('/')) {
    if (part === '..') throw new ErrInvalidPath(`contains ..: ${p}`)
    if (part === '.') throw new ErrInvalidPath(`contains .: ${p}`)
    if (part === '') throw new ErrInvalidPath(`contains empty segment: ${p}`)
  }
}

export const toPath = (p: string): Path => {
  validatePath(p)
  return p as Path
}

export const validatePathSegment = (segment: string): void => {
  if (segment === '') throw new ErrInvalidPath('empty path segment')
  if (segment.includes('\0')) {
    throw new ErrInvalidPath(`path segment contains null byte: ${JSON.stringify(segment)}`)
  }
  if (segment.includes('/')) throw new ErrInvalidPath(`path segment contains slash: ${segment}`)
  if (segment.includes('\\')) {
    throw new ErrInvalidPath(`path segment contains backslash: ${segment}`)
  }
  if (segment === '.' || segment === '..') {
    throw new ErrInvalidPath(`path segment is not canonical: ${segment}`)
  }
}

export const joinPath = (...parts: readonly string[]): Path => {
  const joined = parts.filter((part) => part.length > 0).join('/')
  return toPath(cleanPosix(joined))
}

export const lastSegment = (p: string): string => {
  const idx = p.lastIndexOf('/')
  return idx === -1 ? p : p.slice(idx + 1)
}

export const isGenerated = (p: Path | string): boolean => lastSegment(p).startsWith('_')

export const matchGlob = (pattern: string, name: string): boolean => {
  return matchGlobAt(pattern, 0, name, 0)
}

const cleanPosix = (p: string): string => {
  if (p === '') return '.'
  const rooted = p.startsWith('/')
  const segments = p.split('/')
  const stack: string[] = []
  for (const segment of segments) {
    if (segment === '' || segment === '.') continue
    if (segment === '..') {
      if (stack.length > 0 && stack[stack.length - 1] !== '..') {
        stack.pop()
      } else if (!rooted) {
        stack.push('..')
      }
      continue
    }
    stack.push(segment)
  }
  const out = stack.join('/')
  if (rooted) return `/${out}`
  return out === '' ? '.' : out
}

const matchGlobAt = (pattern: string, pi: number, name: string, ni: number): boolean => {
  let patternIndex = pi
  let nameIndex = ni
  while (patternIndex < pattern.length) {
    const pc = pattern[patternIndex]
    if (pc === '*') {
      while (patternIndex < pattern.length && pattern[patternIndex] === '*') patternIndex += 1
      if (patternIndex === pattern.length) return true
      for (let candidate = nameIndex; candidate <= name.length; candidate += 1) {
        if (matchGlobAt(pattern, patternIndex, name, candidate)) return true
      }
      return false
    }
    if (pc === '?') {
      if (nameIndex >= name.length) return false
      patternIndex += 1
      nameIndex += 1
      continue
    }
    if (pc === '[') {
      const close = pattern.indexOf(']', patternIndex + 1)
      if (close === -1 || nameIndex >= name.length) return false
      const set = pattern.slice(patternIndex + 1, close)
      const target = name[nameIndex] as string
      let negate = false
      let idx = 0
      if (set[0] === '!' || set[0] === '^') {
        negate = true
        idx = 1
      }
      let matched = false
      while (idx < set.length) {
        const left = set[idx] as string
        if (idx + 2 < set.length && set[idx + 1] === '-') {
          const right = set[idx + 2] as string
          if (target >= left && target <= right) matched = true
          idx += 3
        } else {
          if (target === left) matched = true
          idx += 1
        }
      }
      if (matched === negate) return false
      patternIndex = close + 1
      nameIndex += 1
      continue
    }
    if (nameIndex >= name.length || name[nameIndex] !== pc) return false
    patternIndex += 1
    nameIndex += 1
  }
  return nameIndex === name.length
}

type JournalOp =
  | { readonly kind: 'write'; readonly path: Path; readonly content: string }
  | { readonly kind: 'append'; readonly path: Path; readonly content: string }
  | { readonly kind: 'delete'; readonly path: Path }
  | { readonly kind: 'rename'; readonly src: Path; readonly dst: Path }

export const createMobileStore = async (args: {
  root: string
  adapter: FileAdapter
}): Promise<Store> => {
  await args.adapter.ensureDirectory(args.root)
  return new MobileStore(args.root, args.adapter)
}

class MobileStore implements Store {
  private closed = false
  private nextSinkId = 0
  private readonly sinks = new Map<number, EventSink>()
  private writeChain: Promise<unknown> = Promise.resolve()

  constructor(
    readonly root: string,
    private readonly adapter: FileAdapter,
  ) {}

  async read(path: Path): Promise<string> {
    this.ensureOpen()
    validatePath(path)
    const uri = this.resolve(path)
    try {
      return await this.adapter.readText(uri)
    } catch (error) {
      throw wrapNotFound(path, error)
    }
  }

  async write(path: Path, content: string): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      await this.applyWrite(path, content)
    })
  }

  async append(path: Path, content: string): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      await this.applyAppend(path, content)
    })
  }

  async delete(path: Path): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      await this.applyDelete(path)
    })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      await this.applyRename(src, dst)
    })
  }

  async exists(path: Path): Promise<boolean> {
    this.ensureOpen()
    validatePath(path)
    return await this.adapter.exists(this.resolve(path))
  }

  async stat(path: Path): Promise<FileInfo> {
    this.ensureOpen()
    validatePath(path)
    try {
      const info = await this.adapter.stat(this.resolve(path))
      return {
        path,
        size: info.size,
        modTime: info.modTime,
        isDir: info.isDir,
      }
    } catch (error) {
      throw wrapNotFound(path, error)
    }
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    const basePath = dir === '' ? undefined : dir
    const baseUri = basePath === undefined ? this.root : this.resolve(basePath)
    const exists = dir === '' ? true : await this.adapter.exists(baseUri)
    if (!exists) return []
    const results: FileInfo[] = []
    await this.walk(basePath ?? '', baseUri, opts, results)
    results.sort((left, right) => left.path.localeCompare(right.path))
    return results
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    return this.serialise(async () => {
      this.ensureOpen()
      const journal: JournalOp[] = []
      const batch: Batch = {
        read: async (path) => this.read(path),
        write: async (path, content) => {
          journal.push({ kind: 'write', path, content })
        },
        append: async (path, content) => {
          journal.push({ kind: 'append', path, content })
        },
        delete: async (path) => {
          journal.push({ kind: 'delete', path })
        },
        rename: async (src, dst) => {
          journal.push({ kind: 'rename', src, dst })
        },
        exists: async (path) => this.exists(path),
        stat: async (path) => this.stat(path),
        list: async (path, innerOpts) => this.list(path, innerOpts),
      }
      await fn(batch)
      await this.commitJournal(journal, opts.reason)
    })
  }

  subscribe(sink: EventSink): Unsubscribe {
    const id = this.nextSinkId
    this.nextSinkId += 1
    this.sinks.set(id, sink)
    return () => {
      this.sinks.delete(id)
    }
  }

  localPath(path: Path): string | undefined {
    return this.resolve(path)
  }

  async close(): Promise<void> {
    this.closed = true
    this.sinks.clear()
  }

  private async walk(
    logicalDir: Path | '',
    uri: string,
    opts: ListOpts,
    out: FileInfo[],
  ): Promise<void> {
    const entries = await this.adapter.list(uri)
    const recursive = opts.recursive === true
    const includeGenerated = opts.includeGenerated === true
    const glob = opts.glob ?? ''

    for (const entry of entries) {
      const name = this.adapter.basename(entry.uri)
      if (entry.isDir) {
        const childPath = logicalDir === '' ? toPath(name) : joinPath(logicalDir, name)
        if (recursive) {
          await this.walk(childPath, entry.uri, opts, out)
          continue
        }
        if (glob !== '' && !matchGlob(glob, name)) continue
        out.push({
          path: childPath,
          size: entry.size,
          modTime: entry.modTime,
          isDir: true,
        })
        continue
      }

      const childPath = logicalDir === '' ? toPath(name) : joinPath(logicalDir, name)
      if (!includeGenerated && isGenerated(childPath)) continue
      if (glob !== '' && !matchGlob(glob, name)) continue
      out.push({
        path: childPath,
        size: entry.size,
        modTime: entry.modTime,
        isDir: false,
      })
    }
  }

  private async commitJournal(journal: readonly JournalOp[], reason: string): Promise<void> {
    const events: ChangeEvent[] = []
    for (const op of journal) {
      switch (op.kind) {
        case 'write':
          events.push(await this.applyWrite(op.path, op.content, reason))
          break
        case 'append':
          events.push(await this.applyAppend(op.path, op.content, reason))
          break
        case 'delete':
          events.push(await this.applyDelete(op.path, reason))
          break
        case 'rename':
          events.push(await this.applyRename(op.src, op.dst, reason))
          break
      }
    }
    for (const event of events) {
      this.emit(event)
    }
  }

  private resolve(path: Path): string {
    return this.adapter.join(this.root, path)
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }

  private emit(event: ChangeEvent): void {
    for (const sink of this.sinks.values()) sink(event)
  }

  private async applyWrite(path: Path, content: string, reason?: string): Promise<ChangeEvent> {
    validatePath(path)
    const uri = this.resolve(path)
    const existed = await this.adapter.exists(uri)
    await this.adapter.ensureDirectory(parentUri(uri))
    await this.adapter.writeText(uri, content)
    const event: ChangeEvent = {
      kind: existed ? 'updated' : 'created',
      path,
      when: new Date(),
      ...(reason === undefined ? {} : { reason }),
    }
    if (reason === undefined) this.emit(event)
    return event
  }

  private async applyAppend(path: Path, content: string, reason?: string): Promise<ChangeEvent> {
    validatePath(path)
    const uri = this.resolve(path)
    const existed = await this.adapter.exists(uri)
    await this.adapter.ensureDirectory(parentUri(uri))
    await this.adapter.appendText(uri, content)
    const event: ChangeEvent = {
      kind: existed ? 'updated' : 'created',
      path,
      when: new Date(),
      ...(reason === undefined ? {} : { reason }),
    }
    if (reason === undefined) this.emit(event)
    return event
  }

  private async applyDelete(path: Path, reason?: string): Promise<ChangeEvent> {
    validatePath(path)
    const uri = this.resolve(path)
    if (!(await this.adapter.exists(uri))) {
      throw new ErrNotFound(path)
    }
    await this.adapter.delete(uri)
    const event: ChangeEvent = {
      kind: 'deleted',
      path,
      when: new Date(),
      ...(reason === undefined ? {} : { reason }),
    }
    if (reason === undefined) this.emit(event)
    return event
  }

  private async applyRename(src: Path, dst: Path, reason?: string): Promise<ChangeEvent> {
    validatePath(src)
    validatePath(dst)
    const sourceUri = this.resolve(src)
    const targetUri = this.resolve(dst)
    if (!(await this.adapter.exists(sourceUri))) {
      throw new ErrNotFound(src)
    }
    await this.adapter.ensureDirectory(parentUri(targetUri))
    await this.adapter.rename(sourceUri, targetUri)
    const event: ChangeEvent = {
      kind: 'renamed',
      path: dst,
      oldPath: src,
      when: new Date(),
      ...(reason === undefined ? {} : { reason }),
    }
    if (reason === undefined) this.emit(event)
    return event
  }

  private async serialise<T>(fn: () => Promise<T>): Promise<T> {
    const prior = this.writeChain.catch(() => undefined)
    const next = prior.then(fn)
    this.writeChain = next.catch(() => undefined)
    return next
  }
}

const wrapNotFound = (path: string, error: unknown): ErrNotFound => new ErrNotFound(path, error)

const parentUri = (uri: string): string => {
  const trimmed = uri.endsWith('/') ? uri.slice(0, -1) : uri
  const idx = trimmed.lastIndexOf('/')
  return idx === -1 ? trimmed : trimmed.slice(0, idx)
}
