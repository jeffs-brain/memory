import { iterateSSE } from '../llm/sse.js'
import {
  type Batch,
  type BatchOptions,
  type ChangeEvent,
  ErrInvalidPath,
  ErrNotFound,
  ErrPayloadTooLarge,
  ErrReadOnly,
  type EventSink,
  type FileInfo,
  type ListOpts,
  type Path,
  type Store,
  StoreError,
  type Unsubscribe,
  lastSegment,
  matchGlob,
  isGenerated as pathIsGenerated,
  toPath,
  validatePath,
} from './index.js'
import { type DocumentBodyLimits, normaliseDocumentBodyLimits } from './limits.js'

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

export type HttpStoreOptions = {
  readonly baseUrl: string
  readonly brainId: string
  readonly apiKey?: string
  readonly token?: string
  readonly fetch?: FetchLike
  readonly timeoutMs?: number
  readonly bodyLimits?: Partial<DocumentBodyLimits>
}

export type HttpStoreCapabilities = {
  readonly supportsLocalPath: false
  readonly synchronousEvents: false
  readonly bodyLimits: DocumentBodyLimits
}

type RawListEntry = {
  readonly path: string
  readonly size: number
  readonly mtime: string
  readonly is_dir: boolean
}

type RawStat = RawListEntry

type RawChangeEvent = {
  readonly kind: 'created' | 'updated' | 'deleted' | 'renamed'
  readonly path: string
  readonly old_path?: string
  readonly reason?: string
  readonly when: string
}

type ProblemPayload = {
  readonly status?: number
  readonly title?: string
  readonly detail?: string
  readonly code?: string
}

type HttpStoreDeps = {
  readonly baseUrl: string
  readonly brainId: string
  readonly fetch: FetchLike
  readonly timeoutMs: number
  readonly authHeader: string | undefined
  readonly bodyLimits: DocumentBodyLimits
}

type JournalOp =
  | { readonly kind: 'write'; readonly path: Path; readonly content: string }
  | { readonly kind: 'append'; readonly path: Path; readonly content: string }
  | { readonly kind: 'delete'; readonly path: Path }
  | { readonly kind: 'rename'; readonly src: Path; readonly dst: Path }

type BatchState =
  | { readonly kind: 'present'; readonly content: string }
  | { readonly kind: 'deleted' }
  | { readonly kind: 'untouched' }

const DEFAULT_TIMEOUT_MS = 30_000
const DEFAULT_USER_AGENT = '@jeffs-brain/memory-react-native (+HttpStore)'
const encoder = new TextEncoder()
const decoder = new TextDecoder('utf-8')

const resolveFetch = (fetchImpl: FetchLike | undefined): FetchLike => {
  if (fetchImpl !== undefined) return fetchImpl
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as FetchLike
  }
  throw new Error('createHttpStore: no global fetch is available; inject one via the fetch option')
}

const joinUrl = (base: string, path: string): string => {
  const trimmed = base.endsWith('/') ? base.slice(0, -1) : base
  const rooted = path.startsWith('/') ? path : `/${path}`
  return `${trimmed}${rooted}`
}

const appendQuery = (
  url: string,
  query: Readonly<Record<string, string | number | boolean | undefined>>,
): string => {
  const entries = Object.entries(query).filter(([, value]) => value !== undefined)
  if (entries.length === 0) return url
  const params = new URLSearchParams()
  for (const [key, value] of entries) {
    params.append(key, String(value))
  }
  return `${url}${url.includes('?') ? '&' : '?'}${params.toString()}`
}

const parseProblem = async (response: Response): Promise<ProblemPayload> => {
  try {
    return (await response.json()) as ProblemPayload
  } catch {
    return { status: response.status, title: response.statusText }
  }
}

const throwForResponse = async (response: Response, path: string | undefined): Promise<never> => {
  const problem = await parseProblem(response)
  if (response.status === 404) throw new ErrNotFound(path ?? '', problem)
  if (response.status === 400) {
    throw new ErrInvalidPath(problem.detail ?? problem.title ?? 'bad request')
  }
  if (response.status === 413) {
    throw new ErrPayloadTooLarge(problem.detail ?? problem.title ?? 'payload exceeds server limit')
  }
  throw new StoreError(
    `http-store: ${response.status} ${problem.title ?? response.statusText}${problem.detail ? `: ${problem.detail}` : ''}`,
    problem,
  )
}

const buildDeps = (options: HttpStoreOptions): HttpStoreDeps => {
  const authHeader =
    options.apiKey !== undefined && options.apiKey !== ''
      ? `Bearer ${options.apiKey}`
      : options.token !== undefined && options.token !== ''
        ? `Bearer ${options.token}`
        : undefined

  return {
    baseUrl: options.baseUrl,
    brainId: options.brainId,
    fetch: resolveFetch(options.fetch),
    timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    authHeader,
    bodyLimits: normaliseDocumentBodyLimits(options.bodyLimits),
  }
}

const brainPath = (brainId: string, suffix: string): string =>
  `/v1/brains/${encodeURIComponent(brainId)}${suffix}`

const doFetch = async (
  deps: HttpStoreDeps,
  init: {
    readonly method: string
    readonly path: string
    readonly query?: Record<string, string | number | boolean | undefined>
    readonly body?: ArrayBuffer | string
    readonly accept?: string
    readonly contentType?: string
    readonly signal?: AbortSignal
  },
): Promise<Response> => {
  const url = appendQuery(joinUrl(deps.baseUrl, init.path), init.query ?? {})
  const headers = new Headers()
  headers.set('user-agent', DEFAULT_USER_AGENT)
  headers.set('accept', init.accept ?? 'application/json')
  if (init.contentType !== undefined) headers.set('content-type', init.contentType)
  if (deps.authHeader !== undefined) headers.set('authorization', deps.authHeader)

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('request timed out')), deps.timeoutMs)
  const onExternalAbort = (): void => controller.abort(init.signal?.reason)
  if (init.signal !== undefined) {
    if (init.signal.aborted) controller.abort(init.signal.reason)
    else init.signal.addEventListener('abort', onExternalAbort, { once: true })
  }
  const body =
    init.body === undefined
      ? undefined
      : typeof init.body === 'string'
        ? init.body
        : new Blob([init.body])

  try {
    return await deps.fetch(url, {
      method: init.method,
      headers,
      ...(body === undefined ? {} : { body }),
      signal: controller.signal,
    })
  } finally {
    clearTimeout(timer)
    if (init.signal !== undefined) init.signal.removeEventListener('abort', onExternalAbort)
  }
}

const toFileInfo = (raw: RawListEntry | RawStat): FileInfo => ({
  path: toPath(raw.path),
  size: raw.size,
  modTime: new Date(raw.mtime),
  isDir: raw.is_dir,
})

const toChangeEvent = (raw: RawChangeEvent): ChangeEvent => ({
  kind: raw.kind,
  path: toPath(raw.path),
  ...(raw.old_path === undefined ? {} : { oldPath: toPath(raw.old_path) }),
  ...(raw.reason === undefined ? {} : { reason: raw.reason }),
  when: new Date(raw.when),
})

const subscribeSse = (
  deps: HttpStoreDeps,
  onEvent: (event: ChangeEvent) => void,
  onError: (error: unknown) => void,
): (() => void) => {
  const controller = new AbortController()
  let closed = false

  const run = async (): Promise<void> => {
    const response = await doFetch(deps, {
      method: 'GET',
      path: brainPath(deps.brainId, '/events'),
      accept: 'text/event-stream',
      signal: controller.signal,
    })
    if (!response.ok) {
      onError(new StoreError(`http-store: events returned HTTP ${response.status}`))
      return
    }
    if (response.body === null) return

    try {
      for await (const event of iterateSSE(response.body)) {
        if (event.event !== 'change' || event.data === '') continue
        try {
          onEvent(toChangeEvent(JSON.parse(event.data) as RawChangeEvent))
        } catch {
          // Ignore malformed frames.
        }
      }
    } catch (error) {
      if (!closed) onError(error)
    }
  }

  void run().catch((error) => {
    if (!closed) onError(error)
  })

  return () => {
    closed = true
    controller.abort()
  }
}

export const createHttpStore = (options: HttpStoreOptions): HttpStore => new HttpStore(options)

export class HttpStore implements Store {
  readonly root: string
  readonly capabilities: HttpStoreCapabilities

  private readonly deps: HttpStoreDeps
  private readonly sinks = new Map<number, EventSink>()
  private nextSinkId = 0
  private closed = false
  private sseUnsubscribe: (() => void) | undefined

  constructor(options: HttpStoreOptions) {
    this.deps = buildDeps(options)
    this.root = joinUrl(this.deps.baseUrl, brainPath(this.deps.brainId, ''))
    this.capabilities = {
      supportsLocalPath: false,
      synchronousEvents: false,
      bodyLimits: this.deps.bodyLimits,
    }
  }

  async read(path: Path): Promise<string> {
    this.ensureOpen()
    validatePath(path)
    const response = await doFetch(this.deps, {
      method: 'GET',
      path: brainPath(this.deps.brainId, '/documents/read'),
      query: { path },
      accept: 'application/octet-stream',
    })
    if (!response.ok) await throwForResponse(response, path)
    return decoder.decode(await response.arrayBuffer())
  }

  async write(path: Path, content: string): Promise<void> {
    this.ensureOpen()
    validatePath(path)
    this.assertSingleBodySize(byteLength(content), path)
    const response = await doFetch(this.deps, {
      method: 'PUT',
      path: brainPath(this.deps.brainId, '/documents'),
      query: { path },
      contentType: 'application/octet-stream',
      body: toByteBody(content),
    })
    if (!response.ok) await throwForResponse(response, path)
  }

  async append(path: Path, content: string): Promise<void> {
    this.ensureOpen()
    validatePath(path)
    this.assertSingleBodySize(byteLength(content), path)
    const response = await doFetch(this.deps, {
      method: 'POST',
      path: brainPath(this.deps.brainId, '/documents/append'),
      query: { path },
      contentType: 'application/octet-stream',
      body: toByteBody(content),
    })
    if (!response.ok) await throwForResponse(response, path)
  }

  async delete(path: Path): Promise<void> {
    this.ensureOpen()
    validatePath(path)
    const response = await doFetch(this.deps, {
      method: 'DELETE',
      path: brainPath(this.deps.brainId, '/documents'),
      query: { path },
    })
    if (!response.ok) await throwForResponse(response, path)
  }

  async rename(src: Path, dst: Path): Promise<void> {
    this.ensureOpen()
    validatePath(src)
    validatePath(dst)
    const response = await doFetch(this.deps, {
      method: 'POST',
      path: brainPath(this.deps.brainId, '/documents/rename'),
      contentType: 'application/json',
      body: JSON.stringify({ from: src, to: dst }),
    })
    if (!response.ok) await throwForResponse(response, src)
  }

  async exists(path: Path): Promise<boolean> {
    this.ensureOpen()
    validatePath(path)
    const response = await doFetch(this.deps, {
      method: 'HEAD',
      path: brainPath(this.deps.brainId, '/documents'),
      query: { path },
    })
    if (response.status === 200) return true
    if (response.status === 404) return false
    await throwForResponse(response, path)
    return false
  }

  async stat(path: Path): Promise<FileInfo> {
    this.ensureOpen()
    validatePath(path)
    const response = await doFetch(this.deps, {
      method: 'GET',
      path: brainPath(this.deps.brainId, '/documents/stat'),
      query: { path },
    })
    if (!response.ok) await throwForResponse(response, path)
    return toFileInfo((await response.json()) as RawStat)
  }

  async list(dir: Path | '', options: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    const query: Record<string, string | number | boolean | undefined> = {
      dir: dir === '' ? '' : dir,
      recursive: options.recursive === true ? 'true' : 'false',
      include_generated: options.includeGenerated === true ? 'true' : 'false',
      ...(options.glob === undefined || options.glob === '' ? {} : { glob: options.glob }),
    }
    const response = await doFetch(this.deps, {
      method: 'GET',
      path: brainPath(this.deps.brainId, '/documents'),
      query,
    })
    if (!response.ok) await throwForResponse(response, dir === '' ? undefined : dir)
    const body = (await response.json()) as { readonly items: RawListEntry[] }
    return body.items.map(toFileInfo)
  }

  async batch(options: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    this.ensureOpen()
    const journal: JournalOp[] = []
    const batch = new HttpBatch(this, journal)
    await fn(batch)
    if (journal.length === 0) return

    this.assertBatchSize(journal)
    const payload = {
      reason: options.reason,
      ops: journal.map((operation) => {
        if (operation.kind === 'write') {
          return {
            type: 'write',
            path: operation.path,
            content_base64: base64Encode(operation.content),
          }
        }
        if (operation.kind === 'append') {
          return {
            type: 'append',
            path: operation.path,
            content_base64: base64Encode(operation.content),
          }
        }
        if (operation.kind === 'delete') {
          return { type: 'delete', path: operation.path }
        }
        return { type: 'rename', path: operation.src, to: operation.dst }
      }),
    }

    const response = await doFetch(this.deps, {
      method: 'POST',
      path: brainPath(this.deps.brainId, '/documents/batch-ops'),
      contentType: 'application/json',
      body: JSON.stringify(payload),
    })
    if (!response.ok) await throwForResponse(response, undefined)
  }

  subscribe(sink: EventSink): Unsubscribe {
    this.ensureOpen()
    const id = this.nextSinkId
    this.nextSinkId += 1
    this.sinks.set(id, sink)

    if (this.sseUnsubscribe === undefined) {
      this.sseUnsubscribe = subscribeSse(
        this.deps,
        (event) => {
          for (const currentSink of this.sinks.values()) currentSink(event)
        },
        () => {
          // A dropped SSE stream should not crash callers.
        },
      )
    }

    return () => {
      this.sinks.delete(id)
      if (this.sinks.size === 0 && this.sseUnsubscribe !== undefined) {
        this.sseUnsubscribe()
        this.sseUnsubscribe = undefined
      }
    }
  }

  localPath(_path: Path): string | undefined {
    return undefined
  }

  async close(): Promise<void> {
    this.closed = true
    if (this.sseUnsubscribe !== undefined) {
      this.sseUnsubscribe()
      this.sseUnsubscribe = undefined
    }
    this.sinks.clear()
  }

  async _readForBatch(path: Path): Promise<string> {
    return await this.read(path)
  }

  async _existsForBatch(path: Path): Promise<boolean> {
    return await this.exists(path)
  }

  async _statForBatch(path: Path): Promise<FileInfo> {
    return await this.stat(path)
  }

  async _listForBatch(dir: Path | '', options: ListOpts): Promise<FileInfo[]> {
    return await this.list(dir, options)
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }

  private assertSingleBodySize(size: number, path: Path): void {
    if (size > this.deps.bodyLimits.singleDocumentBytes) {
      throw new ErrPayloadTooLarge(
        `${path} exceeds ${this.deps.bodyLimits.singleDocumentBytes} bytes`,
      )
    }
  }

  private assertBatchSize(journal: readonly JournalOp[]): void {
    if (journal.length > this.deps.bodyLimits.batchOpCount) {
      throw new ErrPayloadTooLarge(
        `batch has ${journal.length} ops (cap ${this.deps.bodyLimits.batchOpCount})`,
      )
    }
    let decodedBytes = 0
    for (const operation of journal) {
      if (operation.kind !== 'write' && operation.kind !== 'append') continue
      decodedBytes += byteLength(operation.content)
      if (decodedBytes > this.deps.bodyLimits.batchDecodedBytes) {
        throw new ErrPayloadTooLarge(
          `batch payload exceeds ${this.deps.bodyLimits.batchDecodedBytes} bytes`,
        )
      }
    }
  }
}

class HttpBatch implements Batch {
  constructor(
    private readonly store: HttpStore,
    private readonly journal: JournalOp[],
  ) {}

  async read(path: Path): Promise<string> {
    validatePath(path)
    const state = replay(this.journal, path)
    if (state.kind === 'present') return state.content
    if (state.kind === 'deleted') throw new ErrNotFound(path)
    return await this.store._readForBatch(path)
  }

  async write(path: Path, content: string): Promise<void> {
    validatePath(path)
    this.journal.push({ kind: 'write', path, content })
  }

  async append(path: Path, content: string): Promise<void> {
    validatePath(path)
    const state = replay(this.journal, path)
    let base = ''
    if (state.kind === 'present') {
      base = state.content
    } else if (state.kind === 'untouched') {
      try {
        base = await this.store._readForBatch(path)
      } catch (error) {
        if (!(error instanceof ErrNotFound)) throw error
      }
    }
    this.journal.push({ kind: 'write', path, content: `${base}${content}` })
  }

  async delete(path: Path): Promise<void> {
    validatePath(path)
    const state = replay(this.journal, path)
    if (state.kind === 'present') {
      this.journal.push({ kind: 'delete', path })
      return
    }
    if (state.kind === 'deleted') throw new ErrNotFound(path)
    const exists = await this.store._existsForBatch(path)
    if (!exists) throw new ErrNotFound(path)
    this.journal.push({ kind: 'delete', path })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    validatePath(src)
    validatePath(dst)
    const state = replay(this.journal, src)
    let payload: string
    if (state.kind === 'present') {
      payload = state.content
    } else if (state.kind === 'deleted') {
      throw new ErrNotFound(src)
    } else {
      payload = await this.store._readForBatch(src)
    }
    this.journal.push({ kind: 'write', path: dst, content: payload })
    this.journal.push({ kind: 'delete', path: src })
  }

  async exists(path: Path): Promise<boolean> {
    validatePath(path)
    const state = replay(this.journal, path)
    if (state.kind === 'present') return true
    if (state.kind === 'deleted') return false
    return await this.store._existsForBatch(path)
  }

  async stat(path: Path): Promise<FileInfo> {
    validatePath(path)
    const state = replay(this.journal, path)
    if (state.kind === 'present') {
      return {
        path,
        size: byteLength(state.content),
        modTime: new Date(),
        isDir: false,
      }
    }
    if (state.kind === 'deleted') throw new ErrNotFound(path)
    return await this.store._statForBatch(path)
  }

  async list(dir: Path | '', options: ListOpts = {}): Promise<FileInfo[]> {
    const base = await this.store._listForBatch(dir, options)
    const byPath = new Map<Path, FileInfo>()
    for (const entry of base) {
      byPath.set(entry.path, entry)
    }

    const touched = new Set<Path>()
    for (const operation of this.journal) {
      if (operation.kind === 'rename') {
        touched.add(operation.src)
        touched.add(operation.dst)
        continue
      }
      touched.add(operation.path)
    }

    const recursive = options.recursive === true
    const includeGenerated = options.includeGenerated === true
    const glob = options.glob ?? ''

    for (const path of touched) {
      if (!pathUnderLocal(path, dir, recursive)) continue
      const state = replay(this.journal, path)
      if (state.kind === 'present') {
        if (!includeGenerated && pathIsGenerated(path)) {
          byPath.delete(path)
          continue
        }
        if (glob !== '' && !matchGlob(glob, lastSegment(path))) continue
        byPath.set(path, {
          path,
          size: byteLength(state.content),
          modTime: new Date(),
          isDir: false,
        })
      } else if (state.kind === 'deleted') {
        byPath.delete(path)
      }
    }

    return [...byPath.values()].sort((left, right) => left.path.localeCompare(right.path))
  }
}

const replay = (journal: readonly JournalOp[], path: Path): BatchState => {
  let touched = false
  let present = false
  let content = ''

  for (const operation of journal) {
    if (operation.kind === 'write' && operation.path === path) {
      touched = true
      present = true
      content = operation.content
      continue
    }
    if (operation.kind === 'append' && operation.path === path) {
      touched = true
      present = true
      content = `${content}${operation.content}`
      continue
    }
    if (operation.kind === 'delete' && operation.path === path) {
      touched = true
      present = false
      content = ''
      continue
    }
    if (operation.kind === 'rename' && operation.src === path) {
      touched = true
      present = false
      content = ''
      continue
    }
    if (operation.kind === 'rename' && operation.dst === path) {
      touched = true
      present = true
    }
  }

  if (!touched) return { kind: 'untouched' }
  if (present) return { kind: 'present', content }
  return { kind: 'deleted' }
}

const pathUnderLocal = (path: Path, dir: Path | '', recursive: boolean): boolean => {
  if (dir === '') return true
  const pathValue = String(path)
  const dirValue = String(dir)
  if (pathValue === dirValue) return true
  if (
    !(
      pathValue.length > dirValue.length &&
      pathValue.startsWith(dirValue) &&
      pathValue[dirValue.length] === '/'
    )
  ) {
    return false
  }
  if (recursive) return true
  const rest = pathValue.slice(dirValue.length + 1)
  return !rest.includes('/')
}

const byteLength = (value: string): number => encoder.encode(value).byteLength

const toByteBody = (value: string): ArrayBuffer => {
  const bytes = encoder.encode(value)
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

const base64Encode = (value: string): string => {
  const bytes = encoder.encode(value)
  if (typeof btoa === 'function') {
    let binary = ''
    for (const byte of bytes) {
      binary += String.fromCharCode(byte)
    }
    return btoa(binary)
  }
  const bufferCtor = (globalThis as { readonly Buffer?: typeof Buffer }).Buffer
  if (bufferCtor !== undefined) {
    return bufferCtor.from(bytes).toString('base64')
  }
  throw new Error('http-store: no base64 encoder available in this runtime')
}
