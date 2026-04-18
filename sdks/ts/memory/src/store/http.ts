// SPDX-License-Identifier: Apache-2.0

/**
 * HTTP-backed `Store`.
 *
 * Mounts a remote brain as if it were local by translating every Store op
 * into a REST call. The wire surface is documented under
 * `apps/backend/src/routes/documents-fs.ts` and `.../events.ts`. The
 * implementation deliberately avoids pulling `@jeffs-brain/sdk` as a
 * runtime dependency — `@jeffs-brain/memory` is the lowest layer and may
 * not import up-stack. Instead we reuse the SDK's public HTTP + SSE
 * helpers which are themselves dependency-free.
 *
 * Capability map:
 *
 *   - `localPath` always returns `undefined` (remote store, no local
 *     filesystem to expose).
 *   - `subscribe` installs an SSE listener; events arrive asynchronously
 *     after the server commits a mutation. The contract test tolerates a
 *     short tick of delay between the ack and the event, see
 *     `http.contract.test.ts`.
 *   - `batch` buffers operations locally. `commit()` is a single
 *     `POST /documents/batch-ops` so the server can execute them in one
 *     tenant transaction.
 */

import { ErrInvalidPath, ErrNotFound, ErrReadOnly, StoreError } from './errors.js'
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

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

export type HttpStoreOptions = {
  readonly baseUrl: string
  readonly brainId: string
  readonly apiKey?: string
  readonly token?: string
  readonly fetch?: FetchLike
  readonly timeoutMs?: number
}

export type HttpStoreCapabilities = {
  /** Always `false` for HTTP stores. */
  readonly supportsLocalPath: false
  /**
   * Events arrive asynchronously over SSE; subscribers may need to await
   * a microtask after a mutation completes before their sink fires.
   */
  readonly synchronousEvents: false
}

const DEFAULT_TIMEOUT_MS = 30_000
const DEFAULT_USER_AGENT = '@jeffs-brain/memory (+HttpStore)'

const resolveFetch = (f: FetchLike | undefined): FetchLike => {
  if (f !== undefined) return f
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as FetchLike
  }
  throw new Error(
    'createHttpStore: no global fetch is available; inject one via the `fetch` option',
  )
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
  const entries = Object.entries(query).filter(([, v]) => v !== undefined)
  if (entries.length === 0) return url
  const qs = new URLSearchParams()
  for (const [k, v] of entries) qs.append(k, String(v))
  return `${url}${url.includes('?') ? '&' : '?'}${qs.toString()}`
}

type ProblemPayload = {
  readonly status?: number
  readonly title?: string
  readonly detail?: string
  readonly code?: string
}

const parseProblem = async (resp: Response): Promise<ProblemPayload> => {
  try {
    return (await resp.json()) as ProblemPayload
  } catch {
    return { status: resp.status, title: resp.statusText }
  }
}

const throwForResponse = async (resp: Response, path: string | undefined): Promise<never> => {
  const problem = await parseProblem(resp)
  if (resp.status === 404) throw new ErrNotFound(path ?? '', problem)
  if (resp.status === 400) throw new ErrInvalidPath(problem.detail ?? problem.title ?? 'bad request')
  throw new StoreError(
    `http-store: ${resp.status} ${problem.title ?? resp.statusText}${problem.detail ? `: ${problem.detail}` : ''}`,
    problem,
  )
}

type HttpStoreDeps = {
  readonly baseUrl: string
  readonly brainId: string
  readonly fetch: FetchLike
  readonly timeoutMs: number
  readonly authHeader: string | undefined
}

const buildDeps = (opts: HttpStoreOptions): HttpStoreDeps => {
  const baseUrl = opts.baseUrl
  const authHeader =
    opts.apiKey !== undefined && opts.apiKey !== ''
      ? `Bearer ${opts.apiKey}`
      : opts.token !== undefined && opts.token !== ''
        ? `Bearer ${opts.token}`
        : undefined
  return {
    baseUrl,
    brainId: opts.brainId,
    fetch: resolveFetch(opts.fetch),
    timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    authHeader,
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
    readonly body?: Uint8Array | string | undefined
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
  const onExternal = (): void => controller.abort(init.signal?.reason)
  if (init.signal !== undefined) {
    if (init.signal.aborted) controller.abort(init.signal.reason)
    else init.signal.addEventListener('abort', onExternal, { once: true })
  }
  try {
    return await deps.fetch(url, {
      method: init.method,
      headers,
      body: init.body,
      signal: controller.signal,
    })
  } finally {
    clearTimeout(timer)
    if (init.signal !== undefined) init.signal.removeEventListener('abort', onExternal)
  }
}

type RawListEntry = {
  readonly path: string
  readonly size: number
  readonly mtime: string
  readonly is_dir: boolean
}

type RawStat = {
  readonly path: string
  readonly size: number
  readonly mtime: string
  readonly is_dir: boolean
}

type RawChangeEvent = {
  readonly kind: 'created' | 'updated' | 'deleted' | 'renamed'
  readonly path: string
  readonly old_path?: string
  readonly reason?: string
  readonly when: string
}

const toFileInfo = (raw: RawListEntry | RawStat): FileInfo => ({
  path: raw.path as Path,
  size: raw.size,
  modTime: new Date(raw.mtime),
  isDir: raw.is_dir,
})

const toChangeEvent = (raw: RawChangeEvent): ChangeEvent => ({
  kind: raw.kind,
  path: raw.path as Path,
  ...(raw.old_path !== undefined ? { oldPath: raw.old_path as Path } : {}),
  ...(raw.reason !== undefined ? { reason: raw.reason } : {}),
  when: new Date(raw.when),
})

/**
 * SSE reader. Cancellable via AbortController; we split lines manually
 * rather than pulling the SDK parser so the memory package stays free of
 * a cross-workspace dependency.
 */
const subscribeSse = (
  deps: HttpStoreDeps,
  onEvent: (evt: ChangeEvent) => void,
  onError: (err: unknown) => void,
): (() => void) => {
  const controller = new AbortController()
  let closed = false

  const run = async (): Promise<void> => {
    const resp = await doFetch(deps, {
      method: 'GET',
      path: brainPath(deps.brainId, '/events'),
      accept: 'text/event-stream',
      signal: controller.signal,
    })
    if (!resp.ok) {
      onError(new StoreError(`http-store: events returned HTTP ${resp.status}`))
      return
    }
    if (resp.body === null) return
    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    let currentEvent = ''
    let dataLines: string[] = []
    const dispatch = (): void => {
      if (dataLines.length === 0 && currentEvent === '') return
      if (currentEvent === 'change') {
        try {
          const parsed = JSON.parse(dataLines.join('\n')) as RawChangeEvent
          onEvent(toChangeEvent(parsed))
        } catch {
          /* ignore malformed frame */
        }
      }
      currentEvent = ''
      dataLines = []
    }
    try {
      // biome-ignore lint/suspicious/noConstantCondition: stream loop
      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        while (true) {
          const nl = buffer.indexOf('\n')
          if (nl === -1) break
          let line = buffer.slice(0, nl)
          buffer = buffer.slice(nl + 1)
          if (line.endsWith('\r')) line = line.slice(0, -1)
          if (line === '') {
            dispatch()
            continue
          }
          if (line.startsWith(':')) continue
          const colon = line.indexOf(':')
          const field = colon === -1 ? line : line.slice(0, colon)
          let value = colon === -1 ? '' : line.slice(colon + 1)
          if (value.startsWith(' ')) value = value.slice(1)
          if (field === 'event') currentEvent = value
          else if (field === 'data') dataLines.push(value)
        }
      }
      dispatch()
    } catch (err) {
      if (!closed) onError(err)
    } finally {
      try {
        reader.releaseLock()
      } catch {
        /* ignore */
      }
    }
  }

  void run().catch((err) => {
    if (!closed) onError(err)
  })

  return () => {
    closed = true
    controller.abort()
  }
}

export const createHttpStore = (opts: HttpStoreOptions): HttpStore =>
  new HttpStore(opts)

export class HttpStore implements Store {
  readonly capabilities: HttpStoreCapabilities = {
    supportsLocalPath: false,
    synchronousEvents: false,
  }

  private readonly deps: HttpStoreDeps
  private readonly sinks = new Map<number, EventSink>()
  private nextSinkId = 0
  private closed = false
  private sseUnsubscribe: (() => void) | undefined

  constructor(opts: HttpStoreOptions) {
    this.deps = buildDeps(opts)
  }

  async read(p: Path): Promise<Buffer> {
    this.ensureOpen()
    validatePath(p)
    const resp = await doFetch(this.deps, {
      method: 'GET',
      path: brainPath(this.deps.brainId, '/documents/read'),
      query: { path: p },
      accept: 'application/octet-stream',
    })
    if (!resp.ok) await throwForResponse(resp, p)
    return Buffer.from(await resp.arrayBuffer())
  }

  async exists(p: Path): Promise<boolean> {
    this.ensureOpen()
    validatePath(p)
    const resp = await doFetch(this.deps, {
      method: 'HEAD',
      path: brainPath(this.deps.brainId, '/documents'),
      query: { path: p },
    })
    if (resp.status === 200) return true
    if (resp.status === 404) return false
    await throwForResponse(resp, p)
    return false
  }

  async stat(p: Path): Promise<FileInfo> {
    this.ensureOpen()
    validatePath(p)
    const resp = await doFetch(this.deps, {
      method: 'GET',
      path: brainPath(this.deps.brainId, '/documents/stat'),
      query: { path: p },
    })
    if (!resp.ok) await throwForResponse(resp, p)
    const raw = (await resp.json()) as RawStat
    return toFileInfo(raw)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    this.ensureOpen()
    const query: Record<string, string | number | boolean | undefined> = {
      dir: dir === '' ? '' : dir,
      recursive: opts.recursive === true ? 'true' : 'false',
      include_generated: opts.includeGenerated === true ? 'true' : 'false',
    }
    if (opts.glob !== undefined && opts.glob !== '') query.glob = opts.glob
    const resp = await doFetch(this.deps, {
      method: 'GET',
      path: brainPath(this.deps.brainId, '/documents'),
      query,
    })
    if (!resp.ok) await throwForResponse(resp, dir === '' ? '' : dir)
    const body = (await resp.json()) as { items: RawListEntry[] }
    return body.items.map(toFileInfo)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const resp = await doFetch(this.deps, {
      method: 'PUT',
      path: brainPath(this.deps.brainId, '/documents'),
      query: { path: p },
      contentType: 'application/octet-stream',
      body: new Uint8Array(content),
    })
    if (!resp.ok) await throwForResponse(resp, p)
  }

  async append(p: Path, content: Buffer): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const resp = await doFetch(this.deps, {
      method: 'POST',
      path: brainPath(this.deps.brainId, '/documents/append'),
      query: { path: p },
      contentType: 'application/octet-stream',
      body: new Uint8Array(content),
    })
    if (!resp.ok) await throwForResponse(resp, p)
  }

  async delete(p: Path): Promise<void> {
    this.ensureOpen()
    validatePath(p)
    const resp = await doFetch(this.deps, {
      method: 'DELETE',
      path: brainPath(this.deps.brainId, '/documents'),
      query: { path: p },
    })
    if (!resp.ok) await throwForResponse(resp, p)
  }

  async rename(src: Path, dst: Path): Promise<void> {
    this.ensureOpen()
    validatePath(src)
    validatePath(dst)
    const resp = await doFetch(this.deps, {
      method: 'POST',
      path: brainPath(this.deps.brainId, '/documents/rename'),
      contentType: 'application/json',
      body: JSON.stringify({ from: src, to: dst }),
    })
    if (!resp.ok) await throwForResponse(resp, src)
  }

  async batch(opts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> {
    this.ensureOpen()
    const journal: JournalOp[] = []
    const batch = new HttpBatch(this, journal)
    try {
      await fn(batch)
    } catch (err) {
      // buffered; nothing sent server-side yet
      throw err
    }
    if (journal.length === 0) return
    const payload = {
      reason: opts.reason,
      ...(opts.message !== undefined ? { message: opts.message } : {}),
      ...(opts.author !== undefined ? { author: opts.author } : {}),
      ...(opts.email !== undefined ? { email: opts.email } : {}),
      ops: journal.map((op) => {
        if (op.kind === 'write') {
          return { type: 'write', path: op.path, content_base64: op.content.toString('base64') }
        }
        if (op.kind === 'append') {
          return {
            type: 'append',
            path: op.path,
            content_base64: op.content.toString('base64'),
          }
        }
        if (op.kind === 'delete') {
          return { type: 'delete', path: op.path }
        }
        return { type: 'rename', path: op.src, to: op.dst }
      }),
    }
    const resp = await doFetch(this.deps, {
      method: 'POST',
      path: brainPath(this.deps.brainId, '/documents/batch-ops'),
      contentType: 'application/json',
      body: JSON.stringify(payload),
    })
    if (!resp.ok) await throwForResponse(resp, undefined)
  }

  subscribe(sink: EventSink): Unsubscribe {
    this.ensureOpen()
    const id = this.nextSinkId++
    this.sinks.set(id, sink)
    if (this.sseUnsubscribe === undefined) {
      this.sseUnsubscribe = subscribeSse(
        this.deps,
        (evt) => {
          for (const s of this.sinks.values()) s(evt)
        },
        () => {
          /* swallow: a dropped SSE connection should not crash the store */
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

  localPath(_p: Path): string | undefined {
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

  /**
   * Low-level helper used by {@link HttpBatch.read} to fetch current content
   * when the journal has no buffered state for a path.
   */
  async _readForBatch(p: Path): Promise<Buffer> {
    return this.read(p)
  }

  async _existsForBatch(p: Path): Promise<boolean> {
    return this.exists(p)
  }

  async _listForBatch(dir: Path | '', opts: ListOpts): Promise<FileInfo[]> {
    return this.list(dir, opts)
  }

  async _statForBatch(p: Path): Promise<FileInfo> {
    return this.stat(p)
  }

  private ensureOpen(): void {
    if (this.closed) throw new ErrReadOnly()
  }
}

type JournalOp =
  | { kind: 'write'; path: Path; content: Buffer }
  | { kind: 'append'; path: Path; content: Buffer }
  | { kind: 'delete'; path: Path }
  | { kind: 'rename'; src: Path; dst: Path }

type BatchState =
  | { kind: 'present'; content: Buffer }
  | { kind: 'deleted' }
  | { kind: 'untouched' }

const replay = (journal: readonly JournalOp[], p: Path): BatchState => {
  let present = false
  let content: Buffer = Buffer.alloc(0)
  let touched = false
  for (const op of journal) {
    if (op.kind === 'write' && op.path === p) {
      present = true
      content = op.content
      touched = true
    } else if (op.kind === 'append' && op.path === p) {
      present = true
      content = Buffer.concat([present ? content : Buffer.alloc(0), op.content])
      touched = true
    } else if (op.kind === 'delete' && op.path === p) {
      present = false
      content = Buffer.alloc(0)
      touched = true
    } else if (op.kind === 'rename' && op.src === p) {
      present = false
      content = Buffer.alloc(0)
      touched = true
    } else if (op.kind === 'rename' && op.dst === p) {
      present = true
      touched = true
    }
  }
  if (!touched) return { kind: 'untouched' }
  if (present) return { kind: 'present', content }
  return { kind: 'deleted' }
}

class HttpBatch implements Batch {
  constructor(
    private readonly store: HttpStore,
    private readonly journal: JournalOp[],
  ) {}

  async read(p: Path): Promise<Buffer> {
    validatePath(p)
    const state = replay(this.journal, p)
    if (state.kind === 'present') return Buffer.from(state.content)
    if (state.kind === 'deleted') throw new ErrNotFound(p)
    return this.store._readForBatch(p)
  }

  async write(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    this.journal.push({ kind: 'write', path: p, content: Buffer.from(content) })
  }

  async append(p: Path, content: Buffer): Promise<void> {
    validatePath(p)
    // Materialise so the server sees a single `write` of the concatenated
    // content. This keeps batch semantics independent of server-side
    // append ordering.
    const state = replay(this.journal, p)
    let base: Buffer
    if (state.kind === 'present') base = state.content
    else if (state.kind === 'deleted') base = Buffer.alloc(0)
    else {
      try {
        base = await this.store._readForBatch(p)
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
    const state = replay(this.journal, p)
    if (state.kind === 'present') {
      this.journal.push({ kind: 'delete', path: p })
      return
    }
    if (state.kind === 'deleted') throw new ErrNotFound(p)
    const exists = await this.store._existsForBatch(p)
    if (!exists) throw new ErrNotFound(p)
    this.journal.push({ kind: 'delete', path: p })
  }

  async rename(src: Path, dst: Path): Promise<void> {
    validatePath(src)
    validatePath(dst)
    const state = replay(this.journal, src)
    let payload: Buffer | undefined
    if (state.kind === 'present') payload = state.content
    else if (state.kind === 'deleted') throw new ErrNotFound(src)
    else {
      try {
        payload = await this.store._readForBatch(src)
      } catch (err) {
        if (err instanceof ErrNotFound) throw err
        throw err
      }
    }
    // Materialise as write-then-delete so the server sees a flat sequence.
    this.journal.push({ kind: 'write', path: dst, content: payload })
    this.journal.push({ kind: 'delete', path: src })
  }

  async exists(p: Path): Promise<boolean> {
    validatePath(p)
    const state = replay(this.journal, p)
    if (state.kind === 'present') return true
    if (state.kind === 'deleted') return false
    return this.store._existsForBatch(p)
  }

  async stat(p: Path): Promise<FileInfo> {
    validatePath(p)
    const state = replay(this.journal, p)
    if (state.kind === 'present') {
      return { path: p, size: state.content.length, modTime: new Date(), isDir: false }
    }
    if (state.kind === 'deleted') throw new ErrNotFound(p)
    return this.store._statForBatch(p)
  }

  async list(dir: Path | '', opts: ListOpts = {}): Promise<FileInfo[]> {
    const base = await this.store._listForBatch(dir, opts)
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
      const state = replay(this.journal, p)
      if (state.kind === 'present') {
        if (!includeGenerated && pathIsGenerated(p)) {
          byPath.delete(p)
          continue
        }
        if (glob !== '' && !matchGlob(glob, lastSegment(p))) continue
        byPath.set(p, { path: p, size: state.content.length, modTime: new Date(), isDir: false })
      } else if (state.kind === 'deleted') {
        byPath.delete(p)
      }
    }
    const result = Array.from(byPath.values())
    result.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0))
    return result
  }
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
