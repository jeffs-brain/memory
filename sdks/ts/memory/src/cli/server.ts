// SPDX-License-Identifier: Apache-2.0

/**
 * Minimal HTTP surface for `memory serve`.
 *
 * No framework: just Bun.serve (or node:http when available). Exposes
 * `/search`, `/ingest`, `/extract`, `/recall`, plus a health probe. The
 * server hangs on to a single open Store for its lifetime and closes it
 * when `stop()` is called.
 */

import { basename } from 'node:path'
import { createIngest } from '../knowledge/ingest.js'
import type { Embedder, Message, Provider } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import { type Scope, createMemory, createStoreBackedCursorStore } from '../memory/index.js'
import { ErrInvalidPath } from '../store/errors.js'
import type { Store } from '../store/index.js'
import { validatePathSegment } from '../store/path.js'
import { type SearchMode, isSearchMode, runSearch } from './search-runner.js'

export type ServeOptions = {
  readonly port: number
  readonly hostname?: string
  readonly store: Store
  readonly embedder?: Embedder
  readonly provider?: Provider
}

export type Serving = {
  readonly port: number
  readonly url: string
  stop(): Promise<void>
}

type BunServe = (opts: {
  port: number
  hostname?: string
  fetch: (req: Request) => Promise<Response> | Response
}) => {
  port: number
  hostname: string
  url: URL
  stop: (closeActiveConnections?: boolean) => void
}

type BunGlobal = { serve?: BunServe }

export const startServer = async (opts: ServeOptions): Promise<Serving> => {
  const bun = (globalThis as unknown as { Bun?: BunGlobal }).Bun
  if (bun === undefined || typeof bun.serve !== 'function') {
    throw new Error('memory serve requires Bun runtime (global Bun.serve missing)')
  }
  const handler = buildHandler(opts)
  const server = bun.serve({
    port: opts.port,
    ...(opts.hostname !== undefined ? { hostname: opts.hostname } : {}),
    fetch: handler,
  })
  return {
    port: server.port,
    url: server.url.toString(),
    stop: async () => {
      server.stop(true)
    },
  }
}

export const buildHandler = (opts: ServeOptions) => {
  return async (req: Request): Promise<Response> => {
    const url = new URL(req.url)
    try {
      if (req.method === 'GET' && url.pathname === '/health') {
        return json(200, { ok: true })
      }
      if (req.method === 'POST' && url.pathname === '/search') {
        return await handleSearch(req, opts)
      }
      if (req.method === 'POST' && url.pathname === '/ingest') {
        return await handleIngest(req, opts)
      }
      if (req.method === 'POST' && url.pathname === '/extract') {
        return await handleExtract(req, opts)
      }
      if (req.method === 'POST' && url.pathname === '/recall') {
        return await handleRecall(req, opts)
      }
      return json(404, { error: `no route for ${req.method} ${url.pathname}` })
    } catch (err) {
      return json(500, {
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }
}

const handleSearch = async (req: Request, opts: ServeOptions): Promise<Response> => {
  const body = await readJson<{
    query?: unknown
    mode?: unknown
    limit?: unknown
  }>(req)
  if (body === undefined) return json(400, { error: 'invalid JSON body' })
  const query = typeof body.query === 'string' ? body.query : ''
  if (query === '') return json(400, { error: "missing 'query'" })
  const modeRaw = typeof body.mode === 'string' ? body.mode : 'hybrid'
  const mode: SearchMode = isSearchMode(modeRaw) ? modeRaw : 'hybrid'
  const limit = typeof body.limit === 'number' && body.limit > 0 ? Math.trunc(body.limit) : 10
  const hits = await runSearch(query, {
    store: opts.store,
    ...(opts.embedder !== undefined ? { embedder: opts.embedder } : {}),
    mode,
    limit,
  })
  return json(200, { query, mode, hits })
}

const noopProvider: Provider = {
  name: () => 'noop',
  modelName: () => 'noop',
  async *stream() {
    yield await Promise.reject(new Error('not implemented'))
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

const asScope = (value: unknown): Scope => {
  if (value === 'global' || value === 'project' || value === 'agent') return value
  return 'project'
}

const asActorId = (body: Record<string, unknown>): string => {
  const actorId = typeof body.actorId === 'string' ? body.actorId : body.actor_id
  const resolved = typeof actorId === 'string' && actorId !== '' ? actorId : 'cli'
  validatePathSegment(resolved)
  return resolved
}

const buildMemory = (opts: ServeOptions, scope: Scope, actorId: string) =>
  createMemory({
    store: opts.store,
    provider: opts.provider ?? noopProvider,
    ...(opts.embedder !== undefined ? { embedder: opts.embedder } : {}),
    cursorStore: createStoreBackedCursorStore(opts.store),
    scope,
    actorId,
    extractMinMessages: 1,
  })

const handleRecall = async (req: Request, opts: ServeOptions): Promise<Response> => {
  const body = await readJson<Record<string, unknown>>(req)
  if (body === undefined) return json(400, { error: 'invalid JSON body' })
  const query = typeof body.query === 'string' ? body.query : ''
  if (query === '') return json(400, { error: "missing 'query'" })
  const limitRaw = typeof body.k === 'number' ? body.k : body.limit
  const k = typeof limitRaw === 'number' && limitRaw > 0 ? Math.trunc(limitRaw) : 10
  let actorId: string
  try {
    actorId = asActorId(body)
  } catch (error) {
    if (error instanceof ErrInvalidPath) {
      return json(400, { error: 'invalid actor_id' })
    }
    throw error
  }
  const scope = asScope(body.scope)
  const memory = buildMemory(opts, scope, actorId)
  const hits = await memory.recall({ query, k, scope, actorId })
  return json(200, { query, scope, actor_id: actorId, hits })
}

const handleExtract = async (req: Request, opts: ServeOptions): Promise<Response> => {
  if (opts.provider === undefined) {
    return json(501, { error: 'extract requires a configured provider' })
  }
  const body = await readJson<Record<string, unknown>>(req)
  if (body === undefined) return json(400, { error: 'invalid JSON body' })
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return json(400, { error: "missing 'messages'" })
  }
  const messages: Message[] = []
  for (const raw of body.messages) {
    if (typeof raw !== 'object' || raw === null) {
      return json(400, { error: 'messages must be objects' })
    }
    const role = (raw as { role?: unknown }).role
    const content = (raw as { content?: unknown }).content
    const name = (raw as { name?: unknown }).name
    if (
      (role !== 'system' && role !== 'user' && role !== 'assistant' && role !== 'tool') ||
      typeof content !== 'string' ||
      content === ''
    ) {
      return json(400, { error: 'messages must include valid role and content fields' })
    }
    messages.push({
      role,
      content,
      ...(typeof name === 'string' && name !== '' ? { name } : {}),
    })
  }
  let actorId: string
  try {
    actorId = asActorId(body)
  } catch (error) {
    if (error instanceof ErrInvalidPath) {
      return json(400, { error: 'invalid actor_id' })
    }
    throw error
  }
  const scope = asScope(body.scope)
  const sessionId =
    typeof body.sessionId === 'string'
      ? body.sessionId
      : typeof body.session_id === 'string'
        ? body.session_id
        : undefined
  const sessionDate =
    typeof body.sessionDate === 'string'
      ? body.sessionDate
      : typeof body.session_date === 'string'
        ? body.session_date
        : undefined
  const memory = buildMemory(opts, scope, actorId)
  const extracted = await memory.extract({
    messages,
    actorId,
    scope,
    ...(sessionId !== undefined ? { sessionId } : {}),
    ...(sessionDate !== undefined ? { sessionDate } : {}),
  })
  return json(200, { scope, actor_id: actorId, extracted })
}

const handleIngest = async (req: Request, opts: ServeOptions): Promise<Response> => {
  const body = await readJson<{ name?: unknown; content?: unknown }>(req)
  if (body === undefined) return json(400, { error: 'invalid JSON body' })
  const content = typeof body.content === 'string' ? body.content : ''
  if (content === '') return json(400, { error: "missing 'content'" })
  const name = typeof body.name === 'string' && body.name !== '' ? basename(body.name) : undefined
  const ingest = createIngest({ store: opts.store, logger: noopLogger })
  const result = await ingest(Buffer.from(content, 'utf8'), {
    ...(name !== undefined ? { name } : {}),
  })
  return json(200, result)
}

const readJson = async <T>(req: Request): Promise<T | undefined> => {
  try {
    const raw = await req.text()
    if (raw === '') return {} as T
    return JSON.parse(raw) as T
  } catch {
    return undefined
  }
}

const json = (status: number, body: unknown): Response =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
