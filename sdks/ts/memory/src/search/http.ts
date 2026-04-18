// SPDX-License-Identifier: Apache-2.0

/**
 * HTTP-backed hybrid search index.
 *
 * Mirrors the async shape of {@link PostgresSearchIndex} but talks to
 * `POST /v1/brains/:id/chunks/search` + `/chunks` CRUD rather than running
 * SQL locally. The server decides whether the index is SQLite, Postgres, or
 * something else — callers stay provider-agnostic.
 */

import { StoreError } from '../store/errors.js'
import type { Chunk } from './writer.js'

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

export type HttpSearchIndexOptions = {
  readonly baseUrl: string
  readonly brainId: string
  readonly apiKey?: string
  readonly token?: string
  readonly fetch?: FetchLike
  readonly timeoutMs?: number
}

export type HttpSearchResult = {
  readonly id: string
  readonly score: number
  readonly path: string
  readonly title?: string
  readonly content?: string
}

const DEFAULT_TIMEOUT_MS = 30_000
const DEFAULT_USER_AGENT = '@jeffs-brain/memory (+HttpSearchIndex)'

const resolveFetch = (f: FetchLike | undefined): FetchLike => {
  if (f !== undefined) return f
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as FetchLike
  }
  throw new Error(
    'createHttpSearchIndex: no global fetch is available; inject one via the `fetch` option',
  )
}

const joinUrl = (base: string, path: string): string => {
  const trimmed = base.endsWith('/') ? base.slice(0, -1) : base
  const rooted = path.startsWith('/') ? path : `/${path}`
  return `${trimmed}${rooted}`
}

type Deps = {
  readonly baseUrl: string
  readonly brainId: string
  readonly fetch: FetchLike
  readonly timeoutMs: number
  readonly authHeader: string | undefined
}

const buildDeps = (opts: HttpSearchIndexOptions): Deps => ({
  baseUrl: opts.baseUrl,
  brainId: opts.brainId,
  fetch: resolveFetch(opts.fetch),
  timeoutMs: opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  authHeader:
    opts.apiKey !== undefined && opts.apiKey !== ''
      ? `Bearer ${opts.apiKey}`
      : opts.token !== undefined && opts.token !== ''
        ? `Bearer ${opts.token}`
        : undefined,
})

const doFetch = async (
  deps: Deps,
  init: {
    readonly method: string
    readonly path: string
    readonly body?: string
    readonly contentType?: string
  },
): Promise<Response> => {
  const url = joinUrl(deps.baseUrl, init.path)
  const headers = new Headers()
  headers.set('user-agent', DEFAULT_USER_AGENT)
  headers.set('accept', 'application/json')
  if (init.contentType !== undefined) headers.set('content-type', init.contentType)
  if (deps.authHeader !== undefined) headers.set('authorization', deps.authHeader)
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('request timed out')), deps.timeoutMs)
  try {
    return await deps.fetch(url, {
      method: init.method,
      headers,
      body: init.body,
      signal: controller.signal,
    })
  } finally {
    clearTimeout(timer)
  }
}

const assertOk = async (resp: Response, ctx: string): Promise<void> => {
  if (resp.ok) return
  let detail = ''
  try {
    const payload = (await resp.json()) as { detail?: string; title?: string }
    detail = payload.detail ?? payload.title ?? ''
  } catch {
    detail = resp.statusText
  }
  throw new StoreError(`http-search: ${ctx} returned HTTP ${resp.status}${detail ? `: ${detail}` : ''}`)
}

const brainPath = (brainId: string, suffix: string): string =>
  `/v1/brains/${encodeURIComponent(brainId)}${suffix}`

export type HttpSearchIndex = {
  upsert(chunk: Chunk): Promise<void>
  deleteChunk(id: string): Promise<void>
  deleteByPath(path: string): Promise<void>
  getChunk(id: string): Promise<Chunk | undefined>
  searchBM25(query: string, limit: number): Promise<HttpSearchResult[]>
  searchVector(
    embedding: Float32Array | number[],
    limit: number,
  ): Promise<HttpSearchResult[]>
  searchHybrid(
    query: string,
    embedding: Float32Array | number[],
    limit: number,
  ): Promise<HttpSearchResult[]>
  close(): Promise<void>
}

export const createHttpSearchIndex = (opts: HttpSearchIndexOptions): HttpSearchIndex => {
  const deps = buildDeps(opts)

  const serialiseEmbedding = (embedding: Float32Array | number[]): number[] =>
    embedding instanceof Float32Array ? Array.from(embedding) : embedding

  return {
    async upsert(chunk) {
      const payload: Record<string, unknown> = {
        id: chunk.id,
        path: chunk.path,
        ordinal: chunk.ordinal ?? 0,
        title: chunk.title ?? '',
        summary: chunk.summary ?? '',
        content: chunk.content,
      }
      if (chunk.tags !== undefined) payload.tags = chunk.tags
      if (chunk.metadata !== undefined) payload.metadata = chunk.metadata
      if (chunk.embedding !== undefined) {
        payload.embedding = serialiseEmbedding(chunk.embedding)
      }
      const resp = await doFetch(deps, {
        method: 'POST',
        path: brainPath(deps.brainId, '/chunks'),
        contentType: 'application/json',
        body: JSON.stringify(payload),
      })
      await assertOk(resp, 'upsert')
    },

    async deleteChunk(id) {
      const resp = await doFetch(deps, {
        method: 'DELETE',
        path: brainPath(deps.brainId, `/chunks/${encodeURIComponent(id)}`),
      })
      await assertOk(resp, 'deleteChunk')
    },

    async deleteByPath(path) {
      const url = `${brainPath(deps.brainId, '/chunks')}?path=${encodeURIComponent(path)}`
      const resp = await doFetch(deps, { method: 'DELETE', path: url })
      await assertOk(resp, 'deleteByPath')
    },

    async getChunk(id) {
      const resp = await doFetch(deps, {
        method: 'GET',
        path: brainPath(deps.brainId, `/chunks/${encodeURIComponent(id)}`),
      })
      if (resp.status === 404) return undefined
      await assertOk(resp, 'getChunk')
      return (await resp.json()) as Chunk
    },

    async searchBM25(query, limit) {
      const resp = await doFetch(deps, {
        method: 'POST',
        path: brainPath(deps.brainId, '/chunks/search'),
        contentType: 'application/json',
        body: JSON.stringify({ mode: 'bm25', query, limit }),
      })
      await assertOk(resp, 'searchBM25')
      const body = (await resp.json()) as { chunks: HttpSearchResult[] }
      return body.chunks
    },

    async searchVector(embedding, limit) {
      const resp = await doFetch(deps, {
        method: 'POST',
        path: brainPath(deps.brainId, '/chunks/search'),
        contentType: 'application/json',
        body: JSON.stringify({
          mode: 'vector',
          embedding: serialiseEmbedding(embedding),
          limit,
        }),
      })
      await assertOk(resp, 'searchVector')
      const body = (await resp.json()) as { chunks: HttpSearchResult[] }
      return body.chunks
    },

    async searchHybrid(query, embedding, limit) {
      const resp = await doFetch(deps, {
        method: 'POST',
        path: brainPath(deps.brainId, '/chunks/search'),
        contentType: 'application/json',
        body: JSON.stringify({
          mode: 'hybrid',
          query,
          embedding: serialiseEmbedding(embedding),
          limit,
        }),
      })
      await assertOk(resp, 'searchHybrid')
      const body = (await resp.json()) as { chunks: HttpSearchResult[] }
      return body.chunks
    },

    async close() {
      /* nothing to clean up */
    },
  }
}
