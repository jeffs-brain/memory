// SPDX-License-Identifier: Apache-2.0

/**
 * OpenFGA HTTP adapter.
 *
 * Talks to an OpenFGA server (or any API-compatible proxy) via plain
 * `fetch`. Implements the `AccessControlProvider` surface: `check`,
 * `write`, `read`, `close`.
 *
 * The adapter is stateless and has no transitive runtime dependencies
 * other than `fetch`. Pass `{ fetch: mockFetch }` in tests.
 *
 * `close()` is a no-op for the fetch transport because we do not hold a
 * connection pool or any other releasable resource. Callers should still
 * invoke it for forward compatibility: a future transport (long-lived
 * gRPC client, persistent HTTP/2 session, etc.) may need to release real
 * state, and uniform lifecycle handling avoids leaks when adapters are
 * swapped.
 *
 * Endpoints used (per OpenFGA HTTP API):
 *   POST {apiUrl}/stores/{storeId}/check
 *   POST {apiUrl}/stores/{storeId}/write
 *   POST {apiUrl}/stores/{storeId}/read
 */

import type {
  AccessControlProvider,
  Action,
  CheckResult,
  ReadTuplesQuery,
  Resource,
  Subject,
  Tuple,
  WriteTuplesRequest,
} from '@jeffs-brain/memory/acl'
import { allow, deny } from '@jeffs-brain/memory/acl'

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

export type OpenFgaOptions = {
  readonly apiUrl: string
  readonly storeId: string
  readonly modelId?: string
  readonly token?: string
  readonly fetch?: FetchLike
}

/**
 * Default action -> relation map aligned with `spec/openfga/schema.fga`.
 *
 * Callers can pick different relations for the same action if they have a
 * variant model by passing a pre-computed `Tuple` via `write`/`read`.
 */
const ACTION_TO_RELATION: Readonly<Record<Action, string>> = {
  read: 'reader',
  write: 'writer',
  delete: 'can_delete',
  admin: 'admin',
  export: 'can_export',
}

export class OpenFgaHttpError extends Error {
  override readonly name: string = 'OpenFgaHttpError'
  constructor(
    readonly status: number,
    readonly endpoint: string,
    readonly body: string,
  ) {
    super(`openfga: ${endpoint} failed: HTTP ${status}: ${body.slice(0, 200)}`)
  }
}

export class OpenFgaRequestError extends Error {
  override readonly name: string = 'OpenFgaRequestError'
  constructor(
    readonly endpoint: string,
    override readonly cause: unknown,
  ) {
    super(`openfga: ${endpoint} request failed: ${String(cause)}`)
  }
}

type CheckResponse = {
  readonly allowed?: boolean
  readonly resolution?: string
}

type ReadResponse = {
  readonly tuples?: readonly {
    readonly key?: { readonly user?: string; readonly relation?: string; readonly object?: string }
  }[]
  readonly continuation_token?: string
}

export const createOpenFgaProvider = (opts: OpenFgaOptions): AccessControlProvider => {
  const baseUrl = opts.apiUrl.replace(/\/+$/, '')
  const storePath = `${baseUrl}/stores/${opts.storeId}`
  const doFetch: FetchLike = opts.fetch ?? fetch

  const headers = (): Record<string, string> => {
    const h: Record<string, string> = { 'content-type': 'application/json' }
    if (opts.token !== undefined) h.authorization = `Bearer ${opts.token}`
    return h
  }

  const post = async <T>(endpoint: string, body: unknown): Promise<T> => {
    let res: Response
    try {
      res = await doFetch(`${storePath}${endpoint}`, {
        method: 'POST',
        headers: headers(),
        body: JSON.stringify(body),
      })
    } catch (err) {
      throw new OpenFgaRequestError(endpoint, err)
    }
    if (!res.ok) {
      const text = await res.text().catch(() => '<no body>')
      throw new OpenFgaHttpError(res.status, endpoint, text)
    }
    try {
      return (await res.json()) as T
    } catch (err) {
      throw new OpenFgaRequestError(endpoint, err)
    }
  }

  const check = async (
    subject: Subject,
    action: Action,
    resource: Resource,
  ): Promise<CheckResult> => {
    const body = {
      tuple_key: {
        user: encodeSubject(subject),
        relation: ACTION_TO_RELATION[action],
        object: encodeResource(resource),
      },
      ...(opts.modelId !== undefined ? { authorization_model_id: opts.modelId } : {}),
    }
    const json = await post<CheckResponse>('/check', body)
    if (json.allowed === true) return allow(json.resolution)
    return deny(json.resolution ?? 'openfga: not allowed')
  }

  const write = async (req: WriteTuplesRequest): Promise<void> => {
    if ((req.writes?.length ?? 0) === 0 && (req.deletes?.length ?? 0) === 0) return
    const body: Record<string, unknown> = {}
    if (req.writes && req.writes.length > 0) {
      body.writes = { tuple_keys: req.writes.map(tupleToKey) }
    }
    if (req.deletes && req.deletes.length > 0) {
      body.deletes = { tuple_keys: req.deletes.map(tupleToKey) }
    }
    if (opts.modelId !== undefined) body.authorization_model_id = opts.modelId
    await post<unknown>('/write', body)
  }

  const read = async (query: ReadTuplesQuery): Promise<readonly Tuple[]> => {
    const tupleKey: Record<string, string> = {}
    if (query.subject) tupleKey.user = encodeSubject(query.subject)
    if (query.relation !== undefined) tupleKey.relation = query.relation
    if (query.resource) tupleKey.object = encodeResource(query.resource)
    const body = { tuple_key: tupleKey }
    const json = await post<ReadResponse>('/read', body)
    const rows = json.tuples ?? []
    const out: Tuple[] = []
    for (const row of rows) {
      const key = row.key
      if (!key || !key.user || !key.relation || !key.object) continue
      const subject = decodeSubject(key.user)
      const resource = decodeResource(key.object)
      if (!subject || !resource) continue
      out.push({ subject, relation: key.relation, resource })
    }
    return out
  }

  // No-op for the fetch transport; see the file header for rationale.
  const close = async (): Promise<void> => {}

  return { name: 'openfga', check, write, read, close }
}

const tupleToKey = (t: Tuple): { user: string; relation: string; object: string } => ({
  user: encodeSubject(t.subject),
  relation: t.relation,
  object: encodeResource(t.resource),
})

export const encodeSubject = (s: Subject): string => `${s.kind}:${s.id}`
export const encodeResource = (r: Resource): string => `${r.type}:${r.id}`

export const decodeSubject = (s: string): Subject | undefined => {
  const idx = s.indexOf(':')
  if (idx === -1) return undefined
  const kind = s.slice(0, idx)
  const id = s.slice(idx + 1)
  if (kind !== 'user' && kind !== 'api_key' && kind !== 'service') return undefined
  return { kind, id }
}

export const decodeResource = (s: string): Resource | undefined => {
  const idx = s.indexOf(':')
  if (idx === -1) return undefined
  const type = s.slice(0, idx)
  const id = s.slice(idx + 1)
  if (type !== 'workspace' && type !== 'brain' && type !== 'collection' && type !== 'document') {
    return undefined
  }
  return { type, id }
}
