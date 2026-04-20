// SPDX-License-Identifier: Apache-2.0

/**
 * HTTP router for the memory daemon. Mirrors the Go net/http mux
 * shape so the wire surface is byte-equivalent.
 *
 * The router takes a fetch-style function (Request → Response) and
 * wraps it with:
 *
 *   - `/healthz` probe (unauthenticated)
 *   - Optional bearer-token auth enforced via `JB_AUTH_TOKEN`
 *   - Route dispatch by method + path prefix
 *
 * Everything sits on top of the stock `Request`/`Response` globals so
 * the same handler runs under Bun.serve, Node's HTTP server (via an
 * adapter), or a `fetch` test invocation.
 */

import type { Daemon } from './daemon.js'
import {
  handleAsk,
  handleBatchOps,
  handleConsolidate,
  handleCreateBrain,
  handleDeleteBrain,
  handleDocAppend,
  handleDocDelete,
  handleDocHead,
  handleDocList,
  handleDocRead,
  handleDocRename,
  handleDocStat,
  handleDocWrite,
  handleEvents,
  handleExtract,
  handleGetBrain,
  handleIngestFile,
  handleIngestUrl,
  handleListBrains,
  handleRecall,
  handleReflect,
  handleRemember,
  handleSearch,
} from './handlers.js'
import { forbidden, notFound, unauthorized } from './problem.js'

export type Handler = (req: Request) => Promise<Response> | Response

export const createRouter = (daemon: Daemon): Handler => {
  return async (req: Request): Promise<Response> => {
    const url = new URL(req.url)

    // Health probe sits outside the auth chain.
    if (req.method === 'GET' && url.pathname === '/healthz') {
      return new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (daemon.authToken !== undefined && daemon.authToken !== '') {
      const authn = req.headers.get('authorization') ?? ''
      if (authn === '') {
        return unauthorized('missing Authorization header')
      }
      if (authn !== `Bearer ${daemon.authToken}`) {
        return forbidden('invalid bearer token')
      }
    }

    try {
      return await dispatch(daemon, req, url)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      return new Response(
        JSON.stringify({
          status: 500,
          title: 'Internal Server Error',
          code: 'internal_error',
          detail: message,
        }),
        {
          status: 500,
          headers: { 'content-type': 'application/problem+json' },
        },
      )
    }
  }
}

const dispatch = async (daemon: Daemon, req: Request, url: URL): Promise<Response> => {
  const path = url.pathname

  // Top-level brains.
  if (path === '/v1/brains') {
    if (req.method === 'GET') return handleListBrains(daemon)
    if (req.method === 'POST') return handleCreateBrain(daemon, req)
    return notFound(`no route for ${req.method} ${path}`)
  }

  const brainMatch = path.match(/^\/v1\/brains\/([^/]+)(\/.*)?$/)
  if (brainMatch === null) {
    return notFound(`no route for ${req.method} ${path}`)
  }
  const brainId = decodeURIComponent(brainMatch[1] ?? '')
  const suffix = brainMatch[2] ?? ''

  if (suffix === '') {
    if (req.method === 'GET') return handleGetBrain(daemon, brainId)
    if (req.method === 'DELETE') return handleDeleteBrain(daemon, req, brainId)
    return notFound(`no route for ${req.method} ${path}`)
  }

  switch (suffix) {
    case '/documents/read':
      if (req.method === 'GET') return handleDocRead(daemon, req, brainId, url)
      break
    case '/documents/stat':
      if (req.method === 'GET') return handleDocStat(daemon, brainId, url)
      break
    case '/documents':
      if (req.method === 'GET') return handleDocList(daemon, brainId, url)
      if (req.method === 'HEAD') return handleDocHead(daemon, brainId, url)
      if (req.method === 'PUT') return handleDocWrite(daemon, req, brainId, url)
      if (req.method === 'DELETE') return handleDocDelete(daemon, brainId, url)
      break
    case '/documents/append':
      if (req.method === 'POST') return handleDocAppend(daemon, req, brainId, url)
      break
    case '/documents/rename':
      if (req.method === 'POST') return handleDocRename(daemon, req, brainId)
      break
    case '/documents/batch-ops':
      if (req.method === 'POST') return handleBatchOps(daemon, req, brainId)
      break
    case '/search':
      if (req.method === 'POST') return handleSearch(daemon, req, brainId)
      break
    case '/ask':
      if (req.method === 'POST') return handleAsk(daemon, req, brainId)
      break
    case '/ingest/file':
      if (req.method === 'POST') return handleIngestFile(daemon, req, brainId)
      break
    case '/ingest/url':
      if (req.method === 'POST') return handleIngestUrl(daemon, req, brainId)
      break
    case '/remember':
      if (req.method === 'POST') return handleRemember(daemon, req, brainId)
      break
    case '/recall':
      if (req.method === 'POST') return handleRecall(daemon, req, brainId)
      break
    case '/extract':
      if (req.method === 'POST') return handleExtract(daemon, req, brainId)
      break
    case '/reflect':
      if (req.method === 'POST') return handleReflect(daemon, req, brainId)
      break
    case '/consolidate':
      if (req.method === 'POST') return handleConsolidate(daemon, req, brainId)
      break
    case '/events':
      if (req.method === 'GET') return handleEvents(daemon, req, brainId)
      break
  }

  return notFound(`no route for ${req.method} ${path}`)
}
