// SPDX-License-Identifier: Apache-2.0

/**
 * `memory serve` entry point.
 *
 * Thin wrapper around the {@link Daemon} in `src/http/daemon.ts`: the
 * daemon owns the per-brain store / retrieval / memory bundle; this
 * command resolves its environment, binds a listener, and wires signal
 * handlers to drive graceful shutdown.
 */

import {
  type IncomingMessage,
  type ServerResponse,
  createServer as createNodeServer,
} from 'node:http'
import { defineCommand } from 'citty'

import { Daemon, createRouter, defaultRoot } from '../../http/index.js'
import { resolveIngestRoot } from '../../http/ingest-root.js'
import { isLoopbackHost } from '../../http/router.js'
import type { Logger } from '../../llm/index.js'
import { createContextualPrefixBuilder } from '../../memory/index.js'
import {
  CliUsageError,
  buildEmbedder,
  buildProvider,
  buildReranker,
  embedderFromEnv,
  providerFromEnvOptional,
  rerankerFromEnv,
} from '../config.js'

const DEFAULT_PORT = 8080
const DEFAULT_HOST = '127.0.0.1'

export const serveCommand = defineCommand({
  meta: {
    name: 'serve',
    description: 'Run the memory HTTP daemon (PROTOCOL.md wire surface)',
  },
  args: {
    addr: {
      type: 'string',
      description:
        'Bind address host:port (overrides JB_ADDR; default 127.0.0.1:8080). A non-loopback host requires an auth token',
    },
    port: {
      type: 'string',
      description: 'Port to bind (convenience for --addr :<port>)',
    },
    host: {
      type: 'string',
      description: 'Host to bind (used with --port)',
    },
    root: {
      type: 'string',
      description: 'Daemon root directory (overrides JB_HOME)',
    },
    'auth-token': {
      type: 'string',
      description: 'Shared bearer token (overrides JB_AUTH_TOKEN)',
    },
    'ingest-root': {
      type: 'string',
      description:
        'Directory ingest/file may read server-side paths from (overrides JB_INGEST_ROOT; unset disables path ingest)',
    },
    contextualise: {
      type: 'boolean',
      description: 'Enable live extraction contextualisation.',
    },
    'contextualise-cache-dir': {
      type: 'string',
      description: 'Optional cache directory for live extraction contextualisation.',
    },
  },
  run: async ({ args }) => {
    // `--addr host:port` wins; `--port`/`--host` preserve the older
    // flag surface.
    const portFlag = typeof args.port === 'string' && args.port !== '' ? args.port : undefined
    const hostFlag = typeof args.host === 'string' && args.host !== '' ? args.host : undefined
    const addr =
      typeof args.addr === 'string' && args.addr !== ''
        ? args.addr
        : portFlag !== undefined
          ? `${hostFlag ?? ''}:${portFlag}`
          : (process.env.JB_ADDR ?? `:${DEFAULT_PORT}`)
    const root = typeof args.root === 'string' && args.root !== '' ? args.root : defaultRoot()
    const token =
      typeof args['auth-token'] === 'string' && args['auth-token'] !== ''
        ? args['auth-token']
        : process.env.JB_AUTH_TOKEN
    const { hostname, port } = parseAddr(addr)
    assertBindAllowed(hostname, token)
    const ingestRootFlag =
      typeof args['ingest-root'] === 'string' && args['ingest-root'] !== ''
        ? args['ingest-root']
        : process.env.JB_INGEST_ROOT
    let ingestRoot: string | undefined
    try {
      ingestRoot = await resolveIngestRoot(ingestRootFlag)
    } catch (err) {
      throw new CliUsageError(
        `serve: invalid ingest root: ${err instanceof Error ? err.message : String(err)}`,
      )
    }
    const logger = createStderrLogger()

    const providerSettings = providerFromEnvOptional()
    const provider = providerSettings !== undefined ? buildProvider(providerSettings) : undefined
    const embedderSettings = embedderFromEnv()
    const embedder = embedderSettings !== undefined ? buildEmbedder(embedderSettings) : undefined
    const rerankerSettings = rerankerFromEnv()
    const reranker =
      rerankerSettings !== undefined
        ? buildReranker(rerankerSettings, { ...(provider !== undefined ? { provider } : {}) })
        : undefined
    const contextualise =
      typeof args.contextualise === 'boolean'
        ? args.contextualise
        : envEnabled(process.env.JB_CONTEXTUALISE)
    const contextualiseCacheDir =
      typeof args['contextualise-cache-dir'] === 'string' && args['contextualise-cache-dir'] !== ''
        ? args['contextualise-cache-dir']
        : process.env.JB_CONTEXTUALISE_CACHE_DIR
    const singleDocumentBytes = parseOptionalPositiveInt(process.env.JB_SINGLE_BODY_LIMIT_BYTES)
    const batchDecodedBytes = parseOptionalPositiveInt(process.env.JB_BATCH_BODY_LIMIT_BYTES)
    const batchOpCount = parseOptionalPositiveInt(process.env.JB_BATCH_OP_LIMIT)
    const contextualPrefixBuilder =
      contextualise && provider !== undefined
        ? createContextualPrefixBuilder({
            provider,
            ...(process.env.JB_CONTEXTUALISE_MODEL !== undefined
              ? { model: process.env.JB_CONTEXTUALISE_MODEL }
              : {}),
            ...(contextualiseCacheDir !== undefined ? { cacheDir: contextualiseCacheDir } : {}),
          })
        : undefined

    const daemon = new Daemon({
      root,
      logger,
      ...(token !== undefined ? { authToken: token } : {}),
      ...(ingestRoot !== undefined ? { ingestRoot } : {}),
      ...(provider !== undefined ? { provider } : {}),
      ...(embedder !== undefined ? { embedder } : {}),
      ...(reranker !== undefined ? { reranker } : {}),
      ...(contextualPrefixBuilder !== undefined ? { contextualPrefixBuilder } : {}),
      ...(singleDocumentBytes !== undefined ||
      batchDecodedBytes !== undefined ||
      batchOpCount !== undefined
        ? {
            bodyLimits: {
              ...(singleDocumentBytes !== undefined ? { singleDocumentBytes } : {}),
              ...(batchDecodedBytes !== undefined ? { batchDecodedBytes } : {}),
              ...(batchOpCount !== undefined ? { batchOpCount } : {}),
            },
          }
        : {}),
    })
    await daemon.start()
    const router = createRouter(daemon)

    const server = createNodeServer((nreq, nres) => {
      void handleNodeRequest(router, hostname, port, nreq, nres)
    })

    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(port, hostname, () => {
        server.off('error', reject)
        resolve()
      })
    })

    const displayHost = hostname.includes(':') ? `[${hostname}]` : hostname
    const authNote = token !== undefined && token !== '' ? 'bearer auth' : 'no auth, loopback only'
    process.stderr.write(`memory serve: listening on http://${displayHost}:${port} (${authNote})\n`)

    const shutdown = async (): Promise<void> => {
      process.stderr.write('memory serve: shutting down\n')
      await new Promise<void>((resolve) => server.close(() => resolve()))
      await daemon.close()
      process.exit(0)
    }
    process.once('SIGINT', () => {
      void shutdown()
    })
    process.once('SIGTERM', () => {
      void shutdown()
    })

    await new Promise(() => undefined)
  },
})

export const parseAddr = (addr: string): { hostname: string; port: number } => {
  const trimmed = addr.trim()
  const match = trimmed.match(/^(?:\[?([^\]]*)\]?:)?(\d+)$/)
  if (match === null) {
    throw new CliUsageError(`serve: invalid --addr '${addr}'`)
  }
  const host = match[1] ?? ''
  const port = Number.parseInt(match[2] ?? '', 10)
  if (!Number.isFinite(port) || port <= 0 || port > 65535) {
    throw new CliUsageError(`serve: invalid port in '${addr}'`)
  }
  return { hostname: host !== '' ? host : DEFAULT_HOST, port }
}

/**
 * Bind policy from spec/PROTOCOL.md: without a bearer token the daemon
 * only listens on loopback, so nothing on the network can reach an
 * unauthenticated brain.
 */
export const assertBindAllowed = (hostname: string, token: string | undefined): void => {
  if (token !== undefined && token !== '') return
  if (isLoopbackHost(hostname)) return
  throw new CliUsageError(
    `serve: refusing to listen on ${hostname} without an auth token; set --auth-token or JB_AUTH_TOKEN, or bind to ${DEFAULT_HOST}`,
  )
}

/** One JSON object per line on stderr, so daemon errors are never silent. */
const createStderrLogger = (): Logger => {
  const write =
    (level: string) =>
    (msg: string, ctx?: Record<string, unknown>): void => {
      process.stderr.write(
        `${JSON.stringify({ time: new Date().toISOString(), level, msg, ...(ctx ?? {}) })}\n`,
      )
    }
  return { debug: () => {}, info: write('info'), warn: write('warn'), error: write('error') }
}

const envEnabled = (value: string | undefined): boolean => {
  if (value === undefined) return false
  const lowered = value.trim().toLowerCase()
  return lowered === '1' || lowered === 'true' || lowered === 'yes' || lowered === 'on'
}

const parseOptionalPositiveInt = (value: string | undefined): number | undefined => {
  if (value === undefined || value.trim() === '') return undefined
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new CliUsageError(`serve: invalid positive integer '${value}'`)
  }
  return parsed
}

/** A bare host name, IPv4 literal or bracketed IPv6 literal, optional port. */
const HOST_HEADER =
  /^(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?|\[[0-9A-Fa-f:.]+\])(?::\d{1,5})?$/

/**
 * Translate a Node request into a fetch-style Request, pass it to the
 * router, then stream the Response back onto the Node response.
 *
 * Exported so integration tests can bind a real socket without
 * duplicating the bridge logic here.
 */
export const handleNodeRequest = async (
  router: (req: Request) => Promise<Response> | Response,
  hostname: string,
  port: number,
  nreq: IncomingMessage,
  nres: ServerResponse,
): Promise<void> => {
  const urlPath = nreq.url ?? '/'
  const hostHeader = nreq.headers.host ?? `${hostname}:${port}`
  if (!HOST_HEADER.test(hostHeader)) {
    nres.statusCode = 400
    nres.setHeader('content-type', 'application/problem+json')
    nres.end(
      JSON.stringify({
        status: 400,
        title: 'Bad Request',
        code: 'validation_error',
        detail: 'invalid Host header',
      }),
    )
    return
  }
  const url = `http://${hostHeader}${urlPath.startsWith('/') ? urlPath : `/${urlPath}`}`

  const controller = new AbortController()
  // Note: `nreq.on('close', ...)` fires when the request body is fully
  // consumed, not only on client disconnect. Wiring it to the abort
  // controller would abort the handler before it ever runs. Rely on
  // `nres.on('close', ...)` below to cancel the stream on real client
  // disconnect, and on the socket itself if needed.
  nres.once('close', () => controller.abort())

  const method = nreq.method ?? 'GET'
  const headers = new Headers()
  for (const [k, v] of Object.entries(nreq.headers)) {
    if (v === undefined) continue
    if (Array.isArray(v)) headers.set(k, v.join(', '))
    else headers.set(k, String(v))
  }

  const bodyRequired = method !== 'GET' && method !== 'HEAD'
  const request = new Request(url, {
    method,
    headers,
    body: bodyRequired ? await readBody(nreq) : undefined,
    signal: controller.signal,
  })

  const response = await router(request)
  nres.statusCode = response.status
  response.headers.forEach((value, key) => {
    nres.setHeader(key, value)
  })

  if (response.body === null) {
    nres.end()
    return
  }
  const reader = response.body.getReader()
  nres.once('close', () => {
    void reader.cancel().catch(() => undefined)
  })
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    if (value !== undefined) {
      if (!nres.write(Buffer.from(value))) {
        await new Promise<void>((resolve) => nres.once('drain', () => resolve()))
      }
    }
  }
  nres.end()
}

const readBody = async (nreq: IncomingMessage): Promise<Buffer> => {
  const chunks: Buffer[] = []
  for await (const chunk of nreq) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
  }
  return Buffer.concat(chunks)
}
