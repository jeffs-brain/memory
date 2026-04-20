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

export const serveCommand = defineCommand({
  meta: {
    name: 'serve',
    description: 'Run the memory HTTP daemon (PROTOCOL.md wire surface)',
  },
  args: {
    addr: {
      type: 'string',
      description: 'Bind address host:port (overrides JB_ADDR)',
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
      ...(token !== undefined ? { authToken: token } : {}),
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

    process.stderr.write(`memory serve: listening on http://${hostname}:${port}\n`)

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

const parseAddr = (addr: string): { hostname: string; port: number } => {
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
  return { hostname: host !== '' ? host : '0.0.0.0', port }
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
