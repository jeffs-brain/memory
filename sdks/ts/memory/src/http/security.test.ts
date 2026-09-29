// SPDX-License-Identifier: Apache-2.0

/**
 * Daemon hardening: bearer comparison, the loopback guard for daemons
 * without a token, detail-free 500s, confined server-path ingest and
 * the SSRF guard on URL ingest. Mirrors the Go tests in
 * `go/internal/httpd/local_test.go` and
 * `go/cmd/memory/handler_ingest_test.go`.
 */

import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import type { CompletionResponse, Logger, Provider, StreamEvent } from '../llm/index.js'
import { Daemon, createRouter } from './index.js'
import { resolveIngestRoot } from './ingest-root.js'
import { isLoopbackHost, validBearerToken } from './router.js'

type LogEntry = { level: string; msg: string; ctx: Record<string, unknown> | undefined }

const captureLogger = (entries: LogEntry[]): Logger => {
  const push =
    (level: string) =>
    (msg: string, ctx?: Record<string, unknown>): void => {
      entries.push({ level, msg, ctx })
    }
  return { debug: push('debug'), info: push('info'), warn: push('warn'), error: push('error') }
}

const failingProvider = (message: string): Provider => ({
  name: () => 'failing',
  modelName: () => 'failing-1',
  async *stream() {
    yield { type: 'error', error: new Error(message) } satisfies StreamEvent
  },
  complete: async (): Promise<CompletionResponse> => {
    throw new Error(message)
  },
  supportsStructuredDecoding: () => false,
  structured: async () => {
    throw new Error(message)
  },
})

type Fixture = {
  readonly daemon: Daemon
  readonly handler: (req: Request) => Promise<Response>
  readonly logs: LogEntry[]
}

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  for (const cleanup of cleanups.splice(0)) await cleanup()
})

const tempDir = async (prefix: string): Promise<string> => {
  const dir = await realpath(await mkdtemp(join(tmpdir(), prefix)))
  cleanups.push(async () => rm(dir, { recursive: true, force: true }))
  return dir
}

const makeFixture = async (
  opts: { authToken?: string; ingestRoot?: string; provider?: Provider } = {},
): Promise<Fixture> => {
  const root = await tempDir('memory-security-')
  const logs: LogEntry[] = []
  const daemon = new Daemon({
    root,
    logger: captureLogger(logs),
    ...(opts.authToken !== undefined ? { authToken: opts.authToken } : {}),
    ...(opts.ingestRoot !== undefined ? { ingestRoot: opts.ingestRoot } : {}),
    ...(opts.provider !== undefined ? { provider: opts.provider } : {}),
  })
  await daemon.start()
  cleanups.push(async () => daemon.close())
  const router = createRouter(daemon)
  return { daemon, handler: async (req) => router(req), logs }
}

const request = (
  method: string,
  url: string,
  init: { body?: unknown; headers?: Record<string, string> } = {},
): Request =>
  new Request(url, {
    method,
    headers: {
      ...(init.body !== undefined ? { 'content-type': 'application/json' } : {}),
      ...(init.headers ?? {}),
    },
    ...(init.body !== undefined ? { body: JSON.stringify(init.body) } : {}),
  })

const createBrain = async (fx: Fixture, brainId: string, headers: Record<string, string> = {}) => {
  const resp = await fx.handler(
    request('POST', 'http://localhost/v1/brains', { body: { brainId }, headers }),
  )
  expect(resp.status).toBe(201)
}

describe('validBearerToken', () => {
  it.each([
    ['Bearer secret', true],
    ['bearer secret', true],
    ['BEARER   secret', true],
    ['Bearer secret ', true],
    ['Bearer wrong', false],
    ['Bearer secretsecret', false],
    ['Bearer ', false],
    ['Basic secret', false],
    ['secret', false],
    ['', false],
  ])('%j -> %s', (header, want) => {
    expect(validBearerToken(header, 'secret')).toBe(want)
  })
})

describe('isLoopbackHost', () => {
  it.each([
    ['localhost', true],
    ['LOCALHOST', true],
    ['127.0.0.1', true],
    ['127.10.20.30', true],
    ['::1', true],
    ['[::1]', true],
    ['::ffff:127.0.0.1', true],
    ['[::ffff:7f00:1]', true],
    ['0:0:0:0:0:0:0:1', true],
    ['0.0.0.0', false],
    ['192.0.2.10', false],
    ['::', false],
    ['example.com', false],
    ['localhost.example.com', false],
    ['', false],
  ])('%j -> %s', (host, want) => {
    expect(isLoopbackHost(host)).toBe(want)
  })
})

describe('router without a token', () => {
  it('refuses a foreign Host with 421 (DNS rebinding)', async () => {
    const fx = await makeFixture()
    const resp = await fx.handler(request('GET', 'http://attacker.example/v1/brains'))
    expect(resp.status).toBe(421)
    expect(((await resp.json()) as { code: string }).code).toBe('misdirected_request')
  })

  it('refuses a foreign Origin with 403 (cross-site request)', async () => {
    const fx = await makeFixture()
    const resp = await fx.handler(
      request('GET', 'http://127.0.0.1/v1/brains', {
        headers: { origin: 'https://attacker.example' },
      }),
    )
    expect(resp.status).toBe(403)
  })

  it('accepts loopback hosts and loopback origins', async () => {
    const fx = await makeFixture()
    for (const url of [
      'http://127.0.0.1:8080/v1/brains',
      'http://localhost/v1/brains',
      'http://[::1]/v1/brains',
    ]) {
      const resp = await fx.handler(
        request('GET', url, { headers: { origin: 'http://localhost:3000' } }),
      )
      expect(resp.status, url).toBe(200)
    }
  })

  it('keeps /healthz reachable from any host', async () => {
    const fx = await makeFixture()
    const resp = await fx.handler(request('GET', 'http://attacker.example/healthz'))
    expect(resp.status).toBe(200)
  })
})

describe('router with a token', () => {
  it('serves any host once the bearer token matches', async () => {
    const fx = await makeFixture({ authToken: 'secret' })
    const ok = await fx.handler(
      request('GET', 'http://memory.internal/v1/brains', {
        headers: { authorization: 'bearer secret' },
      }),
    )
    expect(ok.status).toBe(200)
    const wrong = await fx.handler(
      request('GET', 'http://memory.internal/v1/brains', {
        headers: { authorization: 'Bearer secre' },
      }),
    )
    expect(wrong.status).toBe(403)
  })
})

describe('internal errors', () => {
  it('logs the cause and returns a detail-free 500', async () => {
    const fx = await makeFixture()
    fx.daemon.brains.get = async () => {
      throw new Error('open /srv/private/brain.sqlite: permission denied')
    }
    const resp = await fx.handler(
      request('GET', 'http://localhost/v1/brains/any/documents/read?path=a.md'),
    )
    expect(resp.status).toBe(500)
    const text = await resp.text()
    expect(text).not.toContain('/srv/private')
    expect(JSON.parse(text)).toMatchObject({ code: 'internal_error', detail: 'internal error' })
    expect(
      fx.logs.some((e) => e.level === 'error' && String(e.ctx?.err).includes('/srv/private')),
    ).toBe(true)
  })

  it('keeps provider failures out of the ask stream', async () => {
    const fx = await makeFixture({ provider: failingProvider('upstream said: key sk-private') })
    await createBrain(fx, 'ask')
    const resp = await fx.handler(
      request('POST', 'http://localhost/v1/brains/ask/ask', { body: { question: 'anything' } }),
    )
    expect(resp.status).toBe(200)
    const text = await resp.text()
    expect(text).toContain('event: error')
    expect(text).toContain('"code":"llm_error"')
    expect(text).not.toContain('sk-private')
    expect(text).toContain('"ok":false')
    expect(fx.logs.some((e) => String(e.ctx?.err).includes('sk-private'))).toBe(true)
  })
})

describe('POST /ingest/file with a server-side path', () => {
  const ingestPath = async (fx: Fixture, path: string): Promise<Response> =>
    fx.handler(request('POST', 'http://localhost/v1/brains/docs/ingest/file', { body: { path } }))

  it('is refused when no ingest root is configured', async () => {
    const fx = await makeFixture()
    await createBrain(fx, 'docs')
    const resp = await ingestPath(fx, '/etc/passwd')
    expect(resp.status).toBe(403)
  })

  it('reads files inside the ingest root and refuses everything else', async () => {
    const allowed = await tempDir('memory-ingest-root-')
    const outside = await tempDir('memory-ingest-outside-')
    await writeFile(join(allowed, 'note.md'), '# note\n\nInside the root.')
    await mkdir(join(allowed, 'sub'))
    await writeFile(join(outside, 'secret.md'), 'outside')
    await symlink(join(outside, 'secret.md'), join(allowed, 'escape.md'))

    const fx = await makeFixture({ ingestRoot: allowed })
    await createBrain(fx, 'docs')

    expect((await ingestPath(fx, 'note.md')).status).toBe(200)
    expect((await ingestPath(fx, join(allowed, 'note.md'))).status).toBe(200)
    expect((await ingestPath(fx, join(outside, 'secret.md'))).status).toBe(403)
    expect((await ingestPath(fx, '../secret.md')).status).toBe(403)
    expect((await ingestPath(fx, '../does-not-exist.md')).status).toBe(403)
    expect((await ingestPath(fx, 'escape.md')).status).toBe(403)
    expect((await ingestPath(fx, 'missing.md')).status).toBe(400)
    expect((await ingestPath(fx, 'sub')).status).toBe(400)
  })
})

describe('POST /ingest/url', () => {
  it.each([
    'http://127.0.0.1:1/',
    'http://169.254.169.254/latest/meta-data/',
    'file:///etc/passwd',
  ])('refuses %s with 400', async (url) => {
    const fx = await makeFixture()
    await createBrain(fx, 'web')
    const resp = await fx.handler(
      request('POST', 'http://localhost/v1/brains/web/ingest/url', { body: { url } }),
    )
    expect(resp.status).toBe(400)
  })
})

describe('resolveIngestRoot', () => {
  it('treats a blank root as disabled', async () => {
    expect(await resolveIngestRoot(undefined)).toBeUndefined()
    expect(await resolveIngestRoot('  ')).toBeUndefined()
  })

  it('rejects a missing directory or a file', async () => {
    const dir = await tempDir('memory-ingest-root-')
    await writeFile(join(dir, 'file.md'), 'x')
    await expect(resolveIngestRoot(join(dir, 'nope'))).rejects.toThrow()
    await expect(resolveIngestRoot(join(dir, 'file.md'))).rejects.toThrow(/not a directory/)
    expect(await resolveIngestRoot(dir)).toBe(dir)
  })
})
