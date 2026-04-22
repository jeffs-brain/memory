// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import {
  type IncomingMessage,
  type Server,
  type ServerResponse,
  createServer as createNodeServer,
} from 'node:http'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { handleNodeRequest } from '../cli/commands/serve.js'
import { Daemon, createRouter } from '../http/index.js'

import { runConformanceSuite } from './index.js'

type CustomFixtureDocument = {
  readonly placeholders: Readonly<Record<string, string>>
  readonly cases: ReadonlyArray<Record<string, unknown>>
}

type Fixture = {
  readonly server: Server
  readonly baseUrl: string
  cleanup(): Promise<void>
}

let fixtures: Fixture[] = []

const startFixture = async (authToken?: string): Promise<Fixture> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-conformance-'))
  const hostname = '127.0.0.1'
  const daemon = new Daemon({
    root: tempDir,
    ...(authToken !== undefined ? { authToken } : {}),
  })
  await daemon.start()
  const router = createRouter(daemon)

  const server = createNodeServer((request, response) => {
    void handleNodeRequest(router, hostname, 0, request, response)
  })
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, hostname, () => {
      server.off('error', reject)
      resolve()
    })
  })

  const address = server.address()
  if (address === null || typeof address === 'string') {
    throw new Error('conformance test: server.address() returned no port')
  }

  const fixture: Fixture = {
    server,
    baseUrl: `http://${hostname}:${address.port}`,
    cleanup: async (): Promise<void> => {
      await new Promise<void>((resolve) => {
        fixture.server.close(() => resolve())
        fixture.server.closeAllConnections?.()
      })
      await daemon.close()
      await rm(tempDir, { recursive: true, force: true })
    },
  }
  fixtures.push(fixture)
  return fixture
}

type CustomServerOptions = {
  readonly connectDelayMs?: number
  readonly readyEventDelayMs?: number
}

const startCustomFixture = async (options: CustomServerOptions = {}): Promise<Fixture> => {
  const hostname = '127.0.0.1'
  const brains = new Set<string>()
  const subscribers = new Map<string, Set<ServerResponse<IncomingMessage>>>()
  const timers = new Set<ReturnType<typeof setTimeout>>()

  const removeSubscriber = (brainId: string, response: ServerResponse<IncomingMessage>): void => {
    const brainSubscribers = subscribers.get(brainId)
    if (brainSubscribers === undefined) return
    brainSubscribers.delete(response)
    if (brainSubscribers.size === 0) {
      subscribers.delete(brainId)
    }
  }

  const sendJson = (
    response: ServerResponse<IncomingMessage>,
    status: number,
    body: unknown,
  ): void => {
    response.writeHead(status, { 'content-type': 'application/json' })
    response.end(JSON.stringify(body))
  }

  const server = createNodeServer(async (request, response) => {
    const url = new URL(request.url ?? '/', `http://${hostname}`)
    const pathname = url.pathname

    if (request.method === 'GET' && pathname === '/v1/brains') {
      sendJson(response, 200, {
        items: [...brains].sort().map((brainId) => ({ brainId })),
      })
      return
    }

    if (request.method === 'POST' && pathname === '/v1/brains') {
      const chunks: Uint8Array[] = []
      for await (const chunk of request) {
        chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk)
      }
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as { brainId?: string }
      if (typeof body.brainId !== 'string' || body.brainId === '') {
        response.writeHead(400)
        response.end()
        return
      }
      brains.add(body.brainId)
      response.writeHead(201)
      response.end()
      return
    }

    const brainMatch = pathname.match(/^\/v1\/brains\/([^/]+)(?:\/(.*))?$/)
    if (brainMatch === null) {
      response.writeHead(404)
      response.end()
      return
    }

    const brainId = decodeURIComponent(brainMatch[1] ?? '')
    const suffix = brainMatch[2] ?? ''

    if (request.method === 'DELETE' && suffix === '') {
      brains.delete(brainId)
      for (const subscriber of subscribers.get(brainId) ?? []) {
        subscriber.end()
      }
      subscribers.delete(brainId)
      response.writeHead(204)
      response.end()
      return
    }

    if (!brains.has(brainId)) {
      response.writeHead(404)
      response.end()
      return
    }

    if (request.method === 'PUT' && suffix === 'documents') {
      const path = url.searchParams.get('path') ?? ''
      response.writeHead(204)
      response.end()

      if (path !== '') {
        for (const subscriber of subscribers.get(brainId) ?? []) {
          subscriber.write(`event: change\ndata: ${JSON.stringify({ kind: 'created', path })}\n\n`)
        }
      }
      return
    }

    if (request.method === 'GET' && suffix === 'events') {
      const connectTimer = setTimeout(() => {
        timers.delete(connectTimer)
        if (response.writableEnded) return

        response.writeHead(200, {
          'cache-control': 'no-cache',
          connection: 'keep-alive',
          'content-type': 'text/event-stream',
        })
        response.flushHeaders()
        const brainSubscribers = subscribers.get(brainId) ?? new Set()
        brainSubscribers.add(response)
        subscribers.set(brainId, brainSubscribers)

        if (options.readyEventDelayMs !== undefined) {
          const readyTimer = setTimeout(() => {
            timers.delete(readyTimer)
            if (!response.writableEnded) {
              response.write('event: ready\ndata: {}\n\n')
            }
          }, options.readyEventDelayMs)
          timers.add(readyTimer)
        }
      }, options.connectDelayMs ?? 0)
      timers.add(connectTimer)

      const close = (): void => {
        clearTimeout(connectTimer)
        timers.delete(connectTimer)
        removeSubscriber(brainId, response)
      }
      response.on('close', close)
      return
    }

    response.writeHead(404)
    response.end()
  })

  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(0, hostname, () => {
      server.off('error', reject)
      resolve()
    })
  })

  const address = server.address()
  if (address === null || typeof address === 'string') {
    throw new Error('conformance test: custom server.address() returned no port')
  }

  const fixture: Fixture = {
    server,
    baseUrl: `http://${hostname}:${address.port}`,
    cleanup: async (): Promise<void> => {
      for (const timer of timers) {
        clearTimeout(timer)
      }
      timers.clear()
      for (const brainSubscribers of subscribers.values()) {
        for (const subscriber of brainSubscribers) {
          subscriber.end()
        }
      }
      await new Promise<void>((resolve) => {
        fixture.server.close(() => resolve())
        fixture.server.closeAllConnections?.()
      })
    },
  }
  fixtures.push(fixture)
  return fixture
}

const writeCustomFixture = async (document: CustomFixtureDocument): Promise<string> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-conformance-fixture-'))
  const path = join(tempDir, 'http-contract.json')
  await writeFile(path, JSON.stringify(document), 'utf8')
  return path
}

const listBrains = async (baseUrl: string, authToken?: string): Promise<readonly string[]> => {
  const response = await fetch(`${baseUrl}/v1/brains`, {
    headers:
      authToken === undefined
        ? undefined
        : {
            authorization: `Bearer ${authToken}`,
          },
  })
  expect(response.status).toBe(200)
  const body = (await response.json()) as { items: Array<{ brainId: string }> }
  return body.items.map((item) => item.brainId)
}

const assertSuitePassed = (result: Awaited<ReturnType<typeof runConformanceSuite>>): void => {
  if (result.failed > 0) {
    const details = result.cases
      .filter((testCase) => !testCase.ok)
      .map((testCase) => `${testCase.name}: ${testCase.error ?? 'unknown error'}`)
      .join('\n')
    throw new Error(`conformance suite failed:\n${details}`)
  }
}

beforeEach(() => {
  fixtures = []
})

afterEach(async () => {
  for (const fixture of fixtures) {
    await fixture.cleanup()
  }
  fixtures = []
})

describe('runConformanceSuite', () => {
  it('replays the shared fixture against an unauthenticated daemon and cleans up every brain', async () => {
    const fixture = await startFixture()

    const result = await runConformanceSuite({
      baseUrl: fixture.baseUrl,
    })

    expect(result.total).toBeGreaterThanOrEqual(29)
    expect(result.passed).toBe(result.total)
    expect(result.failed).toBe(0)
    assertSuitePassed(result)
    expect(await listBrains(fixture.baseUrl)).toEqual([])
  })

  it('replays the shared fixture against an authenticated daemon when baseUrl includes /v1', async () => {
    const authToken = 'test-token'
    const fixture = await startFixture(authToken)

    const result = await runConformanceSuite({
      baseUrl: `${fixture.baseUrl}/v1`,
      authToken,
    })

    expect(result.total).toBeGreaterThanOrEqual(29)
    expect(result.passed).toBe(result.total)
    expect(result.failed).toBe(0)
    expect(
      result.cases.find((testCase) => testCase.name === 'auth header forwarded when apiKey is set')
        ?.ok,
    ).toBe(true)
    assertSuitePassed(result)
    expect(await listBrains(fixture.baseUrl, authToken)).toEqual([])
  })

  it('treats json-field-equals as exact equality for the selected field value', async () => {
    const fixture = await startFixture()
    const fixturePath = await writeCustomFixture({
      placeholders: {},
      cases: [
        {
          name: 'json-field-equals exact field match',
          setup: [
            {
              method: 'PUT',
              path: '/v1/brains/BRAIN_ID/documents?path=memory%2Fa.md',
              bodyBase64: 'YQ==',
              expectedStatus: 204,
            },
          ],
          request: {
            method: 'GET',
            path: '/v1/brains/BRAIN_ID/documents?dir=memory&recursive=true&include_generated=false',
          },
          expectedResponse: {
            status: 200,
            bodyAssertions: [
              {
                kind: 'json-field-equals',
                field: 'items',
                value: [{ path: 'memory/a.md' }],
              },
            ],
          },
        },
      ],
    })

    try {
      const result = await runConformanceSuite({
        baseUrl: fixture.baseUrl,
        fixturePath,
      })

      expect(result.total).toBe(1)
      expect(result.passed).toBe(0)
      expect(result.failed).toBe(1)
      expect(result.cases[0]?.ok).toBe(false)
      expect(result.cases[0]?.error).toContain('field "items"')
      expect(await listBrains(fixture.baseUrl)).toEqual([])
    } finally {
      await rm(dirname(fixturePath), { recursive: true, force: true })
    }
  })

  it('observes delayed SSE events without dropping chunks while polling', async () => {
    const fixture = await startCustomFixture({ readyEventDelayMs: 350 })
    const fixturePath = await writeCustomFixture({
      placeholders: {},
      cases: [
        {
          name: 'delayed ready event',
          request: {
            method: 'GET',
            path: '/v1/brains/BRAIN_ID/events',
            headers: {
              accept: 'text/event-stream',
            },
          },
          expectedResponse: {
            status: 200,
            streamAssertions: [{ kind: 'expect-event', event: 'ready' }],
          },
        },
      ],
    })

    try {
      const result = await runConformanceSuite({
        baseUrl: fixture.baseUrl,
        fixturePath,
        sseTimeoutMs: 1_000,
      })

      expect(result.total).toBe(1)
      expect(result.passed).toBe(1)
      expect(result.failed).toBe(0)
      assertSuitePassed(result)
      expect(await listBrains(fixture.baseUrl)).toEqual([])
    } finally {
      await rm(dirname(fixturePath), { recursive: true, force: true })
    }
  })

  it('waits for the SSE subscription handshake before follow-up steps run', async () => {
    const fixture = await startCustomFixture({ connectDelayMs: 150 })
    const fixturePath = await writeCustomFixture({
      placeholders: {},
      cases: [
        {
          name: 'open-sse waits for handshake',
          setup: [
            {
              kind: 'open-sse',
              name: 'watcher',
              path: '/v1/brains/BRAIN_ID/events',
            },
          ],
          request: {
            method: 'PUT',
            path: '/v1/brains/BRAIN_ID/documents?path=memory%2Fa.md',
            bodyBase64: 'YQ==',
          },
          expectedResponse: {
            status: 204,
          },
          followUp: [
            {
              kind: 'await-sse-event',
              name: 'watcher',
              event: 'change',
            },
          ],
          teardown: [
            {
              kind: 'close-sse',
              name: 'watcher',
            },
          ],
        },
      ],
    })

    try {
      const result = await runConformanceSuite({
        baseUrl: fixture.baseUrl,
        fixturePath,
      })

      expect(result.total).toBe(1)
      expect(result.passed).toBe(1)
      expect(result.failed).toBe(0)
      assertSuitePassed(result)
      expect(await listBrains(fixture.baseUrl)).toEqual([])
    } finally {
      await rm(dirname(fixturePath), { recursive: true, force: true })
    }
  })
})
