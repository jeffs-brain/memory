// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { type Server, createServer as createNodeServer } from 'node:http'
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
  readonly daemon: Daemon
  readonly tempDir: string
  readonly baseUrl: string
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
    daemon,
    tempDir,
    baseUrl: `http://${hostname}:${address.port}`,
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
    await new Promise<void>((resolve) => fixture.server.close(() => resolve()))
    await fixture.daemon.close()
    await rm(fixture.tempDir, { recursive: true, force: true })
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
})
