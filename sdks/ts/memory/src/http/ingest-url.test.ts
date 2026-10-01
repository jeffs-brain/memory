// SPDX-License-Identifier: Apache-2.0

/**
 * `POST /ingest/url` error mapping with the fetch stubbed: upstream
 * failures are 502 `bad_gateway`, and a successful fetch is ingested.
 * The unstubbed SSRF refusals are covered in `security.test.ts`.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { FetchFailedError, type SafeFetchResult } from '../net/safe-fetch.js'
import { Daemon, createRouter } from './index.js'

const safeFetchMock = vi.hoisted(() => vi.fn<(url: string) => Promise<SafeFetchResult>>())

vi.mock('../net/safe-fetch.js', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../net/safe-fetch.js')>()),
  safeFetch: safeFetchMock,
}))

const cleanups: Array<() => Promise<void>> = []

afterEach(async () => {
  safeFetchMock.mockReset()
  for (const cleanup of cleanups.splice(0)) await cleanup()
})

const makeHandler = async (): Promise<(req: Request) => Promise<Response>> => {
  const root = await mkdtemp(join(tmpdir(), 'memory-ingest-url-'))
  const daemon = new Daemon({ root })
  await daemon.start()
  cleanups.push(async () => {
    await daemon.close()
    await rm(root, { recursive: true, force: true })
  })
  const router = createRouter(daemon)
  const created = await router(
    new Request('http://localhost/v1/brains', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ brainId: 'urls' }),
    }),
  )
  expect(created.status).toBe(201)
  return async (req) => router(req)
}

const ingestUrl = (url: string): Request =>
  new Request('http://localhost/v1/brains/urls/ingest/url', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ url }),
  })

describe('POST /ingest/url', () => {
  it('maps upstream failures to 502 bad_gateway', async () => {
    safeFetchMock.mockRejectedValue(new FetchFailedError('upstream responded HTTP 503', 503))
    const handler = await makeHandler()
    const res = await handler(ingestUrl('https://example.com/'))
    expect(res.status).toBe(502)
    const body = (await res.json()) as { code: string; detail: string }
    expect(body.code).toBe('bad_gateway')
    expect(body.detail).toBe('upstream responded HTTP 503')
  })

  it('ingests a fetched document', async () => {
    safeFetchMock.mockResolvedValue({
      url: 'https://example.com/notes',
      status: 200,
      contentType: 'text/markdown',
      body: Buffer.from('# Notes\n\nFetched through the guard.\n'),
    })
    const handler = await makeHandler()
    const res = await handler(ingestUrl('https://example.com/notes'))
    expect(res.status).toBe(200)
    expect(safeFetchMock).toHaveBeenCalledWith('https://example.com/notes', expect.anything())
  })
})
