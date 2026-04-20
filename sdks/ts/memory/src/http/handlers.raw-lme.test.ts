// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'

import { createHashEmbedder } from '../llm/index.js'
import { Daemon, createRouter } from './index.js'

type Fixture = {
  daemon: Daemon
  handler: (req: Request) => Promise<Response>
  tempDir: string
}

const fixtures: Fixture[] = []

const makeFixture = async (): Promise<Fixture> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-handlers-raw-lme-'))
  const daemon = new Daemon({
    root: tempDir,
    embedder: createHashEmbedder(),
  })
  await daemon.start()
  const router = createRouter(daemon)
  const handler = async (req: Request): Promise<Response> => router(req)
  const fixture: Fixture = { daemon, handler, tempDir }
  fixtures.push(fixture)
  return fixture
}

const makeRequest = (
  method: string,
  path: string,
  init: { body?: BodyInit; headers?: Record<string, string> } = {},
): Request => {
  const headers = new Headers(init.headers ?? {})
  return new Request(`http://daemon${path}`, {
    method,
    headers,
    ...(init.body !== undefined ? { body: init.body } : {}),
  })
}

afterEach(async () => {
  while (fixtures.length > 0) {
    const fixture = fixtures.pop()
    if (fixture === undefined) break
    await fixture.daemon.close()
    await rm(fixture.tempDir, { recursive: true, force: true })
  }
})

describe('handleSearch raw_lme scope', () => {
  it('returns raw LME transcripts without leaking raw documents', async () => {
    const { handler, daemon } = await makeFixture()

    const create = await handler(
      makeRequest('POST', '/v1/brains', {
        body: JSON.stringify({ brainId: 'lme' }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(create.status).toBe(201)

    const brain = await daemon.brains.get('lme')
    await brain.store.write(
      'raw/lme/session-1.md',
      Buffer.from(
        [
          '---',
          'session_id: sess-1',
          'session_date: 2024-03-08',
          '---',
          '[user]: I bought apples.',
          '',
        ].join('\n'),
      ),
    )
    await brain.store.write(
      'raw/documents/apples.md',
      Buffer.from('A raw document about apples.\n'),
    )
    await brain.refresh()

    const resp = await handler(
      makeRequest('POST', '/v1/brains/lme/search', {
        body: JSON.stringify({
          query: 'apples',
          mode: 'bm25',
          filters: { scope: 'raw_lme' },
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    expect(resp.status).toBe(200)
    const payload = (await resp.json()) as { chunks: Array<{ path: string; text: string }> }
    expect(payload.chunks.length).toBe(1)
    expect(payload.chunks[0]?.path).toBe('raw/lme/session-1.md')
    expect(payload.chunks[0]?.text).toContain('[user]: I bought apples.')
  })
})
