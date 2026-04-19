// SPDX-License-Identifier: Apache-2.0

/**
 * Integration tests for the memory HTTP daemon. Mirrors the nine Go
 * tests in `sdks/go/cmd/memory/serve_integration_test.go` so the two
 * SDKs stay behaviourally aligned.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { CompletionResponse, Embedder, Provider, StreamEvent } from '../llm/index.js'
import { createHashEmbedder } from '../llm/index.js'
import { Daemon, createRouter } from './index.js'
import { MEMORY_PACKAGE } from '../index.js'

const makeFakeProvider = (text: string): Provider => ({
  name: () => 'fake',
  modelName: () => 'fake-1',
  async *stream() {
    yield { type: 'text_delta', text } satisfies StreamEvent
    yield { type: 'done', stopReason: 'end_turn' as const } satisfies StreamEvent
  },
  complete: async (): Promise<CompletionResponse> => ({
    content: text,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async () => text,
})

/** Provider that returns the same structured JSON for every call so the
 *  extract/reflect/consolidate pipelines can deterministically hit the
 *  persist path without a real LLM. */
const makeStructuredProvider = (json: string, text = 'ok'): Provider => ({
  name: () => 'fake-structured',
  modelName: () => 'fake-structured-1',
  async *stream() {
    yield { type: 'text_delta', text } satisfies StreamEvent
    yield { type: 'done', stopReason: 'end_turn' as const } satisfies StreamEvent
  },
  complete: async (): Promise<CompletionResponse> => ({
    content: json,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async () => json,
})

type Fixture = {
  daemon: Daemon
  handler: (req: Request) => Promise<Response>
  tempDir: string
}

let fixtures: Fixture[] = []

type MakeDaemonOpts = {
  authToken?: string
  provider?: Provider
  embedder?: Embedder
}

const makeDaemon = async (opts: MakeDaemonOpts = {}): Promise<Fixture> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-daemon-'))
  const daemon = new Daemon({
    root: tempDir,
    provider: opts.provider ?? makeFakeProvider('The hedgehog lives in hedgerows.'),
    ...(opts.embedder !== undefined ? { embedder: opts.embedder } : {}),
    ...(opts.authToken !== undefined ? { authToken: opts.authToken } : {}),
  })
  await daemon.start()
  const router = createRouter(daemon)
  const handler = async (req: Request): Promise<Response> => router(req)
  const fixture: Fixture = { daemon, handler, tempDir }
  fixtures.push(fixture)
  return fixture
}

beforeEach(() => {
  fixtures = []
})

afterEach(async () => {
  for (const f of fixtures) {
    await f.daemon.close()
    await rm(f.tempDir, { recursive: true, force: true })
  }
  fixtures = []
})

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

const createBrain = async (handler: Fixture['handler'], brainId: string): Promise<void> => {
  const resp = await handler(
    makeRequest('POST', '/v1/brains', {
      body: JSON.stringify({ brainId }),
      headers: { 'content-type': 'application/json' },
    }),
  )
  expect(resp.status).toBe(201)
}

describe('memory daemon integration', () => {
  it('1. exposes /healthz', async () => {
    const { handler } = await makeDaemon()
    const resp = await handler(makeRequest('GET', '/healthz'))
    expect(resp.status).toBe(200)
    const body = (await resp.json()) as { ok: boolean }
    expect(body.ok).toBe(true)
  })

  it('2. brain lifecycle (create/get/list/delete)', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'alpha')

    const got = await handler(makeRequest('GET', '/v1/brains/alpha'))
    expect(got.status).toBe(200)

    const list = await handler(makeRequest('GET', '/v1/brains'))
    const listBody = (await list.json()) as { items: { brainId: string }[] }
    expect(listBody.items.some((i) => i.brainId === 'alpha')).toBe(true)

    const noConfirm = await handler(makeRequest('DELETE', '/v1/brains/alpha'))
    expect(noConfirm.status).toBe(428)

    const confirmed = await handler(
      makeRequest('DELETE', '/v1/brains/alpha', {
        headers: { 'x-confirm-delete': 'yes' },
      }),
    )
    expect(confirmed.status).toBe(204)
  })

  it('3. document CRUD round-trip', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'docs')

    const put = await handler(
      makeRequest('PUT', '/v1/brains/docs/documents?path=memory%2Fglobal%2Fa.md', {
        body: new Uint8Array(Buffer.from('hello')),
        headers: { 'content-type': 'application/octet-stream' },
      }),
    )
    expect(put.status).toBe(204)

    const read = await handler(
      makeRequest('GET', '/v1/brains/docs/documents/read?path=memory%2Fglobal%2Fa.md'),
    )
    expect(read.status).toBe(200)
    const buf = new Uint8Array(await read.arrayBuffer())
    expect(Buffer.from(buf).toString('utf8')).toBe('hello')

    const head = await handler(
      makeRequest('HEAD', '/v1/brains/docs/documents?path=memory%2Fglobal%2Fa.md'),
    )
    expect(head.status).toBe(200)

    const list = await handler(
      makeRequest('GET', '/v1/brains/docs/documents?dir=memory%2Fglobal&recursive=true'),
    )
    const listBody = (await list.json()) as { items: { path: string }[] }
    expect(listBody.items.some((i) => i.path === 'memory/global/a.md')).toBe(true)

    const batch = await handler(
      makeRequest('POST', '/v1/brains/docs/documents/batch-ops', {
        body: JSON.stringify({
          reason: 'test',
          ops: [
            {
              type: 'write',
              path: 'memory/global/b.md',
              content_base64: Buffer.from('world').toString('base64'),
            },
            { type: 'delete', path: 'memory/global/a.md' },
          ],
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(batch.status).toBe(200)
    const batchBody = (await batch.json()) as { committed: number }
    expect(batchBody.committed).toBe(2)

    const gone = await handler(
      makeRequest('HEAD', '/v1/brains/docs/documents?path=memory%2Fglobal%2Fa.md'),
    )
    expect(gone.status).toBe(404)
  })

  it('4. ingest then search round-trip', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'ingest')

    const ingest = await handler(
      makeRequest('POST', '/v1/brains/ingest/ingest/file', {
        body: JSON.stringify({
          path: 'hedgehog.md',
          contentType: 'text/markdown',
          contentBase64: Buffer.from('# hedgehog\n\nThe hedgehog lives in hedgerows.').toString(
            'base64',
          ),
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(ingest.status).toBe(200)

    const search = await handler(
      makeRequest('POST', '/v1/brains/ingest/search', {
        body: JSON.stringify({ query: 'hedgehog', topK: 5 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)
    const body = (await search.json()) as { chunks: unknown[] }
    expect(body.chunks.length).toBeGreaterThan(0)
  })

  it('5. /ask streams retrieve → answer_delta → citation → done', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'asksse')
    await handler(
      makeRequest('POST', '/v1/brains/asksse/ingest/file', {
        body: JSON.stringify({
          path: 'hedgehog.md',
          contentBase64: Buffer.from('The hedgehog lives in hedgerows.').toString('base64'),
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    const ask = await handler(
      makeRequest('POST', '/v1/brains/asksse/ask', {
        body: JSON.stringify({ question: 'where does the hedgehog live', topK: 1 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(ask.status).toBe(200)
    const text = await ask.text()
    expect(text).toContain('event: retrieve')
    expect(text).toContain('event: answer_delta')
    expect(text).toContain('event: done')
  })

  it('6. /events streams a ready frame then a change on mutation', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'events')

    const respPromise = handler(makeRequest('GET', '/v1/brains/events/events'))
    const resp = await respPromise
    expect(resp.status).toBe(200)
    const reader = resp.body!.getReader()
    const decoder = new TextDecoder('utf-8')
    let collected = ''
    const readUntil = async (pred: (s: string) => boolean, budgetMs = 2000): Promise<string> => {
      const deadline = Date.now() + budgetMs
      while (!pred(collected) && Date.now() < deadline) {
        const race = Promise.race([
          reader.read(),
          new Promise<{ done: true; value?: undefined }>((resolve) =>
            setTimeout(() => resolve({ done: true }), 250),
          ),
        ])
        const { value, done } = (await race) as { value?: Uint8Array; done?: boolean }
        if (done === true) break
        if (value !== undefined) collected += decoder.decode(value, { stream: true })
      }
      return collected
    }
    await readUntil((s) => s.includes('event: ready'))
    expect(collected).toContain('event: ready')

    // Now mutate the store.
    const put = await handler(
      makeRequest('PUT', '/v1/brains/events/documents?path=evt.md', {
        body: new Uint8Array(Buffer.from('hi')),
        headers: { 'content-type': 'application/octet-stream' },
      }),
    )
    expect(put.status).toBe(204)

    await readUntil((s) => s.includes('event: change'), 2500)
    expect(collected).toContain('event: change')
    expect(collected).toContain('"path":"evt.md"')

    await reader.cancel()
  })

  it('7. maps not-found and oversized body to Problem+JSON', async () => {
    const { handler } = await makeDaemon()

    const missing = await handler(
      makeRequest('GET', '/v1/brains/missing/documents/read?path=a.md'),
    )
    expect(missing.status).toBe(404)
    const missingBody = (await missing.json()) as { code?: string }
    expect(missingBody.code).toBe('not_found')

    await createBrain(handler, 'sized')
    const oversized = new Uint8Array(3 * 1024 * 1024)
    oversized.fill(0x78) // 'x'
    const big = await handler(
      makeRequest('PUT', '/v1/brains/sized/documents?path=memory%2Fbig.md', {
        body: oversized,
        headers: { 'content-type': 'application/octet-stream' },
      }),
    )
    expect(big.status).toBe(413)
    const bigBody = (await big.json()) as { code?: string }
    expect(bigBody.code).toBe('payload_too_large')
  })

  it('8. auth middleware blocks anon and allows a valid token', async () => {
    const { handler } = await makeDaemon({ authToken: 'secret' })

    const hz = await handler(makeRequest('GET', '/healthz'))
    expect(hz.status).toBe(200)

    const anon = await handler(makeRequest('GET', '/v1/brains'))
    expect(anon.status).toBe(401)

    const wrong = await handler(
      makeRequest('GET', '/v1/brains', {
        headers: { authorization: 'Bearer wrong' },
      }),
    )
    expect(wrong.status).toBe(403)

    const ok = await handler(
      makeRequest('GET', '/v1/brains', {
        headers: { authorization: 'Bearer secret' },
      }),
    )
    expect(ok.status).toBe(200)
  })

  it('9. package identifier stays stable for version smoke', async () => {
    expect(MEMORY_PACKAGE).toBe('@jeffs-brain/memory')
    const { handler } = await makeDaemon()
    const hz = await handler(makeRequest('GET', '/healthz'))
    expect(hz.status).toBe(200)
  })

  it('10. ingest → search round-trip returns real chunk content (with embedder)', async () => {
    const embedder = createHashEmbedder()
    const { handler } = await makeDaemon({ embedder })
    await createBrain(handler, 'ingestpipe')

    const markdown = [
      '# Hedgehog facts',
      '',
      'The hedgehog lives in hedgerows.',
      'They eat slugs and beetles.',
      '',
      '## Habitat',
      '',
      'Prefers deciduous woodland and gardens with dense undergrowth.',
    ].join('\n')

    const ingest = await handler(
      makeRequest('POST', '/v1/brains/ingestpipe/ingest/file', {
        body: JSON.stringify({
          path: 'hedgehog.md',
          contentType: 'text/markdown',
          title: 'Hedgehog',
          contentBase64: Buffer.from(markdown).toString('base64'),
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(ingest.status).toBe(200)
    const ingestBody = (await ingest.json()) as {
      documentId: string
      path: string
      chunkCount: number
      bytes: number
      tookMs: number
    }
    expect(ingestBody.documentId).toMatch(/^ingestpipe:/)
    expect(ingestBody.chunkCount).toBeGreaterThan(0)
    expect(ingestBody.bytes).toBe(markdown.length)

    const search = await handler(
      makeRequest('POST', '/v1/brains/ingestpipe/search', {
        body: JSON.stringify({ query: 'hedgehog hedgerows', topK: 5 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)
    const searchBody = (await search.json()) as {
      chunks: { text: string; path: string; title: string }[]
    }
    expect(searchBody.chunks.length).toBeGreaterThan(0)
    const first = searchBody.chunks[0]
    expect(first?.path).toContain('hedgehog')
    expect(first?.text).toMatch(/hedgehog|hedgerow/i)
  })

  it('11. ask with evidence streams citations referencing the ingested path', async () => {
    const embedder = createHashEmbedder()
    const provider = makeFakeProvider('Hedgehogs live in hedgerows and dense undergrowth.')
    const { handler } = await makeDaemon({ embedder, provider })
    await createBrain(handler, 'askevidence')

    await handler(
      makeRequest('POST', '/v1/brains/askevidence/ingest/file', {
        body: JSON.stringify({
          path: 'habitat.md',
          contentType: 'text/markdown',
          contentBase64: Buffer.from(
            '# Habitat\n\nThe hedgehog lives in hedgerows and dense undergrowth.',
          ).toString('base64'),
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    const ask = await handler(
      makeRequest('POST', '/v1/brains/askevidence/ask', {
        body: JSON.stringify({ question: 'where does the hedgehog live', topK: 3 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(ask.status).toBe(200)
    const text = await ask.text()
    expect(text).toContain('event: retrieve')
    expect(text).toContain('event: answer_delta')
    expect(text).toContain('event: citation')
    expect(text).toContain('event: done')
    // The citation should reference the real ingested path.
    expect(text).toMatch(/"path":"raw\/documents\/[^"]*habitat/)
  })

  it('12. remember then recall round-trips through the real memory pipeline', async () => {
    const embedder = createHashEmbedder()
    const { handler } = await makeDaemon({ embedder })
    await createBrain(handler, 'memround')

    const remember = await handler(
      makeRequest('POST', '/v1/brains/memround/remember', {
        body: JSON.stringify({
          note: 'Alex prefers pragmatic, small TypeScript commits.',
          tags: ['preferences', 'workflow'],
          slug: 'alex-prefers-small-commits',
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(remember.status).toBe(201)
    const rememberBody = (await remember.json()) as { path: string; slug: string }
    expect(rememberBody.path).toBe('memory/global/alex-prefers-small-commits.md')

    const recall = await handler(
      makeRequest('POST', '/v1/brains/memround/recall', {
        body: JSON.stringify({ query: 'pragmatic', topK: 3 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(recall.status).toBe(200)
    const recallBody = (await recall.json()) as {
      memories: { path: string; content: string }[]
    }
    expect(recallBody.memories.length).toBeGreaterThan(0)
    const hit = recallBody.memories.find((m) => m.path === rememberBody.path)
    expect(hit).toBeDefined()
    expect(hit!.content).toContain('pragmatic')
  })

  it('13. extract persists structured memories returned by the provider', async () => {
    const structured = JSON.stringify({
      memories: [
        {
          action: 'create',
          filename: 'user-preference-commit-style',
          name: 'User prefers small TypeScript commits',
          description: 'Alex keeps commits narrow and typed.',
          type: 'user',
          content: 'Alex prefers pragmatic, small TypeScript commits.',
          indexEntry: '- [[user-preference-commit-style]]',
          scope: 'global',
          tags: ['preferences'],
        },
      ],
    })
    const provider = makeStructuredProvider(structured)
    const embedder = createHashEmbedder()
    const { handler } = await makeDaemon({ provider, embedder })
    await createBrain(handler, 'extracted')

    const extract = await handler(
      makeRequest('POST', '/v1/brains/extracted/extract', {
        body: JSON.stringify({
          messages: [
            {
              role: 'user',
              content: 'Please keep TypeScript commits small and pragmatic.',
            },
            {
              role: 'assistant',
              content: 'Noted. I will prefer narrow changes.',
            },
          ],
          scope: 'global',
          actorId: 'alex',
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(extract.status).toBe(200)
    const body = (await extract.json()) as {
      memories: Array<{
        filename: string
        name: string
        content: string
        type: string
        scope: string
      }>
    }
    const extracted = body.memories.find(
      (m) => m.filename === 'user-preference-commit-style',
    )
    expect(extracted).toBeDefined()
    expect(extracted!.content).toContain('pragmatic')
    expect(extracted!.scope).toBe('global')
  })

  it('14. search surfaces files pre-seeded on disk before the daemon opens the brain', async () => {
    // Tri-SDK contract: the Go eval runner writes memory facts to
    // $JB_HOME/brains/<id>/memory/global/*.md before any TS daemon is
    // spawned. The daemon must scan the brain root on open and index
    // whatever it finds so a follow-up /search returns hits rather
    // than an empty array. This regression check catches a broken
    // scanBrain pass (e.g. an async-in-async call that throws
    // silently, or a missing scope classifier that filters out valid
    // paths).
    const tempDir = await mkdtemp(join(tmpdir(), 'memory-preseed-'))
    const brainId = 'preseed'
    const seededFile = join(tempDir, 'brains', brainId, 'memory', 'global', 'hedgehog.md')
    const { mkdir, writeFile } = await import('node:fs/promises')
    await mkdir(join(tempDir, 'brains', brainId, 'memory', 'global'), { recursive: true })
    await writeFile(
      seededFile,
      [
        '---',
        'name: Hedgehog',
        'description: Small mammal that lives in hedgerows.',
        '---',
        '',
        'The hedgehog lives in hedgerows across Europe.',
      ].join('\n'),
      'utf8',
    )

    const daemon = new Daemon({
      root: tempDir,
      provider: makeFakeProvider('ok'),
      embedder: createHashEmbedder(),
    })
    await daemon.start()
    const handler = async (req: Request): Promise<Response> =>
      (await createRouter(daemon))(req)
    fixtures.push({ daemon, handler, tempDir })

    // Resolve the brain lazily — the manager opens it in the first
    // /search and kicks off scanBrain under the covers.
    const resp = await handler(
      makeRequest('POST', `/v1/brains/${brainId}/search`, {
        body: JSON.stringify({ query: 'hedgehog hedgerows', topK: 5 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(resp.status).toBe(200)
    const body = (await resp.json()) as { chunks: Array<{ path: string }> }
    expect(body.chunks.length).toBeGreaterThan(0)
    expect(body.chunks.some((c) => c.path.endsWith('hedgehog.md'))).toBe(true)
  })

  it('15. vector backfill populates knowledge_vectors for pre-seeded paths', async () => {
    // Parity with the Go daemon: after the initial FTS scan completes,
    // every indexed path should have a vector pinned under the active
    // embed model so /search runs hybrid (BM25 + vector) rather than
    // BM25-only.
    const tempDir = await mkdtemp(join(tmpdir(), 'memory-backfill-'))
    const brainId = 'backfill'
    const { mkdir, writeFile } = await import('node:fs/promises')
    await mkdir(join(tempDir, 'brains', brainId, 'memory', 'global'), { recursive: true })
    await writeFile(
      join(tempDir, 'brains', brainId, 'memory', 'global', 'alpha.md'),
      'The hedgehog lives in hedgerows.',
      'utf8',
    )
    await writeFile(
      join(tempDir, 'brains', brainId, 'memory', 'global', 'beta.md'),
      'Slugs and beetles are hedgehog food.',
      'utf8',
    )

    const daemon = new Daemon({
      root: tempDir,
      provider: makeFakeProvider('ok'),
      embedder: createHashEmbedder(),
      embedModel: 'hash-1024',
    })
    await daemon.start()
    const handler = async (req: Request): Promise<Response> =>
      (await createRouter(daemon))(req)
    fixtures.push({ daemon, handler, tempDir })

    // Trigger the lazy brain open + scanBrain + backfill chain.
    const search = await handler(
      makeRequest('POST', `/v1/brains/${brainId}/search`, {
        body: JSON.stringify({ query: 'hedgehog', topK: 3 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)

    const br = await daemon.brains.get(brainId)
    expect(br.index).toBeDefined()
    // Both markdown paths should end up with a vector tagged for the
    // active model once the backfill has drained.
    const withVectors = br.index!.chunkIdsWithVectorForModel('hash-1024')
    expect(withVectors.sort()).toEqual([
      'memory/global/alpha.md',
      'memory/global/beta.md',
    ])
  })
})
