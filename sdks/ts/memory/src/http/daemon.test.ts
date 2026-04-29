// SPDX-License-Identifier: Apache-2.0

/**
 * Integration tests for the memory HTTP daemon. Mirrors the nine Go
 * tests in `go/cmd/memory/serve_integration_test.go` so the two
 * SDKs stay behaviourally aligned.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { MEMORY_PACKAGE } from '../index.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Provider,
  StreamEvent,
} from '../llm/index.js'
import { createHashEmbedder } from '../llm/index.js'
import { createContextualPrefixBuilder } from '../memory/index.js'
import type { DocumentBodyLimits } from '../store/index.js'
import { Daemon, createRouter } from './index.js'

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

const makeCapturingProvider = (capture: { prompts: string[] }, text = 'ok'): Provider => ({
  name: () => 'fake-capturing',
  modelName: () => 'fake-capturing-1',
  async *stream(req: CompletionRequest) {
    capture.prompts.push(req.messages.map((message) => message.content ?? '').join('\n\n'))
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

const makeQueuedProvider = (responses: readonly string[]): Provider => {
  let index = 0
  const next = (): string => responses[Math.min(index++, Math.max(0, responses.length - 1))] ?? ''
  return {
    name: () => 'fake-queued',
    modelName: () => 'fake-queued-1',
    async *stream() {
      yield { type: 'done', stopReason: 'end_turn' as const } satisfies StreamEvent
    },
    complete: async (): Promise<CompletionResponse> => ({
      content: next(),
      toolCalls: [],
      usage: { inputTokens: 0, outputTokens: 0 },
      stopReason: 'end_turn',
    }),
    supportsStructuredDecoding: () => false,
    structured: async () => next(),
  }
}

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
  contextualise?: boolean
  contextualiseCacheDir?: string
  bodyLimits?: Partial<DocumentBodyLimits>
}

const makeDaemon = async (opts: MakeDaemonOpts = {}): Promise<Fixture> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-daemon-'))
  const daemon = new Daemon({
    root: tempDir,
    provider: opts.provider ?? makeFakeProvider('The hedgehog lives in hedgerows.'),
    ...(opts.embedder !== undefined ? { embedder: opts.embedder } : {}),
    ...(opts.authToken !== undefined ? { authToken: opts.authToken } : {}),
    ...(opts.contextualise
      ? {
          contextualPrefixBuilder: createContextualPrefixBuilder({
            provider: opts.provider ?? makeFakeProvider('The hedgehog lives in hedgerows.'),
            ...(opts.contextualiseCacheDir !== undefined
              ? { cacheDir: opts.contextualiseCacheDir }
              : {}),
          }),
        }
      : {}),
    ...(opts.bodyLimits !== undefined ? { bodyLimits: opts.bodyLimits } : {}),
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

const sleep = async (ms: number): Promise<void> =>
  await new Promise((resolve) => setTimeout(resolve, ms))

const waitFor = async (
  check: () => void,
  opts: { timeoutMs?: number; intervalMs?: number } = {},
): Promise<void> => {
  const timeoutMs = opts.timeoutMs ?? 5_000
  const intervalMs = opts.intervalMs ?? 25
  const deadline = Date.now() + timeoutMs
  let lastErr: unknown
  while (Date.now() < deadline) {
    try {
      check()
      return
    } catch (err) {
      lastErr = err
      await sleep(intervalMs)
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(`timed out after ${timeoutMs}ms`)
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

  it('4a. leaves untitled raw transcripts untitled in the search index', async () => {
    const fixture = await makeDaemon()
    const { handler, daemon } = fixture
    await createBrain(handler, 'titles')

    const rawTranscript = [
      '---',
      'session_id: answer_sharegpt_example',
      'session_date: 2023/05/30 (Tue) 17:19',
      '---',
      '',
      '[user]: We finally named the Radiation Amplified zombie Fissionator.',
      '',
    ].join('\n')

    const put = await handler(
      makeRequest(
        'PUT',
        '/v1/brains/titles/documents?path=raw%2Flme%2Fanswer_sharegpt_example.md',
        {
          body: Buffer.from(rawTranscript, 'utf8'),
          headers: { 'content-type': 'application/octet-stream' },
        },
      ),
    )
    expect(put.status).toBe(204)

    const res = await daemon.brains.get('titles')
    await res.refresh()
    const chunk = res.index?.getChunk('raw/lme/answer_sharegpt_example.md')
    expect(chunk).toBeDefined()
    expect(chunk?.title).toBe('')
    expect(chunk?.content).toContain('Fissionator')
    expect(chunk?.content).not.toContain('session_id:')
  })

  it('4b. search forwards retrieval filters and returns trace data', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'filters')

    const globalWrite = await handler(
      makeRequest('PUT', '/v1/brains/filters/documents?path=memory%2Fglobal%2Fhedgehog.md', {
        body: Buffer.from('The global hedgehog lives in hedgerows.', 'utf8'),
      }),
    )
    expect(globalWrite.status).toBe(204)

    const projectWrite = await handler(
      makeRequest(
        'PUT',
        '/v1/brains/filters/documents?path=memory%2Fproject%2Fdemo%2Fhedgehog.md',
        {
          body: Buffer.from('The project hedgehog lives in hedgerows.', 'utf8'),
        },
      ),
    )
    expect(projectWrite.status).toBe(204)

    const search = await handler(
      makeRequest('POST', '/v1/brains/filters/search', {
        body: JSON.stringify({
          query: 'hedgehog',
          topK: 5,
          filters: { scope: 'global' },
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)

    const body = (await search.json()) as {
      chunks: Array<{ path: string }>
      trace?: { filtersApplied?: boolean }
      attempts?: unknown[]
    }
    expect(body.chunks).toHaveLength(1)
    expect(body.chunks[0]?.path).toBe('memory/global/hedgehog.md')
    expect(body.trace?.filtersApplied).toBe(true)
    expect(Array.isArray(body.attempts)).toBe(true)
  })

  it('4c. search keeps global and matching project memory when scope is memory', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'memory-scope-filters')

    const writes = [
      {
        path: 'memory/global/hedgehog.md',
        content: 'Global hedgehog facts for the eval memory track.',
      },
      {
        path: 'memory/project/eval-lme/hedgehog.md',
        content: 'Project hedgehog facts for eval-lme.',
      },
      {
        path: 'memory/project/other/hedgehog.md',
        content: 'Project hedgehog facts for another project.',
      },
      {
        path: 'raw/eval-lme/hedgehog.md',
        content: 'Raw hedgehog transcript for eval-lme.',
      },
    ] as const

    for (const { path, content } of writes) {
      const response = await handler(
        makeRequest(
          'PUT',
          `/v1/brains/memory-scope-filters/documents?path=${encodeURIComponent(path)}`,
          {
            body: Buffer.from(content, 'utf8'),
          },
        ),
      )
      expect(response.status).toBe(204)
    }

    const search = await handler(
      makeRequest('POST', '/v1/brains/memory-scope-filters/search', {
        body: JSON.stringify({
          query: 'hedgehog eval-lme',
          topK: 10,
          filters: { scope: 'memory', project: 'eval-lme' },
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)

    const body = (await search.json()) as {
      chunks: Array<{ path: string }>
      trace?: { filtersApplied?: boolean }
    }
    const paths = body.chunks.map((chunk) => chunk.path).sort()
    expect(paths).toEqual(['memory/global/hedgehog.md', 'memory/project/eval-lme/hedgehog.md'])
    expect(body.trace?.filtersApplied).toBe(true)
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

  it('5a. augmented ask renders numbered session facts and strips frontmatter', async () => {
    const capture = { prompts: [] as string[] }
    const provider = makeCapturingProvider(capture)
    const { handler } = await makeDaemon({ provider })
    await createBrain(handler, 'askaugmented')

    const docs = [
      {
        path: 'raw/lme/session-2.md',
        content:
          '---\nsession_id: s2\nsession_date: 2024-02-20\n---\n[user]: The bike is blue now.\n',
      },
      {
        path: 'raw/lme/session-1-a.md',
        content: '---\nsession_id: s1\nsession_date: 2024-01-10\n---\n[user]: I bought a bike.\n',
      },
      {
        path: 'raw/lme/session-1-b.md',
        content:
          '---\nsession_id: s1\nsession_date: 2024-01-10\n---\n[user]: The bike was red at first.\n',
      },
    ] as const

    for (const doc of docs) {
      const put = await handler(
        makeRequest(
          'PUT',
          `/v1/brains/askaugmented/documents?path=${encodeURIComponent(doc.path)}`,
          {
            body: Buffer.from(doc.content, 'utf8'),
            headers: { 'content-type': 'text/markdown' },
          },
        ),
      )
      expect(put.status).toBe(204)
    }

    const ask = await handler(
      makeRequest('POST', '/v1/brains/askaugmented/ask', {
        body: JSON.stringify({
          question: 'What colour is the bike now?',
          topK: 3,
          mode: 'bm25',
          readerMode: 'augmented',
          questionDate: '2024-02-21',
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    expect(ask.status).toBe(200)
    expect(capture.prompts).toHaveLength(1)
    const prompt = capture.prompts[0] ?? ''
    expect(prompt).toContain('Retrieved facts (3):')
    expect(prompt).toContain('[session=s2]')
    expect(prompt).toContain('[session=s1]')
    expect(prompt).toContain('[2024-02-20]')
    expect(prompt).toContain('[2024-01-10]')
    expect(prompt).not.toContain('session_id:')
    expect(prompt).not.toContain('session_date:')
    const firstS1 = prompt.indexOf('[session=s1]')
    const secondS1 = prompt.lastIndexOf('[session=s1]')
    const s2 = prompt.indexOf('[session=s2]')
    expect(s2).toBeGreaterThanOrEqual(0)
    expect(firstS1).toBeGreaterThanOrEqual(0)
    expect(secondS1).toBeGreaterThan(firstS1)
  })

  it('5b. /search forwards candidateK and rerankTopN to retrieval', async () => {
    const { daemon, handler } = await makeDaemon({ embedder: createHashEmbedder() })
    await createBrain(handler, 'searchknobs')
    await handler(
      makeRequest('PUT', '/v1/brains/searchknobs/documents?path=note.md', {
        body: new Uint8Array(Buffer.from('apples and pears')),
        headers: { 'content-type': 'text/plain' },
      }),
    )

    const brain = await daemon.brains.get('searchknobs')
    expect(brain.retrieval).toBeDefined()
    const capture: { request?: Record<string, unknown> } = {}
    const originalSearchRaw = brain.retrieval?.searchRaw.bind(brain.retrieval)
    ;(
      brain.retrieval as {
        searchRaw: (request: Record<string, unknown>) => ReturnType<typeof originalSearchRaw>
      }
    ).searchRaw = async (request) => {
      capture.request = request
      return originalSearchRaw(request)
    }

    const search = await handler(
      makeRequest('POST', '/v1/brains/searchknobs/search', {
        body: JSON.stringify({
          query: 'apples',
          topK: 3,
          candidateK: 80,
          rerankTopN: 40,
          mode: 'hybrid-rerank',
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    expect(search.status).toBe(200)
    expect(capture.request?.candidateK).toBe(80)
    expect(capture.request?.rerankTopN).toBe(40)
  })

  it('6. /events streams a ready frame then a change on mutation', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'events')

    const respPromise = handler(makeRequest('GET', '/v1/brains/events/events'))
    const resp = await respPromise
    expect(resp.status).toBe(200)
    const reader = resp.body?.getReader()
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

  it('6b. /events emits ping heartbeats and stops them on disconnect', async () => {
    vi.useFakeTimers()
    try {
      const { handler } = await makeDaemon()
      await createBrain(handler, 'eventsping')

      const resp = await handler(makeRequest('GET', '/v1/brains/eventsping/events'))
      expect(resp.status).toBe(200)
      const reader = resp.body?.getReader()
      expect(reader).toBeDefined()
      if (reader === undefined) throw new Error('expected response body')
      const decoder = new TextDecoder('utf-8')

      const ready = await reader.read()
      expect(ready.done).toBe(false)
      expect(decoder.decode(ready.value, { stream: true })).toBe(
        'event: ready\nid: 1\ndata: ok\n\n',
      )

      await vi.advanceTimersByTimeAsync(25_000)
      const ping = await reader.read()
      expect(ping.done).toBe(false)
      expect(decoder.decode(ping.value, { stream: true })).toBe(
        'event: ping\nid: 2\ndata: keepalive\n\n',
      )

      await reader.cancel()
      await Promise.resolve()
      expect(vi.getTimerCount()).toBe(0)
    } finally {
      vi.useRealTimers()
    }
  })

  it('7. maps not-found and oversized body to Problem+JSON', async () => {
    const { handler } = await makeDaemon()

    const missing = await handler(makeRequest('GET', '/v1/brains/missing/documents/read?path=a.md'))
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

  it('7b. honours configured batch ceilings on the daemon', async () => {
    const { handler } = await makeDaemon({
      bodyLimits: {
        batchDecodedBytes: 8,
        batchOpCount: 2,
      },
    })
    await createBrain(handler, 'limited')

    const tooManyOps = await handler(
      makeRequest('POST', '/v1/brains/limited/documents/batch-ops', {
        body: JSON.stringify({
          reason: 'test',
          ops: [
            { type: 'write', path: 'a.md', content_base64: Buffer.from('a').toString('base64') },
            { type: 'write', path: 'b.md', content_base64: Buffer.from('b').toString('base64') },
            { type: 'write', path: 'c.md', content_base64: Buffer.from('c').toString('base64') },
          ],
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(tooManyOps.status).toBe(413)

    const tooLarge = await handler(
      makeRequest('POST', '/v1/brains/limited/documents/batch-ops', {
        body: JSON.stringify({
          reason: 'test',
          ops: [
            {
              type: 'write',
              path: 'big.md',
              content_base64: Buffer.from('123456789').toString('base64'),
            },
          ],
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(tooLarge.status).toBe(413)
    const tooLargeBody = (await tooLarge.json()) as { code?: string }
    expect(tooLargeBody.code).toBe('payload_too_large')
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

  it('11a. /search returns retrieval metadata for retrieve-only callers', async () => {
    const { handler } = await makeDaemon()
    await createBrain(handler, 'searchmetadata')

    const put = await handler(
      makeRequest('PUT', '/v1/brains/searchmetadata/documents?path=raw%2Flme%2Fsession-1.md', {
        body: Buffer.from(
          '---\nsession_id: s1\nsession_date: 2024-03-08\n---\n[user]: I bought apples.\n',
          'utf8',
        ),
        headers: { 'content-type': 'text/markdown' },
      }),
    )
    expect(put.status).toBe(204)

    const search = await handler(
      makeRequest('POST', '/v1/brains/searchmetadata/search', {
        body: JSON.stringify({ query: 'apples', topK: 1, mode: 'bm25' }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)
    const body = (await search.json()) as {
      chunks: Array<{
        path: string
        metadata?: Record<string, unknown>
      }>
    }
    expect(body.chunks).toHaveLength(1)
    expect(body.chunks[0]?.path).toBe('raw/lme/session-1.md')
    expect(body.chunks[0]?.metadata?.sessionId).toBe('s1')
    expect(body.chunks[0]?.metadata?.sessionDate).toBe('2024-03-08')
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
    expect(hit?.content).toContain('pragmatic')
  })

  it('13. extract previews structured memories returned by the provider without persisting', async () => {
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
    const { daemon, handler } = await makeDaemon({ provider, embedder })
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
    const extracted = body.memories.find((m) => m.filename === 'user-preference-commit-style')
    expect(extracted).toBeDefined()
    expect(extracted?.content).toContain('pragmatic')
    expect(extracted?.scope).toBe('global')
    const exists = await daemon.brains
      .get('extracted')
      .then((brain) => brain.store.exists('memory/global/alex/user-preference-commit-style.md'))
    expect(exists).toBe(false)
  })

  it('13a. extract carries session metadata and contextual prefixes when enabled', async () => {
    const provider = makeQueuedProvider([
      JSON.stringify({
        memories: [
          {
            action: 'create',
            filename: 'bike-status',
            name: 'Bike status',
            description: 'The user confirmed the bike colour.',
            type: 'project',
            content: 'The bike is blue now.',
            indexEntry: '- [[bike-status]]',
            scope: 'project',
          },
        ],
      }),
      'The session happened on a February check-in about the bike, and this fact records the current colour after the latest update.',
    ])
    const { handler } = await makeDaemon({
      provider,
      embedder: createHashEmbedder(),
      contextualise: true,
      contextualiseCacheDir: join(tmpdir(), 'memory-contextual-prefix-test'),
    })
    await createBrain(handler, 'contextualised')

    const extract = await handler(
      makeRequest('POST', '/v1/brains/contextualised/extract', {
        body: JSON.stringify({
          messages: [
            {
              role: 'system',
              content: 'Session on 2024-02-20 about the bike status.',
            },
            {
              role: 'user',
              content: 'The bike is blue now.',
            },
          ],
          scope: 'project',
          actorId: 'alex',
          sessionId: 'session-42',
          sessionDate: '2024/02/20 (Tue) 09:15',
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    expect(extract.status).toBe(200)
    const body = (await extract.json()) as {
      memories: Array<{
        filename: string
        scope: string
        sessionId?: string
        sessionDate?: string
        contextPrefix?: string
      }>
    }
    expect(body.memories[0]?.sessionId).toBe('session-42')
    expect(body.memories[0]?.sessionDate).toBe('2024-02-20')
    expect(body.memories[0]?.contextPrefix).toContain('bike')

    const exists = await fixtures[fixtures.length - 1]?.daemon.brains
      .get('contextualised')
      .then((brain) => brain.store.exists('memory/project/alex/bike-status.md'))
    expect(exists).toBe(false)
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
    const handler = async (req: Request): Promise<Response> => (await createRouter(daemon))(req)
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
    const handler = async (req: Request): Promise<Response> => (await createRouter(daemon))(req)
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
    await waitFor(() => {
      const withVectors = br.index?.chunkIdsWithVectorForModel('hash-1024')
      expect(withVectors.sort()).toEqual(['memory/global/alpha.md', 'memory/global/beta.md'])
    })
  })

  it('15a. vector backfill honours the active embedder dimension', async () => {
    const tempDir = await mkdtemp(join(tmpdir(), 'memory-backfill-dim-'))
    const brainId = 'backfill-dim'
    const { mkdir, writeFile } = await import('node:fs/promises')
    await mkdir(join(tempDir, 'brains', brainId, 'memory', 'global'), { recursive: true })
    await writeFile(
      join(tempDir, 'brains', brainId, 'memory', 'global', 'alpha.md'),
      'Alpha lives in the hedgerows.',
      'utf8',
    )
    await writeFile(
      join(tempDir, 'brains', brainId, 'memory', 'global', 'beta.md'),
      'Beta likes beetles.',
      'utf8',
    )

    const embedder: Embedder = {
      name: () => 'stub-1536',
      model: () => 'stub-1536',
      dimension: () => 1536,
      async embed(texts) {
        return texts.map((_text, offset) => {
          const vec = new Array<number>(1536).fill(0)
          vec[offset % 1536] = 1
          return vec
        })
      },
    }

    const daemon = new Daemon({
      root: tempDir,
      provider: makeFakeProvider('ok'),
      embedder,
      embedModel: 'stub-1536',
    })
    await daemon.start()
    const handler = async (req: Request): Promise<Response> => (await createRouter(daemon))(req)
    fixtures.push({ daemon, handler, tempDir })

    const search = await handler(
      makeRequest('POST', `/v1/brains/${brainId}/search`, {
        body: JSON.stringify({ query: 'hedgerows', topK: 3 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(search.status).toBe(200)

    const br = await daemon.brains.get(brainId)
    expect(br.index).toBeDefined()
    expect(br.index?.vectorDim).toBe(1536)
    await waitFor(() => {
      expect(br.index?.chunkIdsWithVectorForModel('stub-1536').sort()).toEqual([
        'memory/global/alpha.md',
        'memory/global/beta.md',
      ])
    })
  })

  it('15b. initial scan prunes stale index rows left behind by an older on-disk layout', async () => {
    const tempDir = await mkdtemp(join(tmpdir(), 'memory-stale-index-'))
    const brainId = 'stale-index'
    const { mkdir, writeFile, rm } = await import('node:fs/promises')
    const flatPath = join(tempDir, 'brains', brainId, 'memory', 'project_sentiment_analysis.md')
    await mkdir(join(tempDir, 'brains', brainId, 'memory'), { recursive: true })
    await writeFile(
      flatPath,
      [
        '---',
        'name: Sentiment analysis',
        'description: Flat replay path from an older layout.',
        '---',
        '',
        'The user submitted the sentiment analysis paper in May 2023.',
      ].join('\n'),
      'utf8',
    )

    const firstDaemon = new Daemon({
      root: tempDir,
      provider: makeFakeProvider('ok'),
      embedder: createHashEmbedder(),
    })
    await firstDaemon.start()
    const firstHandler = createRouter(firstDaemon)
    fixtures.push({ daemon: firstDaemon, handler: firstHandler, tempDir })

    const firstSearch = await firstHandler(
      makeRequest('POST', `/v1/brains/${brainId}/search`, {
        body: JSON.stringify({ query: 'sentiment analysis', topK: 5 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(firstSearch.status).toBe(200)
    await firstDaemon.close()

    const nestedPath = join(
      tempDir,
      'brains',
      brainId,
      'memory',
      'project',
      'eval-lme',
      'project_sentiment_analysis.md',
    )
    await rm(flatPath)
    await mkdir(join(tempDir, 'brains', brainId, 'memory', 'project', 'eval-lme'), {
      recursive: true,
    })
    await writeFile(
      nestedPath,
      [
        '---',
        'name: Sentiment analysis',
        'description: Canonical project replay path.',
        '---',
        '',
        'The user submitted the sentiment analysis paper in May 2023.',
      ].join('\n'),
      'utf8',
    )

    const secondDaemon = new Daemon({
      root: tempDir,
      provider: makeFakeProvider('ok'),
      embedder: createHashEmbedder(),
    })
    await secondDaemon.start()
    const secondHandler = createRouter(secondDaemon)
    fixtures.push({ daemon: secondDaemon, handler: secondHandler, tempDir })

    const secondSearch = await secondHandler(
      makeRequest('POST', `/v1/brains/${brainId}/search`, {
        body: JSON.stringify({ query: 'sentiment analysis', topK: 5 }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(secondSearch.status).toBe(200)

    const br = await secondDaemon.brains.get(brainId)
    expect(br.index?.indexedPaths()).toContain(
      'memory/project/eval-lme/project_sentiment_analysis.md',
    )
    expect(br.index?.indexedPaths()).not.toContain('memory/project_sentiment_analysis.md')
  })

  it('15c. first search does not wait for vector backfill to finish', async () => {
    const tempDir = await mkdtemp(join(tmpdir(), 'memory-nonblocking-backfill-'))
    const brainId = 'nonblocking-backfill'
    const { mkdir, writeFile } = await import('node:fs/promises')
    await mkdir(join(tempDir, 'brains', brainId, 'memory', 'global'), { recursive: true })
    await writeFile(
      join(tempDir, 'brains', brainId, 'memory', 'global', 'alpha.md'),
      'The hedgehog lives in hedgerows.',
      'utf8',
    )

    let releaseEmbed: (() => void) | undefined
    const embedGate = new Promise<void>((resolve) => {
      releaseEmbed = resolve
    })
    const embedder: Embedder = {
      name: () => 'gated-hash',
      model: () => 'gated-hash',
      dimension: () => 4,
      async embed(texts) {
        await embedGate
        return texts.map((_text, index) => {
          const vec = [0, 0, 0, 0]
          vec[index % vec.length] = 1
          return vec
        })
      },
    }

    const daemon = new Daemon({
      root: tempDir,
      provider: makeFakeProvider('ok'),
      embedder,
      embedModel: 'gated-hash',
    })
    await daemon.start()
    const handler = async (req: Request): Promise<Response> => (await createRouter(daemon))(req)
    fixtures.push({ daemon, handler, tempDir })

    const searchPromise = handler(
      makeRequest('POST', `/v1/brains/${brainId}/search`, {
        body: JSON.stringify({ query: 'hedgehog hedgerows', topK: 5, mode: 'bm25' }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    const timeoutResponse = new Promise<Response>((_resolve, reject) => {
      setTimeout(() => reject(new Error('search waited on vector backfill')), 150)
    })
    const search = await Promise.race([searchPromise, timeoutResponse])
    expect(search.status).toBe(200)

    releaseEmbed?.()
    const br = await daemon.brains.get(brainId)
    await waitFor(() => {
      expect(br.index?.chunkIdsWithVectorForModel('gated-hash')).toEqual(['memory/global/alpha.md'])
    })
  })
})
