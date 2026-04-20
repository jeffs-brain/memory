// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the augmented LME CoT reader prompt wired into
 * handleAsk. Uses an in-memory fake provider that captures the request
 * payload so we can assert prompt shape + LLM call params without
 * touching any real network.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { createHashEmbedder } from '../llm/index.js'
import type { CompletionRequest, CompletionResponse, Provider, StreamEvent } from '../llm/index.js'

import { Daemon, createRouter } from './index.js'

type CapturedCall = {
  request: CompletionRequest
}

type CapturingProvider = Provider & {
  readonly calls: CapturedCall[]
}

const makeCapturingProvider = (text: string): CapturingProvider => {
  const calls: CapturedCall[] = []
  const provider: CapturingProvider = {
    calls,
    name: () => 'capture',
    modelName: () => 'capture-1',
    async *stream(request) {
      calls.push({ request })
      yield { type: 'text_delta', text } satisfies StreamEvent
      yield { type: 'done', stopReason: 'end_turn' as const } satisfies StreamEvent
    },
    complete: async (request): Promise<CompletionResponse> => {
      calls.push({ request })
      return {
        content: text,
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    },
    supportsStructuredDecoding: () => false,
    structured: async () => text,
  }
  return provider
}

type Fixture = {
  daemon: Daemon
  handler: (req: Request) => Promise<Response>
  provider: CapturingProvider
  tempDir: string
}

let fixtures: Fixture[] = []

const makeFixture = async (): Promise<Fixture> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-handlers-'))
  const provider = makeCapturingProvider('The hedgehog lives in hedgerows.')
  const daemon = new Daemon({
    root: tempDir,
    provider,
    embedder: createHashEmbedder(),
  })
  await daemon.start()
  const router = createRouter(daemon)
  const handler = async (req: Request): Promise<Response> => router(req)
  const fixture: Fixture = { daemon, handler, provider, tempDir }
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

const seedBrain = async (handler: Fixture['handler'], brainId: string): Promise<void> => {
  const create = await handler(
    makeRequest('POST', '/v1/brains', {
      body: JSON.stringify({ brainId }),
      headers: { 'content-type': 'application/json' },
    }),
  )
  expect(create.status).toBe(201)

  const ingest = await handler(
    makeRequest('POST', `/v1/brains/${brainId}/ingest/file`, {
      body: JSON.stringify({
        path: 'hedgehog.md',
        contentType: 'text/markdown',
        title: 'Hedgehog',
        contentBase64: Buffer.from(
          '# Hedgehog\n\nThe hedgehog lives in hedgerows and dense undergrowth.',
        ).toString('base64'),
      }),
      headers: { 'content-type': 'application/json' },
    }),
  )
  expect(ingest.status).toBe(200)
}

const drainAsk = async (
  handler: Fixture['handler'],
  body: Record<string, unknown>,
): Promise<string> => {
  const resp = await handler(
    makeRequest('POST', '/v1/brains/lme/ask', {
      body: JSON.stringify(body),
      headers: { 'content-type': 'application/json' },
    }),
  )
  expect(resp.status).toBe(200)
  return resp.text()
}

const expectDefined = <T>(value: T | undefined, message: string): T => {
  if (value === undefined) throw new Error(message)
  return value
}

const expectString = (value: unknown, message: string): string => {
  if (typeof value !== 'string') throw new Error(message)
  return value
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

describe('handleAsk reader modes', () => {
  it('search forwards exact path filters to retrieval', async () => {
    const { daemon, handler } = await makeFixture()
    await seedBrain(handler, 'lme')

    const brain = await daemon.brains.get('lme')
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

    const resp = await handler(
      makeRequest('POST', '/v1/brains/lme/search', {
        body: JSON.stringify({
          query: 'hedgehog',
          filters: {
            documentPaths: ['hedgehog.md'],
          },
        }),
        headers: { 'content-type': 'application/json' },
      }),
    )

    expect(resp.status).toBe(200)
    expect(capture.request?.filters).toEqual({ paths: ['hedgehog.md'] })
  })

  it('basic mode keeps the existing prompt + params', async () => {
    const { handler, provider } = await makeFixture()
    await seedBrain(handler, 'lme')

    await drainAsk(handler, { question: 'where does the hedgehog live', topK: 3 })

    expect(provider.calls.length).toBe(1)
    const call = expectDefined(provider.calls[0], 'expected provider call')
    expect(call.request.maxTokens).toBe(1024)
    expect(call.request.temperature).toBe(0.2)
    expect(call.request.messages.length).toBe(2)
    expect(call.request.messages[0]?.role).toBe('system')
    expect(call.request.messages[1]?.role).toBe('user')
    const userPrompt = expectString(
      call.request.messages[1]?.content,
      'expected user prompt content',
    )
    expect(userPrompt).toContain('## Evidence')
    expect(userPrompt).toContain('## Question')
    expect(userPrompt).not.toContain('Answer (step by step)')
  })

  it('augmented mode emits the LME CoT reader prompt with recency, enumeration, and temporal guidance', async () => {
    const { handler, provider } = await makeFixture()
    await seedBrain(handler, 'lme')

    await drainAsk(handler, {
      question: 'list every place the hedgehog has been seen recently',
      topK: 3,
      readerMode: 'augmented',
      questionDate: '2024-05-26',
    })

    expect(provider.calls.length).toBe(1)
    const call = expectDefined(provider.calls[0], 'expected provider call')
    expect(call.request.maxTokens).toBe(800)
    expect(call.request.temperature).toBe(0.0)
    expect(call.request.messages.length).toBe(1)
    expect(call.request.messages[0]?.role).toBe('user')
    const prompt = expectString(
      call.request.messages[0]?.content,
      'expected augmented prompt content',
    )

    // CoT framing
    expect(prompt).toContain('Answer the question step by step: first extract')
    expect(prompt).toContain('Answer (step by step):')

    // Recency / supersession guidance
    expect(prompt).toContain('prefer the value from the most recent session date')
    expect(prompt).toContain('One later correction outweighs any number of earlier mentions')

    // Enumeration guidance
    expect(prompt).toContain('When the question asks to list, count, enumerate, or total')
    expect(prompt).toContain('one per line')
    expect(prompt).toContain('If any named part is missing or lacks an amount')
    expect(prompt).toContain('count it once')
    expect(prompt).toContain('prefer direct transactional facts over plans, budgets')

    // Temporal anchor
    expect(prompt).toContain('Today is 2024-05-26 (Sunday)')
    expect(prompt).toContain('Current Date: 2024-05-26')

    // Conflict resolution and abstention guidance
    expect(prompt).toContain('30-minute morning commute')
    expect(prompt).toContain('combine them if the connection is explicit')
    expect(prompt).toContain('state that clearly in the first sentence')
    expect(prompt).toContain('the information provided is not enough to answer the question')

    // Evidence uses the numbered retrieve-only rendering.
    expect(prompt).toContain('Retrieved facts')
    expect(prompt).toContain('[hedgehog-md]')
    expect(prompt).toContain('The hedgehog lives in hedgerows')

    // Question still echoed
    expect(prompt).toContain('list every place the hedgehog has been seen recently')
  })

  it('augmented mode falls back to "unknown" anchor when questionDate is absent', async () => {
    const { handler, provider } = await makeFixture()
    await seedBrain(handler, 'lme')

    await drainAsk(handler, {
      question: 'how many sightings in total',
      topK: 1,
      readerMode: 'augmented',
    })

    const call = expectDefined(provider.calls[0], 'expected provider call')
    const prompt = expectString(
      call.request.messages[0]?.content,
      'expected augmented prompt content',
    )
    expect(prompt).toContain('Today is unknown')
    expect(prompt).toContain('Current Date: unknown')
  })

  it('augmented mode resolves anchored submission dates without calling the provider', async () => {
    const { handler, provider } = await makeFixture()

    const create = await handler(
      makeRequest('POST', '/v1/brains', {
        body: JSON.stringify({ brainId: 'lme' }),
        headers: { 'content-type': 'application/json' },
      }),
    )
    expect(create.status).toBe(201)

    for (const file of [
      {
        path: 'paper.md',
        title: 'Paper note',
        body: '# Paper note\n\nI submitted my research paper on sentiment analysis to ACL.',
      },
      {
        path: 'acl-date.md',
        title: 'ACL date note',
        body: "# ACL date note\n\nI'm reviewing for ACL, and their submission date was February 1st.",
      },
    ]) {
      const ingest = await handler(
        makeRequest('POST', '/v1/brains/lme/ingest/file', {
          body: JSON.stringify({
            path: file.path,
            contentType: 'text/markdown',
            title: file.title,
            contentBase64: Buffer.from(file.body).toString('base64'),
          }),
          headers: { 'content-type': 'application/json' },
        }),
      )
      expect(ingest.status).toBe(200)
    }

    const body = await drainAsk(handler, {
      question: 'When did I submit my research paper on sentiment analysis?',
      topK: 5,
      readerMode: 'augmented',
      mode: 'bm25',
    })

    expect(provider.calls.length).toBe(0)
    expect(body).toContain('February 1st')
    expect(body).toContain('citation')
  })

  it('unknown readerMode value is rejected', async () => {
    const { handler, provider } = await makeFixture()
    await seedBrain(handler, 'lme')

    const res = await handler(
      new Request('http://local/v1/brains/lme/ask', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          question: 'where does the hedgehog live',
          topK: 1,
          readerMode: 'mystery',
        }),
      }),
    )

    expect(res.status).toBe(400)
    await expect(res.text()).resolves.toContain('readerMode')
    expect(provider.calls.length).toBe(0)
  })

  it('ask forwards candidateK, rerankTopN, and filters to retrieval', async () => {
    const { daemon, handler } = await makeFixture()
    await seedBrain(handler, 'lme')

    const brain = await daemon.brains.get('lme')
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

    await drainAsk(handler, {
      question: 'where does the hedgehog live',
      topK: 3,
      candidateK: 80,
      rerankTopN: 40,
      mode: 'hybrid-rerank',
      filters: {
        paths: ['hedgehog.md'],
      },
    })

    expect(capture.request?.candidateK).toBe(80)
    expect(capture.request?.rerankTopN).toBe(40)
    expect(capture.request?.filters).toEqual({ paths: ['hedgehog.md'] })
  })
})
