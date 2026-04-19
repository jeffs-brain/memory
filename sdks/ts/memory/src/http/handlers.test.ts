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
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StreamEvent,
} from '../llm/index.js'

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

const drainAsk = async (handler: Fixture['handler'], body: Record<string, unknown>): Promise<string> => {
  const resp = await handler(
    makeRequest('POST', '/v1/brains/lme/ask', {
      body: JSON.stringify(body),
      headers: { 'content-type': 'application/json' },
    }),
  )
  expect(resp.status).toBe(200)
  return resp.text()
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
  it('basic mode keeps the existing prompt + params', async () => {
    const { handler, provider } = await makeFixture()
    await seedBrain(handler, 'lme')

    await drainAsk(handler, { question: 'where does the hedgehog live', topK: 3 })

    expect(provider.calls.length).toBe(1)
    const call = provider.calls[0]!
    expect(call.request.maxTokens).toBe(1024)
    expect(call.request.temperature).toBe(0.2)
    expect(call.request.messages.length).toBe(2)
    expect(call.request.messages[0]!.role).toBe('system')
    expect(call.request.messages[1]!.role).toBe('user')
    const userPrompt = call.request.messages[1]!.content!
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
    const call = provider.calls[0]!
    expect(call.request.maxTokens).toBe(800)
    expect(call.request.temperature).toBe(0.0)
    expect(call.request.messages.length).toBe(1)
    expect(call.request.messages[0]!.role).toBe('user')
    const prompt = call.request.messages[0]!.content!

    // CoT framing
    expect(prompt).toContain('Answer the question step by step: first extract')
    expect(prompt).toContain('Answer (step by step):')

    // Recency / supersession guidance
    expect(prompt).toContain('prefer the value from the most recent session date')
    expect(prompt).toContain('One later correction outweighs any number of earlier mentions')

    // Enumeration guidance
    expect(prompt).toContain('When the question asks to list, count, enumerate, or total')
    expect(prompt).toContain('one per line')

    // Temporal anchor
    expect(prompt).toContain('Today is 2024-05-26 (Sunday)')
    expect(prompt).toContain('Current Date: 2024-05-26')

    // Evidence chunk uses "### title (path)" formatting
    expect(prompt).toMatch(/### .*\(raw\/documents\/hedgehog[^\)]*\.md\)/)
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

    const call = provider.calls[0]!
    const prompt = call.request.messages[0]!.content!
    expect(prompt).toContain('Today is unknown')
    expect(prompt).toContain('Current Date: unknown')
  })

  it('unknown readerMode value falls back to basic', async () => {
    const { handler, provider } = await makeFixture()
    await seedBrain(handler, 'lme')

    await drainAsk(handler, {
      question: 'where does the hedgehog live',
      topK: 1,
      readerMode: 'mystery',
    })

    const call = provider.calls[0]!
    expect(call.request.maxTokens).toBe(1024)
    expect(call.request.temperature).toBe(0.2)
    expect(call.request.messages.length).toBe(2)
  })
})
