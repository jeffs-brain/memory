// SPDX-License-Identifier: Apache-2.0

/**
 * Real-socket integration test for the memory HTTP daemon.
 *
 * Existing daemon tests invoke the Web API handler directly, which never
 * exercises the node:http bridge in `cli/commands/serve.ts`. A bug
 * there (wiring the abort controller to `nreq.on('close')` rather than
 * `nres.on('close')`) shipped to production because no test bound a
 * real TCP socket.
 *
 * This suite starts the same bridge on port 0, posts `/v1/brains/{id}/ask`
 * over a real fetch, and asserts the SSE frame sequence survives the
 * trip through Node's stream plumbing: retrieve, answer_delta, citation,
 * done. Regressing the abort wiring (so the controller fires on request
 * body end) breaks this test.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { type Server, createServer as createNodeServer } from 'node:http'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { handleNodeRequest } from '../cli/commands/serve.js'
import type { CompletionResponse, Provider, StreamEvent } from '../llm/index.js'
import { Daemon, createRouter } from './index.js'

type Fixture = {
  server: Server
  baseURL: string
  daemon: Daemon
  tempDir: string
}

let fixtures: Fixture[] = []

/** Fake provider that emits several deltas so the socket test can
 *  verify multi-chunk streaming still reaches the client. */
const makeStreamingProvider = (chunks: readonly string[]): Provider => ({
  name: () => 'fake',
  modelName: () => 'fake-stream-1',
  async *stream() {
    for (const c of chunks) {
      yield { type: 'text_delta', text: c } satisfies StreamEvent
    }
    yield { type: 'done', stopReason: 'end_turn' as const } satisfies StreamEvent
  },
  complete: async (): Promise<CompletionResponse> => ({
    content: chunks.join(''),
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async () => chunks.join(''),
})

const startFixture = async (provider: Provider): Promise<Fixture> => {
  const tempDir = await mkdtemp(join(tmpdir(), 'memory-socket-'))
  const daemon = new Daemon({ root: tempDir, provider })
  await daemon.start()
  const router = createRouter(daemon)
  const hostname = '127.0.0.1'

  const server = createNodeServer((nreq, nres) => {
    void handleNodeRequest(router, hostname, 0, nreq, nres)
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
    throw new Error('socket test: server.address() returned no port')
  }
  const baseURL = `http://${hostname}:${address.port}`
  const fx: Fixture = { server, baseURL, daemon, tempDir }
  fixtures.push(fx)
  return fx
}

beforeEach(() => {
  fixtures = []
})

afterEach(async () => {
  for (const fx of fixtures) {
    await new Promise<void>((resolve) => fx.server.close(() => resolve()))
    await fx.daemon.close()
    await rm(fx.tempDir, { recursive: true, force: true })
  }
  fixtures = []
})

const readFrames = async (resp: Response): Promise<readonly { event: string; data: string }[]> => {
  if (resp.body === null) throw new Error('socket test: response body missing')
  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  const frames: { event: string; data: string }[] = []
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    if (value !== undefined) buffer += decoder.decode(value, { stream: true })
    // Drain every fully terminated frame ("event:...\ndata:...\n\n").
    while (true) {
      const idx = buffer.indexOf('\n\n')
      if (idx === -1) break
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const lines = raw.split('\n')
      let event = ''
      const dataLines: string[] = []
      for (const line of lines) {
        if (line.startsWith('event: ')) event = line.slice('event: '.length)
        else if (line.startsWith('data: ')) dataLines.push(line.slice('data: '.length))
      }
      if (event !== '' || dataLines.length > 0) {
        frames.push({ event, data: dataLines.join('\n') })
      }
    }
  }
  return frames
}

describe('memory daemon socket integration', () => {
  it('streams retrieve -> answer_delta -> citation -> done over a real TCP socket', async () => {
    const provider = makeStreamingProvider(['The hedgehog ', 'lives in the ', 'hedgerows.'])
    const { baseURL } = await startFixture(provider)

    const brainId = 'socket-ask'
    const createResp = await fetch(`${baseURL}/v1/brains`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ brainId }),
    })
    expect(createResp.status).toBe(201)

    const ingestResp = await fetch(`${baseURL}/v1/brains/${brainId}/ingest/file`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        path: 'hedgehog.md',
        contentBase64: Buffer.from('# hedgehog\n\nThe hedgehog lives in hedgerows.').toString(
          'base64',
        ),
      }),
    })
    expect(ingestResp.status).toBe(200)

    const askResp = await fetch(`${baseURL}/v1/brains/${brainId}/ask`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'text/event-stream' },
      body: JSON.stringify({ question: 'where does the hedgehog live?', topK: 1 }),
    })
    expect(askResp.status).toBe(200)
    expect(askResp.headers.get('content-type')).toContain('text/event-stream')

    const frames = await readFrames(askResp)
    const events = frames.map((f) => f.event)

    // The exact sequence PROTOCOL.md promises. Any regression that
    // aborts the handler on request-body end collapses this into a
    // truncated stream (most commonly retrieve + done with no deltas,
    // or zero frames).
    expect(events[0]).toBe('retrieve')
    expect(events).toContain('answer_delta')
    expect(events).toContain('citation')
    expect(events[events.length - 1]).toBe('done')

    // Retrieval should be the first frame, all deltas before the first
    // citation, and exactly one done.
    const firstCitation = events.indexOf('citation')
    const lastDelta = events.lastIndexOf('answer_delta')
    expect(lastDelta).toBeGreaterThan(-1)
    expect(firstCitation).toBeGreaterThan(lastDelta)
    expect(events.filter((e) => e === 'done')).toHaveLength(1)

    // Every delta carries text; reassembly matches the provider output.
    const reassembled = frames
      .filter((f) => f.event === 'answer_delta')
      .map((f) => (JSON.parse(f.data) as { text: string }).text)
      .join('')
    expect(reassembled).toBe('The hedgehog lives in the hedgerows.')

    const doneFrame = frames[frames.length - 1]
    expect(doneFrame).toBeDefined()
    if (doneFrame === undefined) throw new Error('unreachable')
    expect((JSON.parse(doneFrame.data) as { ok: boolean }).ok).toBe(true)
  })
})
