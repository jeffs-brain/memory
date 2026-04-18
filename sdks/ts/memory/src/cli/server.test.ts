/**
 * Serve integration test. Boots the HTTP server against a temp brain,
 * hits `/search` with a stub embedder, and shuts down cleanly.
 *
 * `Bun.serve` is a Bun-only API. When the test runner is not Bun we skip
 * the boot assertion and instead exercise the handler directly — the
 * route logic is shared, so this still covers every code path except
 * the transport glue.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import type { CompletionResponse, Embedder, Provider } from '../llm/index.js'
import { toPath } from '../store/index.js'
import { initBrain, openBrain } from './brain.js'
import { buildHandler, startServer } from './server.js'

const createdDirs: string[] = []

afterEach(async () => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir !== undefined) await rm(dir, { recursive: true, force: true })
  }
})

const stubEmbedder = (): Embedder => {
  const deterministic = (text: string): number[] => {
    const dim = 1024
    const out = new Array<number>(dim).fill(0)
    let seed = 0
    for (let i = 0; i < text.length; i += 1) {
      seed = (seed * 31 + text.charCodeAt(i)) >>> 0
    }
    let s = seed || 1
    let norm = 0
    for (let i = 0; i < dim; i += 1) {
      s = (s * 1664525 + 1013904223) >>> 0
      const v = (s / 0xffffffff) * 2 - 1
      out[i] = v
      norm += v * v
    }
    const mag = Math.sqrt(norm) || 1
    for (let i = 0; i < dim; i += 1) out[i] = (out[i] ?? 0) / mag
    return out
  }
  return {
    name: () => 'stub',
    model: () => 'stub-1',
    dimension: () => 1024,
    embed: async (texts: readonly string[]) => texts.map((t) => deterministic(t)),
  }
}

const stubProvider = (content: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield await Promise.reject(new Error('not implemented'))
  },
  complete: async (): Promise<CompletionResponse> => ({
    content,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async () => '',
})

const makeBrain = async () => {
  const root = await mkdtemp(join(tmpdir(), 'jbmem-srv-'))
  createdDirs.push(root)
  const brainDir = join(root, 'brain')
  const initStore = await initBrain(brainDir)
  await initStore.close()
  const ingestStore = await openBrain(brainDir)
  await ingestStore.write(
    toPath('ingested/note.md'),
    Buffer.from('Hedgehogs love mealworms and quiet gardens.', 'utf8'),
  )
  return { brainDir, store: ingestStore }
}

const isBunRuntime = (): boolean => {
  const bun = (globalThis as unknown as { Bun?: { serve?: unknown } }).Bun
  return bun !== undefined && typeof bun.serve === 'function'
}

describe('jbmem serve', () => {
  it('handles /search via the shared handler with a stub embedder', async () => {
    const { store } = await makeBrain()
    try {
      const handler = buildHandler({
        store,
        embedder: stubEmbedder(),
        port: 0,
      })
      const health = await handler(new Request('http://localhost/health', { method: 'GET' }))
      expect(health.status).toBe(200)

      const response = await handler(
        new Request('http://localhost/search', {
          method: 'POST',
          body: JSON.stringify({ query: 'hedgehog', mode: 'bm25', limit: 5 }),
        }),
      )
      expect(response.status).toBe(200)
      const body = (await response.json()) as {
        query: string
        hits: Array<{ path: string }>
      }
      expect(body.query).toBe('hedgehog')
      expect(Array.isArray(body.hits)).toBe(true)
      expect(body.hits.length).toBeGreaterThan(0)
    } finally {
      await store.close()
    }
  })

  it('handles /recall via the memory recall pipeline', async () => {
    const { store } = await makeBrain()
    await store.write(
      toPath('memory/project/cli/hedgehog.md'),
      Buffer.from(
        [
          '---',
          'name: hedgehog',
          'description: desc',
          'type: project',
          'scope: project',
          '---',
          '',
          'Mealworms and quiet gardens.',
        ].join('\n'),
        'utf8',
      ),
    )
    try {
      const handler = buildHandler({
        store,
        embedder: stubEmbedder(),
        port: 0,
      })

      const response = await handler(
        new Request('http://localhost/recall', {
          method: 'POST',
          body: JSON.stringify({ query: 'hedgehog', actor_id: 'cli', scope: 'project', limit: 5 }),
        }),
      )
      expect(response.status).toBe(200)
      const body = (await response.json()) as {
        hits: Array<{ path: string; note: { name: string } }>
      }
      expect(body.hits[0]?.path).toBe('memory/project/cli/hedgehog.md')
      expect(body.hits[0]?.note.name).toBe('hedgehog')
    } finally {
      await store.close()
    }
  })

  it('handles /extract when a provider is configured', async () => {
    const { store } = await makeBrain()
    try {
      const handler = buildHandler({
        store,
        provider: stubProvider(
          JSON.stringify({
            memories: [
              {
                action: 'create',
                filename: 'hedgehog.md',
                name: 'hedgehog',
                description: 'Garden note',
                type: 'project',
                content: 'Mealworms and quiet gardens.',
                index_entry: '- hedgehog',
                scope: 'project',
              },
            ],
          }),
        ),
        port: 0,
      })

      const response = await handler(
        new Request('http://localhost/extract', {
          method: 'POST',
          body: JSON.stringify({
            messages: [{ role: 'user', content: 'Remember the hedgehog note.' }],
            actor_id: 'cli',
            scope: 'project',
          }),
        }),
      )
      expect(response.status).toBe(200)
      const body = (await response.json()) as { extracted: Array<{ filename: string }> }
      expect(body.extracted[0]?.filename).toBe('hedgehog.md')
    } finally {
      await store.close()
    }
  })

  it('rejects invalid actor ids for recall', async () => {
    const { store } = await makeBrain()
    try {
      const handler = buildHandler({
        store,
        embedder: stubEmbedder(),
        port: 0,
      })

      const response = await handler(
        new Request('http://localhost/recall', {
          method: 'POST',
          body: JSON.stringify({ query: 'hedgehog', actor_id: '../global', scope: 'project' }),
        }),
      )
      expect(response.status).toBe(400)
      await expect(response.json()).resolves.toEqual({ error: 'invalid actor_id' })
    } finally {
      await store.close()
    }
  })

  it('boots and shuts down under Bun', async () => {
    if (!isBunRuntime()) {
      // Vitest executed under Node: `Bun.serve` isn't available. Starting
      // the server would throw by design; the handler-level test above
      // already covers the route logic.
      return
    }
    const { store } = await makeBrain()
    try {
      const server = await startServer({
        store,
        embedder: stubEmbedder(),
        port: 0,
        hostname: '127.0.0.1',
      })
      try {
        const res = await fetch(`${server.url.replace(/\/$/, '')}/search`, {
          method: 'POST',
          body: JSON.stringify({ query: 'hedgehog', mode: 'bm25' }),
        })
        expect(res.status).toBe(200)
        const data = (await res.json()) as { hits: unknown[] }
        expect(Array.isArray(data.hits)).toBe(true)
      } finally {
        await server.stop()
      }
    } finally {
      await store.close()
    }
  })
})
