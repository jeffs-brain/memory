// SPDX-License-Identifier: Apache-2.0

/**
 * End-to-end MCP server test. Boots the server in-process against an
 * `InMemoryTransport` pair and drives every tool we can exercise without
 * a live Ollama / OpenAI / Anthropic backend.
 *
 * Integration points that require external services (`memory_ask`,
 * `memory_extract`, `memory_reflect`, `memory_consolidate`) are covered
 * by unit tests that mock the `MemoryClient` rather than the transport.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { resolveConfig } from './config.js'
import { createMemoryClient } from './memory-client.js'
import { createServer } from './server.js'
import type { Tool, ToolContext } from './tools/types.js'
import { tools } from './tools/index.js'

const bootServer = async (home: string): Promise<{
  client: Client
  shutdown: () => Promise<void>
}> => {
  const cfg = resolveConfig({ ...process.env, JB_HOME: home, JB_TOKEN: '' })
  const memoryClient = createMemoryClient(cfg)
  const server = createServer(memoryClient)

  const [serverTransport, clientTransport] = InMemoryTransport.createLinkedPair()
  const client = new Client({ name: 'test', version: '0.0.0' }, { capabilities: {} })

  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)])

  return {
    client,
    shutdown: async () => {
      await client.close().catch(() => undefined)
      await server.close().catch(() => undefined)
      await memoryClient.close().catch(() => undefined)
    },
  }
}

describe('memory-mcp server', () => {
  let tmp: string

  beforeEach(async () => {
    tmp = await mkdtemp(join(tmpdir(), 'jb-mcp-test-'))
  })

  afterEach(async () => {
    await rm(tmp, { recursive: true, force: true })
  })

  it('lists all 11 tools', async () => {
    const { client, shutdown } = await bootServer(tmp)
    try {
      const list = await client.listTools()
      expect(list.tools.length).toBe(11)
      const names = list.tools.map((t) => t.name).sort()
      expect(names).toEqual([
        'memory_ask',
        'memory_consolidate',
        'memory_create_brain',
        'memory_extract',
        'memory_ingest_file',
        'memory_ingest_url',
        'memory_list_brains',
        'memory_recall',
        'memory_reflect',
        'memory_remember',
        'memory_search',
      ])
    } finally {
      await shutdown()
    }
  })

  it('lists empty brains, creates one, then lists it', async () => {
    const { client, shutdown } = await bootServer(tmp)
    try {
      const empty = await client.callTool({ name: 'memory_list_brains', arguments: {} })
      const parsed = JSON.parse((empty.content as { text: string }[])[0]?.text ?? '{}') as {
        items: unknown[]
      }
      expect(parsed.items).toEqual([])

      const created = await client.callTool({
        name: 'memory_create_brain',
        arguments: { name: 'Scratch', slug: 'scratch' },
      })
      const createdPayload = JSON.parse(
        (created.content as { text: string }[])[0]?.text ?? '{}',
      ) as { slug: string }
      expect(createdPayload.slug).toBe('scratch')

      const after = await client.callTool({ name: 'memory_list_brains', arguments: {} })
      const afterPayload = JSON.parse(
        (after.content as { text: string }[])[0]?.text ?? '{}',
      ) as { items: { slug: string }[] }
      expect(afterPayload.items.map((i) => i.slug)).toEqual(['scratch'])
    } finally {
      await shutdown()
    }
  })

  it('ingests a markdown file and finds it via search (BM25 fallback)', async () => {
    const fixture = join(tmp, 'fixture.md')
    await writeFile(
      fixture,
      '# Saturday run\n\nI finished the parkrun 5k in 24 minutes and felt great.\n',
      'utf8',
    )

    const { client, shutdown } = await bootServer(tmp)
    try {
      // Create the brain first so JB_HOME has a tidy directory layout.
      await client.callTool({
        name: 'memory_create_brain',
        arguments: { name: 'default', slug: 'default' },
      })

      const ingest = await client.callTool({
        name: 'memory_ingest_file',
        arguments: { path: fixture, brain: 'default' },
      })
      const ingestPayload = JSON.parse(
        (ingest.content as { text: string }[])[0]?.text ?? '{}',
      ) as { status: string; chunk_count: number }
      expect(ingestPayload.status).toBe('completed')
      expect(ingestPayload.chunk_count).toBeGreaterThanOrEqual(1)

      const search = await client.callTool({
        name: 'memory_search',
        arguments: { query: 'parkrun', brain: 'default' },
      })
      const searchPayload = JSON.parse(
        (search.content as { text: string }[])[0]?.text ?? '{}',
      ) as { hits: { content: string }[] }
      expect(searchPayload.hits.length).toBeGreaterThan(0)
      const combined = searchPayload.hits.map((hit) => hit.content).join('\n')
      expect(combined.toLowerCase()).toContain('parkrun')
    } finally {
      await shutdown()
    }
  })
})

describe('tool handlers (mocked client)', () => {
  const collectHandlers = (): Map<string, Tool> => new Map(tools.map((t) => [t.name, t]))
  const noCtx: ToolContext = {}

  it('forwards memory_remember args to the client', async () => {
    const calls: Array<Record<string, unknown>> = []
    const tool = collectHandlers().get('memory_remember')
    expect(tool).toBeDefined()
    if (tool === undefined) return

    const result = await tool.handler(
      { content: '# Hello\n\nWorld', title: 'Hello', tags: ['x'] },
      {
        mode: 'local',
        remember: async (args) => {
          calls.push(args as unknown as Record<string, unknown>)
          return { id: 'doc_1', path: 'memory/global/hello.md' }
        },
        recall: async () => ({}),
        search: async () => ({}),
        ask: async () => ({}),
        ingestFile: async () => ({}),
        ingestUrl: async () => ({}),
        extract: async () => ({}),
        reflect: async () => ({}),
        consolidate: async () => ({}),
        createBrain: async () => ({}),
        listBrains: async () => ({}),
        close: async () => undefined,
      },
      noCtx,
    )
    expect(calls.length).toBe(1)
    expect(result.structuredContent).toEqual({ id: 'doc_1', path: 'memory/global/hello.md' })
  })

  it('surfaces errors thrown by the client on long-running tools', async () => {
    const tool = collectHandlers().get('memory_ask')
    expect(tool).toBeDefined()
    if (tool === undefined) return

    await expect(
      tool.handler(
        { query: 'why?' },
        {
          mode: 'local',
          remember: async () => ({}),
          recall: async () => ({}),
          search: async () => ({}),
          ask: async () => {
            throw new Error('no provider')
          },
          ingestFile: async () => ({}),
          ingestUrl: async () => ({}),
          extract: async () => ({}),
          reflect: async () => ({}),
          consolidate: async () => ({}),
          createBrain: async () => ({}),
          listBrains: async () => ({}),
          close: async () => undefined,
        },
        noCtx,
      ),
    ).rejects.toThrow('no provider')
  })
})
