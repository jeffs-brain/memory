// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { MemoryClient, ExtractAfterIngestArgs } from '../memory-client.js'
import { ingestFileTool } from './ingest-file.js'
import { ingestUrlTool } from './ingest-url.js'
import type { ToolContext } from './types.js'

const noopClient = (): MemoryClient => ({
  mode: 'local',
  remember: async () => ({}),
  recall: async () => ({}),
  search: async () => ({}),
  ask: async () => ({}),
  ingestFile: async () => ({
    status: 'completed',
    document_id: 'doc-1',
    hash: 'abc',
    byte_size: 100,
  }),
  ingestUrl: async () => ({
    path: 'server',
    result: { status: 'completed', document_id: 'doc-url', hash: 'def' },
  }),
  extract: async () => ({}),
  extractAfterIngest: async () => ({ factsExtracted: 0, memories: [] }),
  reflect: async () => ({}),
  consolidate: async () => ({}),
  createBrain: async () => ({}),
  listBrains: async () => ({}),
  close: async () => undefined,
})

describe('memory_ingest_file with extract option', () => {
  it('returns only ingest result when extract is false (default)', async () => {
    const client = noopClient()
    const result = await ingestFileTool.handler(
      { path: '/test.md' },
      client,
      {},
    )
    const payload = result.structuredContent as Record<string, unknown>
    expect(payload.status).toBe('completed')
    expect(payload).not.toHaveProperty('ingest')
    expect(payload).not.toHaveProperty('extraction')
  })

  it('returns combined ingest + extraction when extract is true', async () => {
    const extractCalls: ExtractAfterIngestArgs[] = []
    const client = noopClient()
    client.extractAfterIngest = async (args: ExtractAfterIngestArgs) => {
      extractCalls.push(args)
      return {
        factsExtracted: 2,
        memories: [
          { filename: 'fact-1.md', content: 'Fact 1' },
          { filename: 'fact-2.md', content: 'Fact 2' },
        ],
      }
    }

    const result = await ingestFileTool.handler(
      { path: '/test.md', extract: true },
      client,
      {},
    )
    const payload = result.structuredContent as {
      ingest: Record<string, unknown>
      extraction: { factsExtracted: number; memories: { filename: string }[] }
    }
    expect(payload.ingest).toBeDefined()
    expect(payload.ingest.status).toBe('completed')
    expect(payload.extraction.factsExtracted).toBe(2)
    expect(payload.extraction.memories).toHaveLength(2)
    expect(extractCalls).toHaveLength(1)
    expect(extractCalls[0]?.path).toBe('/test.md')
  })
})

describe('memory_ingest_url with extract option', () => {
  it('returns only ingest result when extract is false (default)', async () => {
    const client = noopClient()
    const result = await ingestUrlTool.handler(
      { url: 'https://example.com/doc.md' },
      client,
      {},
    )
    const payload = result.structuredContent as Record<string, unknown>
    expect(payload.path).toBe('server')
    expect(payload).not.toHaveProperty('extraction')
  })

  it('returns combined ingest + extraction when extract is true', async () => {
    const extractCalls: ExtractAfterIngestArgs[] = []
    const client = noopClient()
    client.extractAfterIngest = async (args: ExtractAfterIngestArgs) => {
      extractCalls.push(args)
      return {
        factsExtracted: 1,
        memories: [{ filename: 'url-fact.md', content: 'URL fact' }],
      }
    }

    const result = await ingestUrlTool.handler(
      { url: 'https://example.com/doc.md', extract: true },
      client,
      {},
    )
    const payload = result.structuredContent as {
      ingest: Record<string, unknown>
      extraction: { factsExtracted: number }
    }
    expect(payload.ingest).toBeDefined()
    expect(payload.extraction.factsExtracted).toBe(1)
    expect(extractCalls).toHaveLength(1)
    expect(extractCalls[0]?.url).toBe('https://example.com/doc.md')
  })
})
