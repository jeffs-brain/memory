// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { MemoryClient, ExtractAfterIngestArgs } from '../memory-client.js'
import { ingestUrlTool } from './ingest-url.js'

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
    _document_content: 'fetched content from URL',
  }),
  extract: async () => ({}),
  extractAfterIngest: async () => ({ factsExtracted: 0, memories: [] }),
  reflect: async () => ({}),
  consolidate: async () => ({}),
  createBrain: async () => ({}),
  listBrains: async () => ({}),
  close: async () => undefined,
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
    expect(payload).not.toHaveProperty('_document_content')
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
    expect(extractCalls[0]?.content).toBe('fetched content from URL')
    expect(extractCalls[0]?.documentSource).toBe('https://example.com/doc.md')
    // Ensure internal field is stripped from returned result
    expect(payload.ingest).not.toHaveProperty('_document_content')
  })

  it('returns empty extraction when no content is available', async () => {
    const client = noopClient()
    // Simulate hosted mode where no _document_content is returned
    client.ingestUrl = async () => ({
      path: 'server',
      result: { status: 'completed', document_id: 'doc-url', hash: 'def' },
    })

    const result = await ingestUrlTool.handler(
      { url: 'https://example.com/doc.md', extract: true },
      client,
      {},
    )
    const payload = result.structuredContent as {
      ingest: Record<string, unknown>
      extraction: { factsExtracted: number }
    }
    expect(payload.extraction.factsExtracted).toBe(0)
  })
})
