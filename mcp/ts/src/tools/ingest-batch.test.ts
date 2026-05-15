// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { MemoryClient, ProgressEmitter } from '../memory-client.js'
import { ingestBatchTool } from './ingest-batch.js'
import type { ToolContext } from './types.js'

const noopClient = (): MemoryClient => ({
  mode: 'local',
  remember: async () => ({}),
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
})

type IngestFileCall = {
  readonly path: string
  readonly brain: string | undefined
  readonly as: string | undefined
}

describe('memory_ingest_batch tool', () => {
  it('processes a batch of 3 valid files and returns all succeeded', async () => {
    const calls: IngestFileCall[] = []
    const client = noopClient()
    client.ingestFile = async (args: { path: string; brain?: string; as?: string }) => {
      calls.push({ path: args.path, brain: args.brain, as: args.as })
      return {
        status: 'completed',
        document_id: `doc-${args.path}`,
        hash: `hash-${args.path}`,
        byte_size: 100,
      }
    }

    const result = await ingestBatchTool.handler(
      {
        files: [
          { path: '/tmp/a.md' },
          { path: '/tmp/b.md' },
          { path: '/tmp/c.md' },
        ],
      },
      client,
      {},
    )

    const payload = result.structuredContent as {
      total: number
      succeeded: number
      failed: number
      results: { path: string; status: string; documentId: string }[]
    }
    expect(payload.total).toBe(3)
    expect(payload.succeeded).toBe(3)
    expect(payload.failed).toBe(0)
    expect(payload.results).toHaveLength(3)
    expect(calls).toHaveLength(3)
    expect(payload.results[0]?.status).toBe('success')
    expect(payload.results[0]?.documentId).toBe('doc-/tmp/a.md')
  })

  it('isolates per-file errors: file 2 fails but file 1 and 3 succeed', async () => {
    const client = noopClient()
    let callIndex = 0
    client.ingestFile = async (args: { path: string }) => {
      callIndex++
      if (callIndex === 2) {
        throw new Error('file not found')
      }
      return {
        status: 'completed',
        document_id: `doc-${args.path}`,
        hash: `hash-${args.path}`,
        byte_size: 50,
      }
    }

    const result = await ingestBatchTool.handler(
      {
        files: [
          { path: '/tmp/a.md' },
          { path: '/tmp/missing.md' },
          { path: '/tmp/c.md' },
        ],
      },
      client,
      {},
    )

    const payload = result.structuredContent as {
      total: number
      succeeded: number
      failed: number
      results: { path: string; status: string; error?: string }[]
    }
    expect(payload.total).toBe(3)
    expect(payload.succeeded).toBe(2)
    expect(payload.failed).toBe(1)
    expect(payload.results[0]?.status).toBe('success')
    expect(payload.results[1]?.status).toBe('error')
    expect(payload.results[1]?.error).toBe('file not found')
    expect(payload.results[2]?.status).toBe('success')
  })

  it('emits progress notifications for each file processed', async () => {
    const client = noopClient()
    client.ingestFile = async () => ({
      status: 'completed',
      document_id: 'doc',
      hash: 'hash',
      byte_size: 10,
    })

    const progressCalls: { progress: number; message?: string }[] = []
    const ctx: ToolContext = {
      progress: (p: number, m?: string) => {
        progressCalls.push({ progress: p, message: m })
      },
    }

    await ingestBatchTool.handler(
      {
        files: [
          { path: '/a.md' },
          { path: '/b.md' },
          { path: '/c.md' },
        ],
      },
      client,
      ctx,
    )

    expect(progressCalls).toHaveLength(3)
    expect(progressCalls[0]?.progress).toBe(1)
    expect(progressCalls[0]?.message).toBe('1/3 /a.md')
    expect(progressCalls[1]?.progress).toBe(2)
    expect(progressCalls[1]?.message).toBe('2/3 /b.md')
    expect(progressCalls[2]?.progress).toBe(3)
    expect(progressCalls[2]?.message).toBe('3/3 /c.md')
  })

  it('applies brain parameter to all files in the batch', async () => {
    const brainsSeen: (string | undefined)[] = []
    const client = noopClient()
    client.ingestFile = async (args: { brain?: string }) => {
      brainsSeen.push(args.brain)
      return { status: 'completed', document_id: 'doc', hash: 'h', byte_size: 1 }
    }

    await ingestBatchTool.handler(
      {
        files: [{ path: '/a.md' }, { path: '/b.md' }],
        brain: 'test-brain',
      },
      client,
      {},
    )

    expect(brainsSeen).toEqual(['test-brain', 'test-brain'])
  })

  it('handles duplicate file paths: both entries are processed independently', async () => {
    const calls: string[] = []
    const client = noopClient()
    client.ingestFile = async (args: { path: string }) => {
      calls.push(args.path)
      return {
        status: 'completed',
        document_id: `doc-${calls.length}`,
        hash: `hash-${calls.length}`,
        byte_size: 42,
      }
    }

    const result = await ingestBatchTool.handler(
      {
        files: [
          { path: '/data/same.md' },
          { path: '/data/same.md' },
        ],
      },
      client,
      {},
    )

    const payload = result.structuredContent as {
      total: number
      succeeded: number
      failed: number
      results: { path: string; status: string }[]
    }
    expect(payload.total).toBe(2)
    expect(payload.succeeded).toBe(2)
    expect(payload.failed).toBe(0)
    expect(calls).toHaveLength(2)
    expect(calls[0]).toBe('/data/same.md')
    expect(calls[1]).toBe('/data/same.md')
  })

  it('rejects empty files array via zod validation', () => {
    const parseResult = ingestBatchTool.inputSchema.safeParse({ files: [] })
    expect(parseResult.success).toBe(false)
  })

  it('rejects files array exceeding 50 entries via zod validation', () => {
    const files = Array.from({ length: 51 }, (_, i) => ({ path: `/file-${i}.md` }))
    const parseResult = ingestBatchTool.inputSchema.safeParse({ files })
    expect(parseResult.success).toBe(false)
  })

  it('accepts exactly 50 files via zod validation', () => {
    const files = Array.from({ length: 50 }, (_, i) => ({ path: `/file-${i}.md` }))
    const parseResult = ingestBatchTool.inputSchema.safeParse({ files })
    expect(parseResult.success).toBe(true)
  })
})
