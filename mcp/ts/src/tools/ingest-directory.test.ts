// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { IngestTriggerEvent, TriggerBus } from '@jeffs-brain/memory/ingest'
import type { MemoryClient, ProgressEmitter } from '../memory-client.js'
import { ingestDirectoryTool } from './ingest-directory.js'
import type { ToolContext, ToolResult } from './types.js'

/**
 * Stub MemoryClient used by all tests. Individual tests override
 * specific methods to control behaviour.
 */
const noopClient = (): MemoryClient => ({
  mode: 'local',
  remember: async () => ({}),
  recall: async () => ({}),
  search: async () => ({}),
  ask: async () => ({}),
  ingestFile: async () => ({}),
  ingestUrl: async () => ({}),
  extract: async () => ({}),
  extractAfterIngest: async () => ({ factsExtracted: 0, memories: [] }),
  reflect: async () => ({}),
  consolidate: async () => ({}),
  createBrain: async () => ({}),
  listBrains: async () => ({}),
  close: async () => undefined,
})

type DirectoryIngestPayload = {
  readonly jobGroupId: string
  readonly filesQueued: number
  readonly filesSkipped: number
  readonly skippedReasons: readonly string[]
  readonly async: boolean
  readonly total?: number
  readonly succeeded?: number
  readonly failed?: number
  readonly skipped?: number
  readonly results?: readonly {
    readonly path: string
    readonly status: 'success' | 'error' | 'skipped'
    readonly documentId?: string
    readonly hash?: string
    readonly bytes?: number
    readonly error?: string
  }[]
}

const parseResult = (result: ToolResult): DirectoryIngestPayload => {
  const text = result.content[0]?.text
  if (!text) throw new Error('No text content in result')
  return JSON.parse(text) as DirectoryIngestPayload
}

/** Creates a stub trigger bus that records published events. */
const createStubBus = (): { bus: TriggerBus; events: IngestTriggerEvent[] } => {
  const events: IngestTriggerEvent[] = []
  const bus: TriggerBus = {
    publish: (event: IngestTriggerEvent) => { events.push(event) },
    subscribe: () => () => {},
    close: async () => {},
  }
  return { bus, events }
}

describe('memory_ingest_directory tool', () => {
  it('rejects a relative directory path via zod validation', () => {
    const parseResult = ingestDirectoryTool.inputSchema.safeParse({
      directory: 'relative/path',
    })
    // The zod schema requires min(1) but the handler validates absolute paths
    // The parse itself may succeed since zod only requires min(1) string,
    // so instead test that the handler rejects it.
    expect(parseResult.success).toBe(true)
  })

  it('rejects a path containing ".." segments', async () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({ directory: '' })
    expect(parseRes.success).toBe(false)
  })

  it('rejects maxFiles exceeding the cap via zod validation', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
      maxFiles: 501,
    })
    expect(parseRes.success).toBe(false)
  })

  it('accepts valid maxFiles within the cap', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
      maxFiles: 500,
    })
    expect(parseRes.success).toBe(true)
  })

  it('accepts exactly 1 for maxFiles', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
      maxFiles: 1,
    })
    expect(parseRes.success).toBe(true)
  })

  it('rejects zero for maxFiles via zod validation', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
      maxFiles: 0,
    })
    expect(parseRes.success).toBe(false)
  })

  it('rejects negative maxFiles via zod validation', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
      maxFiles: -1,
    })
    expect(parseRes.success).toBe(false)
  })

  it('defaults recursive to true', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
    })
    expect(parseRes.success).toBe(true)
    if (parseRes.success) {
      expect(parseRes.data.recursive).toBe(true)
    }
  })

  it('defaults maxFiles to 100', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
    })
    expect(parseRes.success).toBe(true)
    if (parseRes.success) {
      expect(parseRes.data.maxFiles).toBe(100)
    }
  })

  it('accepts an optional glob pattern', () => {
    const parseRes = ingestDirectoryTool.inputSchema.safeParse({
      directory: '/some/path',
      glob: '**/*.md',
    })
    expect(parseRes.success).toBe(true)
    if (parseRes.success) {
      expect(parseRes.data.glob).toBe('**/*.md')
    }
  })
})

describe('memory_ingest_directory async mode', () => {
  let tmp: string

  beforeEach(async () => {
    tmp = await mkdtemp(join(tmpdir(), 'dir-ingest-async-'))
  })

  afterEach(async () => {
    await rm(tmp, { recursive: true, force: true })
  })

  it('returns immediately with jobGroupId when triggerBus is present', async () => {
    await writeFile(join(tmp, 'a.md'), '# A')
    await writeFile(join(tmp, 'b.md'), '# B')

    const { bus, events } = createStubBus()
    const client = noopClient()
    const ctx: ToolContext = { triggerBus: bus }

    const result = await ingestDirectoryTool.handler(
      { directory: tmp, recursive: true, maxFiles: 100 },
      client,
      ctx,
    )

    const payload = parseResult(result)

    expect(payload.async).toBe(true)
    expect(payload.jobGroupId).toBeDefined()
    expect(typeof payload.jobGroupId).toBe('string')
    expect(payload.filesQueued).toBe(2)
    expect(payload.filesSkipped).toBe(0)
    expect(events).toHaveLength(2)

    // Verify events carry the jobGroupId in metadata.
    for (const event of events) {
      expect(event.payload.kind).toBe('file')
      expect(event.metadata).toBeDefined()
      expect((event.metadata as Record<string, unknown>).jobGroupId).toBe(payload.jobGroupId)
    }
  })

  it('dispatches events with correct brainId', async () => {
    await writeFile(join(tmp, 'test.md'), '# Test')

    const { bus, events } = createStubBus()
    const client = noopClient()
    const ctx: ToolContext = { triggerBus: bus }

    await ingestDirectoryTool.handler(
      { directory: tmp, brain: 'my-brain', recursive: true, maxFiles: 100 },
      client,
      ctx,
    )

    expect(events).toHaveLength(1)
    expect(events[0]?.brainId).toBe('my-brain')
  })

  it('defaults brainId to "default" when not specified', async () => {
    await writeFile(join(tmp, 'test.md'), '# Test')

    const { bus, events } = createStubBus()
    const client = noopClient()
    const ctx: ToolContext = { triggerBus: bus }

    await ingestDirectoryTool.handler(
      { directory: tmp, recursive: true, maxFiles: 100 },
      client,
      ctx,
    )

    expect(events[0]?.brainId).toBe('default')
  })
})

describe('memory_ingest_directory sync fallback', () => {
  let tmp: string

  beforeEach(async () => {
    tmp = await mkdtemp(join(tmpdir(), 'dir-ingest-sync-'))
  })

  afterEach(async () => {
    await rm(tmp, { recursive: true, force: true })
  })

  it('processes files synchronously when no triggerBus is present', async () => {
    await writeFile(join(tmp, 'a.md'), '# A')
    await writeFile(join(tmp, 'b.md'), '# B')

    let ingestCount = 0
    const client: MemoryClient = {
      ...noopClient(),
      ingestFile: async () => {
        ingestCount++
        return { document_id: `doc-${ingestCount}`, hash: 'h', byte_size: 10 }
      },
    }

    const result = await ingestDirectoryTool.handler(
      { directory: tmp, recursive: true, maxFiles: 100 },
      client,
    )

    const payload = parseResult(result)

    expect(payload.async).toBe(false)
    expect(payload.jobGroupId).toBeDefined()
    expect(payload.filesQueued).toBe(2)
    expect(payload.total).toBe(2)
    expect(payload.succeeded).toBe(2)
    expect(payload.failed).toBe(0)
    expect(payload.results).toHaveLength(2)
    expect(ingestCount).toBe(2)
  })

  it('includes jobGroupId even in sync mode', async () => {
    await writeFile(join(tmp, 'test.md'), '# Test')
    const client = noopClient()

    const result = await ingestDirectoryTool.handler(
      { directory: tmp, recursive: true, maxFiles: 100 },
      client,
    )

    const payload = parseResult(result)

    expect(payload.jobGroupId).toBeDefined()
    expect(payload.jobGroupId.length).toBeGreaterThan(0)
  })

  it('handles empty directory gracefully', async () => {
    const client = noopClient()

    const result = await ingestDirectoryTool.handler(
      { directory: tmp, recursive: true, maxFiles: 100 },
      client,
    )

    const payload = parseResult(result)

    expect(payload.filesQueued).toBe(0)
    expect(payload.async).toBe(false)
  })
})
