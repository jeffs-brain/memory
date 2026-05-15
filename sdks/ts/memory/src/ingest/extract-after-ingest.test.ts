// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { extractAfterIngest } from './extract-after-ingest.js'
import type { Memory, ExtractedMemory, ExtractArgs } from '../memory/types.js'

const makeMemoryStub = (
  extractFn: (args: ExtractArgs) => Promise<readonly ExtractedMemory[]>,
): Pick<Memory, 'extract'> => ({
  extract: extractFn,
})

describe('extractAfterIngest', () => {
  it('calls extract with document content as synthetic message', async () => {
    const calls: ExtractArgs[] = []
    const memory = makeMemoryStub(async (args) => {
      calls.push(args)
      return [
        {
          action: 'create',
          filename: 'fact-1.md',
          name: 'Fact 1',
          description: 'A fact',
          type: 'reference',
          content: 'Important fact extracted',
          indexEntry: 'fact-1',
          scope: 'global',
        },
      ]
    })

    const result = await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/readme.md',
      documentContent: '# Project README\n\nThis is a test project.',
      memory,
    })

    expect(result.factsExtracted).toBe(1)
    expect(result.memories).toHaveLength(1)
    expect(result.memories[0]?.content).toBe('Important fact extracted')
    expect(calls).toHaveLength(1)
    expect(calls[0]?.messages[0]?.role).toBe('user')
    expect(calls[0]?.messages[0]?.content).toContain('readme.md')
    expect(calls[0]?.messages[0]?.content).toContain('This is a test project.')
  })

  it('returns empty result when extract is not called (extract=false default)', async () => {
    const memory = makeMemoryStub(async () => [])

    const result = await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/empty.md',
      documentContent: '',
      memory,
    })

    expect(result.factsExtracted).toBe(0)
    expect(result.memories).toHaveLength(0)
  })

  it('truncates content exceeding maxContentChars', async () => {
    let receivedContent = ''
    const memory = makeMemoryStub(async (args) => {
      receivedContent = args.messages[0]?.content ?? ''
      return []
    })

    const longContent = 'x'.repeat(200)
    await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/long.md',
      documentContent: longContent,
      memory,
      maxContentChars: 50,
    })

    // The synthetic message includes the prompt prefix plus truncated content
    expect(receivedContent.length).toBeLessThan(longContent.length + 200)
    expect(receivedContent).not.toContain('x'.repeat(200))
  })

  it('returns empty result when extraction fails (non-fatal)', async () => {
    const memory = makeMemoryStub(async () => {
      throw new Error('LLM unavailable')
    })

    const result = await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/fail.md',
      documentContent: 'Some content',
      memory,
    })

    expect(result.factsExtracted).toBe(0)
    expect(result.memories).toHaveLength(0)
  })

  it('skips extraction for empty content', async () => {
    let called = false
    const memory = makeMemoryStub(async () => {
      called = true
      return []
    })

    const result = await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/empty.md',
      documentContent: '   ',
      memory,
    })

    expect(called).toBe(false)
    expect(result.factsExtracted).toBe(0)
  })

  it('returns empty result for zero-byte file content', async () => {
    let called = false
    const memory = makeMemoryStub(async () => {
      called = true
      return []
    })

    const result = await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/zero.bin',
      documentContent: '',
      memory,
    })

    expect(called).toBe(false)
    expect(result.factsExtracted).toBe(0)
    expect(result.memories).toHaveLength(0)
  })

  it('wraps content in isolation delimiters', async () => {
    let receivedContent = ''
    const memory = makeMemoryStub(async (args) => {
      receivedContent = args.messages[0]?.content ?? ''
      return []
    })

    await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/test.md',
      documentContent: 'Hello world',
      memory,
    })

    expect(receivedContent).toContain('<ingested-document>')
    expect(receivedContent).toContain('</ingested-document>')
  })

  it('passes actorId and sessionId to extract when provided', async () => {
    const calls: ExtractArgs[] = []
    const memory = makeMemoryStub(async (args) => {
      calls.push(args)
      return []
    })

    await extractAfterIngest({
      brainId: 'test-brain',
      documentPath: '/docs/readme.md',
      documentContent: 'Content here',
      memory,
      actorId: 'agent-1',
      sessionId: 'session-abc',
    })

    expect(calls[0]?.actorId).toBe('agent-1')
    expect(calls[0]?.sessionId).toBe('session-abc')
  })
})
