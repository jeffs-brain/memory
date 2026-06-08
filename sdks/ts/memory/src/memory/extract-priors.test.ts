// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Message,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { CodecPriorsError } from './codec-priors.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { createMemory } from './index.js'
import { EXTRACTION_SYSTEM_PROMPT } from './prompts.js'
import type { CodecPriors } from './types.js'

type Capture = { system: string; signalSeen: boolean }

const capturingProvider = (capture: Capture, content: string): Provider & { calls: number } => {
  const provider = {
    calls: 0,
    name: () => 'capture',
    modelName: () => 'capture-model',
    async *stream() {
      yield { type: 'done', stopReason: 'end_turn' as const }
    },
    complete: async (req: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> => {
      provider.calls += 1
      capture.system = req.system ?? ''
      capture.signalSeen = signal !== undefined
      if (signal?.aborted) {
        throw new DOMException('aborted', 'AbortError')
      }
      return {
        content,
        toolCalls: [],
        usage: { inputTokens: 0, outputTokens: 0 },
        stopReason: 'end_turn',
      }
    },
    supportsStructuredDecoding: () => false,
    structured: async (_req: StructuredRequest) => content,
  }
  return provider
}

const messages = (n: number): Message[] => {
  const out: Message[] = []
  for (let i = 0; i < n; i++) {
    out.push({ role: i % 2 === 0 ? 'user' : 'assistant', content: `msg ${i}` })
  }
  return out
}

const note = JSON.stringify({
  memories: [
    {
      action: 'create',
      filename: 'project-note.md',
      name: 'Note',
      description: 'a note',
      type: 'project',
      scope: 'project',
      content: 'A durable project fact.',
      index_entry: '- project-note.md: note',
    },
  ],
})

const newMemory = (capture: Capture, content: string) => {
  const store = createMemStore()
  const provider = capturingProvider(capture, content)
  const mem = createMemory({
    store,
    provider,
    cursorStore: createStoreBackedCursorStore(store),
    scope: 'project',
    actorId: 'tenant-a',
  })
  return { mem, provider, store }
}

describe('extract with codec priors — positive', () => {
  it('injects the priors block into the system prompt passed to the provider', async () => {
    const capture: Capture = { system: '', signalSeen: false }
    const { mem } = newMemory(capture, note)
    const priors: CodecPriors = {
      entities: ['RoyalAWare', 'Sprint'],
      relations: ['dependsOn'],
      domainTerms: ['deployment'],
    }
    await mem.extract({ messages: messages(4), priors })

    expect(capture.system.startsWith(EXTRACTION_SYSTEM_PROMPT)).toBe(true)
    expect(capture.system).toContain('## Project codec priors')
    expect(capture.system).toContain('- RoyalAWare')
    expect(capture.system).toContain('- dependsOn')
    expect(capture.system).toContain('- deployment')
  })
})

describe('extract with codec priors — negative / backward compatibility', () => {
  it('uses the byte-identical baseline prompt when priors are omitted', async () => {
    const capture: Capture = { system: '', signalSeen: false }
    const { mem } = newMemory(capture, note)
    await mem.extract({ messages: messages(4) })
    expect(capture.system).toBe(EXTRACTION_SYSTEM_PROMPT)
  })

  it('uses the byte-identical baseline prompt when priors are empty', async () => {
    const capture: Capture = { system: '', signalSeen: false }
    const { mem } = newMemory(capture, note)
    await mem.extract({ messages: messages(4), priors: { entities: [], relations: [] } })
    expect(capture.system).toBe(EXTRACTION_SYSTEM_PROMPT)
  })

  it('throws a typed error and never calls the provider for malformed priors', async () => {
    const capture: Capture = { system: '', signalSeen: false }
    const { mem, provider } = newMemory(capture, note)
    const malformed = { entities: ['ok', 'bad\ninjection'] } as CodecPriors
    await expect(mem.extract({ messages: messages(4), priors: malformed })).rejects.toBeInstanceOf(
      CodecPriorsError,
    )
    expect(provider.calls).toBe(0)
  })
})

describe('extract with codec priors — edge', () => {
  it('threads the abort signal and honours a pre-aborted signal', async () => {
    const capture: Capture = { system: '', signalSeen: false }
    const { mem, provider } = newMemory(capture, note)
    const controller = new AbortController()
    controller.abort()
    await expect(
      mem.extract({ messages: messages(4), signal: controller.signal }),
    ).rejects.toBeInstanceOf(DOMException)
    // Pre-aborted: provider is never invoked.
    expect(provider.calls).toBe(0)
  })

  it('honours an abort raised by the provider mid-call', async () => {
    const capture: Capture = { system: '', signalSeen: false }
    const store = createMemStore()
    const controller = new AbortController()
    const provider = {
      name: () => 'aborting',
      modelName: () => 'aborting-model',
      async *stream() {
        yield { type: 'done', stopReason: 'end_turn' as const }
      },
      complete: async (
        req: CompletionRequest,
        signal?: AbortSignal,
      ): Promise<CompletionResponse> => {
        capture.signalSeen = signal !== undefined
        controller.abort()
        throw new DOMException('aborted', 'AbortError')
      },
      supportsStructuredDecoding: () => false,
      structured: async (_req: StructuredRequest) => note,
    }
    const mem = createMemory({
      store,
      provider,
      cursorStore: createStoreBackedCursorStore(store),
      scope: 'project',
      actorId: 'tenant-a',
    })
    await expect(
      mem.extract({ messages: messages(4), signal: controller.signal }),
    ).rejects.toBeInstanceOf(DOMException)
    expect(capture.signalSeen).toBe(true)
  })

  it('keeps concurrent extract calls independent', async () => {
    const captureA: Capture = { system: '', signalSeen: false }
    const captureB: Capture = { system: '', signalSeen: false }
    const a = newMemory(captureA, note)
    const b = newMemory(captureB, note)
    await Promise.all([
      a.mem.extract({ messages: messages(4), priors: { entities: ['Alpha'] } }),
      b.mem.extract({ messages: messages(4), priors: { entities: ['Beta'] } }),
    ])
    expect(captureA.system).toContain('- Alpha')
    expect(captureA.system).not.toContain('- Beta')
    expect(captureB.system).toContain('- Beta')
    expect(captureB.system).not.toContain('- Alpha')
  })
})
