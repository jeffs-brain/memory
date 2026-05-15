// SPDX-License-Identifier: Apache-2.0

/**
 * Smoke tests for the @jeffs-brain/memory-pi extension. They spin up a
 * MemStore-backed fixture brain, mount the extension through a stub
 * `ExtensionAPI`, and verify the four lifecycle hooks emit the expected
 * effects:
 *
 *   1. `before_agent_start` injects a `<recalled-memory>` block.
 *   2. `turn_end` enqueues an extract job once the threshold trips.
 *   3. `session_shutdown` drains the queue.
 *
 * The stub deliberately omits TUI / signal plumbing so the test does
 * not need to import any pi runtime internals.
 */

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { BeforeAgentStartEventResult } from '@earendil-works/pi-coding-agent'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  type BeforeAgentStartEvent,
  type ContextEvent,
  type SessionShutdownEvent,
  type TurnEndEvent,
  createMemoryRuntime,
  registerMemoryHooks,
  registerMemoryTools,
  renderRecallBlock,
} from '../src/index.js'

type RegisteredHandlers = {
  before_agent_start?: (
    event: BeforeAgentStartEvent,
  ) => Promise<BeforeAgentStartEventResult | undefined> | BeforeAgentStartEventResult | undefined
  context?: (event: ContextEvent) => unknown
  turn_end?: (event: TurnEndEvent) => unknown
  session_shutdown?: (event: SessionShutdownEvent) => unknown
}

const buildStubExtensionAPI = (): {
  api: ExtensionAPI
  handlers: RegisteredHandlers
  registeredTools: string[]
} => {
  const handlers: RegisteredHandlers = {}
  const registeredTools: string[] = []
  const api = {
    on: ((event: string, handler: (...args: unknown[]) => unknown) => {
      ;(handlers as Record<string, unknown>)[event] = handler
    }) as unknown as ExtensionAPI['on'],
    registerTool: ((tool: { name: string }) => {
      registeredTools.push(tool.name)
    }) as unknown as ExtensionAPI['registerTool'],
    registerCommand: vi.fn(),
    registerShortcut: vi.fn(),
    registerFlag: vi.fn(),
    getFlag: vi.fn(),
    registerMessageRenderer: vi.fn(),
    sendMessage: vi.fn(),
    sendUserMessage: vi.fn(),
    appendEntry: vi.fn(),
    setSessionName: vi.fn(),
    getSessionName: vi.fn(),
    setLabel: vi.fn(),
    exec: vi.fn(),
    getActiveTools: vi.fn(() => []),
    getAllTools: vi.fn(() => []),
    setActiveTools: vi.fn(),
    getCommands: vi.fn(() => []),
    setModel: vi.fn(),
    getThinkingLevel: vi.fn(() => 'medium' as never),
    setThinkingLevel: vi.fn(),
    registerProvider: vi.fn(),
    unregisterProvider: vi.fn(),
    events: { on: vi.fn(), off: vi.fn(), emit: vi.fn() } as never,
  } as unknown as ExtensionAPI
  return { api, handlers, registeredTools }
}

describe('memory-pi smoke', () => {
  let brainRoot: string

  beforeEach(async () => {
    brainRoot = await mkdtemp(join(tmpdir(), 'memory-pi-'))
  })

  afterEach(async () => {
    await rm(brainRoot, { recursive: true, force: true })
  })

  it('builds a runtime against a temp brain root', async () => {
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'smoke',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
      // Provider auto-detection will return undefined in CI; that is
      // fine for recall-only tests.
    })
    expect(runtime.config.brainId).toBe('smoke')
    expect(runtime.embedder).toBeUndefined()
    await runtime.close()
  })

  it('registers all eleven tools', async () => {
    const { api, registeredTools } = buildStubExtensionAPI()
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'smoke',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
    })
    registerMemoryTools(api, runtime)
    expect(registeredTools.sort()).toEqual(
      [
        'memory_remember',
        'memory_recall',
        'memory_search',
        'memory_ask',
        'memory_ingest_file',
        'memory_ingest_url',
        'memory_extract',
        'memory_reflect',
        'memory_consolidate',
        'memory_create_brain',
        'memory_list_brains',
      ].sort(),
    )
    await runtime.close()
  })

  it('respects the tools.expose allowlist', async () => {
    const { api, registeredTools } = buildStubExtensionAPI()
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'smoke',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
      tools: { expose: ['recall', 'search'] },
    })
    registerMemoryTools(api, runtime)
    expect(registeredTools.sort()).toEqual(['memory_recall', 'memory_search'])
    await runtime.close()
  })

  it('injects a recall block into the system prompt on before_agent_start', async () => {
    const { api, handlers } = buildStubExtensionAPI()
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'smoke',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
      recall: { onPrompt: true, cacheFriendly: true },
    })
    // Force a deterministic recall response so we can observe injection.
    runtime.memory.recall = vi.fn(async () => [])
    registerMemoryHooks(api, runtime)
    expect(handlers.before_agent_start).toBeDefined()
    const result = await handlers.before_agent_start?.({
      type: 'before_agent_start',
      prompt: 'tell me about the deploy pipeline',
      systemPrompt: 'You are an assistant.',
      systemPromptOptions: {} as never,
    })
    expect(result).toBeDefined()
    expect(result?.systemPrompt).toContain('<recalled-memory>')
    expect(result?.systemPrompt).toContain('</recalled-memory>')
    // Second call with identical recall set should skip re-injection
    // because cacheFriendly is on and the snapshot signature matches.
    const second = await handlers.before_agent_start?.({
      type: 'before_agent_start',
      prompt: 'tell me about the deploy pipeline',
      systemPrompt: 'You are an assistant.',
      systemPromptOptions: {} as never,
    })
    expect(second).toBeUndefined()
    await runtime.close()
  })

  it('enqueues an extract job on turn_end when threshold trips', async () => {
    const { api, handlers } = buildStubExtensionAPI()
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'smoke',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
      extract: { onTurnEnd: true, minMessages: 1 },
    })
    // Stub the extract pipeline so the queue can drain without an LLM.
    const extractMock = vi.fn(async () => [])
    runtime.memory.extract = extractMock as never
    registerMemoryHooks(api, runtime)

    const turnEvent: TurnEndEvent = {
      type: 'turn_end',
      turnIndex: 0,
      message: {
        role: 'assistant',
        content: 'extract me',
      } as never,
      toolResults: [],
    }
    await handlers.turn_end?.(turnEvent)
    // Drain the queue to confirm the job ran.
    await runtime.extractQueue.flush()
    expect(extractMock).toHaveBeenCalledTimes(1)
    await runtime.close()
  })

  it('flushes the queue on session_shutdown', async () => {
    const { api, handlers } = buildStubExtensionAPI()
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'smoke',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
      reflect: { onSessionEnd: false },
      consolidate: { schedule: 'manual' },
    })
    const extractMock = vi.fn(async () => [])
    runtime.memory.extract = extractMock as never
    runtime.extractQueue.enqueue({
      messages: [{ role: 'user', content: 'pending' }],
      actorId: runtime.config.actorId,
    })
    registerMemoryHooks(api, runtime)
    await handlers.session_shutdown?.({
      type: 'session_shutdown',
      reason: 'quit',
    })
    expect(extractMock).toHaveBeenCalled()
    await runtime.close()
  })

  it('renders a labelled empty recall block when there are no hits', () => {
    const rendered = renderRecallBlock([], 'empty')
    expect(rendered).toBe('<recalled-memory>empty</recalled-memory>')
  })

  it('honours flatLayout and writes the FTS sqlite to the override path', async () => {
    const { mkdir, writeFile } = await import('node:fs/promises')
    const { existsSync } = await import('node:fs')
    await mkdir(join(brainRoot, 'memory'), { recursive: true })
    await writeFile(
      join(brainRoot, 'memory', 'feedback-style.md'),
      '# Feedback\n\nPrefer concise responses.\n',
      'utf8',
    )
    const indexDir = await mkdtemp(join(tmpdir(), 'memory-pi-state-'))
    try {
      const indexPath = join(indexDir, 'memory-pi', 'search.sqlite')
      const runtime = await createMemoryRuntime({
        brainRoot,
        brainId: 'test-brain',
        flatLayout: true,
        searchIndexPath: indexPath,
        store: { kind: 'fs' },
        embedder: { kind: 'off' },
      })
      try {
        expect(runtime.config.flatLayout).toBe(true)
        expect(runtime.config.searchIndexPath).toBe(indexPath)
        expect(existsSync(indexPath)).toBe(true)
        const hits = runtime.searchIndex.searchBM25('concise', 5)
        expect(hits.length).toBeGreaterThan(0)
        expect(hits[0]?.chunk.path).toBe('memory/feedback-style.md')
      } finally {
        await runtime.close()
      }
    } finally {
      await rm(indexDir, { recursive: true, force: true })
    }
  })
})
