// SPDX-License-Identifier: Apache-2.0

/**
 * Unit tests for the auto-detection helpers in `src/defaults.ts`.
 * Each path (Ollama reachable, OpenAI env set, Anthropic env set,
 * nothing detected) gets its own deterministic check using an injected
 * fetch implementation.
 */

import { describe, expect, it, vi } from 'vitest'
import {
  detectEmbedder,
  detectProvider,
  ollamaReachable,
  resolveBrainPaths,
} from '../src/defaults.js'

const ok = () =>
  new Response('{}', { status: 200, headers: { 'content-type': 'application/json' } })

const failing = () =>
  new Response('nope', { status: 500, headers: { 'content-type': 'text/plain' } })

describe('defaults.ollamaReachable', () => {
  it('returns true when /api/tags answers 200', async () => {
    const fetchImpl = vi.fn(async () => ok())
    const reachable = await ollamaReachable({
      baseUrl: 'http://example.test',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    })
    expect(reachable).toBe(true)
    expect(fetchImpl).toHaveBeenCalledWith(
      'http://example.test/api/tags',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
  })

  it('returns false on a non-ok response', async () => {
    const fetchImpl = vi.fn(async () => failing())
    const reachable = await ollamaReachable({
      baseUrl: 'http://example.test',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    })
    expect(reachable).toBe(false)
  })

  it('returns false when fetch throws', async () => {
    const fetchImpl = vi.fn(async () => {
      throw new Error('boom')
    })
    const reachable = await ollamaReachable({
      baseUrl: 'http://example.test',
      fetchImpl: fetchImpl as unknown as typeof fetch,
    })
    expect(reachable).toBe(false)
  })
})

describe('defaults.detectProvider', () => {
  it('prefers OPENAI_API_KEY when set', async () => {
    const detected = await detectProvider({
      env: { OPENAI_API_KEY: 'sk-test' } as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => failing()) as unknown as typeof fetch,
    })
    expect(detected?.kind).toBe('openai')
    expect(detected?.kind === 'openai' && detected.apiKey).toBe('sk-test')
  })

  it('falls back to ANTHROPIC_API_KEY', async () => {
    const detected = await detectProvider({
      env: { ANTHROPIC_API_KEY: 'sk-ant' } as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => failing()) as unknown as typeof fetch,
    })
    expect(detected?.kind).toBe('anthropic')
  })

  it('falls back to Ollama when reachable', async () => {
    const detected = await detectProvider({
      env: {} as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => ok()) as unknown as typeof fetch,
    })
    expect(detected?.kind).toBe('ollama')
    expect(detected?.kind === 'ollama' && detected.baseUrl).toBe('http://localhost:11434')
  })

  it('returns undefined when nothing is configured', async () => {
    const detected = await detectProvider({
      env: {} as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => failing()) as unknown as typeof fetch,
    })
    expect(detected).toBeUndefined()
  })
})

describe('defaults.detectEmbedder', () => {
  it('prefers Ollama when reachable', async () => {
    const detected = await detectEmbedder({
      env: {} as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => ok()) as unknown as typeof fetch,
    })
    expect(detected?.kind).toBe('ollama')
    expect(detected?.kind === 'ollama' && detected.model).toBe('bge-m3')
  })

  it('opts in to OpenAI only when MEMORY_PI_EMBEDDER_OPENAI=true', async () => {
    const detected = await detectEmbedder({
      env: {
        OPENAI_API_KEY: 'sk-test',
        MEMORY_PI_EMBEDDER_OPENAI: 'true',
      } as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => failing()) as unknown as typeof fetch,
    })
    expect(detected?.kind).toBe('openai')
  })

  it('returns undefined without a reachable embedder', async () => {
    const detected = await detectEmbedder({
      env: { OPENAI_API_KEY: 'sk-test' } as NodeJS.ProcessEnv,
      fetchImpl: vi.fn(async () => failing()) as unknown as typeof fetch,
    })
    expect(detected).toBeUndefined()
  })
})

describe('defaults.resolveBrainPaths', () => {
  it('sanitises unsafe brain ids', () => {
    const paths = resolveBrainPaths('/tmp/brains', 'foo/bar')
    expect(paths.id).toBe('foo_bar')
    expect(paths.root.endsWith('/foo_bar')).toBe(true)
    expect(paths.searchIndexPath.endsWith('/foo_bar/search.sqlite')).toBe(true)
  })

  it('falls back to the default id when the cleaned id is empty', () => {
    const paths = resolveBrainPaths('/tmp/brains', '...')
    expect(paths.id).toBe('default')
  })
})
