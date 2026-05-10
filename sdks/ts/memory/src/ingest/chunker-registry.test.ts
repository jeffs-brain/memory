// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  ChunkConfigError,
  createChunkConfig,
  defaultChunkConfig,
} from './chunk-config.js'
import { createChunkerRegistry } from './chunker-registry.js'
import type { Chunk, Chunker } from './chunker-registry.js'
import { markdownChunker } from './chunkers/markdown.js'
import { estimateTokens, recursiveChunker } from './chunkers/recursive.js'

describe('createChunkConfig', () => {
  it('creates a valid config with explicit values', () => {
    const cfg = createChunkConfig(256, 32, 20)
    expect(cfg.maxTokens).toBe(256)
    expect(cfg.overlapTokens).toBe(32)
    expect(cfg.minTokens).toBe(20)
  })

  it('applies defaults for undefined values', () => {
    const cfg = createChunkConfig()
    expect(cfg.maxTokens).toBe(512)
    expect(cfg.overlapTokens).toBe(64)
    expect(cfg.minTokens).toBe(40)
  })

  it('rejects minTokens >= maxTokens', () => {
    expect(() => createChunkConfig(100, 10, 100)).toThrow(ChunkConfigError)
    expect(() => createChunkConfig(100, 10, 200)).toThrow(ChunkConfigError)
  })

  it('rejects overlapTokens >= maxTokens', () => {
    expect(() => createChunkConfig(100, 100, 10)).toThrow(ChunkConfigError)
    expect(() => createChunkConfig(100, 200, 10)).toThrow(ChunkConfigError)
  })
})

describe('recursiveChunker', () => {
  it('returns empty array for blank content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await recursiveChunker('   \n\n  ', cfg)
    expect(chunks).toHaveLength(0)
  })

  it('returns a single chunk for short content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await recursiveChunker('Hello world.', cfg)
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.content).toBe('Hello world.')
    expect(chunks[0]?.metadata.chunker).toBe('recursive')
  })

  it('splits content exceeding maxTokens using separator hierarchy', async () => {
    const paragraphs = Array.from(
      { length: 10 },
      (_, i) => `Paragraph ${i} with some content that uses tokens to fill space.`,
    )
    const content = paragraphs.join('\n\n')
    const cfg = createChunkConfig(64, 16, 10)
    const chunks = await recursiveChunker(content, cfg)
    expect(chunks.length).toBeGreaterThan(1)
  })

  it('applies overlap from previous chunk to next', async () => {
    const paragraphs = [
      'alpha '.repeat(40).trim(),
      'beta '.repeat(40).trim(),
      'gamma '.repeat(40).trim(),
    ]
    const content = paragraphs.join('\n\n')
    const cfg = createChunkConfig(64, 16, 5)
    const chunks = await recursiveChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    // Second chunk should contain overlap tail from first.
    const firstContent = chunks[0]?.content ?? ''
    const overlapChars = 16 * 4
    const expectedTail = firstContent.slice(firstContent.length - overlapChars)
    expect(chunks[1]?.content).toContain(expectedTail)
  })

  it('respects AbortSignal', async () => {
    const cfg = defaultChunkConfig()
    const controller = new AbortController()
    controller.abort()
    await expect(recursiveChunker('test', cfg, controller.signal)).rejects.toThrow()
  })
})

describe('markdownChunker', () => {
  it('returns empty array for blank content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await markdownChunker('', cfg)
    expect(chunks).toHaveLength(0)
  })

  it('splits at heading boundaries preserving heading path', async () => {
    const content = [
      '# Introduction',
      '',
      'Intro text here.',
      '',
      '## Architecture',
      '',
      'Architecture details.',
      '',
      '### Patterns',
      '',
      'Pattern details.',
      '',
      '## Conclusion',
      '',
      'Final thoughts.',
    ].join('\n')
    const cfg = defaultChunkConfig()
    const chunks = await markdownChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(4)

    const introChunk = chunks[0]
    expect(introChunk?.metadata.headingPath).toContain('Introduction')

    const patternsChunk = chunks.find((c) =>
      c.metadata.headingPath.includes('Patterns'),
    )
    expect(patternsChunk).toBeDefined()
    expect(patternsChunk?.metadata.headingPath).toContain('Architecture')
    expect(patternsChunk?.metadata.headingPath).toContain('Patterns')
  })

  it('handles setext-style headings', async () => {
    const content = [
      'Title',
      '=====',
      '',
      'Some intro text.',
      '',
      'Subtitle',
      '--------',
      '',
      'Details here.',
    ].join('\n')
    const cfg = defaultChunkConfig()
    const chunks = await markdownChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    expect(chunks[0]?.metadata.headingPath).toContain('Title')
  })

  it('splits oversized sections respecting maxTokens', async () => {
    const longParagraphs = Array.from(
      { length: 30 },
      (_, i) => `Paragraph ${i} with enough words to accumulate tokens. More filler text here.`,
    )
    const content = '# Big Section\n\n' + longParagraphs.join('\n\n')
    const cfg = createChunkConfig(64, 16, 10)
    const chunks = await markdownChunker(content, cfg)
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) {
      expect(chunk.metadata.headingPath).toContain('Big Section')
    }
  })

  it('respects AbortSignal', async () => {
    const cfg = defaultChunkConfig()
    const controller = new AbortController()
    controller.abort()
    await expect(markdownChunker('# Test\n\nContent', cfg, controller.signal)).rejects.toThrow()
  })
})

describe('createChunkerRegistry', () => {
  it('routes text/markdown to the markdown chunker', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk('# Hello\n\nWorld', 'text/markdown', cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(1)
    expect(chunks[0]?.metadata.chunker).toBe('markdown')
  })

  it('routes text/plain to the recursive chunker', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk('Plain text content.', 'text/plain', cfg)
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.metadata.chunker).toBe('recursive')
  })

  it('falls back to recursive for unknown content types', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk('Some content.', 'application/unknown', cfg)
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.metadata.chunker).toBe('recursive')
  })

  it('normalises content type with charset suffix', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk(
      '# Title\n\nBody',
      'text/markdown; charset=utf-8',
      cfg,
    )
    expect(chunks[0]?.metadata.chunker).toBe('markdown')
  })

  it('allows custom chunker registration', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()

    const customChunker: Chunker = async (content) => {
      return [{ id: 'custom-0', content, metadata: { chunker: 'custom-json' } }]
    }
    reg.register({
      name: 'custom-json',
      contentTypes: ['application/json'],
      chunker: customChunker,
    })

    const chunks = await reg.chunk('{"key": "value"}', 'application/json', cfg)
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.metadata.chunker).toBe('custom-json')
  })

  it('assigns sequential IDs when chunks have no ID', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const content = [
      '# Section A',
      '',
      'Content A.',
      '',
      '# Section B',
      '',
      'Content B.',
    ].join('\n')
    const chunks = await reg.chunk(content, 'text/markdown', cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    expect(chunks[0]?.id).toBe('0')
    expect(chunks[1]?.id).toBe('1')
  })

  it('respects AbortSignal', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const controller = new AbortController()
    controller.abort()
    await expect(
      reg.chunk('test', 'text/plain', cfg, controller.signal),
    ).rejects.toThrow()
  })
})

describe('estimateTokens', () => {
  it('returns 0 for empty string', () => {
    expect(estimateTokens('')).toBe(0)
  })

  it('is monotonic in input length', () => {
    const tokenA = estimateTokens('hello')
    const tokenB = estimateTokens('hello world')
    const tokenC = estimateTokens('hello world how are you today')
    expect(tokenA).toBeLessThanOrEqual(tokenB)
    expect(tokenB).toBeLessThanOrEqual(tokenC)
  })

  it('approximates chars/4', () => {
    expect(estimateTokens('abcd')).toBe(1)
    expect(estimateTokens('abcdefgh')).toBe(2)
  })
})
