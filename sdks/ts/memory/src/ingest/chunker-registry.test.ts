// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  ChunkConfigError,
  createChunkConfig,
  defaultChunkConfig,
} from './chunk-config.js'
import { createChunkerRegistry } from './chunker-registry.js'
import type { Chunk, Chunker } from './chunker-registry.js'
import { codeChunker } from './chunkers/code.js'
import { markdownChunker } from './chunkers/markdown.js'
import { pageLevelChunker } from './chunkers/page-level.js'
import { estimateTokens, recursiveChunker } from './chunkers/recursive.js'
import { tabularChunker } from './chunkers/tabular.js'

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
    expect(cfg.minTokens).toBe(64)
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

describe('codeChunker', () => {
  it('splits Go source at function boundaries', async () => {
    const source = [
      'package main',
      '',
      'import "fmt"',
      '',
      'func hello() {',
      '  fmt.Println("hello")',
      '}',
      '',
      'func world() {',
      '  fmt.Println("world")',
      '}',
      '',
      'func main() {',
      '  hello()',
      '  world()',
      '}',
    ].join('\n')
    const cfg = createChunkConfig(512, 64, 5)
    const chunks = await codeChunker(source, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    for (const c of chunks) {
      expect(c.metadata.chunker).toBe('code')
    }
  })

  it('splits Python source at def/class boundaries', async () => {
    const source = [
      'import os',
      'import sys',
      '',
      'def greet(name):',
      '    print(f"Hello {name}")',
      '',
      'class Greeter:',
      '    def __init__(self, name):',
      '        self.name = name',
      '',
      'def main():',
      '    g = Greeter("world")',
    ].join('\n')
    const cfg = createChunkConfig(512, 64, 5)
    const chunks = await codeChunker(source, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
  })

  it('preserves import block as first chunk', async () => {
    const source = [
      'import "fmt"',
      'import "os"',
      '',
      'func run() {',
      '  fmt.Println(os.Args)',
      '}',
    ].join('\n')
    const cfg = defaultChunkConfig()
    const chunks = await codeChunker(source, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(1)
    expect(chunks[0]?.content).toContain('import')
  })

  it('returns empty array for blank content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await codeChunker('', cfg)
    expect(chunks).toHaveLength(0)
  })
})

describe('tabularChunker', () => {
  it('chunks CSV with header prepended to each chunk', async () => {
    const rows = ['name,age,city']
    for (let i = 0; i < 120; i++) {
      rows.push(`person${i},${20 + i},city${i}`)
    }
    const content = rows.join('\n')
    const cfg = defaultChunkConfig()
    const chunks = await tabularChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    for (const c of chunks) {
      expect(c.content.startsWith('name,age,city')).toBe(true)
      expect(c.metadata.chunker).toBe('tabular')
    }
  })

  it('detects tab delimiter for TSV', async () => {
    const rows = ['name\tage\tcity']
    for (let i = 0; i < 60; i++) {
      rows.push(`person${i}\t${20 + i}\tcity${i}`)
    }
    const content = rows.join('\n')
    const cfg = defaultChunkConfig()
    const chunks = await tabularChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    for (const c of chunks) {
      const firstLine = c.content.split('\n')[0] ?? ''
      expect(firstLine).toContain('\t')
    }
  })

  it('produces single chunk for single row', async () => {
    const content = 'name,age,city\nAlice,30,NYC'
    const cfg = defaultChunkConfig()
    const chunks = await tabularChunker(content, cfg)
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.content.startsWith('name,age,city')).toBe(true)
  })

  it('handles header-only content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await tabularChunker('name,age,city', cfg)
    expect(chunks).toHaveLength(1)
  })

  it('returns empty array for blank content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await tabularChunker('  ', cfg)
    expect(chunks).toHaveLength(0)
  })

  it('respects maxTokens for row count per chunk', async () => {
    const rows = ['name,age,city']
    for (let i = 0; i < 200; i++) {
      rows.push(`person${i},${20 + i},city${i}`)
    }
    const content = rows.join('\n')
    const cfg = createChunkConfig(64, 0, 5)
    const chunks = await tabularChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(4)
  })
})

describe('pageLevelChunker', () => {
  it('splits on form-feed character', async () => {
    const content = 'Page 1 content.\fPage 2 content.\fPage 3 content.'
    const cfg = defaultChunkConfig()
    const chunks = await pageLevelChunker(content, cfg)
    expect(chunks).toHaveLength(3)
    expect(chunks[0]?.metadata.page).toBe('1')
    expect(chunks[1]?.metadata.page).toBe('2')
    expect(chunks[2]?.metadata.page).toBe('3')
    expect(chunks[0]?.metadata.chunker).toBe('page_level')
  })

  it('skips empty pages', async () => {
    const content = 'Page 1\f\f\fPage 4'
    const cfg = defaultChunkConfig()
    const chunks = await pageLevelChunker(content, cfg)
    expect(chunks).toHaveLength(2)
  })

  it('returns empty array for blank content', async () => {
    const cfg = defaultChunkConfig()
    const chunks = await pageLevelChunker('  ', cfg)
    expect(chunks).toHaveLength(0)
  })

  it('splits large pages using recursive strategy', async () => {
    const largePage = 'This is a long sentence with many words. '.repeat(100)
    const content = largePage + '\fSmall page.'
    const cfg = createChunkConfig(64, 16, 10)
    const chunks = await pageLevelChunker(content, cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(3)
  })
})

describe('createChunkerRegistry (new chunkers)', () => {
  it('selects code chunker for text/x-go', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk('func main() {}', 'text/x-go', cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(1)
    expect(chunks[0]?.metadata.chunker).toBe('code')
  })

  it('selects tabular chunker for text/csv', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk('name,age\nAlice,30', 'text/csv', cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(1)
    expect(chunks[0]?.metadata.chunker).toBe('tabular')
  })

  it('selects page-level chunker for application/pdf', async () => {
    const reg = createChunkerRegistry()
    const cfg = defaultChunkConfig()
    const chunks = await reg.chunk('Page 1\fPage 2', 'application/pdf', cfg)
    expect(chunks.length).toBeGreaterThanOrEqual(2)
    expect(chunks[0]?.metadata.chunker).toBe('page_level')
  })
})

describe('createChunkConfig (strategy and separators)', () => {
  it('defaults strategy to empty and separators to undefined', () => {
    const cfg = createChunkConfig()
    expect(cfg.strategy).toBe('')
    expect(cfg.separators).toBeUndefined()
  })

  it('accepts strategy and separators options', () => {
    const cfg = createChunkConfig(512, 64, 64, {
      strategy: 'code',
      separators: ['\n'],
    })
    expect(cfg.strategy).toBe('code')
    expect(cfg.separators).toEqual(['\n'])
  })
})
