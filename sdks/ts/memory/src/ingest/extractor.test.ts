// SPDX-License-Identifier: Apache-2.0

import { Readable } from 'node:stream'
import { describe, expect, it } from 'vitest'
import {
  type ExtractOptions,
  type ExtractResult,
  type Extractor,
  MAX_DECOMPRESSION_RATIO,
  MAX_EXTRACTED_FILES,
  createExtractorRegistry,
  createPlainTextExtractor,
  sanitizeArgs,
} from './extractor.js'

const toReadable = (content: string): Readable => {
  return Readable.from([Buffer.from(content, 'utf8')])
}

describe('createPlainTextExtractor', () => {
  const ext = createPlainTextExtractor()

  it('returns text unchanged for valid UTF-8 input', async () => {
    const input = Buffer.from('hello world', 'utf8')
    const result = await ext.extract(input, { contentType: 'text/plain' })
    expect(result.text).toBe('hello world')
    expect(result.skipped).toBe(false)
    expect(result.metadata).toEqual({})
  })

  it('handles empty input', async () => {
    const result = await ext.extract(Buffer.alloc(0), { contentType: 'text/plain' })
    expect(result.text).toBe('')
    expect(result.skipped).toBe(false)
  })

  it('handles unicode content', async () => {
    const input = Buffer.from('hello 世界 🌍', 'utf8')
    const result = await ext.extract(input, { contentType: 'text/plain' })
    expect(result.text).toBe('hello 世界 🌍')
  })

  it('handles markdown content', async () => {
    const input = Buffer.from('# Title\n\nBody paragraph.', 'utf8')
    const result = await ext.extract(input, { contentType: 'text/markdown' })
    expect(result.text).toBe('# Title\n\nBody paragraph.')
  })

  it('extracts from a stream', async () => {
    const source = toReadable('streamed content')
    const result = await ext.extractStream(source, { contentType: 'text/plain' })
    expect(result.text).toBe('streamed content')
    expect(result.skipped).toBe(false)
  })

  it('respects maxBytes in stream mode', async () => {
    const source = toReadable('abcdefghij')
    const result = await ext.extractStream(source, { contentType: 'text/plain', maxBytes: 5 })
    expect(result.text).toBe('abcde')
  })

  it('declares text/* content types', () => {
    expect(ext.contentTypes).toContain('text/plain')
    expect(ext.contentTypes).toContain('text/markdown')
    expect(ext.contentTypes).toContain('application/json')
  })

  it('has the correct name', () => {
    expect(ext.name).toBe('plain-text')
  })
})

describe('createExtractorRegistry', () => {
  it('routes text/plain to built-in plain text extractor', async () => {
    const registry = createExtractorRegistry()
    const result = await registry.extract(Buffer.from('hello', 'utf8'), {
      contentType: 'text/plain',
    })
    expect(result.text).toBe('hello')
    expect(result.skipped).toBe(false)
  })

  it('routes text/markdown to built-in extractor', async () => {
    const registry = createExtractorRegistry()
    const result = await registry.extract(Buffer.from('# Title', 'utf8'), {
      contentType: 'text/markdown',
    })
    expect(result.text).toBe('# Title')
    expect(result.skipped).toBe(false)
  })

  it('falls back to text/plain for unknown text/* subtypes', async () => {
    const registry = createExtractorRegistry()
    const result = await registry.extract(Buffer.from('<root/>', 'utf8'), {
      contentType: 'text/xml',
    })
    expect(result.skipped).toBe(false)
    expect(result.text).toBe('<root/>')
  })

  it('returns skipped for unsupported content types', async () => {
    const registry = createExtractorRegistry()
    const result = await registry.extract(Buffer.from([0x00, 0x01]), {
      contentType: 'application/octet-stream',
    })
    expect(result.skipped).toBe(true)
    expect(result.reason).toContain('unsupported content type')
  })

  it('strips charset parameter for routing', async () => {
    const registry = createExtractorRegistry()
    const result = await registry.extract(Buffer.from('charset test', 'utf8'), {
      contentType: 'text/plain; charset=utf-8',
    })
    expect(result.skipped).toBe(false)
    expect(result.text).toBe('charset test')
  })

  it('allows custom extractors to override defaults', async () => {
    const registry = createExtractorRegistry()
    const custom: Extractor = {
      name: 'custom-text',
      contentTypes: ['text/plain'],
      async extract(raw: Buffer): Promise<ExtractResult> {
        return {
          text: `CUSTOM:${raw.toString('utf8')}`,
          metadata: { extractor: 'custom' },
          skipped: false,
        }
      },
      async extractStream(source: Readable, opts: ExtractOptions): Promise<ExtractResult> {
        const chunks: Buffer[] = []
        for await (const chunk of source) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array))
        }
        return this.extract(Buffer.concat(chunks), opts)
      },
    }
    registry.register(custom)

    const result = await registry.extract(Buffer.from('hello', 'utf8'), {
      contentType: 'text/plain',
    })
    expect(result.text).toBe('CUSTOM:hello')
    expect(result.metadata).toEqual({ extractor: 'custom' })
  })

  it('routes streams to the correct extractor', async () => {
    const registry = createExtractorRegistry()
    const source = toReadable('stream routing test')
    const result = await registry.extractStream(source, { contentType: 'text/plain' })
    expect(result.text).toBe('stream routing test')
    expect(result.skipped).toBe(false)
  })

  it('returns skipped for unsupported content types in stream mode', async () => {
    const registry = createExtractorRegistry()
    const source = toReadable('binary')
    const result = await registry.extractStream(source, { contentType: 'application/octet-stream' })
    expect(result.skipped).toBe(true)
  })

  it('default stream implementation buffers into extract', async () => {
    const registry = createExtractorRegistry()
    let extractCalled = false
    const custom: Extractor = {
      name: 'buffer-test',
      contentTypes: ['application/x-test'],
      async extract(raw: Buffer): Promise<ExtractResult> {
        extractCalled = true
        return { text: raw.toString('utf8'), metadata: {}, skipped: false }
      },
      async extractStream(source: Readable, opts: ExtractOptions): Promise<ExtractResult> {
        const chunks: Buffer[] = []
        for await (const chunk of source) {
          chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array))
        }
        return this.extract(Buffer.concat(chunks), opts)
      },
    }
    registry.register(custom)

    const source = toReadable('buffered via stream')
    const result = await registry.extractStream(source, { contentType: 'application/x-test' })
    expect(result.text).toBe('buffered via stream')
    expect(extractCalled).toBe(true)
  })
})

describe('sanitizeArgs', () => {
  it('passes through positional arguments unchanged', () => {
    expect(sanitizeArgs(['input.pdf', 'output.txt'])).toEqual(['input.pdf', 'output.txt'])
  })

  it('passes allowlisted short flags', () => {
    expect(sanitizeArgs(['-o', 'output.txt'])).toEqual(['-o', 'output.txt'])
  })

  it('passes allowlisted long flags', () => {
    expect(sanitizeArgs(['--format', 'json'])).toEqual(['--format', 'json'])
  })

  it('rejects disallowed flags', () => {
    expect(() => sanitizeArgs(['--exec', 'rm -rf /'])).toThrow('disallowed argument')
  })

  it('rejects unknown short flags', () => {
    expect(() => sanitizeArgs(['-e', 'evil'])).toThrow('disallowed argument')
  })

  it('handles empty input', () => {
    expect(sanitizeArgs([])).toEqual([])
  })

  it('handles mixed allowed and positional args', () => {
    expect(sanitizeArgs(['file.pdf', '-q', '--output', 'out.txt'])).toEqual([
      'file.pdf',
      '-q',
      '--output',
      'out.txt',
    ])
  })
})

describe('security constants', () => {
  it('defines MAX_DECOMPRESSION_RATIO as 100', () => {
    expect(MAX_DECOMPRESSION_RATIO).toBe(100)
  })

  it('defines MAX_EXTRACTED_FILES as 1000', () => {
    expect(MAX_EXTRACTED_FILES).toBe(1000)
  })
})
