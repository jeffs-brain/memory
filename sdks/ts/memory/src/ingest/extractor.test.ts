// SPDX-License-Identifier: Apache-2.0

import { Readable } from 'node:stream'
import { describe, expect, it } from 'vitest'
import {
  type ExtractOptions,
  type ExtractResult,
  type Extractor,
  type ExtractorCapability,
  MAX_DECOMPRESSION_RATIO,
  MAX_EXTRACTED_FILES,
  createExtractorRegistry,
  createPlainTextExtractor,
  detectEncoding,
  sanitizeArgs,
  transcodeToUTF8,
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
          contentType: 'text/plain',
          encoding: 'UTF-8',
          metadata: { extractor: 'custom' },
          pages: 0,
          language: '',
          confidence: 0,
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
      async available(): Promise<boolean> {
        return true
      },
      capability(): ExtractorCapability {
        return { extensions: ['.txt'], mimeTypes: ['text/plain'], magicBytes: [], requiresBinary: false }
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
        return {
          text: raw.toString('utf8'),
          contentType: 'application/x-test',
          encoding: 'UTF-8',
          metadata: {},
          pages: 0,
          language: '',
          confidence: 0,
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
      async available(): Promise<boolean> {
        return true
      },
      capability(): ExtractorCapability {
        return { extensions: [], mimeTypes: ['application/x-test'], magicBytes: [], requiresBinary: false }
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

describe('createPlainTextExtractor available and capability', () => {
  const ext = createPlainTextExtractor()

  it('reports as available', async () => {
    expect(await ext.available()).toBe(true)
  })

  it('returns capability with extensions and MIME types', () => {
    const cap = ext.capability()
    expect(cap.extensions).toContain('.txt')
    expect(cap.extensions).toContain('.md')
    expect(cap.mimeTypes).toContain('text/plain')
    expect(cap.mimeTypes).toContain('text/markdown')
    expect(cap.requiresBinary).toBe(false)
  })
})

describe('detectEncoding', () => {
  it('detects UTF-8 for valid UTF-8 content', () => {
    const raw = Buffer.from('Hello, world!', 'utf8')
    expect(detectEncoding(raw)).toBe('UTF-8')
  })

  it('detects UTF-8 BOM', () => {
    const raw = Buffer.from([0xef, 0xbb, 0xbf, 0x48, 0x65, 0x6c, 0x6c, 0x6f])
    expect(detectEncoding(raw)).toBe('UTF-8')
  })

  it('detects UTF-16BE BOM', () => {
    const raw = Buffer.from([0xfe, 0xff, 0x00, 0x48, 0x00, 0x65])
    expect(detectEncoding(raw)).toBe('UTF-16BE')
  })

  it('detects UTF-16LE BOM', () => {
    const raw = Buffer.from([0xff, 0xfe, 0x48, 0x00, 0x65, 0x00])
    expect(detectEncoding(raw)).toBe('UTF-16LE')
  })

  it('detects ISO-8859-1 for Latin-1 content', () => {
    const raw = Buffer.from([0x43, 0x61, 0x66, 0xe9, 0x20, 0x63, 0x72, 0xe8, 0x6d, 0x65])
    expect(detectEncoding(raw)).toBe('ISO-8859-1')
  })

  it('detects Windows-1252 for content with C1 controls', () => {
    // 0x93 and 0x94 are smart quotes in Windows-1252
    const raw = Buffer.from([0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x93, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x94])
    expect(detectEncoding(raw)).toBe('Windows-1252')
  })

  it('detects Shift_JIS for Japanese content', () => {
    // Shift_JIS: 日本語
    const raw = Buffer.from([0x93, 0xfa, 0x96, 0x7b, 0x8c, 0xea])
    expect(detectEncoding(raw)).toBe('Shift_JIS')
  })

  it('returns UTF-8 for empty input', () => {
    expect(detectEncoding(Buffer.alloc(0))).toBe('UTF-8')
  })
})

describe('transcodeToUTF8', () => {
  it('transcodes Latin-1 to UTF-8', () => {
    const raw = Buffer.from([0x43, 0x61, 0x66, 0xe9]) // "Cafe" with accent
    const result = transcodeToUTF8(raw, 'ISO-8859-1')
    expect(result.toString('utf8')).toBe('Café')
  })

  it('passes through UTF-8 unchanged', () => {
    const raw = Buffer.from('Already UTF-8', 'utf8')
    const result = transcodeToUTF8(raw, 'UTF-8')
    expect(result.toString('utf8')).toBe('Already UTF-8')
  })

  it('passes through when encoding is empty', () => {
    const raw = Buffer.from('No encoding', 'utf8')
    const result = transcodeToUTF8(raw, '')
    expect(result.toString('utf8')).toBe('No encoding')
  })

  it('transcodes Windows-1252 smart quotes', () => {
    const raw = Buffer.from([0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x93, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x94])
    const result = transcodeToUTF8(raw, 'Windows-1252')
    expect(result.toString('utf8')).toContain('Hello')
    expect(result.toString('utf8')).toContain('world')
  })

  it('transcodes Shift_JIS to UTF-8', () => {
    // Shift_JIS: 日本語
    const raw = Buffer.from([0x93, 0xfa, 0x96, 0x7b, 0x8c, 0xea])
    const result = transcodeToUTF8(raw, 'Shift_JIS')
    expect(result.toString('utf8')).toBe('日本語')
  })

  it('throws for unsupported encoding', () => {
    expect(() => transcodeToUTF8(Buffer.from('data'), 'EBCDIC-37')).toThrow()
  })
})
