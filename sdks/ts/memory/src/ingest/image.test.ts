// SPDX-License-Identifier: Apache-2.0

import { Readable } from 'node:stream'
import { describe, expect, it } from 'vitest'
import type { ExtractOptions, Extractor, ExtractorCapability } from './extractor.js'
import { createImageExtractor, parsePaddleOCROutput } from './image.js'

describe('createImageExtractor', () => {
  const ext = createImageExtractor({
    paddleOcrBinary: 'nonexistent-paddleocr-xyz',
    tesseractBinary: 'nonexistent-tesseract-xyz',
  })

  it('has the correct name', () => {
    expect(ext.name).toBe('image-ocr')
  })

  it('declares image content types', () => {
    expect(ext.contentTypes).toContain('image/png')
    expect(ext.contentTypes).toContain('image/jpeg')
    expect(ext.contentTypes).toContain('image/tiff')
    expect(ext.contentTypes).toContain('image/bmp')
    expect(ext.contentTypes).toContain('image/webp')
  })

  it('returns capability with correct extensions', () => {
    const cap = ext.capability()
    expect(cap.extensions).toContain('.png')
    expect(cap.extensions).toContain('.jpg')
    expect(cap.extensions).toContain('.jpeg')
    expect(cap.extensions).toContain('.tiff')
    expect(cap.extensions).toContain('.tif')
    expect(cap.extensions).toContain('.bmp')
    expect(cap.extensions).toContain('.webp')
  })

  it('returns capability with correct MIME types', () => {
    const cap = ext.capability()
    expect(cap.mimeTypes).toContain('image/png')
    expect(cap.mimeTypes).toContain('image/jpeg')
    expect(cap.mimeTypes).toContain('image/tiff')
    expect(cap.mimeTypes).toContain('image/bmp')
    expect(cap.mimeTypes).toContain('image/webp')
  })

  it('returns capability with magic bytes', () => {
    const cap = ext.capability()
    expect(cap.magicBytes.length).toBeGreaterThan(0)

    // Verify PNG magic bytes: 89 50 4E 47
    const pngMagic = cap.magicBytes.find(
      (m) =>
        m.bytes.length >= 4 &&
        m.bytes[0] === 0x89 &&
        m.bytes[1] === 0x50 &&
        m.bytes[2] === 0x4e &&
        m.bytes[3] === 0x47,
    )
    expect(pngMagic).toBeDefined()

    // Verify JPEG magic bytes: FF D8 FF
    const jpegMagic = cap.magicBytes.find(
      (m) =>
        m.bytes.length >= 3 && m.bytes[0] === 0xff && m.bytes[1] === 0xd8 && m.bytes[2] === 0xff,
    )
    expect(jpegMagic).toBeDefined()
  })

  it('requires binary', () => {
    const cap = ext.capability()
    expect(cap.requiresBinary).toBe(true)
  })

  it('reports unavailable when no OCR binaries exist', async () => {
    const result = await ext.available()
    expect(result).toBe(false)
  })

  it('returns skipped result for empty input', async () => {
    const result = await ext.extract(Buffer.alloc(0), {
      contentType: 'image/png',
    })
    expect(result.skipped).toBe(true)
    expect(result.reason).toBe('empty input')
    expect(result.metadata).toEqual({ ocr_engine: 'none' })
  })

  it('throws descriptive error when no OCR engines available', async () => {
    await expect(
      ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        contentType: 'image/png',
      }),
    ).rejects.toThrow('no OCR engine available')
  })

  it('throws descriptive error mentioning both binary names', async () => {
    await expect(
      ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        contentType: 'image/png',
      }),
    ).rejects.toThrow('nonexistent-paddleocr-xyz')
  })

  it('conforms to the Extractor type', () => {
    const _typed: Extractor = ext
    expect(_typed.name).toBe('image-ocr')
  })
})

describe('createImageExtractor with config overrides', () => {
  it('accepts custom binary paths', () => {
    const ext = createImageExtractor({
      paddleOcrBinary: '/usr/local/bin/paddleocr',
      tesseractBinary: '/usr/local/bin/tesseract',
      defaultLanguage: 'de',
      timeout: 30_000,
    })
    expect(ext.name).toBe('image-ocr')
  })
})

describe('createImageExtractor extractStream', () => {
  it('throws for non-empty stream when no OCR engines available', async () => {
    const ext = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })

    const source = Readable.from([Buffer.from([0x89, 0x50, 0x4e, 0x47])])
    await expect(ext.extractStream(source, { contentType: 'image/png' })).rejects.toThrow(
      'no OCR engine available',
    )
  })

  it('returns skipped for empty stream', async () => {
    const ext = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })

    const source = Readable.from([Buffer.alloc(0)])
    const result = await ext.extractStream(source, { contentType: 'image/png' })
    expect(result.skipped).toBe(true)
  })
})

describe('parsePaddleOCROutput', () => {
  it('parses JSON array format', () => {
    const output = '[["Hello World", 0.95], ["Second Line", 0.88]]'
    const result = parsePaddleOCROutput(output)

    expect(result.text).toContain('Hello World')
    expect(result.text).toContain('Second Line')

    const expectedConf = (0.95 + 0.88) / 2
    expect(result.confidence).toBeCloseTo(expectedConf, 2)
  })

  it('parses tuple format', () => {
    const output = "('Hello World', 0.95)\n('Second Line', 0.88)\n"
    const result = parsePaddleOCROutput(output)

    expect(result.text).toContain('Hello World')
    expect(result.text).toContain('Second Line')

    const expectedConf = (0.95 + 0.88) / 2
    expect(result.confidence).toBeCloseTo(expectedConf, 2)
  })

  it('handles empty output', () => {
    const result = parsePaddleOCROutput('')
    expect(result.text).toBe('')
    expect(result.confidence).toBe(0)
  })

  it('handles single JSON entry', () => {
    const output = '[["Hello", 0.99]]'
    const result = parsePaddleOCROutput(output)
    expect(result.text).toBe('Hello')
    expect(result.confidence).toBeCloseTo(0.99, 2)
  })

  it('handles single tuple entry', () => {
    const output = "('Hello', 0.99)"
    const result = parsePaddleOCROutput(output)
    expect(result.text).toBe('Hello')
    expect(result.confidence).toBeCloseTo(0.99, 2)
  })

  it('handles double-quoted text in tuples', () => {
    const output = '("Hello World", 0.95)'
    const result = parsePaddleOCROutput(output)
    expect(result.text).toBe('Hello World')
    expect(result.confidence).toBeCloseTo(0.95, 2)
  })

  it('ignores non-parseable lines', () => {
    const output = 'Loading model...\nInitializing...\n[["Hello", 0.95]]\nDone.\n'
    const result = parsePaddleOCROutput(output)
    expect(result.text).toBe('Hello')
    expect(result.confidence).toBeCloseTo(0.95, 2)
  })

  it('handles multiple JSON lines', () => {
    const output = '[["Line 1", 0.9]]\n[["Line 2", 0.8]]'
    const result = parsePaddleOCROutput(output)
    expect(result.text).toContain('Line 1')
    expect(result.text).toContain('Line 2')
    expect(result.confidence).toBeCloseTo(0.85, 2)
  })
})

describe('createImageExtractor capability conformance', () => {
  it('capability mimeTypes matches contentTypes', () => {
    const ext = createImageExtractor()
    const cap = ext.capability()
    expect([...cap.mimeTypes].sort()).toEqual([...ext.contentTypes].sort())
  })

  it('capability has at least 6 magic byte signatures', () => {
    const ext = createImageExtractor()
    const cap = ext.capability()
    // PNG, JPEG, TIFF LE, TIFF BE, BMP, WebP
    expect(cap.magicBytes.length).toBeGreaterThanOrEqual(6)
  })
})
