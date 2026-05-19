// SPDX-License-Identifier: Apache-2.0

import { existsSync, writeFileSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { mkdtempSync, readdirSync, statSync } from 'node:fs'
import { Readable } from 'node:stream'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { ExtractOptions, Extractor, ExtractorCapability } from './extractor.js'
import { createImageExtractor, parsePaddleOCROutput } from './image.js'
import { resetBinaryCache } from './subprocess.js'
import type { Logger } from '../llm/types.js'

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

  it('capability has at least 7 magic byte signatures', () => {
    const ext = createImageExtractor()
    const cap = ext.capability()
    // PNG, JPEG, TIFF LE, TIFF BE, BMP, WebP RIFF, WebP WEBP
    expect(cap.magicBytes.length).toBeGreaterThanOrEqual(7)
  })

  it('includes WebP WEBP signature at offset 8', () => {
    const ext = createImageExtractor()
    const cap = ext.capability()
    const webpSig = cap.magicBytes.find(
      (m) =>
        m.offset === 8 &&
        m.bytes.length === 4 &&
        m.bytes[0] === 0x57 &&
        m.bytes[1] === 0x45 &&
        m.bytes[2] === 0x42 &&
        m.bytes[3] === 0x50,
    )
    expect(webpSig).toBeDefined()
  })
})

// --- C2: Language parameter validation ---

describe('language parameter validation', () => {
  it('rejects language starting with single dash', async () => {
    const ext = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })
    await expect(
      ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        contentType: 'image/png',
        language: '-malicious',
      }),
    ).rejects.toThrow('invalid language parameter')
  })

  it('rejects language starting with double dash', async () => {
    const ext = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })
    await expect(
      ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        contentType: 'image/png',
        language: '--exploit',
      }),
    ).rejects.toThrow('invalid language parameter')
  })

  it('accepts valid language codes', async () => {
    const ext = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })
    // Should not throw for valid language; will throw "no OCR engine"
    // which is the expected error (not a validation error).
    await expect(
      ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        contentType: 'image/png',
        language: 'de',
      }),
    ).rejects.toThrow('no OCR engine available')
  })
})

// --- C1: Happy-path mock subprocess tests ---

const writeMockScript = (dir: string, name: string, content: string): string => {
  const path = join(dir, name)
  writeFileSync(path, content, { mode: 0o755 })
  chmodSync(path, 0o755)
  return path
}

const MOCK_PADDLEOCR_SCRIPT = `#!/bin/sh
echo '[["Hello World", 0.95], ["Second Line", 0.88]]'
`

const MOCK_TESSERACT_SCRIPT = `#!/bin/sh
echo 'Hello World from Tesseract'
`

describe('createImageExtractor with mocked PaddleOCR binary', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('extracts text via mocked PaddleOCR subprocess', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', MOCK_PADDLEOCR_SCRIPT)

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: 'nonexistent-tesseract-xyz',
      timeout: 10_000,
    })

    const result = await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
    })

    expect(result.skipped).toBe(false)
    expect(result.text).toContain('Hello World')
    expect(result.text).toContain('Second Line')
    expect(result.metadata.ocr_engine).toBe('paddleocr')
    expect(result.confidence).toBeGreaterThan(0.8)
  })

  it('includes language in metadata when overridden', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', MOCK_PADDLEOCR_SCRIPT)

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: 'nonexistent-tesseract-xyz',
      defaultLanguage: 'en',
      timeout: 10_000,
    })

    const result = await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
      language: 'de',
    })

    expect(result.language).toBe('de')
    expect(result.metadata.ocr_language).toBe('german')
  })
})

describe('createImageExtractor with mocked Tesseract binary', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('extracts text via mocked Tesseract subprocess', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const tesseractPath = writeMockScript(tmpDir, 'tesseract', MOCK_TESSERACT_SCRIPT)

    const ext = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: tesseractPath,
      timeout: 10_000,
    })

    const result = await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
    })

    expect(result.skipped).toBe(false)
    expect(result.text).toContain('Hello World from Tesseract')
    expect(result.metadata.ocr_engine).toBe('tesseract')
  })
})

describe('createImageExtractor PaddleOCR fallback to Tesseract', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('falls back to Tesseract when PaddleOCR fails', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const failScript = '#!/bin/sh\nexit 1\n'
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', failScript)
    const tesseractPath = writeMockScript(tmpDir, 'tesseract', MOCK_TESSERACT_SCRIPT)

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: tesseractPath,
      timeout: 10_000,
    })

    const result = await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
    })

    expect(result.text).toContain('Hello World from Tesseract')
    expect(result.metadata.ocr_engine).toBe('tesseract')
  })

  it('logs warning when PaddleOCR fails and falls back', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const failScript = '#!/bin/sh\nexit 1\n'
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', failScript)
    const tesseractPath = writeMockScript(tmpDir, 'tesseract', MOCK_TESSERACT_SCRIPT)

    const mockLogger: Logger = {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    }

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: tesseractPath,
      timeout: 10_000,
      logger: mockLogger,
    })

    await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
    })

    expect(mockLogger.warn).toHaveBeenCalled()
    const warnCall = (mockLogger.warn as ReturnType<typeof vi.fn>).mock.calls[0]
    expect(warnCall[0]).toContain('paddleocr extraction failed')
  })
})

// --- M3: Temp file cleanup verification ---

describe('temp file cleanup', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('cleans up temp files after successful PaddleOCR extraction', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', MOCK_PADDLEOCR_SCRIPT)

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: 'nonexistent-tesseract-xyz',
      timeout: 10_000,
    })

    await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
    })

    // Check that no memory-ocr- directories remain in the system temp dir.
    const tempEntries = readdirSync(tmpdir()).filter((e) => e.startsWith('memory-ocr-'))
    const recentEntries = tempEntries.filter((e) => {
      const stat = statSync(join(tmpdir(), e))
      return Date.now() - stat.mtimeMs < 5000
    })
    expect(recentEntries).toHaveLength(0)
  })

  it('cleans up temp files after failed extraction', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const failScript = '#!/bin/sh\nexit 1\n'
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', failScript)
    const tesseractPath = writeMockScript(tmpDir, 'tesseract', failScript)

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: tesseractPath,
      timeout: 10_000,
    })

    try {
      await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
        contentType: 'image/png',
      })
    } catch {
      // Expected to throw.
    }

    const tempEntries = readdirSync(tmpdir()).filter((e) => e.startsWith('memory-ocr-'))
    const recentEntries = tempEntries.filter((e) => {
      const stat = statSync(join(tmpdir(), e))
      return Date.now() - stat.mtimeMs < 5000
    })
    expect(recentEntries).toHaveLength(0)
  })
})

// --- M5: Integration test skeleton ---

describe('image extractor integration', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('extracts from a real image when OCR binary is available', async () => {
    const ext = createImageExtractor()
    const available = await ext.available()
    if (!available) {
      return // Skip: no OCR binary available on this system.
    }

    // Minimal 1x1 white PNG.
    const pngData = Buffer.from([
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
      0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
      0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8,
      0xcf, 0xc0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, 0x33, 0x00, 0x00, 0x00,
      0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ])

    const result = await ext.extract(pngData, {
      contentType: 'image/png',
      fileName: 'test.png',
    })

    expect(result.skipped).toBe(false)
    const engine = result.metadata.ocr_engine
    expect(['paddleocr', 'tesseract']).toContain(engine)
  })
})

// --- M1: language field is used instead of encoding ---

describe('resolveLanguage uses language field', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('uses language field not encoding field', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'image-test-'))
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', MOCK_PADDLEOCR_SCRIPT)

    const ext = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: 'nonexistent-tesseract-xyz',
      defaultLanguage: 'en',
      timeout: 10_000,
    })

    // Providing encoding should NOT affect OCR language.
    const result = await ext.extract(Buffer.from([0x89, 0x50, 0x4e, 0x47]), {
      contentType: 'image/png',
      encoding: 'UTF-8',
    })

    // Should use default "en", not "UTF-8" from encoding field.
    expect(result.metadata.ocr_language).toBe('en')
    expect(result.language).toBe('en')
  })
})

// --- TS direct helper function tests (m7 parity with Go) ---

describe('parsePaddleOCROutput with interleaved log lines', () => {
  it('ignores non-parseable lines and extracts valid results', () => {
    const output = 'Loading model...\nInitializing engine...\n[["Hello", 0.95]]\nDone.\n'
    const result = parsePaddleOCROutput(output)
    expect(result.text).toBe('Hello')
    expect(result.confidence).toBeCloseTo(0.95, 2)
  })
})
