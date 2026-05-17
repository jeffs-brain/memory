// SPDX-License-Identifier: Apache-2.0

import { chmodSync, mkdtempSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Readable } from 'node:stream'
import { afterEach, describe, expect, it, vi } from 'vitest'
import type { Extractor, ExtractOptions } from './extractor.js'
import { createPDFExtractor, isSubstantialText } from './pdf.js'
import { createImageExtractor } from './image.js'
import { resetBinaryCache } from './subprocess.js'
import type { Logger } from '../llm/types.js'

const writeMockScript = (dir: string, name: string, content: string): string => {
  const path = join(dir, name)
  writeFileSync(path, content, { mode: 0o755 })
  chmodSync(path, 0o755)
  return path
}

const MOCK_PDFTOTEXT_EMPTY = `#!/bin/sh
echo ""
`

const MOCK_PDFTOTEXT_WITH_TEXT = `#!/bin/sh
echo "This is a PDF document with a genuine text layer that contains enough characters to pass the scanned threshold test."
`

const MOCK_PDFTOPPM = `#!/bin/sh
# Creates a fake PNG file to simulate page image output.
# pdftoppm args: -png -r 300 -l <maxPages> <pdfPath> <outputPrefix>
# The output prefix is the last argument.
prefix=""
for arg; do
  prefix="$arg"
done
printf '\\x89PNG' > "\${prefix}-1.png"
`

const MOCK_PADDLEOCR = `#!/bin/sh
echo '[["OCR text from scanned PDF", 0.92]]'
`

describe('createPDFExtractor', () => {
  const ext = createPDFExtractor({
    pdftoppmBinary: 'nonexistent-pdftoppm-xyz',
    pdftotextBinary: 'nonexistent-pdftotext-xyz',
  })

  it('has the correct name', () => {
    expect(ext.name).toBe('pdf-ocr')
  })

  it('declares PDF content type', () => {
    expect(ext.contentTypes).toContain('application/pdf')
    expect(ext.contentTypes).toHaveLength(1)
  })

  it('returns capability with .pdf extension', () => {
    const cap = ext.capability()
    expect(cap.extensions).toContain('.pdf')
    expect(cap.extensions).toHaveLength(1)
  })

  it('returns capability with PDF MIME type', () => {
    const cap = ext.capability()
    expect(cap.mimeTypes).toContain('application/pdf')
  })

  it('returns capability with PDF magic bytes', () => {
    const cap = ext.capability()
    expect(cap.magicBytes.length).toBeGreaterThan(0)
    const pdfMagic = cap.magicBytes.find(
      (m) =>
        m.offset === 0 &&
        m.bytes.length === 4 &&
        m.bytes[0] === 0x25 &&
        m.bytes[1] === 0x50 &&
        m.bytes[2] === 0x44 &&
        m.bytes[3] === 0x46,
    )
    expect(pdfMagic).toBeDefined()
  })

  it('requires binary', () => {
    const cap = ext.capability()
    expect(cap.requiresBinary).toBe(true)
  })

  it('reports unavailable when binaries are absent', async () => {
    const result = await ext.available()
    expect(result).toBe(false)
  })

  it('returns skipped result for empty input', async () => {
    const result = await ext.extract(Buffer.alloc(0), {
      contentType: 'application/pdf',
    })
    expect(result.skipped).toBe(true)
    expect(result.reason).toBe('empty input')
  })

  it('conforms to the Extractor type', () => {
    const _typed: Extractor = ext
    expect(_typed.name).toBe('pdf-ocr')
  })
})

describe('isSubstantialText', () => {
  it('returns false for empty string', () => {
    expect(isSubstantialText('')).toBe(false)
  })

  it('returns false for whitespace only', () => {
    expect(isSubstantialText('   \n\t  ')).toBe(false)
  })

  it('returns false for short artefact text', () => {
    expect(isSubstantialText('  \f  ')).toBe(false)
  })

  it('returns false just under threshold', () => {
    expect(isSubstantialText('a'.repeat(49))).toBe(false)
  })

  it('returns true at threshold', () => {
    expect(isSubstantialText('a'.repeat(50))).toBe(true)
  })

  it('returns true for substantial text', () => {
    expect(
      isSubstantialText(
        'This is a real paragraph with enough words to pass the threshold.',
      ),
    ).toBe(true)
  })
})

describe('createPDFExtractor with text-based PDF', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('extracts text via pdftotext for text-based PDFs', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'pdf-test-'))
    const pdftotextPath = writeMockScript(tmpDir, 'pdftotext', MOCK_PDFTOTEXT_WITH_TEXT)

    const imgExt = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })

    const ext = createPDFExtractor({
      pdftotextBinary: pdftotextPath,
      pdftoppmBinary: 'nonexistent-pdftoppm-xyz',
      imageExtractor: imgExt,
      timeout: 10_000,
    })

    const pdfData = Buffer.from('%PDF-1.4 dummy content for testing')
    const result = await ext.extract(pdfData, {
      contentType: 'application/pdf',
    })

    expect(result.skipped).toBe(false)
    expect(result.text).toContain('genuine text layer')
    expect(result.metadata.pdf_method).toBe('pdftotext')
    expect(result.metadata.extractor).toBe('pdf-ocr')
  })
})

describe('createPDFExtractor with scanned PDF', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('falls back to OCR for scanned PDFs', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'pdf-test-'))
    const pdftotextPath = writeMockScript(tmpDir, 'pdftotext', MOCK_PDFTOTEXT_EMPTY)
    const pdftoppmPath = writeMockScript(tmpDir, 'pdftoppm', MOCK_PDFTOPPM)
    const paddlePath = writeMockScript(tmpDir, 'paddleocr', MOCK_PADDLEOCR)

    const imgExt = createImageExtractor({
      paddleOcrBinary: paddlePath,
      tesseractBinary: 'nonexistent-tesseract-xyz',
      timeout: 10_000,
    })

    const ext = createPDFExtractor({
      pdftotextBinary: pdftotextPath,
      pdftoppmBinary: pdftoppmPath,
      imageExtractor: imgExt,
      timeout: 10_000,
    })

    const pdfData = Buffer.from('%PDF-1.4 dummy scanned content')
    const result = await ext.extract(pdfData, {
      contentType: 'application/pdf',
    })

    expect(result.skipped).toBe(false)
    expect(result.metadata.pdf_method).toBe('pdftoppm+ocr')
    expect(result.metadata.extractor).toBe('pdf-ocr')
  })
})

describe('createPDFExtractor without imageExtractor', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('reports unavailable when imageExtractor is missing', async () => {
    const ext = createPDFExtractor({
      pdftoppmBinary: 'nonexistent-pdftoppm-xyz',
      pdftotextBinary: 'nonexistent-pdftotext-xyz',
    })
    const available = await ext.available()
    expect(available).toBe(false)
  })

  it('throws when scanned PDF needs OCR but no imageExtractor', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'pdf-test-'))
    const pdftotextPath = writeMockScript(tmpDir, 'pdftotext', MOCK_PDFTOTEXT_EMPTY)

    const ext = createPDFExtractor({
      pdftotextBinary: pdftotextPath,
      pdftoppmBinary: 'nonexistent-pdftoppm-xyz',
      timeout: 10_000,
    })

    const pdfData = Buffer.from('%PDF-1.4 dummy scanned content')
    await expect(
      ext.extract(pdfData, { contentType: 'application/pdf' }),
    ).rejects.toThrow('imageExtractor')
  })
})

describe('createPDFExtractor extractStream', () => {
  it('returns skipped for empty stream', async () => {
    const ext = createPDFExtractor({
      pdftoppmBinary: 'nonexistent-pdftoppm-xyz',
      pdftotextBinary: 'nonexistent-pdftotext-xyz',
    })

    const source = Readable.from([Buffer.alloc(0)])
    const result = await ext.extractStream(source, { contentType: 'application/pdf' })
    expect(result.skipped).toBe(true)
  })
})

describe('createPDFExtractor temp file cleanup', () => {
  afterEach(() => {
    resetBinaryCache()
  })

  it('cleans up temp files after extraction', async () => {
    const tmpDir = mkdtempSync(join(tmpdir(), 'pdf-test-'))
    const pdftotextPath = writeMockScript(tmpDir, 'pdftotext', MOCK_PDFTOTEXT_WITH_TEXT)

    const imgExt = createImageExtractor({
      paddleOcrBinary: 'nonexistent-paddleocr-xyz',
      tesseractBinary: 'nonexistent-tesseract-xyz',
    })

    const ext = createPDFExtractor({
      pdftotextBinary: pdftotextPath,
      pdftoppmBinary: 'nonexistent-pdftoppm-xyz',
      imageExtractor: imgExt,
      timeout: 10_000,
    })

    await ext.extract(Buffer.from('%PDF-1.4 dummy content'), {
      contentType: 'application/pdf',
    })

    const tempEntries = readdirSync(tmpdir()).filter((e) => e.startsWith('memory-pdf-'))
    const recentEntries = tempEntries.filter((e) => {
      const stat = statSync(join(tmpdir(), e))
      return Date.now() - stat.mtimeMs < 5000
    })
    expect(recentEntries).toHaveLength(0)
  })
})

describe('createPDFExtractor config', () => {
  it('accepts custom config values', () => {
    const ext = createPDFExtractor({
      pdftoppmBinary: '/custom/pdftoppm',
      pdftotextBinary: '/custom/pdftotext',
      maxPages: 50,
      timeout: 90_000,
    })
    expect(ext.name).toBe('pdf-ocr')
  })
})
