// SPDX-License-Identifier: Apache-2.0

/**
 * PDF OCR extractor. Text-based PDFs are handled via pdftotext.
 * Scanned PDFs (no text layer or minimal text) are converted
 * page-by-page to images via pdftoppm (poppler-utils), then OCR-ed
 * via the ImageExtractor delegate. Implements the Extractor interface.
 */

import { readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { mkdtemp } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Readable } from 'node:stream'
import type {
  ExtractOptions,
  ExtractResult,
  Extractor,
  ExtractorCapability,
  MagicSignature,
} from './extractor.js'
import { DEFAULT_SUBPROCESS_TIMEOUT_MS, checkBinaryAvailable, runSubprocess } from './subprocess.js'
import type { Logger } from '../llm/types.js'
import { noopLogger } from '../llm/types.js'

/** Configuration for the PDF OCR extractor. */
export type PDFExtractorConfig = {
  /** Path or name of the pdftoppm binary. Default: "pdftoppm". */
  readonly pdftoppmBinary?: string
  /** Path or name of the pdftotext binary. Default: "pdftotext". */
  readonly pdftotextBinary?: string
  /** Delegate extractor for OCR on page images. */
  readonly imageExtractor?: Extractor
  /** Subprocess timeout in milliseconds. Default: 120000. */
  readonly timeout?: number
  /** Maximum pages to extract. Default: 100. */
  readonly maxPages?: number
  /** Logger for diagnostic output. */
  readonly logger?: Logger
}

const DEFAULT_PDF_TIMEOUT = 120_000
const DEFAULT_MAX_PDF_PAGES = 100
const PDF_CONTENT_TYPE = 'application/pdf'
const SCANNED_TEXT_THRESHOLD = 50

/** PDF magic bytes: %PDF */
const PDF_MAGIC_BYTES: readonly MagicSignature[] = [
  { offset: 0, bytes: new Uint8Array([0x25, 0x50, 0x44, 0x46]) },
]

const envOrDefault = (envKey: string, fallback: string): string => {
  const value = process.env[envKey]
  return value !== undefined && value !== '' ? value : fallback
}

const resolveTimeout = (configTimeout?: number): number => {
  if (configTimeout !== undefined && configTimeout > 0) return configTimeout
  const envMs = process.env.MEMORY_EXTRACTOR_TIMEOUT_MS
  if (envMs !== undefined && envMs !== '') {
    const parsed = Number.parseInt(envMs, 10)
    if (!Number.isNaN(parsed) && parsed > 0) return parsed
  }
  return DEFAULT_PDF_TIMEOUT
}

/**
 * Checks whether extracted text has enough content to be considered
 * a genuine text layer rather than artefacts from a scanned PDF.
 */
export const isSubstantialText = (text: string): boolean =>
  text.trim().length >= SCANNED_TEXT_THRESHOLD

/**
 * Creates a PDFExtractor that extracts text from PDF files.
 * Text-based PDFs use pdftotext directly. Scanned PDFs are converted
 * to images via pdftoppm and then OCR-ed via the image extractor.
 */
export const createPDFExtractor = (config?: PDFExtractorConfig): Extractor => {
  const pdftoppmBinary =
    config?.pdftoppmBinary ?? envOrDefault('MEMORY_PDFTOPPM_PATH', 'pdftoppm')
  const pdftotextBinary =
    config?.pdftotextBinary ?? envOrDefault('MEMORY_PDFTOTEXT_PATH', 'pdftotext')
  const imageExtractor = config?.imageExtractor
  const timeout = resolveTimeout(config?.timeout)
  const maxPages = config?.maxPages ?? DEFAULT_MAX_PDF_PAGES
  const logger = config?.logger ?? noopLogger

  const extractWithPdftotext = async (
    pdfPath: string,
    signal?: AbortSignal,
  ): Promise<string> => {
    const subprocessOpts = signal !== undefined ? { timeout, signal } : { timeout }
    const result = await runSubprocess(
      pdftotextBinary,
      ['-layout', pdfPath, '-'],
      undefined,
      subprocessOpts,
    )
    if (result.exitCode !== 0) {
      throw new Error(`pdftotext exited with code ${result.exitCode}: ${result.stderr}`)
    }
    return result.stdout.toString('utf-8').trim()
  }

  const extractViaOCR = async (
    pdfPath: string,
    tmpDir: string,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult> => {
    if (imageExtractor === undefined) {
      throw new Error('ingest: PDF OCR requires an imageExtractor dependency')
    }

    const outputPrefix = join(tmpDir, 'page')
    const subprocessOpts = signal !== undefined ? { timeout, signal } : { timeout }

    const convResult = await runSubprocess(
      pdftoppmBinary,
      ['-png', '-r', '300', '-l', String(maxPages), pdfPath, outputPrefix],
      undefined,
      subprocessOpts,
    )
    if (convResult.exitCode !== 0) {
      throw new Error(`pdftoppm exited with code ${convResult.exitCode}: ${convResult.stderr}`)
    }

    const files = await readdir(tmpDir)
    const pageImages = files
      .filter((f) => f.startsWith('page') && f.endsWith('.png'))
      .sort()

    if (pageImages.length === 0) {
      throw new Error('ingest: pdftoppm produced no page images')
    }

    const pageTexts: string[] = []
    let totalConfidence = 0
    let pageCount = 0

    for (const imageFile of pageImages) {
      const imagePath = join(tmpDir, imageFile)
      let imageData: Buffer
      try {
        imageData = await readFile(imagePath)
      } catch (readErr: unknown) {
        const message = readErr instanceof Error ? readErr.message : String(readErr)
        logger.warn(`skipping unreadable page image: ${message}`, { path: imagePath })
        continue
      }

      const pageOpts: ExtractOptions = {
        contentType: 'image/png',
        fileName: imageFile,
        ...(opts.language !== undefined ? { language: opts.language } : {}),
      }

      try {
        const pageResult = await imageExtractor.extract(imageData, pageOpts, signal)
        if (!pageResult.skipped && pageResult.text.length > 0) {
          pageTexts.push(pageResult.text)
          totalConfidence += pageResult.confidence
          pageCount++
        }
      } catch (ocrErr: unknown) {
        const message = ocrErr instanceof Error ? ocrErr.message : String(ocrErr)
        logger.warn(`OCR failed for page ${imageFile}: ${message}`)
      }
    }

    const avgConfidence = pageCount > 0 ? totalConfidence / pageCount : 0

    return {
      text: pageTexts.join('\n\n'),
      contentType: 'text/plain',
      encoding: 'UTF-8',
      metadata: {
        extractor: 'pdf-ocr',
        pdf_method: 'pdftoppm+ocr',
        pdf_pages: String(pageImages.length),
        ocr_pages: String(pageCount),
        ocr_confidence: avgConfidence.toFixed(4),
      },
      pages: pageImages.length,
      language: opts.language ?? '',
      confidence: avgConfidence,
      skipped: false,
    }
  }

  const extractor: Extractor = {
    name: 'pdf-ocr',

    contentTypes: [PDF_CONTENT_TYPE],

    async extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult> {
      if (raw.length === 0) {
        return {
          text: '',
          contentType: PDF_CONTENT_TYPE,
          encoding: '',
          metadata: { extractor: 'pdf-ocr' },
          pages: 0,
          language: '',
          confidence: 0,
          skipped: true,
          reason: 'empty input',
        }
      }

      const tmpDir = await mkdtemp(join(tmpdir(), 'memory-pdf-'))
      try {
        const pdfPath = join(tmpDir, 'input.pdf')
        await writeFile(pdfPath, raw, { mode: 0o600 })

        // Try pdftotext first to detect if scanned.
        let textResult = ''
        try {
          textResult = await extractWithPdftotext(pdfPath, signal)
        } catch {
          // pdftotext failed; treat as scanned.
        }

        if (isSubstantialText(textResult)) {
          return {
            text: textResult,
            contentType: 'text/plain',
            encoding: 'UTF-8',
            metadata: {
              extractor: 'pdf-ocr',
              pdf_method: 'pdftotext',
            },
            pages: 0,
            language: opts.language ?? '',
            confidence: 0,
            skipped: false,
          }
        }

        logger.info(
          `pdf appears scanned or has minimal text layer, falling back to OCR (text_length=${textResult.length})`,
          { file: opts.fileName },
        )

        return await extractViaOCR(pdfPath, tmpDir, opts, signal)
      } finally {
        await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
      }
    },

    async extractStream(
      source: Readable,
      opts: ExtractOptions,
      signal?: AbortSignal,
    ): Promise<ExtractResult> {
      const chunks: Buffer[] = []
      let totalBytes = 0

      for await (const chunk of source) {
        const buf = Buffer.isBuffer(chunk)
          ? chunk
          : chunk instanceof Uint8Array
            ? Buffer.from(chunk)
            : Buffer.from(String(chunk))

        if (opts.maxBytes !== undefined && opts.maxBytes > 0) {
          const remaining = opts.maxBytes - totalBytes
          if (remaining <= 0) break
          chunks.push(buf.subarray(0, remaining))
          totalBytes += Math.min(buf.length, remaining)
          if (totalBytes >= opts.maxBytes) break
        } else {
          chunks.push(buf)
          totalBytes += buf.length
        }
      }

      return extractor.extract(Buffer.concat(chunks), opts, signal)
    },

    async available(): Promise<boolean> {
      const hasPdftotext = await checkBinaryAvailable(pdftotextBinary)
      if (!hasPdftotext) return false

      const hasPdftoppm = await checkBinaryAvailable(pdftoppmBinary)
      if (!hasPdftoppm) return false

      if (imageExtractor === undefined) return false

      return imageExtractor.available()
    },

    capability(): ExtractorCapability {
      return {
        extensions: ['.pdf'],
        mimeTypes: [PDF_CONTENT_TYPE],
        magicBytes: PDF_MAGIC_BYTES,
        requiresBinary: true,
      }
    },
  }

  return extractor
}
