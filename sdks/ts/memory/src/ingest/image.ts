// SPDX-License-Identifier: Apache-2.0

/**
 * Image OCR extractor using PaddleOCR (primary) with Tesseract as
 * fallback. Both engines are invoked as subprocesses — no native
 * bindings required. Implements the Extractor interface from P1-5.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
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

/** Configuration for the image OCR extractor. */
export type ImageExtractorConfig = {
  /** Path or name of the PaddleOCR binary. Default: "paddleocr". */
  readonly paddleOcrBinary?: string
  /** Path or name of the Tesseract binary. Default: "tesseract". */
  readonly tesseractBinary?: string
  /** Default OCR language hint. Default: "en". */
  readonly defaultLanguage?: string
  /** Subprocess timeout in milliseconds. Default: 60000. */
  readonly timeout?: number
}

/** Maps ISO 639-1 codes to PaddleOCR language parameters. */
const PADDLE_OCR_LANGUAGE_MAP: Readonly<Record<string, string>> = {
  en: 'en',
  zh: 'ch',
  ja: 'japan',
  ko: 'korean',
  fr: 'fr',
  de: 'german',
  es: 'es',
  pt: 'pt',
  it: 'it',
  ru: 'ru',
  ar: 'ar',
  hi: 'hi',
  ta: 'ta',
  te: 'te',
}

/** Maps ISO 639-1 codes to Tesseract 3-letter language codes. */
const TESSERACT_LANGUAGE_MAP: Readonly<Record<string, string>> = {
  en: 'eng',
  zh: 'chi_sim',
  ja: 'jpn',
  ko: 'kor',
  fr: 'fra',
  de: 'deu',
  es: 'spa',
  pt: 'por',
  it: 'ita',
  ru: 'rus',
  ar: 'ara',
  hi: 'hin',
  ta: 'tam',
  te: 'tel',
}

/** Maps MIME types to file extensions for temp files. */
const CONTENT_TYPE_EXTENSION_MAP: Readonly<Record<string, string>> = {
  'image/png': '.png',
  'image/jpeg': '.jpg',
  'image/tiff': '.tiff',
  'image/bmp': '.bmp',
  'image/webp': '.webp',
}

/** Image content types handled by this extractor. */
const IMAGE_CONTENT_TYPES: readonly string[] = [
  'image/png',
  'image/jpeg',
  'image/tiff',
  'image/bmp',
  'image/webp',
]

/** File extensions handled by this extractor. */
const IMAGE_EXTENSIONS: readonly string[] = [
  '.png',
  '.jpg',
  '.jpeg',
  '.tiff',
  '.tif',
  '.bmp',
  '.webp',
]

/** Magic byte signatures for supported image formats. */
const IMAGE_MAGIC_BYTES: readonly MagicSignature[] = [
  { offset: 0, bytes: new Uint8Array([0x89, 0x50, 0x4e, 0x47]) }, // PNG
  { offset: 0, bytes: new Uint8Array([0xff, 0xd8, 0xff]) }, // JPEG
  { offset: 0, bytes: new Uint8Array([0x49, 0x49, 0x2a, 0x00]) }, // TIFF LE
  { offset: 0, bytes: new Uint8Array([0x4d, 0x4d, 0x00, 0x2a]) }, // TIFF BE
  { offset: 0, bytes: new Uint8Array([0x42, 0x4d]) }, // BMP
  { offset: 0, bytes: new Uint8Array([0x52, 0x49, 0x46, 0x46]) }, // WebP (RIFF)
]

/**
 * Resolves a configuration value with env var override support.
 */
const envOrDefault = (envKey: string, fallback: string): string => {
  const value = process.env[envKey]
  return value !== undefined && value !== '' ? value : fallback
}

/**
 * Resolves the effective timeout from config and environment.
 */
const resolveTimeout = (configTimeout?: number): number => {
  if (configTimeout !== undefined && configTimeout > 0) return configTimeout
  const envMs = process.env.MEMORY_EXTRACTOR_TIMEOUT_MS
  if (envMs !== undefined && envMs !== '') {
    const parsed = Number.parseInt(envMs, 10)
    if (!Number.isNaN(parsed) && parsed > 0) return parsed
  }
  return DEFAULT_SUBPROCESS_TIMEOUT_MS
}

/**
 * Maps an ISO 639-1 language code to PaddleOCR's format.
 */
const mapLanguageToPaddleOCR = (lang: string): string => {
  const normalised = lang.trim().toLowerCase()
  return PADDLE_OCR_LANGUAGE_MAP[normalised] ?? normalised
}

/**
 * Maps an ISO 639-1 language code to Tesseract's 3-letter format.
 */
const mapLanguageToTesseract = (lang: string): string => {
  const normalised = lang.trim().toLowerCase()
  return TESSERACT_LANGUAGE_MAP[normalised] ?? 'eng'
}

/**
 * Determines the file extension from extract options.
 */
const extensionFromOpts = (opts: ExtractOptions): string => {
  const fromMime = CONTENT_TYPE_EXTENSION_MAP[opts.contentType]
  if (fromMime !== undefined) return fromMime

  if (opts.fileName !== undefined) {
    const dotIdx = opts.fileName.lastIndexOf('.')
    if (dotIdx >= 0) return opts.fileName.slice(dotIdx)
  }

  return '.png'
}

/**
 * Writes raw bytes to a secure temporary file and returns the path.
 * Creates a dedicated temp directory to prevent collisions.
 */
const writeTempFile = async (
  raw: Buffer,
  extension: string,
): Promise<{ path: string; dir: string }> => {
  const dir = await mkdtemp(join(tmpdir(), 'memory-ocr-'))
  const path = join(dir, `input${extension}`)
  await writeFile(path, raw)
  return { path, dir }
}

/**
 * Strips surrounding single or double quotes from a string.
 */
const stripQuotes = (s: string): string => {
  if (s.length < 2) return s
  const first = s[0]
  const last = s[s.length - 1]
  if ((first === "'" && last === "'") || (first === '"' && last === '"')) {
    return s.slice(1, -1)
  }
  return s
}

/** Parsed OCR result with text blocks and average confidence. */
type ParsedOCRResult = {
  readonly text: string
  readonly confidence: number
}

/**
 * Parses PaddleOCR output. Handles both JSON array format and
 * tuple-style line format.
 */
export const parsePaddleOCROutput = (output: string): ParsedOCRResult => {
  const lines = output.split('\n')
  const textBlocks: string[] = []
  let totalConfidence = 0
  let blockCount = 0

  for (const rawLine of lines) {
    const line = rawLine.trim()
    if (line === '') continue

    // Try JSON array format: [["text", confidence], ...]
    try {
      const entries = JSON.parse(line) as unknown
      if (Array.isArray(entries)) {
        for (const entry of entries) {
          if (Array.isArray(entry) && entry.length >= 2) {
            const text = entry[0]
            const conf = entry[1]
            if (typeof text === 'string' && typeof conf === 'number') {
              textBlocks.push(text)
              totalConfidence += conf
              blockCount++
            }
          }
        }
        continue
      }
    } catch {
      // Not JSON; try tuple format.
    }

    // Try tuple format: ('text', 0.95)
    if (line.startsWith('(') && line.endsWith(')')) {
      const inner = line.slice(1, -1)
      const lastComma = inner.lastIndexOf(',')
      if (lastComma > 0) {
        const textPart = stripQuotes(inner.slice(0, lastComma).trim())
        const confPart = inner.slice(lastComma + 1).trim()
        const conf = Number.parseFloat(confPart)
        if (!Number.isNaN(conf)) {
          textBlocks.push(textPart)
          totalConfidence += conf
          blockCount++
        }
      }
    }
  }

  return {
    text: textBlocks.join('\n'),
    confidence: blockCount > 0 ? totalConfidence / blockCount : 0,
  }
}

/**
 * Creates an ImageExtractor that extracts text from images using
 * PaddleOCR (primary) with Tesseract as fallback.
 */
export const createImageExtractor = (config?: ImageExtractorConfig): Extractor => {
  const paddleOcrBinary =
    config?.paddleOcrBinary ?? envOrDefault('MEMORY_PADDLEOCR_PATH', 'paddleocr')
  const tesseractBinary =
    config?.tesseractBinary ?? envOrDefault('MEMORY_TESSERACT_PATH', 'tesseract')
  const defaultLanguage = config?.defaultLanguage ?? 'en'
  const timeout = resolveTimeout(config?.timeout)

  const resolveLanguage = (opts: ExtractOptions): string => opts.encoding ?? defaultLanguage

  const extractWithPaddleOCR = async (
    raw: Buffer,
    language: string,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult> => {
    const ext = extensionFromOpts(opts)
    const { path: tmpPath, dir: tmpDir } = await writeTempFile(raw, ext)

    try {
      const paddleLang = mapLanguageToPaddleOCR(language)

      const result = await runSubprocess(
        paddleOcrBinary,
        ['--image_dir', tmpPath, '--use_angle_cls', 'true', '--lang', paddleLang, '--type', 'ocr'],
        undefined,
        { timeout, signal },
      )

      if (result.exitCode !== 0) {
        throw new Error(`paddleocr exited with code ${result.exitCode}: ${result.stderr}`)
      }

      const parsed = parsePaddleOCROutput(result.stdout.toString('utf-8'))

      return {
        text: parsed.text,
        contentType: opts.contentType,
        encoding: 'UTF-8',
        metadata: {
          ocr_engine: 'paddleocr',
          ocr_confidence: parsed.confidence.toFixed(4),
          ocr_language: paddleLang,
        },
        pages: 0,
        language,
        confidence: parsed.confidence,
        skipped: false,
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  const extractWithTesseract = async (
    raw: Buffer,
    language: string,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult> => {
    const ext = extensionFromOpts(opts)
    const { path: tmpPath, dir: tmpDir } = await writeTempFile(raw, ext)

    try {
      const tesseractLang = mapLanguageToTesseract(language)

      const result = await runSubprocess(
        tesseractBinary,
        [tmpPath, 'stdout', '-l', tesseractLang, '--oem', '3', '--psm', '3'],
        undefined,
        { timeout, signal },
      )

      if (result.exitCode !== 0) {
        throw new Error(`tesseract exited with code ${result.exitCode}: ${result.stderr}`)
      }

      const text = result.stdout.toString('utf-8').trim()

      return {
        text,
        contentType: opts.contentType,
        encoding: 'UTF-8',
        metadata: {
          ocr_engine: 'tesseract',
          ocr_language: tesseractLang,
        },
        pages: 0,
        language,
        confidence: 0,
        skipped: false,
      }
    } finally {
      await rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    }
  }

  const extractor: Extractor = {
    name: 'image-ocr',

    contentTypes: IMAGE_CONTENT_TYPES,

    async extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult> {
      if (raw.length === 0) {
        return {
          text: '',
          contentType: opts.contentType,
          encoding: '',
          metadata: { ocr_engine: 'none' },
          pages: 0,
          language: '',
          confidence: 0,
          skipped: true,
          reason: 'empty input',
        }
      }

      const language = resolveLanguage(opts)

      // Try PaddleOCR first.
      if (await checkBinaryAvailable(paddleOcrBinary)) {
        try {
          return await extractWithPaddleOCR(raw, language, opts, signal)
        } catch {
          // Fall through to Tesseract.
        }
      }

      // Fallback to Tesseract.
      if (await checkBinaryAvailable(tesseractBinary)) {
        try {
          return await extractWithTesseract(raw, language, opts, signal)
        } catch (err: unknown) {
          throw new Error(
            `ingest: image extraction failed with both OCR engines: ${err instanceof Error ? err.message : String(err)}`,
          )
        }
      }

      throw new Error(
        `ingest: no OCR engine available (tried ${paddleOcrBinary}, ${tesseractBinary})`,
      )
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
      const paddleAvailable = await checkBinaryAvailable(paddleOcrBinary)
      if (paddleAvailable) return true
      return checkBinaryAvailable(tesseractBinary)
    },

    capability(): ExtractorCapability {
      return {
        extensions: IMAGE_EXTENSIONS,
        mimeTypes: IMAGE_CONTENT_TYPES,
        magicBytes: IMAGE_MAGIC_BYTES,
        requiresBinary: true,
      }
    },
  }

  return extractor
}
