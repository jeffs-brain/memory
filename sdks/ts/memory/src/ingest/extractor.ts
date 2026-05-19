// SPDX-License-Identifier: Apache-2.0

/**
 * Extractor registry with content-type routing and streaming support.
 * Routes incoming content to the appropriate extractor based on MIME type,
 * with fallback behaviour for unknown types. Mirrors the Go implementation
 * in go/ingest/extractor.go.
 */

import type { Readable } from 'node:stream'
import { TextDecoder as NodeTextDecoder } from 'node:util'

/**
 * Security constants for downstream extractors (Phase 4) that decompress
 * archives or spawn subprocesses.
 */

/** Caps the ratio of decompressed-to-compressed size to prevent ZIP bomb attacks. */
export const MAX_DECOMPRESSION_RATIO = 100

/** Limits the number of files extracted from an archive to prevent resource exhaustion. */
export const MAX_EXTRACTED_FILES = 1000

/**
 * Allowlisted flags that may be passed to subprocess extractors. Anything
 * starting with '-' that is not in this set is rejected.
 */
const SANITIZE_ARGS_ALLOWLIST: ReadonlySet<string> = new Set([
  '-o',
  '--output',
  '-f',
  '--format',
  '-q',
  '--quiet',
  '-v',
  '--verbose',
  '--stdin',
  '--stdout',
  '--no-color',
])

/**
 * Filters a list of command-line arguments, rejecting any flag (starting
 * with '-') that is not in the hardcoded allowlist. Prevents injection of
 * dangerous flags into subprocess extractors.
 *
 * @throws Error when a disallowed flag is encountered
 */
export const sanitizeArgs = (args: readonly string[]): string[] => {
  const sanitized: string[] = []
  for (const arg of args) {
    if (arg.startsWith('-') && !SANITIZE_ARGS_ALLOWLIST.has(arg)) {
      throw new Error(`ingest: disallowed argument "${arg}"`)
    }
    sanitized.push(arg)
  }
  return sanitized
}

/** Identifies a file format by magic bytes at a given offset. */
export type MagicSignature = {
  readonly offset: number
  readonly bytes: Uint8Array
}

/**
 * Describes what content types an extractor handles. Used by the
 * registry for routing and by callers to inspect extractor capabilities
 * without triggering extraction.
 */
export type ExtractorCapability = {
  readonly extensions: readonly string[]
  readonly mimeTypes: readonly string[]
  readonly magicBytes: readonly MagicSignature[]
  readonly requiresBinary: boolean
}

/** Options provided to an extractor about the content being processed. */
export type ExtractOptions = {
  readonly contentType: string
  readonly fileName?: string
  readonly encoding?: string
  /** ISO 639-1 language hint for OCR extractors (e.g. "en", "de"). */
  readonly language?: string
  readonly maxBytes?: number
}

/** The output of an extraction operation. */
export type ExtractResult = {
  readonly text: string
  readonly contentType: string
  readonly encoding: string
  readonly metadata: Readonly<Record<string, string>>
  readonly pages: number
  readonly language: string
  readonly confidence: number
  readonly skipped: boolean
  readonly reason?: string
}

/**
 * Contract for content extraction. Implementations declare the MIME types
 * they handle and provide both buffered and streaming extraction methods.
 */
export type Extractor = {
  /** Converts buffered raw bytes into text content. */
  extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult>
  /**
   * Processes content from a readable stream. Default implementations
   * buffer into extract().
   */
  extractStream(
    source: Readable,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult>
  /** The MIME types this extractor handles. */
  readonly contentTypes: readonly string[]
  /** Human-readable identifier for this extractor. */
  readonly name: string
  /**
   * Reports whether this extractor's external dependencies are present
   * (e.g. PaddleOCR, FFmpeg). Extractors with no external dependencies
   * always resolve to true.
   */
  available(): Promise<boolean>
  /**
   * Describes what content types, file extensions, and magic byte
   * signatures this extractor handles. Used by the registry for routing
   * and by callers for introspection.
   */
  capability(): ExtractorCapability
}

/**
 * Collects all chunks from a Readable into a Buffer, respecting an
 * optional byte limit.
 */
const bufferStream = async (source: Readable, maxBytes?: number): Promise<Buffer> => {
  const chunks: Buffer[] = []
  let totalBytes = 0

  for await (const chunk of source) {
    const buf = Buffer.isBuffer(chunk)
      ? chunk
      : typeof chunk === 'string'
        ? Buffer.from(chunk)
        : chunk instanceof Uint8Array
          ? Buffer.from(chunk)
          : Buffer.from(String(chunk))
    if (maxBytes !== undefined && maxBytes > 0) {
      const remaining = maxBytes - totalBytes
      if (remaining <= 0) break
      chunks.push(buf.subarray(0, remaining))
      totalBytes += Math.min(buf.length, remaining)
      if (totalBytes >= maxBytes) break
    } else {
      chunks.push(buf)
      totalBytes += buf.length
    }
  }

  return Buffer.concat(chunks)
}

/**
 * PlainTextExtractor handles text/* content types by returning the raw
 * bytes as UTF-8 text.
 */
export const createPlainTextExtractor = (): Extractor => ({
  name: 'plain-text',

  contentTypes: [
    'text/plain',
    'text/markdown',
    'text/csv',
    'text/x-yaml',
    'application/json',
    'application/x-yaml',
  ],

  async extract(raw: Buffer, _opts: ExtractOptions): Promise<ExtractResult> {
    const textDecoder = new TextDecoder('utf-8', { fatal: true })
    try {
      const text = textDecoder.decode(raw)
      return {
        text,
        contentType: _opts.contentType,
        encoding: 'UTF-8',
        metadata: {},
        pages: 0,
        language: '',
        confidence: 0,
        skipped: false,
      }
    } catch {
      return {
        text: '',
        contentType: _opts.contentType,
        encoding: '',
        metadata: {},
        pages: 0,
        language: '',
        confidence: 0,
        skipped: true,
        reason: 'invalid utf-8 content',
      }
    }
  },

  async extractStream(source: Readable, opts: ExtractOptions): Promise<ExtractResult> {
    const raw = await bufferStream(source, opts.maxBytes)
    return this.extract(raw, opts)
  },

  async available(): Promise<boolean> {
    return true
  },

  capability(): ExtractorCapability {
    return {
      extensions: ['.txt', '.text', '.log', '.md', '.markdown', '.csv', '.yaml', '.yml', '.json'],
      mimeTypes: this.contentTypes,
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

/** Normalises a content type by stripping parameters and lowercasing. */
const normaliseContentType = (ct: string): string => {
  const semiIdx = ct.indexOf(';')
  const base = semiIdx >= 0 ? ct.slice(0, semiIdx) : ct
  return base.trim().toLowerCase()
}

/**
 * ExtractorRegistry maps content types to extractors and routes incoming
 * content to the correct handler. Unknown content types return a skipped
 * result rather than throwing.
 */
export type ExtractorRegistry = {
  /** Registers an extractor for all content types it declares. */
  register(ext: Extractor): void
  /** Routes raw bytes to the appropriate extractor. */
  extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult>
  /** Routes a readable stream to the appropriate extractor. */
  extractStream(
    source: Readable,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult>
}

/**
 * Creates a new ExtractorRegistry pre-loaded with the built-in
 * PlainTextExtractor for all text/* content types.
 */
export const createExtractorRegistry = (): ExtractorRegistry => {
  const extractors = new Map<string, Extractor>()

  const resolve = (contentType: string): Extractor | undefined => {
    const normalised = normaliseContentType(contentType)

    // Exact match.
    const exact = extractors.get(normalised)
    if (exact !== undefined) return exact

    // Fallback: match text/* family via text/plain.
    if (normalised.startsWith('text/')) {
      const fallback = extractors.get('text/plain')
      if (fallback !== undefined) return fallback
    }

    return undefined
  }

  const registry: ExtractorRegistry = {
    register(ext: Extractor): void {
      for (const ct of ext.contentTypes) {
        extractors.set(normaliseContentType(ct), ext)
      }
    },

    async extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult> {
      const ext = resolve(opts.contentType)
      if (ext === undefined) {
        return {
          text: '',
          contentType: opts.contentType,
          encoding: '',
          metadata: {},
          pages: 0,
          language: '',
          confidence: 0,
          skipped: true,
          reason: `unsupported content type: ${opts.contentType}`,
        }
      }
      return ext.extract(raw, opts, signal)
    },

    async extractStream(
      source: Readable,
      opts: ExtractOptions,
      signal?: AbortSignal,
    ): Promise<ExtractResult> {
      const ext = resolve(opts.contentType)
      if (ext === undefined) {
        return {
          text: '',
          contentType: opts.contentType,
          encoding: '',
          metadata: {},
          pages: 0,
          language: '',
          confidence: 0,
          skipped: true,
          reason: `unsupported content type: ${opts.contentType}`,
        }
      }
      return ext.extractStream(source, opts, signal)
    },
  }

  // Pre-register the built-in plain text extractor.
  registry.register(createPlainTextExtractor())

  return registry
}

/**
 * Detect the character encoding of raw bytes. Returns the encoding name
 * (e.g. "UTF-8", "ISO-8859-1", "Shift_JIS"). Uses heuristic analysis:
 * BOM markers first, then UTF-8 validation, then byte frequency patterns.
 */
export const detectEncoding = (raw: Buffer): string => {
  if (raw.length === 0) return 'UTF-8'

  // Check for BOM markers.
  if (raw.length >= 3 && raw[0] === 0xef && raw[1] === 0xbb && raw[2] === 0xbf) {
    return 'UTF-8'
  }
  if (raw.length >= 2 && raw[0] === 0xfe && raw[1] === 0xff) {
    return 'UTF-16BE'
  }
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    return 'UTF-16LE'
  }

  // Valid UTF-8 check via TextDecoder.
  try {
    const decoder = new TextDecoder('utf-8', { fatal: true })
    decoder.decode(raw)
    return 'UTF-8'
  } catch {
    // Not valid UTF-8; fall through to heuristics.
  }

  return detectByByteFrequency(raw)
}

/**
 * Use byte frequency analysis to distinguish common single-byte encodings
 * when content is not valid UTF-8.
 */
const detectByByteFrequency = (raw: Buffer): string => {
  // First pass: identify Shift_JIS double-byte pairs and track which
  // positions are consumed.
  const paired = new Uint8Array(raw.length)
  let shiftJISPairs = 0

  for (let i = 0; i < raw.length; i++) {
    const b = raw[i]!
    if ((b >= 0x81 && b <= 0x9f) || (b >= 0xe0 && b <= 0xef)) {
      if (i + 1 < raw.length) {
        const trail = raw[i + 1]!
        if ((trail >= 0x40 && trail <= 0x7e) || (trail >= 0x80 && trail <= 0xfc)) {
          paired[i] = 1
          paired[i + 1] = 1
          shiftJISPairs++
          i++
        }
      }
    }
  }

  // Second pass: count unpaired high bytes and C1 controls.
  let unpairedHighBytes = 0
  let unpairedC1Controls = 0

  for (let i = 0; i < raw.length; i++) {
    const b = raw[i]!
    if (b >= 0x80 && !paired[i]) {
      unpairedHighBytes++
      if (b <= 0x9f) unpairedC1Controls++
    }
  }

  // All high bytes consumed by Shift_JIS pairs: Japanese text.
  if (shiftJISPairs >= 2 && unpairedHighBytes === 0) {
    return 'Shift_JIS'
  }

  // Unpaired C1 controls: Windows-1252 (smart quotes, em dashes).
  if (unpairedC1Controls > 0) {
    return 'Windows-1252'
  }

  return 'ISO-8859-1'
}

/**
 * Transcode raw bytes from the specified encoding to UTF-8. Returns the
 * buffer unchanged when the source encoding is already UTF-8. Uses the
 * built-in TextDecoder which supports all WHATWG encoding labels.
 *
 * @throws Error when the encoding is unsupported or content cannot be decoded
 */
export const transcodeToUTF8 = (raw: Buffer, fromEncoding: string): Buffer => {
  const normalised = fromEncoding.trim().toLowerCase()
  if (normalised === 'utf-8' || normalised === '') return raw

  // Map common encoding names to WHATWG labels understood by TextDecoder.
  const labelMap: Readonly<Record<string, string>> = {
    'iso-8859-1': 'iso-8859-1',
    'latin-1': 'iso-8859-1',
    'latin1': 'iso-8859-1',
    'windows-1252': 'windows-1252',
    'windows-1251': 'windows-1251',
    'shift_jis': 'shift_jis',
    'shift-jis': 'shift_jis',
    'euc-jp': 'euc-jp',
    'euc-kr': 'euc-kr',
    'gbk': 'gbk',
    'gb2312': 'gb2312',
    'gb18030': 'gb18030',
    'big5': 'big5',
    'iso-2022-jp': 'iso-2022-jp',
    'koi8-r': 'koi8-r',
    'koi8-u': 'koi8-u',
    'utf-16be': 'utf-16be',
    'utf-16le': 'utf-16le',
  }

  const label = labelMap[normalised] ?? normalised

  try {
    const decoder = new NodeTextDecoder(label, { fatal: true })
    const text = decoder.decode(raw)
    return Buffer.from(text, 'utf-8')
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    throw new Error(`ingest: transcoding from ${fromEncoding} to UTF-8: ${message}`)
  }
}
