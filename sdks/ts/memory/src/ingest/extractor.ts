// SPDX-License-Identifier: Apache-2.0

/**
 * Pluggable content extractor type and implementations for structured
 * data formats (CSV, JSON, JSONL, XML) and plain text. Each extractor
 * wraps the corresponding extraction function with name, capability,
 * extract, and available methods for registry-based routing.
 *
 * Also provides an ExtractorRegistry that routes extraction requests
 * to the correct extractor based on MIME type, plus encoding detection
 * and transcoding helpers.
 */

import type { Readable } from 'node:stream'

import {
  type ExtractResult,
  type CsvExtractorConfig,
  type JsonExtractorConfig,
  extractCSV,
  extractJSON,
  extractJSONL,
} from './structured.js'
export type { ExtractResult } from './structured.js'
import { type XmlExtractorConfig, extractXML } from './xml.js'

// ---------------------------------------------------------------------------
// Security constants
// ---------------------------------------------------------------------------

/** Maximum ratio of decompressed-to-compressed size before aborting. */
export const MAX_DECOMPRESSION_RATIO = 100

/** Maximum number of files extracted from an archive before aborting. */
export const MAX_EXTRACTED_FILES = 1000

// ---------------------------------------------------------------------------
// Argument sanitisation
// ---------------------------------------------------------------------------

const sanitizeArgsAllowlist: ReadonlySet<string> = new Set([
  '-o', '--output', '-f', '--format', '-q', '--quiet',
  '-v', '--verbose', '--stdin', '--stdout', '--no-color',
])

/**
 * Validates a list of command-line arguments against a hardcoded
 * allowlist. Positional arguments (not starting with `-`) are passed
 * through. Flags starting with `-` must appear in the allowlist or
 * an error is thrown.
 */
export const sanitizeArgs = (args: readonly string[]): string[] => {
  const out: string[] = []
  for (const arg of args) {
    if (arg.startsWith('-') && !sanitizeArgsAllowlist.has(arg)) {
      throw new Error(`disallowed argument: ${arg}`)
    }
    out.push(arg)
  }
  return out
}

// ---------------------------------------------------------------------------
// Encoding detection and transcoding
// ---------------------------------------------------------------------------

/**
 * Detect the character encoding of a raw buffer by inspecting BOMs
 * and byte patterns. Returns a canonical encoding name string.
 */
export const detectEncoding = (raw: Buffer): string => {
  if (raw.length === 0) return 'UTF-8'

  // UTF-8 BOM
  if (raw.length >= 3 && raw[0] === 0xef && raw[1] === 0xbb && raw[2] === 0xbf) {
    return 'UTF-8'
  }

  // UTF-16BE BOM
  if (raw.length >= 2 && raw[0] === 0xfe && raw[1] === 0xff) {
    return 'UTF-16BE'
  }

  // UTF-16LE BOM
  if (raw.length >= 2 && raw[0] === 0xff && raw[1] === 0xfe) {
    return 'UTF-16LE'
  }

  // Heuristic: check for Shift_JIS double-byte sequences.
  // Shift_JIS lead bytes: 0x81-0x9F, 0xE0-0xEF followed by 0x40-0xFC.
  let shiftJISPairs = 0
  let c1Controls = 0
  let highLatin = 0
  for (let i = 0; i < raw.length; i++) {
    const b = raw[i] ?? 0
    if (((b >= 0x81 && b <= 0x9f) || (b >= 0xe0 && b <= 0xef)) && i + 1 < raw.length) {
      const next = raw[i + 1] ?? 0
      if (next >= 0x40 && next <= 0xfc && next !== 0x7f) {
        shiftJISPairs++
        i++ // skip trail byte
        continue
      }
    }
    // C1 control range (0x80-0x9F) used in Windows-1252 but not Latin-1
    if (b >= 0x80 && b <= 0x9f) c1Controls++
    // High Latin-1 range (0xA0-0xFF)
    if (b >= 0xa0 && b <= 0xff) highLatin++
  }

  // If we found multiple Shift_JIS pairs, that's the best guess.
  if (shiftJISPairs >= 2) return 'Shift_JIS'

  // Try UTF-8 validity via round-trip.
  const utf8Text = raw.toString('utf8')
  const roundTrip = Buffer.from(utf8Text, 'utf8')
  if (roundTrip.length === raw.length && raw.equals(roundTrip)) {
    return 'UTF-8'
  }

  // Windows-1252 if C1 controls are present (smart quotes etc.)
  if (c1Controls > 0) return 'Windows-1252'

  // Fallback: ISO-8859-1
  if (highLatin > 0) return 'ISO-8859-1'

  return 'UTF-8'
}

/**
 * Transcode a buffer from the specified encoding to UTF-8. If the
 * encoding is already UTF-8 or empty, the buffer is returned as-is.
 * Throws for unsupported encodings.
 */
export const transcodeToUTF8 = (raw: Buffer, encoding: string): Buffer => {
  if (encoding === '' || encoding === 'UTF-8') return raw

  const decoderLabel = encodingToDecoderLabel(encoding)
  // TextDecoder accepts any WHATWG encoding label at runtime;
  // the TS lib types restrict the parameter to the Encoding union,
  // so we construct via the untyped global to avoid narrowing.
  const ctor = TextDecoder as { new (label: string): TextDecoder }
  const decoder = new ctor(decoderLabel)
  const text = decoder.decode(raw)
  return Buffer.from(text, 'utf8')
}

const encodingToDecoderLabel = (encoding: string): string => {
  const map: Record<string, string> = {
    'ISO-8859-1': 'iso-8859-1',
    'Windows-1252': 'windows-1252',
    'UTF-16LE': 'utf-16le',
    'UTF-16BE': 'utf-16be',
    'Shift_JIS': 'shift_jis',
  }
  const label = map[encoding]
  if (label === undefined) {
    throw new Error(`unsupported encoding: ${encoding}`)
  }
  return label
}

/** Identifies a file format by magic bytes at a given offset. */
export type MagicSignature = {
  readonly offset: number
  readonly bytes: Uint8Array
}

/**
 * Describes what content types an extractor handles. Used by the
 * registry for routing and by callers to inspect extractor capabilities.
 */
export type ExtractorCapability = {
  readonly extensions: readonly string[]
  readonly mimeTypes: readonly string[]
  readonly magicBytes: readonly MagicSignature[]
  readonly requiresBinary: boolean
}

/** Options provided to an extractor about the content being processed. */
export type ExtractOptions = {
  readonly contentType?: string
  readonly fileName?: string
  readonly encoding?: string
  /** ISO 639-1 language hint for OCR extractors (e.g. "en", "de"). */
  readonly language?: string
  readonly maxBytes?: number
}

/**
 * Contract for content extraction. Implementations declare the MIME
 * types they handle and provide both buffered and streaming extraction
 * methods. Mirrors the canonical Extractor interface from P1-5.
 */
export type Extractor = {
  /** Converts buffered raw bytes into text content. */
  extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult>
  /** Processes content from a readable stream. */
  extractStream(
    source: Readable,
    opts: ExtractOptions,
    signal?: AbortSignal,
  ): Promise<ExtractResult>
  /** The MIME types this extractor handles. */
  readonly contentTypes: readonly string[]
  /** Human-readable identifier for this extractor. */
  readonly name: string
  /** Reports whether this extractor's external dependencies are present. */
  available(): Promise<boolean>
  /** Describes what content types, file extensions, and magic byte signatures this extractor handles. */
  capability(): ExtractorCapability
}

/**
 * Collects all chunks from a Readable into a Buffer, respecting an
 * optional byte limit.
 */
export const bufferStream = async (source: Readable, maxBytes?: number): Promise<Buffer> => {
  const chunks: Buffer[] = []
  let total = 0
  for await (const chunk of source) {
    const buf = Buffer.isBuffer(chunk)
      ? chunk
      : typeof chunk === 'string'
        ? Buffer.from(chunk)
        : chunk instanceof Uint8Array
          ? Buffer.from(chunk)
          : Buffer.from(String(chunk))
    if (maxBytes !== undefined && total + buf.length > maxBytes) {
      const remaining = maxBytes - total
      if (remaining > 0) chunks.push(buf.subarray(0, remaining))
      total = maxBytes
      break
    }
    chunks.push(buf)
    total += buf.length
  }
  return Buffer.concat(chunks)
}

/** Create a CSV extractor implementing the canonical Extractor interface. */
export const createCSVExtractor = (config: CsvExtractorConfig = {}): Extractor => ({
  name: 'csv',
  contentTypes: ['text/csv', 'text/tab-separated-values'],

  async extract(raw: Buffer): Promise<ExtractResult> {
    return extractCSV(raw, config)
  },

  async extractStream(source: Readable): Promise<ExtractResult> {
    const raw = await bufferStream(source)
    return extractCSV(raw, config)
  },

  async available(): Promise<boolean> {
    return true
  },

  capability(): ExtractorCapability {
    return {
      extensions: ['.csv', '.tsv'],
      mimeTypes: ['text/csv', 'text/tab-separated-values'],
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

/** Create a JSON extractor implementing the canonical Extractor interface. */
export const createJSONExtractor = (config: JsonExtractorConfig = {}): Extractor => ({
  name: 'json',
  contentTypes: ['application/json'],

  async extract(raw: Buffer): Promise<ExtractResult> {
    return extractJSON(raw, config)
  },

  async extractStream(source: Readable): Promise<ExtractResult> {
    const raw = await bufferStream(source)
    return extractJSON(raw, config)
  },

  async available(): Promise<boolean> {
    return true
  },

  capability(): ExtractorCapability {
    return {
      extensions: ['.json'],
      mimeTypes: ['application/json'],
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

/** Create a JSONL extractor implementing the canonical Extractor interface. */
export const createJSONLExtractor = (config: JsonExtractorConfig = {}): Extractor => ({
  name: 'jsonl',
  contentTypes: ['application/jsonl', 'application/x-ndjson'],

  async extract(raw: Buffer): Promise<ExtractResult> {
    return extractJSONL(raw, config)
  },

  async extractStream(source: Readable): Promise<ExtractResult> {
    const raw = await bufferStream(source)
    return extractJSONL(raw, config)
  },

  async available(): Promise<boolean> {
    return true
  },

  capability(): ExtractorCapability {
    return {
      extensions: ['.jsonl', '.ndjson'],
      mimeTypes: ['application/jsonl', 'application/x-ndjson'],
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

/** Create an XML extractor implementing the canonical Extractor interface. */
export const createXMLExtractor = (config: XmlExtractorConfig = {}): Extractor => ({
  name: 'xml',
  contentTypes: ['application/xml', 'text/xml'],

  async extract(raw: Buffer): Promise<ExtractResult> {
    return extractXML(raw, config)
  },

  async extractStream(source: Readable): Promise<ExtractResult> {
    const raw = await bufferStream(source)
    return extractXML(raw, config)
  },

  async available(): Promise<boolean> {
    return true
  },

  capability(): ExtractorCapability {
    return {
      extensions: ['.xml'],
      mimeTypes: ['application/xml', 'text/xml'],
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

// ---------------------------------------------------------------------------
// Plain text extractor
// ---------------------------------------------------------------------------

/** Create a plain text extractor that passes content through as-is. */
export const createPlainTextExtractor = (): Extractor => ({
  name: 'plain-text',
  contentTypes: [
    'text/plain',
    'text/markdown',
    'text/html',
    'text/css',
    'text/javascript',
    'application/json',
    'application/javascript',
  ],

  async extract(raw: Buffer): Promise<ExtractResult> {
    return {
      text: raw.toString('utf8'),
      contentType: 'text/plain',
      encoding: 'UTF-8',
      metadata: {},
      pages: 0,
      language: '',
      confidence: 0,
      skipped: false,
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
      extensions: ['.txt', '.md', '.html', '.css', '.js'],
      mimeTypes: ['text/plain', 'text/markdown', 'text/html'],
      magicBytes: [],
      requiresBinary: false,
    }
  },
})

// ---------------------------------------------------------------------------
// Extractor registry
// ---------------------------------------------------------------------------

/**
 * Registry that routes extraction requests to the correct extractor
 * based on MIME type. Ships with a built-in plain text extractor and
 * allows custom extractors to be registered (overriding defaults).
 */
export type ExtractorRegistry = {
  /** Register a custom extractor. Later registrations override earlier ones. */
  register(extractor: Extractor): void
  /** Extract from a buffer, routing by content type. */
  extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult>
  /** Extract from a stream, routing by content type. */
  extractStream(source: Readable, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult>
}

const skippedResult = (reason: string): ExtractResult => ({
  text: '',
  contentType: '',
  encoding: '',
  metadata: {},
  pages: 0,
  language: '',
  confidence: 0,
  skipped: true,
  reason,
})

const stripCharset = (contentType: string): string => {
  const idx = contentType.indexOf(';')
  return idx >= 0 ? contentType.slice(0, idx).trim() : contentType.trim()
}

/** Create a new extractor registry with the built-in plain text extractor. */
export const createExtractorRegistry = (): ExtractorRegistry => {
  const extractors = new Map<string, Extractor>()

  // Seed with built-in plain text extractor.
  const plainText = createPlainTextExtractor()
  for (const ct of plainText.contentTypes) {
    extractors.set(ct, plainText)
  }

  const findExtractor = (contentType: string | undefined): Extractor | undefined => {
    if (contentType === undefined) return undefined
    const base = stripCharset(contentType)
    const direct = extractors.get(base)
    if (direct !== undefined) return direct
    // Fall back to text/plain for any text/* subtype.
    if (base.startsWith('text/')) return extractors.get('text/plain')
    return undefined
  }

  return {
    register(extractor: Extractor): void {
      for (const ct of extractor.contentTypes) {
        extractors.set(ct, extractor)
      }
    },

    async extract(raw: Buffer, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult> {
      const ext = findExtractor(opts.contentType)
      if (ext === undefined) {
        return skippedResult(`unsupported content type: ${opts.contentType ?? 'unknown'}`)
      }
      return ext.extract(raw, opts, signal)
    },

    async extractStream(source: Readable, opts: ExtractOptions, signal?: AbortSignal): Promise<ExtractResult> {
      const ext = findExtractor(opts.contentType)
      if (ext === undefined) {
        return skippedResult(`unsupported content type: ${opts.contentType ?? 'unknown'}`)
      }
      return ext.extractStream(source, opts, signal)
    },
  }
}
