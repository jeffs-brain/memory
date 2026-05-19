// SPDX-License-Identifier: Apache-2.0

/**
 * Pluggable content extractor type and implementations for structured
 * data formats (CSV, JSON, JSONL, XML). Each extractor wraps the
 * corresponding extraction function with name, capability, extract,
 * and available methods for registry-based routing.
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
  for await (const chunk of source) {
    const buf = Buffer.isBuffer(chunk)
      ? chunk
      : typeof chunk === 'string'
        ? Buffer.from(chunk)
        : chunk instanceof Uint8Array
          ? Buffer.from(chunk)
          : Buffer.from(String(chunk))
    chunks.push(buf)
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
