// SPDX-License-Identifier: Apache-2.0

/**
 * Extractor registry with content-type routing and streaming support.
 * Routes incoming content to the appropriate extractor based on MIME type,
 * with fallback behaviour for unknown types. Mirrors the Go implementation
 * in go/ingest/extractor.go.
 */

import type { Readable } from 'node:stream'

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

/** Options provided to an extractor about the content being processed. */
export type ExtractOptions = {
  readonly contentType: string
  readonly fileName?: string
  readonly maxBytes?: number
}

/** The output of an extraction operation. */
export type ExtractResult = {
  readonly text: string
  readonly metadata: Readonly<Record<string, string>>
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
}

/**
 * Collects all chunks from a Readable into a Buffer, respecting an
 * optional byte limit.
 */
const bufferStream = async (source: Readable, maxBytes?: number): Promise<Buffer> => {
  const chunks: Buffer[] = []
  let totalBytes = 0

  for await (const chunk of source) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array)
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
    const text = raw.toString('utf8')
    return {
      text,
      metadata: {},
      skipped: false,
    }
  },

  async extractStream(source: Readable, opts: ExtractOptions): Promise<ExtractResult> {
    const raw = await bufferStream(source, opts.maxBytes)
    return this.extract(raw, opts)
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
          metadata: {},
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
          metadata: {},
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
