// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for the pluggable extractor subsystem. Mirrors the
 * canonical types from go/ingest/extractor.go and the P1-5 TS
 * extractor.ts definitions.
 */

import type { Readable } from 'node:stream'

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
  readonly contentType?: string
  readonly fileName?: string
  readonly encoding?: string
  readonly language?: string
  readonly maxBytes?: number
  readonly signal?: AbortSignal
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
 * Contract for content extraction. Implementations declare the MIME
 * types they handle and provide both buffered and streaming extraction
 * methods. Mirrors the canonical Extractor interface from P1-5.
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
   * (e.g. FFmpeg, faster-whisper). Extractors with no external
   * dependencies always resolve to true.
   */
  available(): Promise<boolean>
  /**
   * Describes what content types, file extensions, and magic byte
   * signatures this extractor handles.
   */
  capability(): ExtractorCapability
}
