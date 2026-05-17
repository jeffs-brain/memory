// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for the pluggable extractor subsystem. Each extractor
 * converts a specific media type into searchable plain text.
 */

export type ExtractionResult = {
  readonly content: string
  readonly mime: string
  readonly metadata: Readonly<Record<string, unknown>>
  readonly pages?: number
  readonly language?: string
  readonly confidence?: number
}

export type MagicSignature = {
  readonly offset: number
  readonly bytes: Uint8Array
}

export type ExtractorCapability = {
  readonly extensions: ReadonlySet<string>
  readonly mimeTypes: ReadonlySet<string>
  readonly magicBytes?: readonly MagicSignature[]
  readonly requiresBinary: boolean
}

export type ExtractOptions = {
  readonly filename?: string
  readonly mime?: string
  readonly language?: string
  readonly signal?: AbortSignal
  readonly timeout?: number
}

export type Extractor = {
  readonly name: string
  readonly capability: ExtractorCapability
  extract(input: Buffer, opts: ExtractOptions): Promise<ExtractionResult>
  available(): Promise<boolean>
}
