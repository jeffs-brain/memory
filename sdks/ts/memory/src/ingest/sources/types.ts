// SPDX-License-Identifier: Apache-2.0

/**
 * Shared types for ingest source adapters. Each adapter accepts a raw
 * payload and produces a `LoadedSource` that the pipeline can hand off to
 * the chunker.
 */

export type SourceLoadOptions = {
  readonly title?: string
  readonly filename?: string
  readonly mime?: string
}

export type LoadedSource = {
  readonly content: Buffer
  readonly mime: string
  readonly title?: string
  readonly meta?: Readonly<Record<string, unknown>>
}

export type SourceFetchLike = (
  input: string,
  init?: {
    readonly method?: string
    readonly headers?: Readonly<Record<string, string>>
    readonly body?: unknown
    readonly signal?: AbortSignal
  },
) => Promise<{
  readonly ok: boolean
  readonly status: number
  readonly statusText: string
  readonly headers: { get(name: string): string | null }
  arrayBuffer(): Promise<ArrayBuffer>
  text(): Promise<string>
}>
