// SPDX-License-Identifier: Apache-2.0

/**
 * Chunker registry with content-type routing. Provides a pluggable
 * architecture for content-type-specific chunking strategies. The
 * registry pre-registers built-in chunkers (markdown, recursive) and
 * allows custom chunkers to be registered at runtime.
 */

import type { ChunkConfig } from './chunk-config.js'
import { codeChunker } from './chunkers/code.js'
import { markdownChunker } from './chunkers/markdown.js'
import { pageLevelChunker } from './chunkers/page-level.js'
import { recursiveChunker } from './chunkers/recursive.js'
import { tabularChunker } from './chunkers/tabular.js'

/** A single chunk produced by a Chunker. */
export type Chunk = {
  readonly id: string
  readonly content: string
  readonly metadata: Readonly<Record<string, string>>
}

/**
 * Chunker function signature. Async to support LLM-powered chunkers
 * and cancellation via AbortSignal. Returns an immutable array of
 * chunks respecting the provided ChunkConfig bounds.
 */
export type Chunker = (
  content: string,
  cfg: ChunkConfig,
  signal?: AbortSignal,
) => Promise<readonly Chunk[]>

/** Descriptor pairing a chunker with its supported content types and name. */
export type ChunkerDescriptor = {
  readonly name: string
  readonly contentTypes: readonly string[]
  readonly chunker: Chunker
}

/** Registry that routes content to the appropriate chunker by MIME type. */
export type ChunkerRegistry = {
  /** Register a custom chunker for the given content types. */
  readonly register: (descriptor: ChunkerDescriptor) => void
  /** Chunk content using the registered chunker for contentType. */
  readonly chunk: (
    content: string,
    contentType: string,
    cfg: ChunkConfig,
    signal?: AbortSignal,
  ) => Promise<readonly Chunk[]>
}

/**
 * Creates a ChunkerRegistry pre-loaded with built-in chunkers:
 * - text/markdown, text/x-markdown -> markdown chunker
 * - text/plain, application/octet-stream -> recursive chunker (also fallback)
 */
export const createChunkerRegistry = (): ChunkerRegistry => {
  const routes = new Map<string, ChunkerDescriptor>()

  const mdDescriptor: ChunkerDescriptor = {
    name: 'markdown',
    contentTypes: ['text/markdown', 'text/x-markdown'],
    chunker: markdownChunker,
  }
  for (const ct of mdDescriptor.contentTypes) {
    routes.set(ct, mdDescriptor)
  }

  const recursiveDescriptor: ChunkerDescriptor = {
    name: 'recursive',
    contentTypes: ['text/plain', 'application/octet-stream'],
    chunker: recursiveChunker,
  }
  for (const ct of recursiveDescriptor.contentTypes) {
    routes.set(ct, recursiveDescriptor)
  }

  const codeDescriptor: ChunkerDescriptor = {
    name: 'code',
    contentTypes: [
      'text/x-go',
      'text/x-python',
      'text/x-typescript',
      'text/x-javascript',
      'text/x-java',
      'text/x-c',
      'text/x-c++',
      'text/x-rust',
      'application/x-typescript',
      'application/javascript',
    ],
    chunker: codeChunker,
  }
  for (const ct of codeDescriptor.contentTypes) {
    routes.set(ct, codeDescriptor)
  }

  const tabularDescriptor: ChunkerDescriptor = {
    name: 'tabular',
    contentTypes: ['text/csv', 'text/tab-separated-values', 'text/tsv'],
    chunker: tabularChunker,
  }
  for (const ct of tabularDescriptor.contentTypes) {
    routes.set(ct, tabularDescriptor)
  }

  const pageLevelDescriptor: ChunkerDescriptor = {
    name: 'page_level',
    contentTypes: ['application/pdf', 'text/x-pdf-text'],
    chunker: pageLevelChunker,
  }
  for (const ct of pageLevelDescriptor.contentTypes) {
    routes.set(ct, pageLevelDescriptor)
  }

  const register = (descriptor: ChunkerDescriptor): void => {
    for (const ct of descriptor.contentTypes) {
      routes.set(ct.toLowerCase().trim(), descriptor)
    }
  }

  const chunk = async (
    content: string,
    contentType: string,
    cfg: ChunkConfig,
    signal?: AbortSignal,
  ): Promise<readonly Chunk[]> => {
    signal?.throwIfAborted()
    const normalised = normaliseContentType(contentType)
    const descriptor = routes.get(normalised) ?? recursiveDescriptor
    const chunks = await descriptor.chunker(content, cfg, signal)
    return chunks.map((c, idx) => ({
      ...c,
      id: c.id || String(idx),
    }))
  }

  return { register, chunk }
}

/** Strips charset suffix and normalises content type for lookup. */
const normaliseContentType = (raw: string): string => {
  const lower = raw.toLowerCase().trim()
  const semicolonIdx = lower.indexOf(';')
  if (semicolonIdx >= 0) {
    return lower.slice(0, semicolonIdx).trim()
  }
  return lower
}
