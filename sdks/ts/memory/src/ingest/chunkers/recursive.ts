// SPDX-License-Identifier: Apache-2.0

/**
 * Recursive chunker: the default strategy that splits content using a
 * separator hierarchy. Splits by paragraph breaks first, then single
 * newlines, then sentence endings, then words, then characters. Applies
 * overlap by copying trailing text from the previous chunk to the next.
 */

import type { ChunkConfig } from '../chunk-config.js'
import type { Chunk, Chunker } from '../chunker-registry.js'

/** Separator hierarchy from coarsest to finest split points. */
const SEPARATORS: readonly string[] = ['\n\n', '\n', '. ', ' ', '']

/**
 * Approximate token count using chars/4 heuristic. Monotonic in text
 * length. Matches the Go SDK estimateTokens for parity.
 */
export const estimateTokens = (text: string): number => {
  if (text.length === 0) return 0
  return Math.ceil(text.length / 4)
}

/**
 * Recursive chunker function. Splits content using the separator
 * hierarchy until each piece fits within maxTokens, then applies overlap.
 */
export const recursiveChunker: Chunker = async (
  content: string,
  cfg: ChunkConfig,
  signal?: AbortSignal,
): Promise<readonly Chunk[]> => {
  signal?.throwIfAborted()
  if (content.trim() === '') return []

  const pieces = recursiveSplit(content, cfg.maxTokens, 0)
  return applyOverlapAndBuild(pieces, cfg)
}

/** Recursively splits text using the separator at sepIdx until pieces fit. */
const recursiveSplit = (text: string, maxTokens: number, sepIdx: number): readonly string[] => {
  if (estimateTokens(text) <= maxTokens) {
    return [text]
  }
  if (sepIdx >= SEPARATORS.length) {
    return hardSplit(text, maxTokens)
  }
  const sep = SEPARATORS[sepIdx]
  if (sep === undefined || sep === '') {
    return hardSplit(text, maxTokens)
  }
  const parts = text.split(sep)
  const merged: string[] = []
  let current = ''

  for (const part of parts) {
    const candidate = current === '' ? part : current + sep + part
    if (estimateTokens(candidate) > maxTokens) {
      if (current !== '') {
        merged.push(current)
        current = ''
      }
      if (estimateTokens(part) > maxTokens) {
        const sub = recursiveSplit(part, maxTokens, sepIdx + 1)
        merged.push(...sub)
      } else {
        current = part
      }
    } else {
      current = candidate
    }
  }
  if (current !== '') {
    merged.push(current)
  }
  return merged
}

/** Character-level hard split as a last resort. */
const hardSplit = (text: string, maxTokens: number): readonly string[] => {
  const step = maxTokens * 4
  const out: string[] = []
  for (let i = 0; i < text.length; i += step) {
    out.push(text.slice(i, i + step))
  }
  return out
}

/** Applies overlap between consecutive chunks and merges undersized pieces. */
const applyOverlapAndBuild = (pieces: readonly string[], cfg: ChunkConfig): readonly Chunk[] => {
  if (pieces.length === 0) return []
  const chunks: Chunk[] = []
  let prevTail = ''

  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i]
    if (piece === undefined) continue
    const trimmed = piece.trim()
    if (trimmed === '') continue

    let chunkContent = trimmed
    if (i > 0 && cfg.overlapTokens > 0 && prevTail !== '') {
      chunkContent = prevTail + '\n' + trimmed
    }

    // Merge undersized chunks into the previous one.
    if (estimateTokens(trimmed) < cfg.minTokens && chunks.length > 0) {
      const prev = chunks[chunks.length - 1]
      if (prev !== undefined) {
        chunks[chunks.length - 1] = {
          ...prev,
          content: prev.content + '\n' + trimmed,
        }
        prevTail = extractTail(prev.content + '\n' + trimmed, cfg.overlapTokens)
        continue
      }
    }

    chunks.push({
      id: '',
      content: chunkContent,
      metadata: { chunker: 'recursive' },
    })
    prevTail = extractTail(trimmed, cfg.overlapTokens)
  }
  return chunks
}

/** Extracts approximately overlapTokens worth of characters from the end. */
const extractTail = (text: string, overlapTokens: number): string => {
  const chars = overlapTokens * 4
  if (chars >= text.length) return text
  return text.slice(text.length - chars)
}
