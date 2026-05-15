// SPDX-License-Identifier: Apache-2.0

/**
 * Page-level chunker: splits content at page boundaries indicated by
 * the form-feed character (\f). PDF text extractors typically insert
 * \f between pages. Pages that exceed maxTokens are split further
 * using the recursive separator hierarchy.
 */

import type { ChunkConfig } from '../chunk-config.js'
import type { Chunk, Chunker } from '../chunker-registry.js'
import { estimateTokens } from './recursive.js'

const SEPARATORS: readonly string[] = ['\n\n', '\n', '. ', ' ', '']

export const pageLevelChunker: Chunker = async (
  content: string,
  cfg: ChunkConfig,
  signal?: AbortSignal,
): Promise<readonly Chunk[]> => {
  signal?.throwIfAborted()
  if (content.trim() === '') return []

  const pages = content.split('\f')
  const chunks: Chunk[] = []

  for (let i = 0; i < pages.length; i++) {
    signal?.throwIfAborted()
    const page = pages[i]
    if (page === undefined) continue
    const trimmed = page.trim()
    if (trimmed === '') continue

    const pageNum = String(i + 1)

    if (estimateTokens(trimmed) <= cfg.maxTokens) {
      chunks.push({
        id: '',
        content: trimmed,
        metadata: { chunker: 'page_level', page: pageNum },
      })
      continue
    }

    const subPieces = recursiveSplitLocal(trimmed, cfg.maxTokens, 0)
    for (const piece of subPieces) {
      const t = piece.trim()
      if (t === '') continue
      chunks.push({
        id: '',
        content: t,
        metadata: { chunker: 'page_level', page: pageNum },
      })
    }
  }
  return chunks
}

const recursiveSplitLocal = (text: string, maxTokens: number, sepIdx: number): readonly string[] => {
  if (estimateTokens(text) <= maxTokens) return [text]
  if (sepIdx >= SEPARATORS.length) return hardSplit(text, maxTokens)
  const sep = SEPARATORS[sepIdx]
  if (sep === undefined || sep === '') return hardSplit(text, maxTokens)
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
        merged.push(...recursiveSplitLocal(part, maxTokens, sepIdx + 1))
      } else {
        current = part
      }
    } else {
      current = candidate
    }
  }
  if (current !== '') merged.push(current)
  return merged
}

const hardSplit = (text: string, maxTokens: number): readonly string[] => {
  const step = maxTokens * 4
  const out: string[] = []
  for (let i = 0; i < text.length; i += step) {
    out.push(text.slice(i, i + step))
  }
  return out
}
