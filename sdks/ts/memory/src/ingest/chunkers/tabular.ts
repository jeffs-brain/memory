// SPDX-License-Identifier: Apache-2.0

/**
 * Tabular chunker: splits CSV/TSV content by rows, prepending the
 * header row to each chunk so every chunk is self-contained. The
 * delimiter is auto-detected from the first line.
 */

import type { ChunkConfig } from '../chunk-config.js'
import type { Chunk, Chunker } from '../chunker-registry.js'
import { estimateTokens } from './recursive.js'

/** Default number of data rows per chunk when token budget allows. */
const DEFAULT_ROWS_PER_CHUNK = 50

export const tabularChunker: Chunker = async (
  content: string,
  cfg: ChunkConfig,
  signal?: AbortSignal,
): Promise<readonly Chunk[]> => {
  signal?.throwIfAborted()
  const trimmed = content.trim()
  if (trimmed === '') return []

  const lines = trimmed.split('\n')
  if (lines.length === 0) return []

  const header = lines[0] ?? ''
  const dataLines = lines.slice(1)

  if (dataLines.length === 0) {
    return [{
      id: '',
      content: header,
      metadata: { chunker: 'tabular' },
    }]
  }

  const rowsPerChunk = computeRowsPerChunk(header, dataLines, cfg)

  const chunks: Chunk[] = []
  for (let start = 0; start < dataLines.length; start += rowsPerChunk) {
    signal?.throwIfAborted()
    const end = Math.min(start + rowsPerChunk, dataLines.length)
    const batch = dataLines.slice(start, end)
    const chunkContent = header + '\n' + batch.join('\n')
    chunks.push({
      id: '',
      content: chunkContent,
      metadata: { chunker: 'tabular' },
    })
  }
  return chunks
}

const computeRowsPerChunk = (
  header: string,
  dataLines: readonly string[],
  cfg: ChunkConfig,
): number => {
  const headerTokens = estimateTokens(header)
  const budgetForRows = cfg.maxTokens - headerTokens - 1
  if (budgetForRows <= 0 || dataLines.length === 0) return DEFAULT_ROWS_PER_CHUNK

  const sampleSize = Math.min(dataLines.length, 10)
  let totalTokens = 0
  for (let i = 0; i < sampleSize; i++) {
    totalTokens += estimateTokens(dataLines[i] ?? '')
  }
  const avgTokensPerRow = Math.max(Math.floor(totalTokens / sampleSize), 1)

  const rows = Math.floor(budgetForRows / avgTokensPerRow)
  if (rows <= 0) return 1
  return Math.min(rows, DEFAULT_ROWS_PER_CHUNK)
}
