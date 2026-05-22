// SPDX-License-Identifier: Apache-2.0

import type { Embedder, SqliteSearchIndex } from '@jeffs-brain/memory'
import { type RuntimeLogger, noopRuntimeLogger } from './runtime-logger.js'

const DEFAULT_BACKFILL_BATCH_SIZE = 64
const DEFAULT_BACKFILL_TEXT_CAP = 8192

type IndexedChunkRow = {
  readonly id: string
  readonly path: string
  readonly ordinal: number | bigint
  readonly title: string
  readonly summary: string
  readonly tags: string
  readonly content: string
  readonly metadata_json: string | null
}

export type VectorBackfillOptions = {
  readonly brainId: string
  readonly searchIndex: SqliteSearchIndex
  readonly embedder: Embedder
  readonly model?: string
  readonly batchSize?: number
  readonly textCap?: number
  readonly signal?: AbortSignal
  readonly logger?: RuntimeLogger
}

export type VectorBackfillResult = {
  readonly scanned: number
  readonly embedded: number
  readonly skipped: number
  readonly model: string
  readonly durationMs: number
}

const parseMetadata = (raw: string | null): Readonly<Record<string, unknown>> | undefined => {
  if (raw === null || raw.trim() === '') return undefined
  try {
    const parsed = JSON.parse(raw) as unknown
    if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Readonly<Record<string, unknown>>
    }
  } catch {
    return undefined
  }
  return undefined
}

const rowOrdinal = (value: number | bigint): number =>
  typeof value === 'bigint' ? Number(value) : value

const rowText = (row: IndexedChunkRow, cap: number): string => {
  const text = [row.title, row.summary, row.content]
    .map((part) => part.trim())
    .filter((part) => part !== '')
    .join('\n\n')
  return text.length > cap ? text.slice(0, cap) : text
}

const rowsMissingVector = (
  index: SqliteSearchIndex,
  model: string,
): readonly IndexedChunkRow[] => {
  const existing = new Set(index.chunkIdsWithVectorForModel(model))
  const rows = index.db
    .prepare(
      `SELECT id, path, ordinal, title, summary, tags, content, metadata_json
         FROM knowledge_chunks
        ORDER BY path, ordinal, id`,
    )
    .all() as IndexedChunkRow[]
  return rows.filter((row) => !existing.has(row.id))
}

export const resolveBackfillModel = (
  embedder: Embedder | undefined,
  configuredModel?: string,
): string => {
  if (embedder === undefined) return ''
  const explicit = configuredModel?.trim()
  if (explicit !== undefined && explicit !== '') return explicit
  const reported = embedder.model().trim()
  return reported !== '' ? reported : 'unknown-embedder'
}

export const backfillSearchIndexVectors = async (
  options: VectorBackfillOptions,
): Promise<VectorBackfillResult> => {
  const logger = options.logger ?? noopRuntimeLogger
  const model = resolveBackfillModel(options.embedder, options.model)
  const started = Date.now()
  if (model === '') {
    return {
      scanned: 0,
      embedded: 0,
      skipped: 0,
      model,
      durationMs: Date.now() - started,
    }
  }

  const batchSize = options.batchSize ?? DEFAULT_BACKFILL_BATCH_SIZE
  const textCap = options.textCap ?? DEFAULT_BACKFILL_TEXT_CAP
  const rows = rowsMissingVector(options.searchIndex, model)
  if (rows.length === 0) {
    logger.info('memory-pi vectors: up to date', {
      brainId: options.brainId,
      model,
    })
    return {
      scanned: 0,
      embedded: 0,
      skipped: 0,
      model,
      durationMs: Date.now() - started,
    }
  }

  logger.info('memory-pi vectors: backfill start', {
    brainId: options.brainId,
    model,
    count: rows.length,
  })

  let embedded = 0
  let skipped = 0
  for (let i = 0; i < rows.length; i += batchSize) {
    if (options.signal?.aborted === true) {
      logger.info('memory-pi vectors: backfill cancelled', {
        brainId: options.brainId,
        embedded,
        skipped,
      })
      break
    }

    const batch = rows.slice(i, i + batchSize)
    const texts = batch.map((row) => rowText(row, textCap))
    let vectors: number[][]
    try {
      vectors = await options.embedder.embed(texts, options.signal)
    } catch (err) {
      skipped += batch.length
      logger.warn('memory-pi vectors: embed batch failed', {
        brainId: options.brainId,
        model,
        err: err instanceof Error ? err.message : String(err),
      })
      continue
    }

    if (vectors.length !== batch.length) {
      skipped += batch.length
      logger.warn('memory-pi vectors: embedder returned mismatched count', {
        brainId: options.brainId,
        model,
        got: vectors.length,
        want: batch.length,
      })
      continue
    }

    for (let offset = 0; offset < batch.length; offset += 1) {
      const row = batch[offset]
      const vector = vectors[offset]
      if (row === undefined || vector === undefined || vector.length === 0) {
        skipped++
        continue
      }
      if (vector.length !== options.searchIndex.vectorDim) {
        skipped += batch.length - offset
        logger.warn('memory-pi vectors: embedding dim mismatch', {
          brainId: options.brainId,
          model,
          got: vector.length,
          want: options.searchIndex.vectorDim,
        })
        break
      }
      const metadata = parseMetadata(row.metadata_json)
      options.searchIndex.upsertChunk({
        id: row.id,
        path: row.path,
        ordinal: rowOrdinal(row.ordinal),
        title: row.title,
        summary: row.summary,
        tags: row.tags,
        content: row.content,
        ...(metadata !== undefined ? { metadata } : {}),
        embedding: vector,
        embeddingModel: model,
      })
      embedded++
    }
  }

  const result: VectorBackfillResult = {
    scanned: rows.length,
    embedded,
    skipped,
    model,
    durationMs: Date.now() - started,
  }
  logger.info('memory-pi vectors: backfill complete', {
    ...result,
    brainId: options.brainId,
  })
  return result
}
