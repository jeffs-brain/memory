// SPDX-License-Identifier: Apache-2.0

import { type Path, type Store, joinPath, pathUnder } from '../store/index.js'
import { RAW_DOCUMENTS_PREFIX, hashContent } from './ingest.js'

export const RAW_DOCUMENTS_PROCESSED_PREFIX = joinPath(RAW_DOCUMENTS_PREFIX, '_processed')

export type ProcessedSourceMarker = {
  readonly sourcePath: string
  readonly contentHash: string
  readonly processedAt: string
  readonly writtenPaths: readonly string[]
}

export const processedMarkerPath = (sourceId: string): Path =>
  joinPath(RAW_DOCUMENTS_PROCESSED_PREFIX, `${sourceId}.json`)

export const parseProcessedMarker = (content: string): ProcessedSourceMarker | undefined => {
  try {
    const parsed = JSON.parse(content) as Partial<ProcessedSourceMarker>
    if (
      typeof parsed.sourcePath !== 'string' ||
      typeof parsed.contentHash !== 'string' ||
      typeof parsed.processedAt !== 'string' ||
      !Array.isArray(parsed.writtenPaths)
    ) {
      return undefined
    }
    const writtenPaths = parsed.writtenPaths.filter(
      (value): value is string => typeof value === 'string' && value.trim() !== '',
    )
    return {
      sourcePath: parsed.sourcePath,
      contentHash: parsed.contentHash,
      processedAt: parsed.processedAt,
      writtenPaths,
    }
  } catch {
    return undefined
  }
}

export const serialiseProcessedMarker = (marker: ProcessedSourceMarker): string =>
  `${JSON.stringify(marker, null, 2)}\n`

export const readProcessedMarker = async (
  store: Store,
  sourceId: string,
): Promise<ProcessedSourceMarker | undefined> => {
  const path = processedMarkerPath(sourceId)
  const exists = await store.exists(path).catch(() => false)
  if (!exists) return undefined
  const raw = await store.read(path).catch(() => undefined)
  if (raw === undefined) return undefined
  return parseProcessedMarker(raw.toString('utf8'))
}

export const isProcessedMarkerPath = (path: string): boolean =>
  pathUnder(path as Path, RAW_DOCUMENTS_PROCESSED_PREFIX, true)

export const isProcessedSourceContent = (
  marker: ProcessedSourceMarker | undefined,
  sourcePath: string,
  content: Buffer,
): boolean =>
  marker !== undefined &&
  marker.sourcePath === sourcePath &&
  marker.contentHash === hashContent(content)
