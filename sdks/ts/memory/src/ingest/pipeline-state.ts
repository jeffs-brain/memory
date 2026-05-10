// SPDX-License-Identifier: Apache-2.0

/**
 * Pipeline state tracking for crash recovery. Persists the last completed
 * stage of each document's ingest to the Store, enabling the pipeline to
 * resume from the point of failure rather than silently skipping documents
 * whose raw content was written but never indexed.
 *
 * State files live at `raw/.pipeline-state/{hash}.json` alongside the
 * raw document content. On successful full completion the state file is
 * left in place at `stage: 'indexed'` so subsequent ingests of the same
 * content can be deduplicated (`reused: true`). It is only deleted when
 * chunking yields zero chunks (no usable content).
 */

import type { Logger } from '../llm/types.js'
import type { Batch, Path, Store } from '../store/index.js'
import { isNotFound, joinPath, toPath } from '../store/index.js'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PIPELINE_STATE_PREFIX = 'raw/.pipeline-state'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Ordered stages of the ingest pipeline. Each stage implies all prior
 * stages completed successfully.
 */
export type PipelineStage = 'stored' | 'chunked' | 'embedded' | 'indexed'

export type PipelineState = {
  readonly documentId: string
  readonly hash: string
  readonly stage: PipelineStage
  readonly updatedAt: string
  readonly chunkCount?: number
}

// ---------------------------------------------------------------------------
// Stage ordering
// ---------------------------------------------------------------------------

const STAGE_ORDER: Readonly<Record<PipelineStage, number>> = {
  stored: 0,
  chunked: 1,
  embedded: 2,
  indexed: 3,
}

/** Returns true when `current` is at or beyond `target` in the pipeline. */
export const isStageComplete = (current: PipelineStage, target: PipelineStage): boolean =>
  STAGE_ORDER[current] >= STAGE_ORDER[target]

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

export const pipelineStatePath = (documentHash: string): Path =>
  joinPath(PIPELINE_STATE_PREFIX, `${documentHash}.json`)

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

export const readPipelineState = async (
  store: Store,
  documentHash: string,
  logger: Logger,
): Promise<PipelineState | undefined> => {
  const path = pipelineStatePath(documentHash)
  try {
    const buf = await store.read(path)
    const parsed: unknown = JSON.parse(buf.toString('utf8'))
    if (!isValidState(parsed)) {
      logger.warn('pipeline state file contains invalid data, treating as absent', {
        path: path as string,
      })
      return undefined
    }
    return parsed
  } catch (err: unknown) {
    if (isNotFound(err)) return undefined
    logger.warn('failed to read pipeline state, treating as absent', {
      path: path as string,
      error: String(err),
    })
    return undefined
  }
}

export const writePipelineState = async (
  writer: Store | Batch,
  state: PipelineState,
): Promise<void> => {
  const path = pipelineStatePath(state.hash)
  const content = Buffer.from(JSON.stringify(state), 'utf8')
  await writer.write(toPath(path as string), content)
}

export const deletePipelineState = async (
  store: Store,
  documentHash: string,
  logger: Logger,
): Promise<void> => {
  const path = pipelineStatePath(documentHash)
  try {
    await store.delete(toPath(path as string))
  } catch (err: unknown) {
    if (isNotFound(err)) return
    logger.warn('failed to delete pipeline state file', {
      path: path as string,
      error: String(err),
    })
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

const VALID_STAGES: ReadonlySet<string> = new Set(['stored', 'chunked', 'embedded', 'indexed'])

const isValidState = (value: unknown): value is PipelineState => {
  if (value === null || typeof value !== 'object') return false
  const obj = value as Record<string, unknown>
  if (typeof obj['documentId'] !== 'string') return false
  if (typeof obj['hash'] !== 'string') return false
  if (typeof obj['stage'] !== 'string') return false
  if (!VALID_STAGES.has(obj['stage'])) return false
  if (typeof obj['updatedAt'] !== 'string') return false
  return true
}
