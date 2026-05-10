// SPDX-License-Identifier: Apache-2.0

/**
 * Pipeline state store interface and file-based implementation.
 *
 * Tracks the progress of each document through the ingest pipeline so that
 * crash recovery can resume from the last completed stage rather than
 * re-processing from scratch or (worse) permanently skipping the document.
 *
 * The FilePipelineStateStore persists state as JSON files inside the brain's
 * Store at `raw/.pipeline-state/{hash}.json`.
 */

import type { Store } from '../store/index.js'
import { isNotFound, toPath } from '../store/index.js'

/** Ordered stages a document passes through during ingestion. */
export type PipelineStage =
  | 'received'
  | 'stored'
  | 'chunked'
  | 'embedded'
  | 'indexed'
  | 'completed'
  | 'failed'

/** Terminal stages that indicate a document is no longer in-flight. */
const TERMINAL_STAGES: ReadonlySet<PipelineStage> = new Set(['completed', 'failed'])

/** A single document's pipeline state record. */
export type PipelineStateEntry = {
  readonly documentHash: string
  readonly brainId: string
  readonly stage: PipelineStage
  readonly retryCount: number
  readonly lastError?: string
  readonly createdAt: Date
  readonly updatedAt: Date
  readonly completedAt?: Date
}

/**
 * Persistence interface for pipeline state tracking. Implementations must
 * be safe for sequential use within a single pipeline run.
 */
export type PipelineStateStore = {
  /** Retrieve state for a document. Returns undefined when no record exists. */
  get(documentHash: string, signal?: AbortSignal): Promise<PipelineStateEntry | undefined>

  /** Create or overwrite state for a document. */
  set(entry: PipelineStateEntry, signal?: AbortSignal): Promise<void>

  /** Return all entries not in a terminal stage, optionally filtered by brainId. */
  listIncomplete(brainId: string, signal?: AbortSignal): Promise<readonly PipelineStateEntry[]>

  /** Remove the state record for a document. No-op if the record does not exist. */
  delete(documentHash: string, signal?: AbortSignal): Promise<void>
}

const STATE_PREFIX = 'raw/.pipeline-state'

const statePath = (documentHash: string) => toPath(`${STATE_PREFIX}/${documentHash}.json`)

/** JSON wire format for serialisation. Dates are stored as ISO strings. */
type SerializedEntry = {
  readonly documentHash: string
  readonly brainId: string
  readonly stage: PipelineStage
  readonly retryCount: number
  readonly lastError?: string
  readonly createdAt: string
  readonly updatedAt: string
  readonly completedAt?: string
}

const serialize = (entry: PipelineStateEntry): Buffer => {
  const wire: SerializedEntry = {
    documentHash: entry.documentHash,
    brainId: entry.brainId,
    stage: entry.stage,
    retryCount: entry.retryCount,
    ...(entry.lastError !== undefined ? { lastError: entry.lastError } : {}),
    createdAt: entry.createdAt.toISOString(),
    updatedAt: entry.updatedAt.toISOString(),
    ...(entry.completedAt !== undefined ? { completedAt: entry.completedAt.toISOString() } : {}),
  }
  return Buffer.from(JSON.stringify(wire, null, 2), 'utf8')
}

const deserialize = (buf: Buffer): PipelineStateEntry => {
  const raw: unknown = JSON.parse(buf.toString('utf8'))
  const wire = raw as SerializedEntry
  return {
    documentHash: wire.documentHash,
    brainId: wire.brainId,
    stage: wire.stage,
    retryCount: wire.retryCount,
    ...(wire.lastError !== undefined ? { lastError: wire.lastError } : {}),
    createdAt: new Date(wire.createdAt),
    updatedAt: new Date(wire.updatedAt),
    ...(wire.completedAt !== undefined ? { completedAt: new Date(wire.completedAt) } : {}),
  }
}

/**
 * File-based pipeline state store. Persists each document's state as a JSON
 * file at `raw/.pipeline-state/{documentHash}.json` using the brain Store
 * interface.
 */
export class FilePipelineStateStore implements PipelineStateStore {
  private readonly store: Store

  constructor(store: Store) {
    this.store = store
  }

  async get(documentHash: string): Promise<PipelineStateEntry | undefined> {
    try {
      const buf = await this.store.read(statePath(documentHash))
      return deserialize(buf)
    } catch (err: unknown) {
      if (isNotFound(err)) return undefined
      throw err
    }
  }

  async set(entry: PipelineStateEntry): Promise<void> {
    await this.store.write(statePath(entry.documentHash), serialize(entry))
  }

  async listIncomplete(brainId: string): Promise<readonly PipelineStateEntry[]> {
    const dir = toPath(STATE_PREFIX)
    const exists = await this.store.exists(dir)
    if (!exists) return []

    const files = await this.store.list(dir, { recursive: false, glob: '*.json' })
    const entries: PipelineStateEntry[] = []

    for (const file of files) {
      if (file.isDir) continue
      const buf = await this.store.read(file.path)
      const entry = deserialize(buf)
      if (entry.brainId === brainId && !TERMINAL_STAGES.has(entry.stage)) {
        entries.push(entry)
      }
    }

    return entries
  }

  async delete(documentHash: string): Promise<void> {
    try {
      await this.store.delete(statePath(documentHash))
    } catch (err: unknown) {
      if (isNotFound(err)) return
      throw err
    }
  }
}
