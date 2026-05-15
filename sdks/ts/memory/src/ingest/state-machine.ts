// SPDX-License-Identifier: Apache-2.0

/**
 * Pipeline state machine implemented as plain TypeScript. Defines the
 * formal stage transitions for the ingestion pipeline with retry logic
 * and dead letter handling. No external dependencies.
 *
 * Stages: received -> stored -> chunked -> embedded -> indexed
 * Any stage may transition to "dead_letter" when retries are exhausted.
 */

/** Pipeline stages representing document processing progress. */
export type PipelineStage =
  | 'received'
  | 'stored'
  | 'chunked'
  | 'embedded'
  | 'indexed'
  | 'dead_letter'

/** Ordered stages for validation (excludes dead_letter). */
export const STAGE_ORDER: readonly PipelineStage[] = [
  'received',
  'stored',
  'chunked',
  'embedded',
  'indexed',
]

/** Ordinal position of each stage for comparison. */
const STAGE_ORDINAL: ReadonlyMap<PipelineStage, number> = new Map([
  ['received', 0],
  ['stored', 1],
  ['chunked', 2],
  ['embedded', 3],
  ['indexed', 4],
])

/** Valid single-step forward transitions. */
const VALID_TRANSITIONS: ReadonlyMap<PipelineStage, PipelineStage> = new Map([
  ['received', 'stored'],
  ['stored', 'chunked'],
  ['chunked', 'embedded'],
  ['embedded', 'indexed'],
])

/**
 * Reports whether moving from current to target is a valid single-step
 * forward transition.
 */
export const isValidTransition = (current: PipelineStage, target: PipelineStage): boolean => {
  return VALID_TRANSITIONS.get(current) === target
}

/** Persisted state for a single document in the pipeline. */
export type PipelineStateEntry = {
  readonly documentHash: string
  readonly stage: PipelineStage
  readonly retryCount: number
  readonly lastError: string
  readonly createdAt: string
  readonly updatedAt: string
}

/** Persistence interface for pipeline state entries. */
export type PipelineStateStore = {
  readonly load: (documentHash: string) => Promise<PipelineStateEntry | undefined>
  readonly save: (entry: PipelineStateEntry) => Promise<void>
  readonly listIncomplete: () => Promise<readonly PipelineStateEntry[]>
}

/** Callback invoked on every state transition for observability. */
export type TransitionCallback = (
  documentHash: string,
  from: PipelineStage,
  to: PipelineStage,
  event: string,
) => void

/** Maximum retry count before moving to the dead_letter stage. */
const MAX_DEFAULT_RETRIES = 3

/** Configuration for creating a PipelineStateMachine instance. */
export type PipelineStateMachineConfig = {
  readonly stateStore: PipelineStateStore
  readonly maxRetries?: number
  readonly onTransition?: TransitionCallback
}

/** V1 pipeline state entry from P0-1 for migration support. */
export type V1PipelineStateEntry = {
  readonly documentId: string
  readonly hash: string
  readonly stage: 'stored' | 'chunked' | 'embedded' | 'indexed'
  readonly updatedAt: string
  readonly chunkCount?: number
}

/** Map from V1 stage names to V2 PipelineStage. */
const V1_STAGE_MAP: ReadonlyMap<string, PipelineStage> = new Map([
  ['stored', 'stored'],
  ['chunked', 'chunked'],
  ['embedded', 'embedded'],
  ['indexed', 'indexed'],
])

/**
 * Migrates a V1 pipeline state entry to the V2 format used by the
 * state machine. V1 entries from P0-1 lack retry tracking and use a
 * different type shape.
 */
export const migrateFromV1 = (v1: V1PipelineStateEntry): PipelineStateEntry => {
  const stage = V1_STAGE_MAP.get(v1.stage) ?? 'received'
  return {
    documentHash: v1.hash,
    stage,
    retryCount: 0,
    lastError: '',
    createdAt: v1.updatedAt,
    updatedAt: v1.updatedAt,
  }
}

/**
 * Creates a pipeline state machine manager that orchestrates stage
 * transitions with persistence and observability. Uses a plain
 * switch-based approach matching the Go implementation pattern.
 */
export const createPipelineStateMachine = (config: PipelineStateMachineConfig) => {
  const maxRetries = config.maxRetries ?? MAX_DEFAULT_RETRIES

  /**
   * Resolves the next stage for a forward advance from the current stage.
   * Returns the current stage if no valid forward transition exists.
   */
  const nextStage = (current: PipelineStage): PipelineStage => {
    return VALID_TRANSITIONS.get(current) ?? current
  }

  /**
   * Determines whether the current stage is at or past the target
   * in the forward chain.
   */
  const isAtOrPast = (current: PipelineStage, target: PipelineStage): boolean => {
    const currentOrd = STAGE_ORDINAL.get(current)
    const targetOrd = STAGE_ORDINAL.get(target)
    if (currentOrd === undefined || targetOrd === undefined) return false
    return currentOrd >= targetOrd
  }

  /**
   * Advance a document to the next stage. Validates the transition,
   * persists the new state, and invokes the transition callback.
   */
  const advance = async (
    documentHash: string,
    targetStage: PipelineStage,
  ): Promise<PipelineStateEntry> => {
    const existing = await config.stateStore.load(documentHash)
    const now = new Date().toISOString()

    const currentStage: PipelineStage = existing?.stage ?? 'received'

    // Terminal state: no transitions allowed.
    if (currentStage === 'dead_letter') {
      return existing ?? {
        documentHash,
        stage: 'dead_letter',
        retryCount: 0,
        lastError: '',
        createdAt: now,
        updatedAt: now,
      }
    }

    // Idempotent: already at or past the target.
    if (isAtOrPast(currentStage, targetStage)) {
      return existing ?? {
        documentHash,
        stage: currentStage,
        retryCount: 0,
        lastError: '',
        createdAt: now,
        updatedAt: now,
      }
    }

    // Validate: only single-step forward transitions.
    const next = nextStage(currentStage)
    if (next !== targetStage) {
      return existing ?? {
        documentHash,
        stage: currentStage,
        retryCount: 0,
        lastError: '',
        createdAt: now,
        updatedAt: now,
      }
    }

    const entry: PipelineStateEntry = {
      documentHash,
      stage: next,
      retryCount: existing?.retryCount ?? 0,
      lastError: existing?.lastError ?? '',
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    }

    await config.stateStore.save(entry)

    if (config.onTransition !== undefined && currentStage !== next) {
      config.onTransition(documentHash, currentStage, next, 'advance')
    }

    return entry
  }

  /**
   * Record a failure for the document. Increments the retry counter.
   * If retries are exhausted, moves to dead_letter.
   */
  const recordFailure = async (
    documentHash: string,
    error: string,
  ): Promise<PipelineStateEntry> => {
    const existing = await config.stateStore.load(documentHash)
    const now = new Date().toISOString()
    const currentStage: PipelineStage = existing?.stage ?? 'received'
    const currentRetry = existing?.retryCount ?? 0
    const newRetryCount = currentRetry + 1

    if (newRetryCount >= maxRetries) {
      const entry: PipelineStateEntry = {
        documentHash,
        stage: 'dead_letter',
        retryCount: newRetryCount,
        lastError: error,
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
      }
      await config.stateStore.save(entry)
      if (config.onTransition !== undefined && currentStage !== 'dead_letter') {
        config.onTransition(documentHash, currentStage, 'dead_letter', 'retry_exhausted')
      }
      return entry
    }

    const entry: PipelineStateEntry = {
      documentHash,
      stage: currentStage,
      retryCount: newRetryCount,
      lastError: error,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    }
    await config.stateStore.save(entry)
    return entry
  }

  /**
   * Reports whether a document has not yet exhausted its retry budget.
   */
  const shouldRetry = (entry: PipelineStateEntry): boolean => {
    return entry.retryCount < maxRetries
  }

  /**
   * Unconditionally marks a document as dead_letter regardless of retry count.
   */
  const markDeadLetter = async (
    documentHash: string,
    error: string,
  ): Promise<PipelineStateEntry> => {
    const existing = await config.stateStore.load(documentHash)
    const now = new Date().toISOString()
    const currentStage: PipelineStage = existing?.stage ?? 'received'

    const entry: PipelineStateEntry = {
      documentHash,
      stage: 'dead_letter',
      retryCount: existing?.retryCount ?? 0,
      lastError: error,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    }
    await config.stateStore.save(entry)

    if (config.onTransition !== undefined && currentStage !== 'dead_letter') {
      config.onTransition(documentHash, currentStage, 'dead_letter', 'mark_dead_letter')
    }
    return entry
  }

  /**
   * Returns all entries not in a terminal stage (indexed or dead_letter).
   */
  const listIncomplete = async (): Promise<readonly PipelineStateEntry[]> => {
    return config.stateStore.listIncomplete()
  }

  return {
    advance,
    recordFailure,
    shouldRetry,
    markDeadLetter,
    listIncomplete,
    maxRetries,
  } as const
}
