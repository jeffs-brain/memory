// SPDX-License-Identifier: Apache-2.0

/**
 * Pipeline state machine using XState v5. Defines the formal stage
 * transitions for the ingestion pipeline with retry logic and dead
 * letter handling.
 *
 * Stages: received -> stored -> chunked -> embedded -> indexed -> completed
 * Any stage may transition to "failed" when retries are exhausted.
 */

import { assign, createActor, setup } from 'xstate'

/** Pipeline stages representing document processing progress. */
export type PipelineStage =
  | 'received'
  | 'stored'
  | 'chunked'
  | 'embedded'
  | 'indexed'
  | 'completed'
  | 'failed'

/** Events that drive state machine transitions. */
export type PipelineMachineEvent =
  | { readonly type: 'STORE_COMPLETE' }
  | { readonly type: 'CHUNK_COMPLETE' }
  | { readonly type: 'EMBED_COMPLETE' }
  | { readonly type: 'INDEX_COMPLETE' }
  | { readonly type: 'COMPLETE' }
  | { readonly type: 'FAIL'; readonly error: string }
  | { readonly type: 'RETRY_EXHAUSTED'; readonly error: string }

/** Context tracked by the state machine for each document. */
export type PipelineMachineContext = {
  readonly documentHash: string
  readonly retryCount: number
  readonly maxRetries: number
  readonly lastError: string
}

/** Input required when creating a pipeline machine actor. */
export type PipelineMachineInput = {
  readonly documentHash: string
  readonly retryCount: number
  readonly maxRetries: number
  readonly lastError: string
}

/** Persistence interface for pipeline state entries. */
export type PipelineStateStore = {
  readonly load: (documentHash: string) => Promise<PipelineStateEntry | undefined>
  readonly save: (entry: PipelineStateEntry) => Promise<void>
  readonly listIncomplete: () => Promise<readonly PipelineStateEntry[]>
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

/** Callback invoked on every state transition for observability. */
export type TransitionCallback = (
  documentHash: string,
  from: PipelineStage,
  to: PipelineStage,
  event: PipelineMachineEvent['type'],
) => void

/** Maximum retry count before moving to the failed stage. */
const MAX_DEFAULT_RETRIES = 3

/** Ordered stages for validation. */
export const STAGE_ORDER: readonly PipelineStage[] = [
  'received',
  'stored',
  'chunked',
  'embedded',
  'indexed',
  'completed',
]

/** Valid single-step forward transitions. */
const VALID_TRANSITIONS: ReadonlyMap<PipelineStage, PipelineStage> = new Map([
  ['received', 'stored'],
  ['stored', 'chunked'],
  ['chunked', 'embedded'],
  ['embedded', 'indexed'],
  ['indexed', 'completed'],
])

/**
 * Reports whether moving from current to target is a valid single-step
 * forward transition.
 */
export const isValidTransition = (current: PipelineStage, target: PipelineStage): boolean => {
  return VALID_TRANSITIONS.get(current) === target
}

/** Shared FAIL transition definition used by each non-terminal state. */
const failTransitions = [
  {
    guard: 'canRetry' as const,
    actions: ['recordErrorFromEvent' as const, 'incrementRetry' as const],
  },
  { target: 'failed' as const, actions: ['recordErrorFromEvent' as const] },
]

/** Shared RETRY_EXHAUSTED transition used by each non-terminal state. */
const retryExhaustedTransition = {
  target: 'failed' as const,
  actions: ['recordErrorFromEvent' as const],
}

/**
 * XState v5 machine definition for the ingestion pipeline state machine.
 * Each state handles FAIL events for retry logic, and forward transitions
 * on stage completion events.
 */
export const pipelineMachine = setup({
  types: {
    context: {} as PipelineMachineContext,
    events: {} as PipelineMachineEvent,
    input: {} as PipelineMachineInput,
  },
  guards: {
    canRetry: ({ context }) => context.retryCount < context.maxRetries,
  },
  actions: {
    incrementRetry: assign({
      retryCount: ({ context }) => context.retryCount + 1,
    }),
    recordErrorFromEvent: assign({
      lastError: ({ event }) => {
        if ('error' in event) return event.error
        return ''
      },
    }),
  },
}).createMachine({
  id: 'pipeline',
  initial: 'received',
  context: ({ input }) => ({
    documentHash: input.documentHash,
    retryCount: input.retryCount,
    maxRetries: input.maxRetries,
    lastError: input.lastError,
  }),
  states: {
    received: {
      on: {
        STORE_COMPLETE: { target: 'stored' },
        FAIL: failTransitions,
        RETRY_EXHAUSTED: retryExhaustedTransition,
      },
    },
    stored: {
      on: {
        CHUNK_COMPLETE: { target: 'chunked' },
        FAIL: failTransitions,
        RETRY_EXHAUSTED: retryExhaustedTransition,
      },
    },
    chunked: {
      on: {
        EMBED_COMPLETE: { target: 'embedded' },
        FAIL: failTransitions,
        RETRY_EXHAUSTED: retryExhaustedTransition,
      },
    },
    embedded: {
      on: {
        INDEX_COMPLETE: { target: 'indexed' },
        FAIL: failTransitions,
        RETRY_EXHAUSTED: retryExhaustedTransition,
      },
    },
    indexed: {
      on: {
        COMPLETE: { target: 'completed' },
        FAIL: failTransitions,
        RETRY_EXHAUSTED: retryExhaustedTransition,
      },
    },
    completed: {
      type: 'final',
    },
    failed: {
      type: 'final',
    },
  },
})

/** Configuration for creating a PipelineStateMachine instance. */
export type PipelineStateMachineConfig = {
  readonly stateStore: PipelineStateStore
  readonly maxRetries?: number
  readonly onTransition?: TransitionCallback
}

/**
 * Creates a pipeline state machine manager that wraps XState actors with
 * persistence via PipelineStateStore and transition observability.
 */
export const createPipelineStateMachine = (config: PipelineStateMachineConfig) => {
  const maxRetries = config.maxRetries ?? MAX_DEFAULT_RETRIES

  /**
   * Resolves the persisted state to an XState snapshot for actor restoration.
   * For the initial state (received), returns undefined so the machine starts
   * normally. For other stages, uses machine.resolveState to hydrate.
   */
  const resolveSnapshot = (stage: PipelineStage, context: PipelineMachineContext) => {
    if (stage === 'received') return undefined
    return pipelineMachine.resolveState({
      value: stage,
      context,
    })
  }

  /**
   * Advance a document to the next stage by sending the appropriate event.
   * Validates the transition, persists the new state, and invokes the
   * transition callback.
   */
  const advanceStage = async (
    documentHash: string,
    event: PipelineMachineEvent,
  ): Promise<PipelineStateEntry> => {
    const existing = await config.stateStore.load(documentHash)
    const currentStage: PipelineStage = existing?.stage ?? 'received'
    const currentRetryCount = existing?.retryCount ?? 0

    const machineContext: PipelineMachineContext = {
      documentHash,
      retryCount: currentRetryCount,
      maxRetries,
      lastError: existing?.lastError ?? '',
    }

    const snapshot = resolveSnapshot(currentStage, machineContext)

    const actor = createActor(pipelineMachine, {
      input: machineContext,
      ...(snapshot !== undefined ? { snapshot } : {}),
    })

    actor.start()
    const beforeState = actor.getSnapshot().value as PipelineStage
    actor.send(event)
    const afterSnapshot = actor.getSnapshot()
    const afterState = afterSnapshot.value as PipelineStage
    actor.stop()

    const now = new Date().toISOString()
    const entry: PipelineStateEntry = {
      documentHash,
      stage: afterState,
      retryCount: afterSnapshot.context.retryCount,
      lastError: afterSnapshot.context.lastError,
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    }

    await config.stateStore.save(entry)

    if (config.onTransition !== undefined && beforeState !== afterState) {
      config.onTransition(documentHash, beforeState, afterState, event.type)
    }

    return entry
  }

  /**
   * Reports whether a document has not yet exhausted its retry budget.
   */
  const shouldRetry = (entry: PipelineStateEntry): boolean => {
    return entry.retryCount < maxRetries
  }

  /**
   * Unconditionally marks a document as failed regardless of retry count.
   */
  const markFailed = async (documentHash: string, error: string): Promise<PipelineStateEntry> => {
    return advanceStage(documentHash, { type: 'RETRY_EXHAUSTED', error })
  }

  return {
    advanceStage,
    shouldRetry,
    markFailed,
    maxRetries,
  } as const
}
