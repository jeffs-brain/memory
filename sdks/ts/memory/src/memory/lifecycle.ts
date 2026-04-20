// SPDX-License-Identifier: Apache-2.0

import {
  appendL0Observation,
  defaultL0BufferConfig,
  exportL0BufferSnapshot,
  observeMessages,
  renderL0Reminder,
  restoreL0BufferSnapshot,
} from './buffer.js'
import type { RecordEpisodeResult } from './episodes.js'
import type {
  ConsolidateArgs,
  ConsolidationReport,
  ContextualiseArgs,
  ExtractArgs,
  ExtractedMemory,
  L0BufferConfig,
  L0BufferSnapshot,
  L0Observation,
  Memory,
  PromptContext,
  ReflectArgs,
  ReflectionResult,
} from './types.js'

export type MemoryLifecycle = {
  beforeTurn(args: ContextualiseArgs): Promise<PromptContext>
  afterTurn(args: ExtractArgs): Promise<readonly ExtractedMemory[]>
  endSession(args: MemoryLifecycleEndSessionArgs): Promise<MemoryLifecycleEndSessionResult>
  exportL0BufferSnapshot(args?: { readonly createdAt?: Date | string }): L0BufferSnapshot
  restoreL0BufferSnapshot(snapshot: unknown): void
}

export type MemoryLifecycleEndSessionArgs = ReflectArgs & {
  readonly consolidate?: boolean | ConsolidateArgs
}

export type MemoryLifecycleEndSessionResult = {
  readonly reflection?: ReflectionResult | undefined
  readonly episode?: RecordEpisodeResult | undefined
  readonly consolidation?: ConsolidationReport | undefined
}

export type CreateMemoryLifecycleOptions = {
  readonly memory: Pick<
    Memory,
    | 'contextualise'
    | 'extract'
    | 'reflect'
    | 'consolidate'
    | 'recordEpisode'
    | 'detectAndPersistProceduralRecords'
  >
  readonly l0Buffer?: boolean | Partial<L0BufferConfig>
}

export const createMemoryLifecycle = (opts: CreateMemoryLifecycleOptions): MemoryLifecycle => {
  const l0Enabled = opts.l0Buffer !== false
  const l0Config =
    typeof opts.l0Buffer === 'object'
      ? defaultL0BufferConfig(opts.l0Buffer)
      : defaultL0BufferConfig()
  let observations: readonly L0Observation[] = []

  return {
    beforeTurn: async (args) => {
      const prompt = await opts.memory.contextualise(args)
      if (!l0Enabled || observations.length === 0) {
        return prompt
      }

      const l0Reminder = renderL0Reminder(observations, l0Config)
      if (l0Reminder === '') {
        return prompt
      }

      return {
        ...prompt,
        systemReminder:
          prompt.systemReminder.trim() === ''
            ? l0Reminder
            : `${l0Reminder}\n\n${prompt.systemReminder}`,
      }
    },

    afterTurn: async (args) => {
      const extracted = await opts.memory.extract(args)
      if (opts.memory.detectAndPersistProceduralRecords !== undefined) {
        try {
          await opts.memory.detectAndPersistProceduralRecords({
            messages: args.messages,
            ...(args.actorId !== undefined ? { actorId: args.actorId } : {}),
            ...(args.sessionId !== undefined ? { sessionId: args.sessionId } : {}),
          })
        } catch {
          // Fail open. Procedural persistence should not block post-turn extraction.
        }
      }
      if (!l0Enabled) {
        return extracted
      }

      const observation = observeMessages(args.messages, l0Config)
      if (observation === undefined) {
        return extracted
      }

      observations = appendL0Observation(observations, observation, l0Config).observations
      return extracted
    },

    exportL0BufferSnapshot: (args = {}) => exportL0BufferSnapshot(observations, args),

    restoreL0BufferSnapshot: (snapshot) => {
      observations = restoreL0BufferSnapshot(snapshot)
    },

    endSession: async (args) => {
      try {
        const reflection = await opts.memory.reflect(args)
        let episode: RecordEpisodeResult | undefined
        if (opts.memory.recordEpisode !== undefined && reflection?.shouldRecordEpisode !== false) {
          try {
            episode = await opts.memory.recordEpisode({
              messages: args.messages,
              sessionId: args.sessionId,
              reflection: {
                outcome: reflection?.outcome ?? 'unknown',
                summary: reflection?.summary ?? '',
                retryFeedback: reflection?.retryFeedback ?? '',
                shouldRecordEpisode: reflection?.shouldRecordEpisode ?? true,
                openQuestions: reflection?.openQuestions ?? [],
                heuristics: reflection?.heuristics ?? [],
              },
              ...(args.actorId !== undefined ? { actorId: args.actorId } : {}),
              ...(args.scope !== undefined ? { scope: args.scope } : {}),
            })
          } catch {
            // Fail open. Episode capture should not block reflection or consolidation.
          }
        }
        const consolidate = args.consolidate
        if (consolidate === undefined || consolidate === false) {
          return { reflection, ...(episode !== undefined ? { episode } : {}) }
        }

        const consolidation = await opts.memory.consolidate(
          consolidate === true
            ? {
                ...(args.actorId !== undefined ? { actorId: args.actorId } : {}),
                ...(args.scope !== undefined ? { scope: args.scope } : {}),
              }
            : consolidate,
        )
        return {
          reflection,
          ...(episode !== undefined ? { episode } : {}),
          consolidation,
        }
      } finally {
        observations = []
      }
    },
  }
}
