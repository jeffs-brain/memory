/**
 * Memory module entry point. `createMemory` wires the Store, Provider,
 * Embedder, CursorStore, logger, and plugins into the five canonical
 * stages. Every call returns a fresh instance — no process-wide singleton,
 * no hidden caches.
 */

import { noopLogger } from '../llm/index.js'
import type { ChangeEvent } from '../store/index.js'
import { createConsolidate } from './consolidate.js'
import { createContextualise } from './contextualise.js'
import { createEpisodeRecorder } from './episodes.js'
import { createExtract, defaultExtractConfig } from './extract.js'
import { createMemoryLifecycle } from './lifecycle.js'
import { createStoreBackedProceduralStore } from './procedural-store.js'
import { createRecall } from './recall.js'
import { createReflect } from './reflect.js'
import type { Memory, MemoryOpts } from './types.js'

export const createMemory = (opts: MemoryOpts): Memory => {
  const logger = opts.logger ?? noopLogger
  const plugins = opts.plugins ?? []
  const defaults = defaultExtractConfig()
  const extractMinMessages = opts.extractMinMessages ?? defaults.minMessages
  const extractMaxRecent = opts.extractMaxRecent ?? defaults.maxRecent

  const recall = createRecall({
    store: opts.store,
    provider: opts.provider,
    ...(opts.embedder ? { embedder: opts.embedder } : {}),
    ...(opts.searchIndex ? { searchIndex: opts.searchIndex } : {}),
    logger,
    defaultScope: opts.scope,
    defaultActorId: opts.actorId,
  })

  const extract = createExtract({
    store: opts.store,
    provider: opts.provider,
    cursorStore: opts.cursorStore,
    logger,
    plugins,
    defaultScope: opts.scope,
    defaultActorId: opts.actorId,
    minMessages: extractMinMessages,
    maxRecent: extractMaxRecent,
  })

  const reflect = createReflect({
    store: opts.store,
    provider: opts.provider,
    logger,
    plugins,
    defaultScope: opts.scope,
    defaultActorId: opts.actorId,
  })

  const consolidate = createConsolidate({
    store: opts.store,
    provider: opts.provider,
    logger,
    plugins,
    defaultScope: opts.scope,
    defaultActorId: opts.actorId,
  })

  const contextualise = createContextualise({
    recall,
    defaultScope: opts.scope,
    defaultActorId: opts.actorId,
  })
  const episodes = createEpisodeRecorder({
    store: opts.store,
    logger,
    defaultScope: opts.scope,
    defaultActorId: opts.actorId,
  })
  const proceduralStore = createStoreBackedProceduralStore(opts.store)

  const subscribe = (sink: (event: ChangeEvent) => void): (() => void) =>
    opts.store.subscribe(sink)

  const unsubscribe = (handle: () => void): void => {
    handle()
  }

  return {
    extract,
    recall,
    reflect,
    consolidate,
    contextualise,
    recordEpisode: episodes.record,
    getEpisode: episodes.get,
    listEpisodes: episodes.list,
    queryEpisodes: episodes.query,
    persistProceduralRecords: async (args) =>
      proceduralStore.persist({
        records: args.records,
        actorId: args.actorId ?? opts.actorId,
        ...(args.sessionId !== undefined ? { sessionId: args.sessionId } : {}),
        ...(args.reason !== undefined ? { reason: args.reason } : {}),
      }),
    detectAndPersistProceduralRecords: async (args) =>
      proceduralStore.detectAndPersist({
        actorId: args.actorId ?? opts.actorId,
        messages: args.messages,
        ...(args.observedAt !== undefined ? { observedAt: args.observedAt } : {}),
        ...(args.maxContextLength !== undefined
          ? { maxContextLength: args.maxContextLength }
          : {}),
        ...(args.sessionId !== undefined ? { sessionId: args.sessionId } : {}),
        ...(args.reason !== undefined ? { reason: args.reason } : {}),
      }),
    listProceduralRecords: async (args = {}) =>
      proceduralStore.list({
        actorId: args.actorId ?? opts.actorId,
        ...(args.sessionId !== undefined ? { sessionId: args.sessionId } : {}),
        ...(args.tier !== undefined ? { tier: args.tier } : {}),
        ...(args.outcome !== undefined ? { outcome: args.outcome } : {}),
        ...(args.name !== undefined ? { name: args.name } : {}),
        ...(args.tags !== undefined ? { tags: args.tags } : {}),
        ...(args.since !== undefined ? { since: args.since } : {}),
        ...(args.until !== undefined ? { until: args.until } : {}),
        ...(args.limit !== undefined ? { limit: args.limit } : {}),
        ...(args.sort !== undefined ? { sort: args.sort } : {}),
      }),
    queryProceduralRecords: async (args) =>
      proceduralStore.query({
        actorId: args.actorId ?? opts.actorId,
        text: args.text,
        ...(args.sessionId !== undefined ? { sessionId: args.sessionId } : {}),
        ...(args.tier !== undefined ? { tier: args.tier } : {}),
        ...(args.outcome !== undefined ? { outcome: args.outcome } : {}),
        ...(args.name !== undefined ? { name: args.name } : {}),
        ...(args.tags !== undefined ? { tags: args.tags } : {}),
        ...(args.since !== undefined ? { since: args.since } : {}),
        ...(args.until !== undefined ? { until: args.until } : {}),
        ...(args.limit !== undefined ? { limit: args.limit } : {}),
        ...(args.sort !== undefined ? { sort: args.sort } : {}),
      }),
    subscribe,
    unsubscribe,
  }
}

export type {
  ConsolidateArgs,
  ConsolidationOp,
  ConsolidationPayload,
  ConsolidationReport,
  ContextualiseArgs,
  CursorStore,
  ExtractArgs,
  ExtractedMemory,
  ExtractionPayload,
  Heuristic,
  L0BufferAppendResult,
  L0BufferCompactionResult,
  L0BufferConfig,
  L0Intent,
  L0Observation,
  L0Outcome,
  Memory,
  MemoryDetectAndPersistProceduralRecordsArgs,
  MemoryNote,
  MemoryOpts,
  MemoryPersistProceduralRecordsArgs,
  ObserveMessagesOptions,
  Plugin,
  ProceduralOutcome,
  ProceduralRecord,
  ProceduralTier,
  PromptContext,
  RecallHit,
  RecallOpts,
  RecallSelectorMode,
  ReflectArgs,
  ReflectionPayload,
  ReflectionResult,
  Scope,
  SearchHit,
  SearchIndex,
  DetectProceduralRecordsOptions,
} from './types.js'

export {
  CONTEXTUAL_PREFIX_MARKER,
  CONTEXTUAL_PREFIX_SYSTEM_PROMPT,
  DEDUPLICATION_SYSTEM_PROMPT,
  EXTRACTION_SYSTEM_PROMPT,
  RECALL_SELECTOR_SYSTEM_PROMPT,
  REFLECTION_SYSTEM_PROMPT,
} from './prompts.js'

export {
  MEMORY_AGENT_PREFIX,
  MEMORY_GLOBAL_PREFIX,
  MEMORY_PROJECTS_PREFIX,
  REFLECTIONS_PREFIX,
  reflectionPath,
  scopeIndex,
  scopePrefix,
  scopeTopic,
} from './paths.js'

export { StoreBackedCursorStore, createStoreBackedCursorStore } from './cursor.js'
export {
  createMemoryLifecycle,
  type CreateMemoryLifecycleOptions,
  type MemoryLifecycle,
  type MemoryLifecycleEndSessionArgs,
  type MemoryLifecycleEndSessionResult,
} from './lifecycle.js'
export {
  isRecentMemoryQuery,
  isTimeSensitiveMemoryQuery,
  mergeRecallHits,
  sortRecallHitsChronologically,
  type RecallHitSortMode,
} from './recall.js'
export {
  appendL0Observation,
  compactL0Buffer,
  defaultL0BufferConfig,
  estimateL0BufferTokens,
  formatL0Observation,
  needsL0BufferCompaction,
  observeMessages,
  renderL0Buffer,
  renderL0Reminder,
} from './buffer.js'
export { detectProceduralRecords, formatProceduralRecord } from './procedural.js'
export {
  createEpisodeRecorder,
  defaultEpisodeRecorderConfig,
  episodePath,
  type EpisodeGateDecision,
  type EpisodeGateReason,
  type EpisodeListOptions,
  type EpisodeOutcome,
  type EpisodeQueryHit,
  type EpisodeQueryOptions,
  type EpisodeRecorderDeps,
  type EpisodeRecord,
  type EpisodeRecordArgs,
  type EpisodeRecorder,
  type EpisodeRecorderConfig,
  type EpisodeSignals,
  type RecordEpisodeResult,
} from './episodes.js'
export {
  PROCEDURAL_RECORDS_PREFIX,
  StoreBackedProceduralStore,
  createProceduralStore,
  createStoreBackedProceduralStore,
  proceduralActorPrefix,
  proceduralSessionPrefix,
  type DetectAndPersistProceduralRecordsArgs,
  type ListProceduralRecordsArgs,
  type PersistProceduralRecordsArgs,
  type ProceduralQueryHit,
  type ProceduralStore,
  type QueryProceduralRecordsArgs,
  type StoredProceduralRecord,
} from './procedural-store.js'
