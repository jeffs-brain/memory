/**
 * Public types for the memory-stage layer.
 *
 * This layer composes the Store (persistence) and Provider/Embedder (LLM
 * surface) into the five canonical stages: extract, recall, reflect,
 * consolidate, contextualise. No singletons — every call site constructs
 * its own `Memory` instance via `createMemory`.
 */

import type { Embedder, Logger, Message, Provider } from '../llm/index.js'
import type { Path, Store } from '../store/index.js'
import type {
  EpisodeListOptions,
  EpisodeQueryHit,
  EpisodeQueryOptions,
  EpisodeRecord,
  EpisodeRecordArgs,
  RecordEpisodeResult,
} from './episodes.js'
import type {
  DetectAndPersistProceduralRecordsArgs,
  ListProceduralRecordsArgs,
  PersistProceduralRecordsArgs,
  ProceduralQueryHit,
  QueryProceduralRecordsArgs,
  StoredProceduralRecord,
} from './procedural-store.js'

/** Scope of a memory note. Mirrors the Go `scope` enum. */
export type Scope = 'global' | 'project' | 'agent'

/** A single extracted memory note. Mirrors Go `ExtractedMemory`. */
export type ExtractedMemory = {
  readonly action: 'create' | 'update'
  readonly filename: string
  readonly name: string
  readonly description: string
  readonly type: 'user' | 'feedback' | 'project' | 'reference'
  readonly content: string
  readonly indexEntry: string
  readonly scope: Scope
  readonly supersedes?: string
  readonly tags?: readonly string[]
  readonly sessionId?: string
  readonly observedOn?: string
  readonly sessionDate?: string
  readonly contextPrefix?: string
  readonly modifiedOverride?: string
}

/** Memory note as persisted (with resolved path). */
export type MemoryNote = {
  readonly path: Path
  readonly name: string
  readonly description: string
  readonly type: string
  readonly scope: Scope
  readonly tags: readonly string[]
  readonly content: string
  readonly modified?: string
  readonly created?: string
  readonly sessionId?: string
  readonly sessionDate?: string
  readonly observedOn?: string
}

/** Recall hit surfaced to callers. */
export type RecallHit = {
  readonly path: Path
  readonly score: number
  readonly content: string
  readonly note: MemoryNote
}

export type RecallSelectorMode = 'off' | 'auto'

/** Options passed to `recall`. */
export type RecallOpts = {
  readonly query: string
  readonly k?: number
  readonly scope?: Scope
  readonly actorId?: string
  readonly excludedPaths?: readonly Path[]
  readonly surfacedPaths?: readonly Path[]
  readonly selector?: RecallSelectorMode
}

/** Reflection outcome persisted + returned. */
export type ReflectionResult = {
  readonly outcome: 'success' | 'partial' | 'failure' | 'unknown'
  readonly summary: string
  readonly retryFeedback?: string
  readonly shouldRecordEpisode?: boolean
  readonly openQuestions: readonly string[]
  readonly heuristics: readonly Heuristic[]
  readonly path: Path
}

export type Heuristic = {
  readonly rule: string
  readonly context: string
  readonly confidence: 'low' | 'medium' | 'high'
  readonly category: string
  readonly scope: Scope
  readonly antiPattern: boolean
}

export type ProceduralTier = 'skill' | 'agent'

export type ProceduralOutcome = 'ok' | 'error' | 'partial'

export type ProceduralRecord = {
  readonly tier: ProceduralTier
  readonly name: string
  readonly taskContext: string
  readonly outcome: ProceduralOutcome
  readonly observedAt: string
  readonly toolCalls: readonly string[]
  readonly tags: readonly string[]
}

export type DetectProceduralRecordsOptions = {
  readonly observedAt?: Date | string
  readonly maxContextLength?: number
}

export type L0Intent = 'ask' | 'edit' | 'read' | 'plan' | 'chat'

export type L0Outcome = 'ok' | 'error' | 'partial'

export type L0Observation = {
  readonly at: string
  readonly intent: L0Intent
  readonly entities: readonly string[]
  readonly outcome: L0Outcome
  readonly summary: string
}

export type L0BufferConfig = {
  readonly tokenBudget: number
  readonly compactThresholdPercent: number
  readonly keepRecentPercent: number
  readonly maxObservationLength: number
  readonly maxEntities: number
  readonly reminderHeading: string
}

export type ObserveMessagesOptions = {
  readonly observedAt?: Date | string
  readonly maxObservationLength?: number
  readonly maxEntities?: number
}

export type MemoryPersistProceduralRecordsArgs = Omit<
  PersistProceduralRecordsArgs,
  'actorId'
> & {
  readonly actorId?: string
}

export type MemoryDetectAndPersistProceduralRecordsArgs = Omit<
  DetectAndPersistProceduralRecordsArgs,
  'actorId'
> & {
  readonly actorId?: string
}

export type L0BufferCompactionResult = {
  readonly observations: readonly L0Observation[]
  readonly removed: number
}

export type L0BufferAppendResult = {
  readonly observations: readonly L0Observation[]
  readonly removed: number
  readonly compacted: boolean
}

/** Consolidation operation log entry. */
export type ConsolidationOp =
  | { readonly kind: 'merge'; readonly keeper: Path; readonly donor: Path }
  | { readonly kind: 'delete'; readonly path: Path; readonly reason: string }
  | { readonly kind: 'promote'; readonly path: Path }
  | { readonly kind: 'rewrite'; readonly path: Path }

export type ConsolidationReport = {
  readonly merged: number
  readonly deleted: number
  readonly promoted: number
  readonly ops: readonly ConsolidationOp[]
  readonly errors: readonly string[]
}

/** Search index contract used by `recall`. Injectable per call. */
export type SearchIndex = {
  search(
    query: string,
    embedding: readonly number[] | undefined,
    opts: { readonly k: number; readonly scope?: Scope; readonly actorId?: string },
  ): Promise<readonly SearchHit[]>
}

export type SearchHit = {
  readonly path: Path
  readonly score: number
}

export type CursorScope = {
  readonly sessionId?: string
}

/** Persistent cursor store interface. Implementations MUST persist each set. */
export type CursorStore = {
  get(actorId: string, scope?: CursorScope): Promise<number>
  set(actorId: string, cursor: number, scope?: CursorScope): Promise<void>
}

/** Per-stage plugin hook payloads. Mirrors the extract/reflect/consolidate events. */
export type ExtractionPayload = {
  readonly actorId: string
  readonly scope: Scope
  readonly messages: readonly Message[]
  readonly extracted: readonly ExtractedMemory[]
}

export type ReflectionPayload = {
  readonly actorId: string
  readonly scope: Scope
  readonly messages: readonly Message[]
  readonly result?: ReflectionResult
}

export type ConsolidationPayload = {
  readonly scope: Scope
  readonly report?: ConsolidationReport
}

/**
 * Plugin hooks — a fresh list passed at construction time. Plugins fire in
 * registration order for "Start" events, in reverse order for "End" events.
 * Any hook that throws is logged and swallowed so a bad plugin never kills
 * the pipeline.
 */
export type Plugin = {
  readonly name: string
  onExtractionStart?: (ctx: ExtractionPayload) => Promise<void> | void
  onExtractionEnd?: (ctx: ExtractionPayload) => Promise<void> | void
  onReflectionStart?: (ctx: ReflectionPayload) => Promise<void> | void
  onReflectionEnd?: (ctx: ReflectionPayload) => Promise<void> | void
  onConsolidationStart?: (ctx: ConsolidationPayload) => Promise<void> | void
  onConsolidationEnd?: (ctx: ConsolidationPayload) => Promise<void> | void
}

/** Context object for injecting recalled memories into a prompt. */
export type PromptContext = {
  /** The original user message untouched. */
  readonly userMessage: string
  /** Top-N recalled memories in score order. */
  readonly memories: readonly RecallHit[]
  /** Formatted system reminder block suitable for direct injection. */
  readonly systemReminder: string
}

/** Options passed to `createMemory`. No singletons — every tenant / request
 *  wires a fresh instance. */
export type MemoryOpts = {
  readonly store: Store
  readonly provider: Provider
  readonly embedder?: Embedder
  readonly logger?: Logger
  readonly plugins?: readonly Plugin[]
  /**
   * Default scope used when stages do not receive an explicit one. The
   * memory module is scope-aware rather than scope-locked: per-call overrides
   * still work even when a default is set.
   */
  readonly scope: Scope
  /** Actor identifier used as the default for cursor + recall calls. */
  readonly actorId: string
  /** Persistent cursor store — required so no in-memory cursor leaks across
   *  instances. */
  readonly cursorStore: CursorStore
  /** Injectable search index used by `recall`. When absent, recall falls
   *  back to listing store-backed notes and scoring them purely by
   *  embedding cosine distance (if an embedder is configured). */
  readonly searchIndex?: SearchIndex
  /** Minimum messages between extract calls. Defaults to 6 (Go parity). */
  readonly extractMinMessages?: number
  /** Cap on recent messages fed to the extractor. Defaults to 40. */
  readonly extractMaxRecent?: number
}

/** Arguments accepted by `extract`. */
export type ExtractArgs = {
  readonly messages: readonly Message[]
  readonly actorId?: string
  readonly scope?: Scope
  readonly sessionId?: string
  readonly sessionDate?: string
}

/** Arguments accepted by `reflect`. */
export type ReflectArgs = {
  readonly messages: readonly Message[]
  readonly sessionId: string
  readonly actorId?: string
  readonly scope?: Scope
}

/** Arguments accepted by `consolidate`. */
export type ConsolidateArgs = {
  readonly scope?: Scope
  readonly actorId?: string
}

/** Arguments accepted by `contextualise`. */
export type ContextualiseArgs = {
  readonly message: string
  readonly topK?: number
  readonly scope?: Scope
  readonly actorId?: string
  readonly excludedPaths?: readonly Path[]
  readonly surfacedPaths?: readonly Path[]
  readonly selector?: RecallSelectorMode
}

/** Public surface returned by `createMemory`. */
export type Memory = {
  extract(args: ExtractArgs): Promise<readonly ExtractedMemory[]>
  recall(opts: RecallOpts): Promise<readonly RecallHit[]>
  reflect(args: ReflectArgs): Promise<ReflectionResult | undefined>
  consolidate(args?: ConsolidateArgs): Promise<ConsolidationReport>
  contextualise(args: ContextualiseArgs): Promise<PromptContext>
  recordEpisode?: (args: EpisodeRecordArgs) => Promise<RecordEpisodeResult>
  getEpisode?: (sessionId: string) => Promise<EpisodeRecord | undefined>
  listEpisodes?: (args?: EpisodeListOptions) => Promise<readonly EpisodeRecord[]>
  queryEpisodes?: (args: EpisodeQueryOptions) => Promise<readonly EpisodeQueryHit[]>
  persistProceduralRecords?: (
    args: MemoryPersistProceduralRecordsArgs,
  ) => Promise<readonly StoredProceduralRecord[]>
  detectAndPersistProceduralRecords?: (
    args: MemoryDetectAndPersistProceduralRecordsArgs,
  ) => Promise<readonly StoredProceduralRecord[]>
  listProceduralRecords?: (
    args?: ListProceduralRecordsArgs,
  ) => Promise<readonly StoredProceduralRecord[]>
  queryProceduralRecords?: (
    args: QueryProceduralRecordsArgs,
  ) => Promise<readonly ProceduralQueryHit[]>
  /** Subscribe to the underlying Store's change stream. */
  subscribe(sink: (event: import('../store/index.js').ChangeEvent) => void): () => void
  /** Explicit alias for the unsubscribe callback returned by subscribe. */
  unsubscribe(handle: () => void): void
}
