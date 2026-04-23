import type { Embedder, Logger, Message, Provider } from '../llm/types.js'
import type { Retrieval } from '../retrieval/index.js'
import type { SearchIndex } from '../search/index.js'
import type { Path, Store } from '../store/index.js'
import type {
  EpisodeListOptions,
  EpisodeQueryHit,
  EpisodeQueryOptions,
  EpisodeRecord,
  EpisodeRecordArgs,
  RecordEpisodeResult,
} from './episodes.js'
import type { Scope } from './paths.js'
import type {
  DetectAndPersistProceduralRecordsArgs,
  ListProceduralRecordsArgs,
  PersistProceduralRecordsArgs,
  ProceduralQueryHit,
  QueryProceduralRecordsArgs,
  StoredProceduralRecord,
} from './procedural-store.js'

export type MemoryNoteType = 'user' | 'feedback' | 'project' | 'reference' | 'reflection'

export type StoredMemoryNote = {
  readonly path: Path
  readonly name: string
  readonly description: string
  readonly type: MemoryNoteType
  readonly scope: Scope
  readonly actorId: string
  readonly tags: readonly string[]
  readonly content: string
  readonly created: string
  readonly modified: string
  readonly sessionId?: string
  readonly sessionDate?: string
  readonly observedOn?: string
}

export type RememberArgs = {
  readonly filename: string
  readonly name: string
  readonly description: string
  readonly content: string
  readonly type?: MemoryNoteType
  readonly scope?: Scope
  readonly actorId?: string
  readonly tags?: readonly string[]
  readonly created?: string
  readonly modified?: string
  readonly sessionId?: string
  readonly sessionDate?: string
  readonly observedOn?: string
}

export type RecallArgs = {
  readonly query: string
  readonly topK?: number
  readonly scope?: Scope
  readonly actorId?: string
}

export type RecallHit = {
  readonly path: Path
  readonly score: number
  readonly note: StoredMemoryNote
}

export type ExtractArgs = {
  readonly messages: readonly Message[]
  readonly scope?: Scope
  readonly actorId?: string
  readonly sessionId?: string
  readonly sessionDate?: string
  readonly observedOn?: string
}

export type ExtractResult = {
  readonly created: readonly StoredMemoryNote[]
  readonly skipped: boolean
  readonly reason?: string
}

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

export type PromptContext = {
  readonly userMessage: string
  readonly memories: readonly RecallHit[]
  readonly systemReminder: string
}

export type MemoryPersistProceduralRecordsArgs = Omit<PersistProceduralRecordsArgs, 'actorId'> & {
  readonly actorId?: string
}

export type MemoryDetectAndPersistProceduralRecordsArgs = Omit<
  DetectAndPersistProceduralRecordsArgs,
  'actorId'
> & {
  readonly actorId?: string
}

export type MemoryListProceduralRecordsArgs = Omit<ListProceduralRecordsArgs, 'actorId'> & {
  readonly actorId?: string
}

export type MemoryQueryProceduralRecordsArgs = Omit<QueryProceduralRecordsArgs, 'actorId'> & {
  readonly actorId?: string
}

export type MemoryClient = {
  readonly brainId: string
  readonly store: Store
  readonly searchIndex: SearchIndex
  readonly retrieval: Retrieval
  remember(args: RememberArgs): Promise<StoredMemoryNote>
  forget(path: Path): Promise<void>
  listNotes(args?: { readonly scope?: Scope; readonly actorId?: string }): Promise<
    readonly StoredMemoryNote[]
  >
  rebuildIndex(args?: { readonly scope?: Scope; readonly actorId?: string }): Promise<void>
  recall(args: RecallArgs): Promise<readonly RecallHit[]>
  contextualise(args: RecallArgs & { readonly userMessage: string }): Promise<PromptContext>
  extract(args: ExtractArgs): Promise<ExtractResult>
  reflect(args: ExtractArgs): Promise<ReflectionResult | null>
  consolidate(args?: {
    readonly scope?: Scope
    readonly actorId?: string
  }): Promise<ConsolidationReport>
  recordEpisode(args: EpisodeRecordArgs): Promise<RecordEpisodeResult>
  getEpisode(sessionId: string): Promise<EpisodeRecord | undefined>
  listEpisodes(args?: EpisodeListOptions): Promise<readonly EpisodeRecord[]>
  queryEpisodes(args: EpisodeQueryOptions): Promise<readonly EpisodeQueryHit[]>
  persistProceduralRecords(
    args: MemoryPersistProceduralRecordsArgs,
  ): Promise<readonly StoredProceduralRecord[]>
  detectAndPersistProceduralRecords(
    args: MemoryDetectAndPersistProceduralRecordsArgs,
  ): Promise<readonly StoredProceduralRecord[]>
  listProceduralRecords(
    args?: MemoryListProceduralRecordsArgs,
  ): Promise<readonly StoredProceduralRecord[]>
  queryProceduralRecords(
    args: MemoryQueryProceduralRecordsArgs,
  ): Promise<readonly ProceduralQueryHit[]>
  close(): Promise<void>
}

export type CreateMemoryClientOptions = {
  readonly brainId: string
  readonly store: Store
  readonly searchIndex: SearchIndex
  readonly retrieval: Retrieval
  readonly provider?: Provider
  readonly embedder?: Embedder
  readonly logger?: Logger
  readonly defaultScope?: Scope
  readonly defaultActorId?: string
}
