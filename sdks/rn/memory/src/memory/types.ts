import type { Embedder, Logger, Message, Provider } from '../llm/types.js'
import type { Retrieval } from '../retrieval/index.js'
import type { SearchIndex } from '../search/index.js'
import type { Path, Store } from '../store/index.js'
import type { Scope } from './paths.js'

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
  readonly summary: string
  readonly openQuestions: readonly string[]
  readonly path: Path
}

export type ConsolidationReport = {
  readonly merged: number
  readonly deleted: number
  readonly rewritten: readonly Path[]
}

export type PromptContext = {
  readonly userMessage: string
  readonly memories: readonly RecallHit[]
  readonly systemReminder: string
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
