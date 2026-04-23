import type { Embedder, Logger, Provider } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import type { Store } from '../store/index.js'
import {
  type PruneArchivedSourcesOptions,
  type PruneArchivedSourcesResult,
  type SourceArchiveStats,
  createSourceArchive,
} from './archive.js'
import { DRAFTS_PREFIX, createCompile, listIngested } from './compile.js'
import { type DedupOptions, createDedup } from './dedup.js'
import type { HashContentInput } from './hash.js'
import { RAW_DOCUMENTS_PREFIX, createIngest, hashContent, rawDocumentPath } from './ingest.js'
import { createLintFix } from './lint-fix.js'
import { createLint } from './lint.js'
import { LOG_PATH, appendLog, appendLogInBatch, formatEntry, parseLog, readLog } from './log.js'
import {
  RAW_DOCUMENTS_PROCESSED_PREFIX,
  parseProcessedMarker,
  processedMarkerPath,
  readProcessedMarker,
  serialiseProcessedMarker,
} from './processed.js'
import { type PromoteResult, WIKI_PREFIX, createPromote } from './promote.js'
import { createQuery } from './query.js'
import { DEFAULT_PATTERNS, applyPatterns, createScrub } from './scrub.js'
import type {
  CompileOptions,
  CompileResult,
  DedupReport,
  IngestOptions,
  IngestResult,
  KnowledgeEventSink,
  KnowledgeQueryRetriever,
  LintReport,
  PromoteOptions,
  QueryOptions,
  QueryResult,
  ScrubOptions,
  ScrubResult,
} from './types.js'

export * from './types.js'
export { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'
export { appendLog, appendLogInBatch, formatEntry, LOG_PATH, parseLog, readLog } from './log.js'
export { DEFAULT_PATTERNS, applyPatterns, createScrub } from './scrub.js'
export { createIngest, hashContent, rawDocumentPath, RAW_DOCUMENTS_PREFIX }
export { DRAFTS_PREFIX, createCompile, listIngested } from './compile.js'
export { createKnowledgeEventEmitter } from './events.js'
export { WIKI_PREFIX, createPromote } from './promote.js'
export { createLintFix } from './lint-fix.js'
export {
  normaliseKnowledgeArticleStem,
  normaliseWikiRelativeArticlePath,
  tryNormaliseKnowledgeArticleStem,
  tryNormaliseWikiRelativeArticlePath,
} from './paths.js'
export { createQuery } from './query.js'
export { createDedup } from './dedup.js'
export {
  archivedSourcePath,
  createSourceArchive,
  RAW_DOCUMENTS_ARCHIVE_PREFIX,
  type PruneArchivedSourcesOptions,
  type PruneArchivedSourcesResult,
  type SourceArchiveStats,
} from './archive.js'
export {
  RAW_DOCUMENTS_PROCESSED_PREFIX,
  processedMarkerPath,
  readProcessedMarker,
  parseProcessedMarker,
  serialiseProcessedMarker,
} from './processed.js'

export type KnowledgeOptions = {
  store: Store
  provider: Provider
  embedder?: Embedder
  onEvent?: KnowledgeEventSink
  retriever?: KnowledgeQueryRetriever
  logger?: Logger
}

export type Knowledge = {
  ingest: (input: HashContentInput, opts?: IngestOptions) => Promise<IngestResult>
  compile: (opts?: CompileOptions) => Promise<CompileResult>
  query: (question: string, opts?: QueryOptions) => Promise<QueryResult>
  promote: (slug: string, opts?: PromoteOptions) => Promise<PromoteResult>
  lint: () => Promise<LintReport>
  lintFix: ReturnType<typeof createLintFix>
  dedup: (opts?: DedupOptions) => Promise<DedupReport>
  scrub: (opts?: ScrubOptions) => Promise<readonly ScrubResult[]>
  sourcesInfo: () => Promise<SourceArchiveStats>
  pruneSources: (opts: PruneArchivedSourcesOptions) => Promise<PruneArchivedSourcesResult>
}

export const createKnowledge = (opts: KnowledgeOptions): Knowledge => {
  const store = opts.store
  const provider = opts.provider
  const logger = opts.logger ?? noopLogger
  const archive = createSourceArchive({ store })
  const compile = createCompile({ store, provider, logger })
  const lintFix = createLintFix({ store, compile })

  return {
    ingest: createIngest({ store, logger }),
    compile,
    query: createQuery({
      store,
      provider,
      logger,
      ...(opts.retriever !== undefined ? { retriever: opts.retriever } : {}),
    }),
    promote: createPromote({
      store,
      logger,
      ...(opts.onEvent !== undefined ? { onEvent: opts.onEvent } : {}),
    }),
    lint: createLint({ store }),
    lintFix,
    dedup: createDedup({ store }),
    scrub: createScrub({ store, logger }),
    sourcesInfo: archive.info,
    pruneSources: archive.prune,
  }
}
