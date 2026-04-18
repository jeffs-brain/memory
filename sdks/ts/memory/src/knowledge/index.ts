/**
 * Knowledge pipeline factory. Returns per-call bound instances of the
 * ingest / compile / query / promote / lint / lint-fix / dedup / scrub
 * entry points.
 * No singleton state; everything the pipeline needs is passed in.
 */

import type { Embedder, Logger, Provider } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import type { Store } from '../store/index.js'
import {
  createSourceArchive,
  type PruneArchivedSourcesOptions,
  type PruneArchivedSourcesResult,
  type SourceArchiveStats,
} from './archive.js'
import { createCompile } from './compile.js'
import { createDedup, type DedupOptions } from './dedup.js'
import { createIngest } from './ingest.js'
import { createLint } from './lint.js'
import { createLintFix } from './lint-fix.js'
import { createPromote } from './promote.js'
import { createQuery } from './query.js'
import { createScrub } from './scrub.js'
import type {
  CompileOptions,
  CompileResult,
  DedupReport,
  IngestOptions,
  IngestResult,
  LintReport,
  PromoteOptions,
  QueryOptions,
  QueryResult,
  ScrubOptions,
  ScrubResult,
} from './types.js'
import type { PromoteResult } from './promote.js'

export * from './types.js'
export { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'
export { appendLog, appendLogInBatch, formatEntry, LOG_PATH, parseLog, readLog } from './log.js'
export { DEFAULT_PATTERNS, applyPatterns } from './scrub.js'
export { createIngest, hashContent, ingestedPath, INGESTED_PREFIX } from './ingest.js'
export { DRAFTS_PREFIX } from './compile.js'
export { WIKI_PREFIX } from './promote.js'
export { createLintFix } from './lint-fix.js'
export {
  normaliseKnowledgeArticleStem,
  normaliseWikiRelativeArticlePath,
  tryNormaliseKnowledgeArticleStem,
  tryNormaliseWikiRelativeArticlePath,
} from './paths.js'
export { createQuery } from './query.js'
export {
  archivedSourcePath,
  createSourceArchive,
  INGESTED_ARCHIVE_PREFIX,
  type PruneArchivedSourcesOptions,
  type PruneArchivedSourcesResult,
  type SourceArchiveStats,
} from './archive.js'
export {
  INGESTED_PROCESSED_PREFIX,
  processedMarkerPath,
  readProcessedMarker,
  parseProcessedMarker,
  serialiseProcessedMarker,
} from './processed.js'

export type KnowledgeOptions = {
  store: Store
  provider: Provider
  /** Embedder is currently unused by the structural dedup path but is
   * accepted so future semantic dedup can be wired up without changing
   * the public factory signature. */
  embedder?: Embedder
  retriever?: import('./types.js').KnowledgeQueryRetriever
  logger?: Logger
}

export type Knowledge = {
  ingest: (input: string | Buffer, opts?: IngestOptions) => Promise<IngestResult>
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
    promote: createPromote({ store, logger }),
    lint: createLint({ store }),
    lintFix,
    dedup: createDedup({ store }),
    scrub: createScrub({ store, logger }),
    sourcesInfo: archive.info,
    pruneSources: archive.prune,
  }
}
