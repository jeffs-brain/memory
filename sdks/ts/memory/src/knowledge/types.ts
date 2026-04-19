// SPDX-License-Identifier: Apache-2.0

/**
 * Public types for the knowledge pipeline. Ported (and simplified) from
 * apps/jeff/internal/knowledge/{knowledge,ingest,compile,promote,lint,
 * dedup,scrub,frontmatter}.go.
 *
 * Storage layout used across the brain:
 *   - raw/documents/<hash>.md  -- raw ingested inputs
 *   - drafts/<slug>.md         -- compiled intermediates
 *   - wiki/<slug>.md           -- promoted final articles
 *   - _log.md                  -- operations log
 */

import type { Path } from '../store/index.js'

export type Frontmatter = {
  title: string
  summary: string
  tags: readonly string[]
  sources: readonly string[]
  created?: string
  modified?: string
  archived?: boolean
  supersededBy?: string
}

export type Article = {
  path: Path
  frontmatter: Frontmatter
  body: string
}

export type QuerySortMode = 'relevance' | 'recency' | 'relevance_then_recency'

export type KnowledgeQueryRetrieverChunk = {
  readonly path: string
  readonly score: number
  readonly metadata?: Readonly<Record<string, string | number | boolean | null>>
}

export type KnowledgeQueryRetriever = {
  retrieve: (req: {
    readonly query: string
    readonly limit: number
    readonly mode?: 'hybrid'
    readonly candidateLimit?: number
  }) => Promise<{
    readonly chunks: readonly KnowledgeQueryRetrieverChunk[]
  }>
}

export type QueryOptions = {
  /** Maximum articles to feed into synthesis. Defaults to 5. */
  maxSources?: number
  /** Retrieval ordering before synthesis. Defaults to relevance. */
  sort?: QuerySortMode
}

export type QueryResult = {
  answer: string
  sourcePaths: readonly Path[]
  searchHits: number
}

export type OperationKind =
  | 'ingest'
  | 'compile.plan'
  | 'compile.write'
  | 'compile.update'
  | 'compile.crossref'
  | 'promote'
  | 'lint'
  | 'lint.fix'
  | 'dedup'
  | 'scrub'

export type Operation = {
  kind: OperationKind
  title: string
  detail: string
  /** ISO-8601 UTC timestamp. */
  when: string
}

export type IngestResult = {
  path: Path
  hash: string
  bytes: number
  skipped?: 'duplicate'
}

export type CompilePlanArticle = {
  slug: string
  title: string
  summary: string
  sourceHashes: readonly string[]
}

export type CompilePlanUpdate = {
  path: string
  reason: string
  sourceHashes: readonly string[]
}

export type CompilePlanCrossReference = {
  path: string
  reason: string
  sourceHashes: readonly string[]
}

export type CompilePlan = {
  articles: readonly CompilePlanArticle[]
  newArticles: readonly CompilePlanArticle[]
  updates: readonly CompilePlanUpdate[]
  crossReferences: readonly CompilePlanCrossReference[]
  concepts: readonly string[]
  processedSources: readonly string[]
}

export type CompileResult = {
  plan: CompilePlan
  written: readonly Path[]
}

export type LintIssueKind =
  | 'missing_frontmatter_field'
  | 'orphan_draft'
  | 'broken_link'
  | 'empty_body'
  | 'duplicate_title'
  | 'stub_article'
  | 'zero_link_article'
  | 'stale_source'

export type LintIssue = {
  kind: LintIssueKind
  path: Path
  message: string
  details?: {
    field?: string
    relatedPaths?: readonly Path[]
    sourcePath?: string
    target?: string
    titleKey?: string
    wordCount?: number
  }
}

export type LintReport = {
  ok: boolean
  issues: readonly LintIssue[]
}

export type LintFixActionKind = 'rehydrate_stub' | 'archive_duplicate_title'

export type LintFixAction =
  | {
      kind: 'rehydrate_stub'
      articlePath: Path
      sourcePaths: readonly Path[]
      markerPaths: readonly Path[]
    }
  | {
      kind: 'archive_duplicate_title'
      titleKey: string
      canonicalPath: Path
      archivePaths: readonly Path[]
    }

export type LintFixSkipReason =
  | 'already_pending'
  | 'already_superseded'
  | 'limit_reached'
  | 'missing_article'
  | 'no_active_duplicates'
  | 'no_rehydratable_sources'
  | 'unsafe_duplicate_group'

export type LintFixSkippedItem = {
  kind: LintFixActionKind
  reason: LintFixSkipReason
  detail: string
  path?: Path
  paths?: readonly Path[]
}

export type LintFixPlan = {
  actions: readonly LintFixAction[]
  skipped: readonly LintFixSkippedItem[]
  summary: {
    stubRehydrates: number
    duplicateGroups: number
    skipped: number
  }
}

export type LintFixBuildOptions = {
  maxStubRehydrates?: number
  maxDuplicateGroups?: number
}

export type LintFixApplyOptions = {
  dryRun?: boolean
  runCompile?: boolean
}

export type LintFixResult = {
  dryRun: boolean
  planned: readonly LintFixAction[]
  applied: readonly LintFixAction[]
  skipped: readonly LintFixSkippedItem[]
  reopenedSources: readonly Path[]
  clearedMarkers: readonly Path[]
  archivedDuplicates: readonly Path[]
  compileTriggered: boolean
  compileResult?: CompileResult
}

export type DedupSuggestion = {
  keep: Path
  merge: readonly Path[]
  score: number
  reason: 'title' | 'content_hash' | 'both'
}

export type DedupReport = {
  suggestions: readonly DedupSuggestion[]
}

export type ScrubPattern = {
  name: string
  pattern: RegExp
  replacement: string
}

export type ScrubResult = {
  path: Path
  before: string
  after: string
  matches: readonly { name: string; count: number }[]
}

export type IngestOptions = {
  /** Optional filename (stem). Defaults to the content hash. */
  name?: string
  /** Operation log title. Defaults to the hash. */
  logTitle?: string
  /** Operation log detail. Defaults to "ingested N bytes". */
  logDetail?: string
}

export type CompileOptions = {
  /** Cap on articles planned per run. Defaults to 50. */
  maxArticles?: number
  /** Override the model. Provider default is used when unset. */
  model?: string
}

export type PromoteOptions = {
  /** Override the operation log detail. */
  detail?: string
}

export type ScrubOptions = {
  /** Whether the default PII patterns (email, phone) should run. Default true. */
  useDefaults?: boolean
  /** Extra patterns applied on top of the defaults. */
  patterns?: readonly ScrubPattern[]
  /** Only report; never mutate. */
  dryRun?: boolean
}
