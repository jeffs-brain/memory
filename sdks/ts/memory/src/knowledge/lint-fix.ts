// SPDX-License-Identifier: Apache-2.0

import { pathUnder, type Path, type Store } from '../store/index.js'
import { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'
import { INGESTED_PREFIX } from './ingest.js'
import { appendLogInBatch } from './log.js'
import { processedMarkerPath, readProcessedMarker } from './processed.js'
import type {
  CompileResult,
  LintFixAction,
  LintFixApplyOptions,
  LintFixBuildOptions,
  LintFixPlan,
  LintFixResult,
  LintFixSkippedItem,
  LintReport,
} from './types.js'

const DEFAULT_MAX_STUB_REHYDRATES = 20
const DEFAULT_MAX_DUPLICATE_GROUPS = 10
const ARCHIVED_DUPLICATE_SUFFIX = ' (archived duplicate)'

type LintFixDeps = {
  store: Store
  compile?: () => Promise<CompileResult>
}

type LoadedArticle = {
  path: Path
  present: boolean
  frontmatter: ReturnType<typeof parseFrontmatter>['frontmatter']
  body: string
}

export const createLintFix = (deps: LintFixDeps) => {
  const { store, compile } = deps

  return {
    buildPlan: async (report: LintReport, opts: LintFixBuildOptions = {}): Promise<LintFixPlan> => {
      const skipped: LintFixSkippedItem[] = []
      const actions: LintFixAction[] = []

      actions.push(
        ...(await buildStubRehydrateActions({
          store,
          report,
          skipped,
          maxActions: clampPositive(opts.maxStubRehydrates, DEFAULT_MAX_STUB_REHYDRATES),
        })),
      )
      actions.push(
        ...(await buildDuplicateArchiveActions({
          store,
          report,
          skipped,
          maxActions: clampPositive(opts.maxDuplicateGroups, DEFAULT_MAX_DUPLICATE_GROUPS),
        })),
      )

      return {
        actions,
        skipped,
        summary: {
          stubRehydrates: actions.filter((action) => action.kind === 'rehydrate_stub').length,
          duplicateGroups: actions.filter((action) => action.kind === 'archive_duplicate_title').length,
          skipped: skipped.length,
        },
      }
    },

    applyPlan: async (
      plan: LintFixPlan,
      opts: LintFixApplyOptions = {},
    ): Promise<LintFixResult> => {
      const dryRun = opts.dryRun === true
      if (dryRun || plan.actions.length === 0) {
        return {
          dryRun,
          planned: plan.actions,
          applied: [],
          skipped: plan.skipped,
          reopenedSources: [],
          clearedMarkers: [],
          archivedDuplicates: [],
          compileTriggered: false,
        }
      }

      const runtimeSkipped = [...plan.skipped]
      const reopenedSources = new Set<Path>()
      const clearedMarkers = new Set<Path>()
      const archivedDuplicates = new Set<Path>()
      const applied: LintFixAction[] = []

      await store.batch({ reason: 'lint-fix' }, async (batch) => {
        for (const action of plan.actions) {
          if (action.kind === 'rehydrate_stub') {
            let clearedAny = false
            for (const markerPath of action.markerPaths) {
              const exists = await batch.exists(markerPath)
              if (!exists) continue
              await batch.delete(markerPath)
              clearedMarkers.add(markerPath)
              clearedAny = true
            }
            if (!clearedAny) {
              runtimeSkipped.push({
                kind: 'rehydrate_stub',
                reason: 'already_pending',
                detail: `all processed markers were already absent for ${action.articlePath}`,
                path: action.articlePath,
              })
              continue
            }
            for (const sourcePath of action.sourcePaths) reopenedSources.add(sourcePath)
            applied.push(action)
            continue
          }

          let archivedAny = false
          for (const archivePath of action.archivePaths) {
            const article = await readArticleFromBatch(batch, archivePath)
            if (article === undefined || !article.present) {
              runtimeSkipped.push({
                kind: 'archive_duplicate_title',
                reason: 'unsafe_duplicate_group',
                detail: `skipped duplicate archive for ${archivePath} because the article is missing or lacks frontmatter`,
                path: archivePath,
              })
              continue
            }

            const nextFrontmatter = {
              ...article.frontmatter,
              title: archiveDuplicateTitle(article.frontmatter.title),
              archived: true,
              supersededBy: action.canonicalPath,
              modified: new Date().toISOString(),
            }
            const nextBody = prefixArchivedDuplicateBody(article.body, action.canonicalPath)
            await batch.write(
              archivePath,
              Buffer.from(serialiseFrontmatter(nextFrontmatter, nextBody), 'utf8'),
            )
            archivedDuplicates.add(archivePath)
            archivedAny = true
          }

          if (archivedAny) {
            applied.push(action)
            continue
          }

          runtimeSkipped.push({
            kind: 'archive_duplicate_title',
            reason: 'no_active_duplicates',
            detail: `no duplicate articles remained to archive for title group ${action.titleKey}`,
            paths: [action.canonicalPath, ...action.archivePaths],
          })
        }

        if (applied.length === 0) return

        await appendLogInBatch(batch, {
          kind: 'lint.fix',
          title: 'Applied lint fix plan',
          detail: formatLintFixLogDetail({
            reopenedSources,
            clearedMarkers,
            archivedDuplicates,
            duplicateGroups: applied.filter((action) => action.kind === 'archive_duplicate_title').length,
          }),
          when: new Date().toISOString(),
        })
      })

      let compileResult: CompileResult | undefined
      const shouldRunCompile =
        reopenedSources.size > 0 && opts.runCompile !== false && compile !== undefined
      if (shouldRunCompile) {
        compileResult = await compile()
      }

      return {
        dryRun: false,
        planned: plan.actions,
        applied,
        skipped: runtimeSkipped,
        reopenedSources: sortPaths(reopenedSources),
        clearedMarkers: sortPaths(clearedMarkers),
        archivedDuplicates: sortPaths(archivedDuplicates),
        compileTriggered: compileResult !== undefined,
        ...(compileResult !== undefined ? { compileResult } : {}),
      }
    },
  }
}

const buildStubRehydrateActions = async (input: {
  store: Store
  report: LintReport
  skipped: LintFixSkippedItem[]
  maxActions: number
}): Promise<LintFixAction[]> => {
  const paths = uniquePaths(
    input.report.issues
      .filter((issue) => issue.kind === 'stub_article')
      .map((issue) => issue.path)
      .sort(),
  )
  const actions: LintFixAction[] = []

  for (const articlePath of paths) {
    if (actions.length >= input.maxActions) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'limit_reached',
        detail: `stub rehydrate limit reached for ${articlePath}`,
        path: articlePath,
      })
      continue
    }

    const article = await readArticle(input.store, articlePath)
    if (article === undefined) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'missing_article',
        detail: `stub article no longer exists: ${articlePath}`,
        path: articlePath,
      })
      continue
    }
    if (!article.present) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'no_rehydratable_sources',
        detail: `stub article has no frontmatter to inspect for sources: ${articlePath}`,
        path: articlePath,
      })
      continue
    }
    if (isSupersededArticle(article.frontmatter)) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'already_superseded',
        detail: `stub article is already archived or superseded: ${articlePath}`,
        path: articlePath,
      })
      continue
    }

    const sourcePaths = uniquePaths(
      article.frontmatter.sources
        .filter((sourcePath): sourcePath is Path => isIngestedMarkdownPath(sourcePath))
        .sort(),
    )
    if (sourcePaths.length === 0) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'no_rehydratable_sources',
        detail: `stub article has no ingested sources to reopen: ${articlePath}`,
        path: articlePath,
      })
      continue
    }

    const availableSources: Path[] = []
    const markerPaths: Path[] = []
    for (const sourcePath of sourcePaths) {
      const exists = await input.store.exists(sourcePath).catch(() => false)
      if (!exists) continue
      availableSources.push(sourcePath)
      const sourceId = sourceIdFromIngestedPath(sourcePath)
      if (sourceId === undefined) continue
      const marker = await readProcessedMarker(input.store, sourceId)
      if (marker !== undefined) markerPaths.push(processedMarkerPath(sourceId))
    }

    if (availableSources.length === 0) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'no_rehydratable_sources',
        detail: `stub article sources no longer exist in ingested storage: ${articlePath}`,
        path: articlePath,
      })
      continue
    }
    if (markerPaths.length === 0) {
      input.skipped.push({
        kind: 'rehydrate_stub',
        reason: 'already_pending',
        detail: `stub article sources are already pending compile: ${articlePath}`,
        path: articlePath,
      })
      continue
    }

    actions.push({
      kind: 'rehydrate_stub',
      articlePath,
      sourcePaths: availableSources,
      markerPaths,
    })
  }

  return actions
}

const buildDuplicateArchiveActions = async (input: {
  store: Store
  report: LintReport
  skipped: LintFixSkippedItem[]
  maxActions: number
}): Promise<LintFixAction[]> => {
  const groups = new Map<string, Set<Path>>()
  for (const issue of input.report.issues) {
    if (issue.kind !== 'duplicate_title') continue
    let titleKey = issue.details?.titleKey?.trim() ?? ''
    if (titleKey === '') {
      const article = await readArticle(input.store, issue.path)
      titleKey = article === undefined ? '' : normaliseTitleForGrouping(article.frontmatter.title)
    }
    if (titleKey === '') continue
    const group = groups.get(titleKey) ?? new Set<Path>()
    group.add(issue.path)
    for (const relatedPath of issue.details?.relatedPaths ?? []) group.add(relatedPath)
    groups.set(titleKey, group)
  }

  const actions: LintFixAction[] = []
  for (const titleKey of [...groups.keys()].sort()) {
    const groupPaths = uniquePaths([...(groups.get(titleKey) ?? [])]).sort()
    if (actions.length >= input.maxActions) {
      input.skipped.push({
        kind: 'archive_duplicate_title',
        reason: 'limit_reached',
        detail: `duplicate archive limit reached for title group ${titleKey}`,
        paths: groupPaths,
      })
      continue
    }

    const articles: LoadedArticle[] = []
    let unsafeGroup = false
    for (const path of groupPaths) {
      const article = await readArticle(input.store, path)
      if (article === undefined || !article.present) {
        unsafeGroup = true
        break
      }
      articles.push(article)
    }
    if (unsafeGroup) {
      input.skipped.push({
        kind: 'archive_duplicate_title',
        reason: 'unsafe_duplicate_group',
        detail: `duplicate title group ${titleKey} could not be repaired safely`,
        paths: groupPaths,
      })
      continue
    }

    const activeArticles = articles.filter((article) => !isSupersededArticle(article.frontmatter))
    if (activeArticles.length < 2) {
      input.skipped.push({
        kind: 'archive_duplicate_title',
        reason: 'no_active_duplicates',
        detail: `duplicate title group ${titleKey} no longer has multiple active articles`,
        paths: groupPaths,
      })
      continue
    }

    const canonical = pickCanonicalArticle(activeArticles)
    const archivePaths = activeArticles
      .map((article) => article.path)
      .filter((path) => path !== canonical.path)

    if (archivePaths.length === 0) {
      input.skipped.push({
        kind: 'archive_duplicate_title',
        reason: 'no_active_duplicates',
        detail: `duplicate title group ${titleKey} had no remaining archive targets`,
        paths: groupPaths,
      })
      continue
    }

    actions.push({
      kind: 'archive_duplicate_title',
      titleKey,
      canonicalPath: canonical.path,
      archivePaths,
    })
  }

  return actions
}

const readArticle = async (store: Store, path: Path): Promise<LoadedArticle | undefined> => {
  const exists = await store.exists(path).catch(() => false)
  if (!exists) return undefined
  const raw = await store.read(path).catch(() => undefined)
  if (raw === undefined) return undefined
  const parsed = parseFrontmatter(raw.toString('utf8'))
  return {
    path,
    present: parsed.present,
    frontmatter: parsed.frontmatter,
    body: parsed.body,
  }
}

const readArticleFromBatch = async (
  batch: Pick<Store, 'read'>,
  path: Path,
): Promise<LoadedArticle | undefined> => {
  const raw = await batch.read(path).catch(() => undefined)
  if (raw === undefined) return undefined
  const parsed = parseFrontmatter(raw.toString('utf8'))
  return {
    path,
    present: parsed.present,
    frontmatter: parsed.frontmatter,
    body: parsed.body,
  }
}

const pickCanonicalArticle = (articles: readonly LoadedArticle[]): LoadedArticle => {
  return [...articles].sort((left, right) => {
    if (right.body.length !== left.body.length) return right.body.length - left.body.length
    return left.path.localeCompare(right.path)
  })[0]!
}

const isSupersededArticle = (frontmatter: LoadedArticle['frontmatter']): boolean =>
  frontmatter.archived === true ||
  (typeof frontmatter.supersededBy === 'string' && frontmatter.supersededBy.trim() !== '')

const isIngestedMarkdownPath = (path: string): path is Path =>
  pathUnder(path as Path, INGESTED_PREFIX, true) && path.endsWith('.md')

const sourceIdFromIngestedPath = (path: Path): string | undefined => {
  if (!isIngestedMarkdownPath(path)) return undefined
  const relative = path.slice(INGESTED_PREFIX.length + 1)
  if (relative.startsWith('_processed/')) return undefined
  return relative.slice(0, -'.md'.length)
}

const archiveDuplicateTitle = (title: string): string => {
  const trimmed = title.trim()
  if (trimmed === '') return `Archived duplicate${ARCHIVED_DUPLICATE_SUFFIX}`
  if (trimmed.toLowerCase().endsWith(ARCHIVED_DUPLICATE_SUFFIX)) return trimmed
  return `${trimmed}${ARCHIVED_DUPLICATE_SUFFIX}`
}

const prefixArchivedDuplicateBody = (body: string, canonicalPath: Path): string => {
  const target = wikilinkTarget(canonicalPath)
  const note = `This article is an archived duplicate of [[${target}]]. Consult [[${target}]] for the canonical entry.`
  const trimmed = body.trim()
  if (trimmed.startsWith(note)) return trimmed
  if (trimmed === '') return note
  return `${note}\n\n${trimmed}`
}

const wikilinkTarget = (path: Path): string => {
  if (!path.startsWith('wiki/') || !path.endsWith('.md')) return path
  return path.slice('wiki/'.length, -'.md'.length)
}

const normaliseTitleForGrouping = (title: string): string => {
  const source = title.trim().toLowerCase()
  if (source === '') return ''
  let out = ''
  let prevSpace = false
  for (const r of source) {
    const isAlnum = /[\p{L}\p{N}]/u.test(r)
    if (isAlnum) {
      out += r
      prevSpace = false
      continue
    }
    if (!prevSpace && out.length > 0) {
      out += ' '
      prevSpace = true
    }
  }
  return out.trim()
}

const uniquePaths = (paths: readonly Path[]): Path[] => [...new Set(paths)]

const sortPaths = (paths: ReadonlySet<Path>): readonly Path[] => [...paths].sort()

const clampPositive = (value: number | undefined, fallback: number): number => {
  if (value === undefined) return fallback
  if (!Number.isFinite(value) || value < 1) return fallback
  return Math.floor(value)
}

const formatLintFixLogDetail = (input: {
  reopenedSources: ReadonlySet<Path>
  clearedMarkers: ReadonlySet<Path>
  archivedDuplicates: ReadonlySet<Path>
  duplicateGroups: number
}): string =>
  [
    `Reopened sources: ${input.reopenedSources.size}`,
    `Cleared processed markers: ${input.clearedMarkers.size}`,
    `Archived duplicate articles: ${input.archivedDuplicates.size}`,
    `Duplicate groups repaired: ${input.duplicateGroups}`,
  ].join(', ')
