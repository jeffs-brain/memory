// SPDX-License-Identifier: Apache-2.0

import { type Logger, type Provider, extractJSON } from '../llm/index.js'
import { type Path, type Store, joinPath, pathUnder, toPath } from '../store/index.js'
import { RAW_DOCUMENTS_ARCHIVE_PREFIX, archivedSourcePath } from './archive.js'
import { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'
import { RAW_DOCUMENTS_PREFIX, hashContent } from './ingest.js'
import { appendLogInBatch } from './log.js'
import { tryNormaliseKnowledgeArticleStem, tryNormaliseWikiRelativeArticlePath } from './paths.js'
import {
  isProcessedSourceContent,
  processedMarkerPath,
  readProcessedMarker,
  serialiseProcessedMarker,
} from './processed.js'
import { WIKI_PREFIX } from './promote.js'
import type {
  CompileOptions,
  CompilePlan,
  CompilePlanArticle,
  CompilePlanCrossReference,
  CompilePlanUpdate,
  CompileResult,
  Frontmatter,
} from './types.js'

export const DRAFTS_PREFIX = 'drafts'

const DEFAULT_MAX_ARTICLES = 50

const PLAN_SYSTEM = `You are a knowledge curator. Given the current ingested notes and the existing wiki index, decide what to create and what to update.

Respond with ONLY a JSON object in this shape:
{
  "new_articles": [
    {
      "slug": "kebab-case-slug",
      "title": "Human-readable title",
      "summary": "One-line summary",
      "source_hashes": ["<hash>", "<hash>"]
    }
  ],
  "updates": [
    {
      "path": "existing/article.md",
      "reason": "Why this article should be updated",
      "source_hashes": ["<hash>"]
    }
  ],
  "cross_references": [
    {
      "path": "existing/article.md",
      "reason": "Why this article should gain a lightweight link",
      "source_hashes": ["<hash>"]
    }
  ],
  "concepts": ["concept1", "concept2"],
  "processed_hashes": ["<hash>", "<hash>"]
}

Rules:
- Each ingested hash must appear in at most one new article.
- Slugs must be unique, lowercase, kebab-case.
- Prefer updates for existing wiki articles rather than creating duplicates.
- Include source_hashes for updates and cross references when specific notes drove that change.
- processed_hashes should include every ingested hash that this compile run fully handled, including notes that led to no write because the wiki already covers them.
- Produce at most {MAX} new articles.`

const CREATE_SYSTEM = `You are a technical writer. Produce the body of a single wiki article from the provided sources.

Respond with ONLY a JSON object:
{
  "title": "Final title",
  "summary": "One-line summary",
  "tags": ["tag1", "tag2"],
  "body": "Markdown body, no frontmatter"
}`

const UPDATE_SYSTEM = `You are a knowledge base editor. You are integrating new information into an existing wiki article.

Respond with ONLY a JSON object:
{
  "title": "Final title",
  "summary": "One-line summary",
  "tags": ["tag1", "tag2"],
  "body": "Markdown body, no frontmatter"
}

Rules:
- Preserve useful existing content unless the new sources clearly supersede it.
- Add or keep wikilinks where relevant.
- Use British English and no em dashes.
- Keep the body under 3000 words.
- Do not include YAML frontmatter.`

const CROSSREF_SYSTEM = `You are a knowledge base editor adding lightweight cross-references between wiki articles.

Respond with ONLY a JSON object:
{
  "title": "Final title",
  "summary": "One-line summary",
  "tags": ["tag1", "tag2"],
  "body": "Markdown body, no frontmatter"
}

Rules:
- Keep additions minimal. Add only the smallest useful contextual sentence or clause.
- Do not remove or rewrite existing information unless a tiny adjustment is required to fit the new link naturally.
- Add or keep wikilinks where relevant.
- Use British English and no em dashes.
- Keep the body under 3000 words.
- Do not include YAML frontmatter.`

type CompileDeps = {
  store: Store
  provider: Provider
  logger: Logger
}

type RawPlanArticle = {
  slug?: string
  title?: string
  summary?: string
  source_hashes?: readonly string[]
  sourceHashes?: readonly string[]
}

type RawPlanUpdate = {
  path?: string
  reason?: string
  source_hashes?: readonly string[]
  sourceHashes?: readonly string[]
}

type RawPlanCrossReference = {
  path?: string
  reason?: string
  source_hashes?: readonly string[]
  sourceHashes?: readonly string[]
}

type RawPlan = {
  new_articles?: readonly RawPlanArticle[]
  newArticles?: readonly RawPlanArticle[]
  articles?: readonly RawPlanArticle[]
  updates?: readonly RawPlanUpdate[]
  cross_references?: readonly RawPlanCrossReference[]
  crossReferences?: readonly RawPlanCrossReference[]
  concepts?: readonly string[]
  processed_hashes?: readonly string[]
  processedHashes?: readonly string[]
}

type RawWrite = {
  title?: string
  summary?: string
  tags?: readonly string[]
  body?: string
}

type WikiIndexArticle = {
  path: string
  title: string
  summary: string
  tags: readonly string[]
  body: string
}

type PendingWrite = {
  path: Path
  body: string
  kind: 'compile.write' | 'compile.update' | 'compile.crossref'
  title: string
  detail: string
}

type PendingArticleRewrite = {
  path: Path
  existing: {
    frontmatter: Frontmatter
    body: string
  }
  mode: 'crossref' | 'update'
  reason: string
  sourceHashes: readonly string[]
}

type PendingProcessedSource = {
  sourceId: string
  sourcePath: Path
  contentHash: string
  rawContent: Buffer
}

type IngestedSource = PendingProcessedSource & {
  body: string
}

export const createCompile = (deps: CompileDeps) => {
  const { store, provider, logger } = deps

  return async (opts: CompileOptions = {}): Promise<CompileResult> => {
    const max = opts.maxArticles ?? DEFAULT_MAX_ARTICLES
    const ingested = await listPendingIngestedSources(store)
    if (ingested.length === 0) {
      logger.debug('compile: no ingested entries')
      const emptyPlan: CompilePlan = {
        articles: [],
        newArticles: [],
        updates: [],
        crossReferences: [],
        concepts: [],
        processedSources: [],
      }
      return { plan: emptyPlan, written: [] }
    }

    const ingestedById = new Map(ingested.map((source) => [source.sourceId, source]))
    const wikiIndex = await listWikiIndex(store)
    const planPrompt = buildPlanPrompt(ingested, wikiIndex)
    const planRaw = await provider.complete({
      ...(opts.model !== undefined ? { model: opts.model } : {}),
      system: PLAN_SYSTEM.replace('{MAX}', String(max)),
      messages: [{ role: 'user', content: planPrompt }],
      jsonMode: true,
      temperature: 0.1,
      maxTokens: 4096,
    })
    const plan = parsePlan(
      planRaw.content,
      max,
      ingested.map((source) => source.sourceId),
    )
    const pendingWrites: PendingWrite[] = []
    const pendingArticleRewrites = new Map<Path, PendingArticleRewrite>()

    const queueArticleRewrite = async (
      planned: CompilePlanUpdate | CompilePlanCrossReference,
      mode: 'crossref' | 'update',
    ): Promise<void> => {
      const articlePath = normaliseWikiRelativePath(planned.path)
      if (articlePath === '') return

      const existing = await readWikiArticle(store, articlePath)
      if (existing === undefined) return

      const path = joinPath(WIKI_PREFIX, articlePath)
      const current = pendingArticleRewrites.get(path)
      if (current === undefined) {
        pendingArticleRewrites.set(path, {
          path,
          existing,
          mode,
          reason: planned.reason,
          sourceHashes: planned.sourceHashes,
        })
        return
      }

      if (mode === 'update') {
        current.mode = 'update'
      }
      current.reason = mergeReasons(current.reason, planned.reason)
      current.sourceHashes = mergeStrings(current.sourceHashes, planned.sourceHashes)
    }

    for (const article of plan.newArticles) {
      const existingWikiPath = `${article.slug}.md`
      if (await readWikiArticle(store, existingWikiPath)) {
        await queueArticleRewrite(
          {
            path: existingWikiPath,
            reason: [
              `A planned new article already exists at ${existingWikiPath}.`,
              'Merge the new source material into the existing article instead of creating a duplicate draft.',
              `Planned title: ${article.title}`,
            ].join('\n'),
            sourceHashes: article.sourceHashes,
          },
          'update',
        )
        continue
      }
      const sourceBodies = article.sourceHashes.map(
        (sourceId) => ingestedById.get(sourceId)?.body ?? '',
      )
      const writePrompt = buildCreatePrompt(article, sourceBodies)
      const resp = await provider.complete({
        ...(opts.model !== undefined ? { model: opts.model } : {}),
        system: CREATE_SYSTEM,
        messages: [{ role: 'user', content: writePrompt }],
        jsonMode: true,
        temperature: 0.2,
        maxTokens: 4096,
      })
      const parsed = parseWrite(resp.content)
      const fm: Frontmatter = {
        title: parsed.title || article.title,
        summary: parsed.summary || article.summary,
        tags: parsed.tags,
        sources: article.sourceHashes.map((sourceId) => archivedSourcePath(sourceId)),
        created: new Date().toISOString(),
        modified: new Date().toISOString(),
      }
      pendingWrites.push({
        path: joinPath(DRAFTS_PREFIX, `${article.slug}.md`),
        body: serialiseFrontmatter(fm, parsed.body),
        kind: 'compile.write',
        title: article.title,
        detail: `wrote ${joinPath(DRAFTS_PREFIX, `${article.slug}.md`)}`,
      })
    }

    for (const planned of plan.updates) {
      await queueArticleRewrite(planned, 'update')
    }

    for (const planned of plan.crossReferences) {
      await queueArticleRewrite(planned, 'crossref')
    }

    const ingestedBodies = new Map(ingested.map((source) => [source.sourceId, source.body]))
    for (const rewrite of pendingArticleRewrites.values()) {
      const sourceIds =
        rewrite.sourceHashes.length > 0
          ? rewrite.sourceHashes
          : ingested.map((source) => source.sourceId)
      const sourceBodies = sourceIds.map((sourceId) => ingestedBodies.get(sourceId) ?? '')
      const prompt =
        rewrite.mode === 'crossref'
          ? buildCrossReferencePrompt(rewrite.existing, rewrite.reason, sourceBodies)
          : buildUpdatePrompt(rewrite.existing, rewrite.reason, sourceBodies)
      const resp = await provider.complete({
        ...(opts.model !== undefined ? { model: opts.model } : {}),
        system: rewrite.mode === 'crossref' ? CROSSREF_SYSTEM : UPDATE_SYSTEM,
        messages: [{ role: 'user', content: prompt }],
        jsonMode: true,
        temperature: 0.2,
        maxTokens: 4096,
      })
      const parsed = parseWrite(resp.content)
      const fm: Frontmatter = {
        title: parsed.title || rewrite.existing.frontmatter.title,
        summary: parsed.summary || rewrite.existing.frontmatter.summary,
        tags: parsed.tags.length > 0 ? parsed.tags : rewrite.existing.frontmatter.tags,
        sources: mergeStrings(
          rewrite.existing.frontmatter.sources,
          sourceIds.map((sourceId) => archivedSourcePath(sourceId)),
        ),
        ...(rewrite.existing.frontmatter.created !== undefined
          ? { created: rewrite.existing.frontmatter.created }
          : {}),
        modified: new Date().toISOString(),
        ...(rewrite.existing.frontmatter.archived !== undefined
          ? { archived: rewrite.existing.frontmatter.archived }
          : {}),
        ...(rewrite.existing.frontmatter.supersededBy !== undefined
          ? { supersededBy: rewrite.existing.frontmatter.supersededBy }
          : {}),
      }
      pendingWrites.push({
        path: rewrite.path,
        body: serialiseFrontmatter(fm, parsed.body || rewrite.existing.body),
        kind: rewrite.mode === 'crossref' ? 'compile.crossref' : 'compile.update',
        title: parsed.title || rewrite.existing.frontmatter.title,
        detail:
          rewrite.mode === 'crossref'
            ? `cross-referenced ${rewrite.path}`
            : `updated ${rewrite.path}`,
      })
    }

    await assertNoPendingPathConflicts(store, pendingWrites)

    const pendingProcessedSources = plan.processedSources.flatMap((sourceId) => {
      const source = ingestedById.get(sourceId)
      return source === undefined ? [] : [source]
    })
    const written: Path[] = []
    await store.batch({ reason: 'compile' }, async (batch) => {
      await appendLogInBatch(batch, {
        kind: 'compile.plan',
        title: `${plan.newArticles.length} articles planned`,
        detail: [
          `new: ${plan.newArticles.length}`,
          `updates: ${plan.updates.length}`,
          `cross refs: ${plan.crossReferences.length}`,
          `processed: ${pendingProcessedSources.length}`,
        ].join('\n'),
        when: new Date().toISOString(),
      })

      for (const w of pendingWrites) {
        await batch.write(w.path, Buffer.from(w.body, 'utf8'))
        written.push(w.path)
        await appendLogInBatch(batch, {
          kind: w.kind,
          title: w.title,
          detail: w.detail,
          when: new Date().toISOString(),
        })
      }

      const processedAt = new Date().toISOString()
      for (const source of pendingProcessedSources) {
        await batch.write(archivedSourcePath(source.sourceId), source.rawContent)
        const marker = serialiseProcessedMarker({
          sourcePath: source.sourcePath,
          contentHash: source.contentHash,
          processedAt,
          writtenPaths: written,
        })
        await batch.write(processedMarkerPath(source.sourceId), Buffer.from(marker, 'utf8'))
      }
    })

    logger.info('compile complete', {
      planned: plan.newArticles.length + plan.updates.length + plan.crossReferences.length,
      written: written.length,
      processed: pendingProcessedSources.length,
    })
    return { plan, written }
  }
}

export const listIngested = async (store: Store): Promise<readonly string[]> => {
  const pending = await listPendingIngestedSources(store)
  return pending.map((source) => source.sourceId)
}

const listPendingIngestedSources = async (store: Store): Promise<readonly IngestedSource[]> => {
  const prefix = toPath(RAW_DOCUMENTS_PREFIX)
  const exists = await store.exists(prefix).catch(() => false)
  if (!exists) return []
  const entries = await store.list(prefix, { recursive: true })
  const sources: IngestedSource[] = []
  for (const e of entries) {
    if (e.isDir) continue
    if (!pathUnder(e.path, RAW_DOCUMENTS_PREFIX, true)) continue
    if (pathUnder(e.path, RAW_DOCUMENTS_ARCHIVE_PREFIX, true)) continue
    if (!e.path.endsWith('.md')) continue
    const sourceId = e.path.slice(RAW_DOCUMENTS_PREFIX.length + 1, -'.md'.length)
    const content = await store.read(e.path)
    const marker = await readProcessedMarker(store, sourceId)
    if (isProcessedSourceContent(marker, e.path, content)) {
      continue
    }
    sources.push({
      sourceId,
      sourcePath: e.path,
      contentHash: hashContent(content),
      rawContent: Buffer.from(content),
      body: content.toString('utf8'),
    })
  }
  sources.sort((left, right) => left.sourceId.localeCompare(right.sourceId))
  return sources
}

const listWikiIndex = async (store: Store): Promise<readonly WikiIndexArticle[]> => {
  const prefix = toPath(WIKI_PREFIX)
  const exists = await store.exists(prefix).catch(() => false)
  if (!exists) return []
  const entries = await store.list(prefix, { recursive: true })
  const out: WikiIndexArticle[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!pathUnder(entry.path, WIKI_PREFIX, true)) continue
    if (!entry.path.endsWith('.md')) continue
    const raw = (await store.read(entry.path)).toString('utf8')
    const parsed = parseFrontmatter(raw)
    out.push({
      path: entry.path,
      title: parsed.frontmatter.title || lastSegment(entry.path).replace(/\.md$/, ''),
      summary: parsed.frontmatter.summary,
      tags: parsed.frontmatter.tags,
      body: parsed.body,
    })
  }
  out.sort((a, b) => a.path.localeCompare(b.path))
  return out
}

const readWikiArticle = async (
  store: Store,
  relativePath: string,
): Promise<{ frontmatter: Frontmatter; body: string } | undefined> => {
  const fullPath = joinPath(WIKI_PREFIX, relativePath)
  if (!(await store.exists(fullPath))) return undefined
  const content = (await store.read(fullPath)).toString('utf8')
  const parsed = parseFrontmatter(content)
  return {
    frontmatter: parsed.frontmatter,
    body: parsed.body,
  }
}

const buildPlanPrompt = (
  ingested: readonly IngestedSource[],
  wikiIndex: readonly WikiIndexArticle[],
): string => {
  const parts: string[] = ['Existing wiki index:', '']
  if (wikiIndex.length === 0) {
    parts.push('(empty)', '')
  } else {
    for (const article of wikiIndex) {
      parts.push(
        `### ${article.path}`,
        `Title: ${article.title}`,
        `Summary: ${article.summary}`,
        article.tags.length > 0 ? `Tags: ${article.tags.join(', ')}` : 'Tags: []',
        article.body.length > 1200
          ? `${article.body.slice(0, 1200)}\n[...truncated]`
          : article.body,
        '',
      )
    }
  }

  parts.push('Ingested notes:', '')
  for (const source of ingested) {
    const content = source.body
    const snippet = content.length > 2000 ? `${content.slice(0, 2000)}\n[...truncated]` : content
    parts.push(`### ${source.sourceId}`, snippet, '')
  }
  return parts.join('\n')
}

const buildCreatePrompt = (
  article: CompilePlanArticle,
  sourceBodies: readonly string[],
): string => {
  const parts: string[] = [
    `Title: ${article.title}`,
    `Summary: ${article.summary}`,
    '',
    'Sources:',
    '',
  ]
  for (let i = 0; i < sourceBodies.length; i++) {
    parts.push(`### ${article.sourceHashes[i] ?? ''}`, sourceBodies[i] ?? '', '')
  }
  return parts.join('\n')
}

const buildUpdatePrompt = (
  existing: { frontmatter: Frontmatter; body: string },
  reason: string,
  sourceBodies: readonly string[],
): string => {
  const parts: string[] = [
    'Existing article:',
    `Title: ${existing.frontmatter.title}`,
    `Summary: ${existing.frontmatter.summary}`,
    `Tags: ${existing.frontmatter.tags.join(', ')}`,
    '',
    existing.body,
    '',
    'Reason for update:',
    reason,
    '',
    'Sources:',
    '',
  ]
  for (let i = 0; i < sourceBodies.length; i++) {
    parts.push(`### source-${i + 1}`, sourceBodies[i] ?? '', '')
  }
  return parts.join('\n')
}

const buildCrossReferencePrompt = (
  existing: { frontmatter: Frontmatter; body: string },
  reason: string,
  sourceBodies: readonly string[],
): string => {
  const parts: string[] = [
    'Existing article:',
    `Title: ${existing.frontmatter.title}`,
    `Summary: ${existing.frontmatter.summary}`,
    `Tags: ${existing.frontmatter.tags.join(', ')}`,
    '',
    existing.body,
    '',
    'Cross-reference goal:',
    reason,
    '',
    'Recent source material:',
    '',
  ]
  for (let i = 0; i < sourceBodies.length; i++) {
    parts.push(`### source-${i + 1}`, sourceBodies[i] ?? '', '')
  }
  return parts.join('\n')
}

const parsePlan = (
  content: string,
  max: number,
  availableSourceIds: readonly string[],
): CompilePlan => {
  let raw: RawPlan
  try {
    raw = JSON.parse(extractJSON(content)) as RawPlan
  } catch {
    return {
      articles: [],
      newArticles: [],
      updates: [],
      crossReferences: [],
      concepts: [],
      processedSources: [],
    }
  }

  const newArticles = parseCreateList(raw, max)
  const updates = parseUpdateList(raw, max)
  const crossReferences = parseCrossReferenceList(raw, max)
  const concepts = parseConcepts(raw)
  const processedSources = parseProcessedSources(raw, availableSourceIds, {
    newArticles,
    updates,
    crossReferences,
  })

  return {
    articles: newArticles,
    newArticles,
    updates,
    crossReferences,
    concepts,
    processedSources,
  }
}

const parseCreateList = (raw: RawPlan, max: number): CompilePlanArticle[] => {
  const slugs = new Set<string>()
  const articles: CompilePlanArticle[] = []
  const source = raw.new_articles ?? raw.newArticles ?? raw.articles ?? []
  for (const a of source) {
    if (articles.length >= max) break
    const slug = normaliseSlug(a.slug ?? '')
    const title = typeof a.title === 'string' ? a.title.trim() : ''
    const summary = typeof a.summary === 'string' ? a.summary.trim() : ''
    const hashes = a.source_hashes ?? a.sourceHashes ?? []
    if (slug === '' || title === '' || slugs.has(slug)) continue
    const clean = uniqueStrings(
      hashes
        .filter((h): h is string => typeof h === 'string' && h.trim() !== '')
        .map((h) => h.trim()),
    )
    if (clean.length === 0) continue
    slugs.add(slug)
    articles.push({ slug, title, summary, sourceHashes: clean })
  }
  return articles
}

const parseUpdateList = (raw: RawPlan, max: number): CompilePlanUpdate[] => {
  const paths = new Set<string>()
  const updates: CompilePlanUpdate[] = []
  for (const a of raw.updates ?? []) {
    if (updates.length >= max) break
    const path = normaliseWikiRelativePath(a.path ?? '')
    const reason = typeof a.reason === 'string' ? a.reason.trim() : ''
    if (path === '' || reason === '' || paths.has(path)) continue
    paths.add(path)
    updates.push({
      path,
      reason,
      sourceHashes: parseSourceIds(a.source_hashes ?? a.sourceHashes ?? []),
    })
  }
  return updates
}

const parseCrossReferenceList = (raw: RawPlan, max: number): CompilePlanCrossReference[] => {
  const paths = new Set<string>()
  const crossReferences: CompilePlanCrossReference[] = []
  const source = raw.cross_references ?? raw.crossReferences ?? []
  for (const a of source) {
    if (crossReferences.length >= max) break
    const path = normaliseWikiRelativePath(a.path ?? '')
    const reason = typeof a.reason === 'string' ? a.reason.trim() : ''
    if (path === '' || reason === '' || paths.has(path)) continue
    paths.add(path)
    crossReferences.push({
      path,
      reason,
      sourceHashes: parseSourceIds(a.source_hashes ?? a.sourceHashes ?? []),
    })
  }
  return crossReferences
}

const parseProcessedSources = (
  raw: RawPlan,
  availableSourceIds: readonly string[],
  plan: {
    readonly newArticles: readonly CompilePlanArticle[]
    readonly updates: readonly CompilePlanUpdate[]
    readonly crossReferences: readonly CompilePlanCrossReference[]
  },
): readonly string[] => {
  const available = new Set(availableSourceIds)
  const explicit = parseSourceIds(raw.processed_hashes ?? raw.processedHashes ?? []).filter(
    (sourceId) => available.has(sourceId),
  )
  const inferred = uniqueStrings([
    ...plan.newArticles.flatMap((article) => article.sourceHashes),
    ...plan.updates.flatMap((update) =>
      update.sourceHashes.length > 0 ? update.sourceHashes : availableSourceIds,
    ),
    ...plan.crossReferences.flatMap((crossReference) =>
      crossReference.sourceHashes.length > 0 ? crossReference.sourceHashes : availableSourceIds,
    ),
  ]).filter((sourceId) => available.has(sourceId))
  return uniqueStrings([...explicit, ...inferred])
}

const parseConcepts = (raw: RawPlan): readonly string[] => {
  const seen = new Set<string>()
  const concepts: string[] = []
  for (const concept of raw.concepts ?? []) {
    if (typeof concept !== 'string') continue
    const clean = concept.trim()
    if (clean === '' || seen.has(clean)) continue
    seen.add(clean)
    concepts.push(clean)
  }
  return concepts
}

const parseWrite = (
  content: string,
): { title: string; summary: string; tags: readonly string[]; body: string } => {
  let raw: RawWrite
  try {
    raw = JSON.parse(extractJSON(content)) as RawWrite
  } catch {
    return { title: '', summary: '', tags: [], body: content.trim() }
  }
  const tags = Array.isArray(raw.tags)
    ? raw.tags
        .filter((t): t is string => typeof t === 'string' && t.trim() !== '')
        .map((t) => t.trim())
    : []
  return {
    title: (raw.title ?? '').trim(),
    summary: (raw.summary ?? '').trim(),
    tags,
    body: (raw.body ?? '').trim(),
  }
}

const normaliseSlug = (value: string): string => {
  return tryNormaliseKnowledgeArticleStem(value) ?? ''
}

const normaliseWikiRelativePath = (value: string): string => {
  return tryNormaliseWikiRelativeArticlePath(value) ?? ''
}

const parseSourceIds = (values: readonly string[]): readonly string[] =>
  uniqueStrings(
    values
      .filter((value): value is string => typeof value === 'string' && value.trim() !== '')
      .map((value) => value.trim()),
  )

const uniqueStrings = (values: readonly string[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    if (seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
}

const mergeStrings = (first: readonly string[], second: readonly string[]): string[] => {
  return uniqueStrings([...first, ...second])
}

const mergeReasons = (current: string, next: string): string => {
  if (current === next) return current
  if (current.includes(next)) return current
  if (next.includes(current)) return next
  return `${current}\n\nAdditional context:\n${next}`
}

const lastSegment = (value: string): string => {
  const idx = value.lastIndexOf('/')
  return idx >= 0 ? value.slice(idx + 1) : value
}

const assertNoPendingPathConflicts = async (
  store: Store,
  pendingWrites: readonly PendingWrite[],
): Promise<void> => {
  const seen = new Set<string>()
  const pendingPaths = new Set(pendingWrites.map((write) => String(write.path)))
  const conflicts = new Set<string>()

  for (const write of pendingWrites) {
    const target = String(write.path)
    if (seen.has(target)) {
      conflicts.add(`duplicate compile target: ${target}`)
      continue
    }
    seen.add(target)
  }

  for (const target of pendingPaths) {
    for (const blocker of blockingMarkdownAncestors(target)) {
      if (pendingPaths.has(blocker)) {
        conflicts.add(`${blocker} blocks ${target}`)
        continue
      }
      if (await store.exists(toPath(blocker)).catch(() => false)) {
        conflicts.add(`${blocker} blocks ${target}`)
      }
    }
  }

  if (conflicts.size > 0) {
    const detail = [...conflicts].sort().join('\n- ')
    throw new Error(`compile: path conflicts detected:\n- ${detail}`)
  }
}

const blockingMarkdownAncestors = (path: string): readonly string[] => {
  if (!path.endsWith('.md')) return []
  const stem = path.slice(0, -'.md'.length)
  const segments = stem.split('/')
  if (segments.length < 3) return []
  const blockers: string[] = []
  for (let index = 1; index < segments.length - 1; index++) {
    blockers.push(`${segments.slice(0, index + 1).join('/')}.md`)
  }
  return blockers
}
