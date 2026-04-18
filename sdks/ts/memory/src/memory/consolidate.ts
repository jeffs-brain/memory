// SPDX-License-Identifier: Apache-2.0

/**
 * Consolidate stage. Walks the scope prefix, finds candidate duplicates
 * (Jaccard similarity over significant words from the filenames), asks
 * the provider to rule on each pair, then applies the verdict inside a
 * single Store batch with `Reason: 'consolidate'`.
 *
 * This stage also keeps the scope healthy by rebuilding `MEMORY.md`,
 * flagging stale notes, and rebalancing heuristic confidence from the
 * frontmatter history already present on note files.
 */

import type { Logger, Provider } from '../llm/index.js'
import { ErrNotFound } from '../store/errors.js'
import type { Batch, FileInfo, ListOpts, Store } from '../store/index.js'
import { lastSegment, type Path } from '../store/path.js'
import { buildFrontmatter, parseFrontmatter, type Frontmatter } from './frontmatter.js'
import { scopeIndex, scopePrefix } from './paths.js'
import { fireConsolidationEnd, fireConsolidationStart } from './plugins.js'
import { DEDUPLICATION_SYSTEM_PROMPT } from './prompts.js'
import type {
  ConsolidateArgs,
  ConsolidationOp,
  ConsolidationReport,
  Plugin,
  Scope,
} from './types.js'

const DEDUP_MAX_TOKENS = 512
const DEDUP_TEMPERATURE = 0.0
const JACCARD_CUTOFF = 0.3
const STALE_AFTER_DAYS = 90
const DEEP_STALE_AFTER_DAYS = 180
const HEURISTIC_REINFORCEMENT_DAYS = 14
const HEURISTIC_STRONG_REINFORCEMENT_DAYS = 45
const INDEX_ENTRY_LIMIT = 180
const HEURISTIC_TAG = 'heuristic'
const STALE_TAG = 'stale'
const CONFIDENCE_LEVELS = ['low', 'medium', 'high'] as const

type Confidence = (typeof CONFIDENCE_LEVELS)[number]

type StoreReader = {
  list(dir: Path | '', opts?: ListOpts): Promise<FileInfo[]>
  read(path: Path): Promise<Buffer>
}

type NoteRecord = {
  readonly path: Path
  readonly relativePath: string
  readonly content: string
  readonly body: string
  readonly frontmatter: Frontmatter
  readonly observedAt: Date
  readonly createdAt: Date
}

export type ConsolidateDeps = {
  readonly store: Store
  readonly provider: Provider
  readonly logger: Logger
  readonly plugins: readonly Plugin[]
  readonly defaultScope: Scope
  readonly defaultActorId: string
}

export const createConsolidate = (deps: ConsolidateDeps) => {
  return async (args: ConsolidateArgs = {}): Promise<ConsolidationReport> => {
    const scope = args.scope ?? deps.defaultScope
    const actorId = args.actorId ?? deps.defaultActorId
    const prefix = scopePrefix(scope, actorId)
    const indexPath = scopeIndex(scope, actorId)
    const now = new Date()

    await fireConsolidationStart(deps.plugins, { scope }, deps.logger)

    let topics: readonly Path[]
    try {
      topics = await listTopicPaths(deps.store, prefix)
    } catch (err) {
      if (err instanceof ErrNotFound) {
        const report: ConsolidationReport = {
          merged: 0,
          deleted: 0,
          promoted: 0,
          ops: [],
          errors: [],
        }
        await fireConsolidationEnd(deps.plugins, { scope, report }, deps.logger)
        return report
      }
      throw err
    }

    const ops: ConsolidationOp[] = []
    const errors: string[] = []
    let merged = 0
    let deleted = 0

    try {
      await deps.store.batch({ reason: 'consolidate' }, async (b) => {
        const pairs = candidatePairs(topics)
        for (const [aPath, bPath] of pairs) {
          // Skip if either was already removed in this pass.
          if (!(await b.exists(aPath)) || !(await b.exists(bPath))) continue
          const contentA = safeRead(await b.read(aPath))
          const contentB = safeRead(await b.read(bPath))
          if (contentA === undefined || contentB === undefined) continue
          let verdict: string
          try {
            verdict = await askVerdict(deps.provider, aPath, contentA, bPath, contentB)
          } catch (err) {
            errors.push(
              `dedup LLM failed for ${aPath} + ${bPath}: ${err instanceof Error ? err.message : String(err)}`,
            )
            continue
          }
          switch (verdict) {
            case 'keep_first':
              await b.delete(bPath)
              deleted++
              ops.push({ kind: 'delete', path: bPath, reason: 'keep_first' })
              break
            case 'keep_second':
              await b.delete(aPath)
              deleted++
              ops.push({ kind: 'delete', path: aPath, reason: 'keep_second' })
              break
            case 'merge': {
              const mergePlan = planMerge(aPath, contentA, bPath, contentB)
              const combined = mergeBodies(
                mergePlan.keeperContent,
                mergePlan.donorContent,
                mergePlan.donorPath,
              )
              await b.write(mergePlan.keeperPath, Buffer.from(combined, 'utf8'))
              await b.delete(mergePlan.donorPath)
              merged++
              ops.push({
                kind: 'merge',
                keeper: mergePlan.keeperPath,
                donor: mergePlan.donorPath,
              })
              break
            }
            default:
              // "distinct" or unknown → leave both in place.
              break
          }
        }

        const maintenance = await runMaintenancePass(b, {
          prefix,
          indexPath,
          now,
        })
        ops.push(...maintenance.ops)
        errors.push(...maintenance.errors)
      })
    } catch (err) {
      errors.push(
        `consolidate batch failed: ${err instanceof Error ? err.message : String(err)}`,
      )
    }

    const report: ConsolidationReport = {
      merged,
      deleted,
      promoted: 0,
      ops,
      errors,
    }
    await fireConsolidationEnd(deps.plugins, { scope, report }, deps.logger)
    return report
  }
}

const candidatePairs = (topics: readonly Path[]): Array<[Path, Path]> => {
  const pairs: Array<[Path, Path]> = []
  for (let i = 0; i < topics.length; i++) {
    for (let j = i + 1; j < topics.length; j++) {
      const a = topics[i]
      const b = topics[j]
      if (a === undefined || b === undefined) continue
      const aWords = significantWords(lastSegment(a))
      const bWords = significantWords(lastSegment(b))
      if (jaccard(aWords, bWords) >= JACCARD_CUTOFF) {
        pairs.push([a, b])
      }
    }
  }
  return pairs
}

const significantWords = (filename: string): Set<string> => {
  const base = filename.toLowerCase().replace(/\.md$/, '')
  const words = base.split(/[^a-z0-9]+/).filter((w) => w.length > 2)
  return new Set(words)
}

const jaccard = (a: ReadonlySet<string>, b: ReadonlySet<string>): number => {
  if (a.size === 0 && b.size === 0) return 0
  let intersect = 0
  for (const v of a) if (b.has(v)) intersect++
  const union = a.size + b.size - intersect
  return union === 0 ? 0 : intersect / union
}

const safeRead = (buf: Buffer | undefined): string | undefined => {
  if (!buf) return undefined
  return buf.toString('utf8')
}

const listTopicPaths = async (
  reader: Pick<Store, 'list'>,
  prefix: Path,
): Promise<readonly Path[]> => {
  const entries = await reader.list(prefix, { recursive: true, includeGenerated: true })
  const topics: Path[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    const name = lastSegment(entry.path)
    if (!name.endsWith('.md') || name === 'MEMORY.md') continue
    topics.push(entry.path)
  }
  return topics
}

const askVerdict = async (
  provider: Provider,
  aPath: Path,
  aContent: string,
  bPath: Path,
  bContent: string,
): Promise<string> => {
  const prompt = `## File 1: ${lastSegment(aPath)}\n\n${aContent}\n\n---\n\n## File 2: ${lastSegment(bPath)}\n\n${bContent}`
  const resp = await provider.complete({
    messages: [{ role: 'user', content: prompt }],
    system: DEDUPLICATION_SYSTEM_PROMPT,
    maxTokens: DEDUP_MAX_TOKENS,
    temperature: DEDUP_TEMPERATURE,
  })
  const trimmed = resp.content.trim()
  const first = trimmed.indexOf('{')
  const last = trimmed.lastIndexOf('}')
  if (first < 0 || last <= first) return 'distinct'
  try {
    const parsed = JSON.parse(trimmed.slice(first, last + 1)) as { verdict?: string }
    return parsed.verdict ?? 'distinct'
  } catch {
    return 'distinct'
  }
}

const planMerge = (
  aPath: Path,
  aContent: string,
  bPath: Path,
  bContent: string,
): {
  readonly keeperPath: Path
  readonly keeperContent: string
  readonly donorPath: Path
  readonly donorContent: string
} => {
  const aObservedAt = observedAtFromContent(aContent)
  const bObservedAt = observedAtFromContent(bContent)
  if (bObservedAt.getTime() > aObservedAt.getTime()) {
    return {
      keeperPath: bPath,
      keeperContent: bContent,
      donorPath: aPath,
      donorContent: aContent,
    }
  }
  return {
    keeperPath: aPath,
    keeperContent: aContent,
    donorPath: bPath,
    donorContent: bContent,
  }
}

const mergeBodies = (keeper: string, donor: string, donorPath: Path): string => {
  const { body: donorBody } = parseFrontmatter(donor)
  const trimmedKeeper = keeper.trim()
  return `${trimmedKeeper}\n\n---\n\n*Merged from ${lastSegment(donorPath)}:*\n\n${donorBody.trim()}\n`
}

const runMaintenancePass = async (
  batch: Batch,
  input: {
    readonly prefix: Path
    readonly indexPath: Path
    readonly now: Date
  },
): Promise<{ readonly ops: readonly ConsolidationOp[]; readonly errors: readonly string[] }> => {
  const ops: ConsolidationOp[] = []
  const errors: string[] = []
  const notes = await listNotes(batch, input.prefix, errors)

  for (const note of notes) {
    const rewrite = rewriteNote(note, input.now)
    if (!rewrite.changed) continue
    await batch.write(note.path, Buffer.from(rewrite.content, 'utf8'))
    ops.push({ kind: 'rewrite', path: note.path })
  }

  const refreshedNotes = await listNotes(batch, input.prefix, errors)
  const nextIndex = buildIndexContent(refreshedNotes)
  if (await writeIfChanged(batch, input.indexPath, nextIndex)) {
    ops.push({ kind: 'rewrite', path: input.indexPath })
  }

  return { ops, errors }
}

const listNotes = async (
  reader: StoreReader,
  prefix: Path,
  errors: string[],
): Promise<readonly NoteRecord[]> => {
  let entries: readonly FileInfo[]
  try {
    entries = await reader.list(prefix, { recursive: true, includeGenerated: true })
  } catch (err) {
    if (err instanceof ErrNotFound) return []
    errors.push(
      `listing ${prefix} for maintenance failed: ${err instanceof Error ? err.message : String(err)}`,
    )
    return []
  }

  const notes: NoteRecord[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    const filename = lastSegment(entry.path)
    if (!filename.endsWith('.md') || filename === 'MEMORY.md') continue
    try {
      const content = (await reader.read(entry.path)).toString('utf8')
      const parsed = parseFrontmatter(content)
      notes.push({
        path: entry.path,
        relativePath: relativePath(prefix, entry.path),
        content,
        body: parsed.body,
        frontmatter: parsed.frontmatter,
        observedAt: resolveObservedAt(parsed.frontmatter, entry.modTime),
        createdAt: resolveCreatedAt(parsed.frontmatter, entry.modTime),
      })
    } catch (err) {
      errors.push(
        `reading ${entry.path} for maintenance failed: ${err instanceof Error ? err.message : String(err)}`,
      )
    }
  }

  notes.sort(compareNotesForIndex)
  return notes
}

const rewriteNote = (
  note: NoteRecord,
  now: Date,
): { readonly changed: boolean; readonly content: string } => {
  const originalTags = normaliseTags(note.frontmatter.tags)
  let nextTags = [...originalTags]
  let changed = false

  const stale = isStale(note.observedAt, now)
  nextTags = stale ? upsertTag(nextTags, STALE_TAG) : removeTag(nextTags, STALE_TAG)

  const nextExtra = { ...note.frontmatter.extra }
  if (stale) {
    if (!nextExtra.stale_since) {
      nextExtra.stale_since = now.toISOString()
      changed = true
    }
  } else if (nextExtra.stale_since) {
    delete nextExtra.stale_since
    changed = true
  }

  let nextConfidence = note.frontmatter.confidence?.trim()
  if (hasTag(nextTags, HEURISTIC_TAG)) {
    const derivedConfidence = deriveHeuristicConfidence(note, now)
    if (derivedConfidence !== nextConfidence) {
      nextConfidence = derivedConfidence
      changed = true
    }
    nextTags = replaceConfidenceTags(nextTags, derivedConfidence)
  }

  if (!sameStringArray(nextTags, originalTags)) {
    changed = true
  }

  if (!changed) {
    return { changed: false, content: note.content }
  }

  const {
    confidence: _originalConfidence,
    tags: _originalTags,
    ...restFrontmatter
  } = note.frontmatter
  const content = buildNoteContent(
    {
      ...restFrontmatter,
      modified: note.frontmatter.modified?.trim() || note.observedAt.toISOString(),
      extra: nextExtra,
      ...(nextConfidence !== undefined ? { confidence: nextConfidence } : {}),
      ...(nextTags.length > 0 ? { tags: nextTags } : {}),
    },
    note.body,
  )

  return { changed: content !== note.content, content }
}

const buildNoteContent = (frontmatter: Frontmatter, body: string): string => {
  const built = buildFrontmatter(frontmatter)
  const trimmedBody = body.trim()
  return trimmedBody === '' ? `${built}\n` : `${built}\n${trimmedBody}\n`
}

const buildIndexContent = (notes: readonly NoteRecord[]): string => {
  const lines = notes.map((note) => buildIndexEntry(note))
  return `${lines.join('\n')}\n`
}

const buildIndexEntry = (note: NoteRecord): string => {
  const qualifiers: string[] = []
  const confidence = note.frontmatter.confidence?.trim().toLowerCase()
  if (confidence !== undefined && isConfidence(confidence)) qualifiers.push(confidence)
  if (hasTag(note.frontmatter.tags, STALE_TAG)) qualifiers.push(STALE_TAG)
  const summary = truncateOneLine(
    note.frontmatter.description?.trim() ||
      firstMeaningfulLine(note.body) ||
      note.frontmatter.type?.trim() ||
      'memory note',
    INDEX_ENTRY_LIMIT,
  )
  const suffix = qualifiers.length > 0 ? ` [${qualifiers.join(', ')}]` : ''
  return `- ${note.relativePath}: ${summary}${suffix}`
}

const writeIfChanged = async (batch: Batch, path: Path, content: string): Promise<boolean> => {
  let existing = ''
  try {
    existing = (await batch.read(path)).toString('utf8')
  } catch (err) {
    if (!(err instanceof ErrNotFound)) throw err
  }
  if (existing === content) return false
  await batch.write(path, Buffer.from(content, 'utf8'))
  return true
}

const compareNotesForIndex = (a: NoteRecord, b: NoteRecord): number => {
  const observedDelta = b.observedAt.getTime() - a.observedAt.getTime()
  if (observedDelta !== 0) return observedDelta
  return a.relativePath.localeCompare(b.relativePath)
}

const relativePath = (prefix: Path, path: Path): string => {
  const value = String(path)
  const prefixValue = String(prefix)
  if (!value.startsWith(`${prefixValue}/`)) return lastSegment(path)
  return value.slice(prefixValue.length + 1)
}

const resolveObservedAt = (frontmatter: Frontmatter, fallback: Date): Date =>
  parseDate(frontmatter.modified) ??
  parseDate(frontmatter.observed_on) ??
  parseDate(frontmatter.session_date) ??
  parseDate(frontmatter.created) ??
  fallback

const resolveCreatedAt = (frontmatter: Frontmatter, fallback: Date): Date =>
  parseDate(frontmatter.created) ??
  parseDate(frontmatter.session_date) ??
  parseDate(frontmatter.observed_on) ??
  parseDate(frontmatter.modified) ??
  fallback

const observedAtFromContent = (content: string): Date =>
  resolveObservedAt(parseFrontmatter(content).frontmatter, new Date(0))

const parseDate = (value: string | undefined): Date | undefined => {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  if (trimmed === '') return undefined
  const parsed = new Date(trimmed)
  return Number.isNaN(parsed.getTime()) ? undefined : parsed
}

const isStale = (observedAt: Date, now: Date): boolean =>
  diffDays(observedAt, now) >= STALE_AFTER_DAYS

const deriveHeuristicConfidence = (note: NoteRecord, now: Date): Confidence => {
  const current = normaliseConfidence(note.frontmatter.confidence)
  const ageDays = diffDays(note.observedAt, now)
  const reinforcementDays = Math.max(diffDays(note.createdAt, note.observedAt), 0)

  if (ageDays >= DEEP_STALE_AFTER_DAYS) return 'low'
  if (ageDays >= STALE_AFTER_DAYS) {
    return current === 'high' ? 'medium' : 'low'
  }
  if (reinforcementDays >= HEURISTIC_STRONG_REINFORCEMENT_DAYS) return 'high'
  if (reinforcementDays >= HEURISTIC_REINFORCEMENT_DAYS) {
    return current === 'high' ? 'high' : 'medium'
  }
  return current
}

const diffDays = (start: Date, end: Date): number =>
  Math.floor((end.getTime() - start.getTime()) / (24 * 60 * 60 * 1000))

const normaliseConfidence = (value: string | undefined): Confidence => {
  const trimmed = value?.trim().toLowerCase()
  return isConfidence(trimmed) ? trimmed : 'low'
}

const isConfidence = (value: string | undefined): value is Confidence =>
  value !== undefined && CONFIDENCE_LEVELS.some((confidence) => confidence === value)

const normaliseTags = (tags: readonly string[] | undefined): string[] => {
  const out: string[] = []
  const seen = new Set<string>()
  for (const raw of tags ?? []) {
    const tag = raw.trim()
    if (tag === '') continue
    const key = tag.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(tag)
  }
  return out
}

const hasTag = (tags: readonly string[] | undefined, target: string): boolean =>
  (tags ?? []).some((tag) => tag.trim().toLowerCase() === target)

const upsertTag = (tags: readonly string[], target: string): string[] => {
  const next = removeTag(tags, target)
  next.push(target)
  return next
}

const removeTag = (tags: readonly string[], target: string): string[] =>
  tags.filter((tag) => tag.trim().toLowerCase() !== target)

const replaceConfidenceTags = (tags: readonly string[], confidence: Confidence): string[] => {
  const next = tags.filter((tag) => !isConfidence(tag.trim().toLowerCase()))
  next.push(confidence)
  return normaliseTags(next)
}

const sameStringArray = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((value, index) => value === b[index])

const firstMeaningfulLine = (body: string): string => {
  for (const raw of body.split('\n')) {
    const line = raw.trim()
    if (line !== '') return line
  }
  return ''
}

const truncateOneLine = (value: string, limit: number): string => {
  const collapsed = value.replace(/\s+/g, ' ').trim()
  if (collapsed.length <= limit) return collapsed
  return `${collapsed.slice(0, Math.max(limit - 3, 1)).trimEnd()}...`
}
