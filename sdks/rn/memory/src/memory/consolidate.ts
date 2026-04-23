import type { Logger, Provider } from '../llm/types.js'
import { ErrNotFound } from '../store/index.js'
import type { Batch, FileInfo, ListOpts, Path, Store } from '../store/index.js'
import { lastSegment } from '../store/index.js'
import { type Frontmatter, buildFrontmatter, parseFrontmatter } from './frontmatter.js'
import { type Scope, scopeIndex, scopePrefix } from './paths.js'
import type { ConsolidationOp, ConsolidationReport } from './types.js'

const DEDUPLICATION_SYSTEM_PROMPT = `You are analysing two memory files for overlap. Determine whether they cover the same topic or are distinct.

Respond with ONLY a JSON object:
{
  "verdict": "keep_first" | "keep_second" | "merge" | "distinct",
  "reason": "brief explanation"
}

- "distinct": files cover different topics, keep both
- "keep_first": files overlap, the first is more complete, delete the second
- "keep_second": files overlap, the second is more complete, delete the first
- "merge": files have complementary information, combine into one

Respond with ONLY valid JSON, no other text.`

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
  readonly list: (dir: Path | '', opts?: ListOpts) => Promise<FileInfo[]>
  readonly read: (path: Path) => Promise<string>
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

export type ConsolidateArgs = {
  readonly scope?: Scope
  readonly actorId?: string
}

export type ConsolidateDeps = {
  readonly store: Store
  readonly provider?: Provider
  readonly logger: Logger
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

    let topics: readonly Path[]
    try {
      topics = await listTopicPaths(deps.store, prefix)
    } catch (error) {
      if (error instanceof ErrNotFound) {
        return {
          merged: 0,
          deleted: 0,
          promoted: 0,
          ops: [],
          errors: [],
        }
      }
      throw error
    }

    const ops: ConsolidationOp[] = []
    const errors: string[] = []
    let merged = 0
    let deleted = 0

    try {
      await deps.store.batch({ reason: 'consolidate' }, async (batch) => {
        if (deps.provider !== undefined) {
          const pairs = candidatePairs(topics)
          for (const [leftPath, rightPath] of pairs) {
            if (!(await batch.exists(leftPath)) || !(await batch.exists(rightPath))) continue
            const leftContent = await safeRead(batch, leftPath)
            const rightContent = await safeRead(batch, rightPath)
            if (leftContent === undefined || rightContent === undefined) continue

            let verdict: string
            try {
              verdict = await askVerdict(
                deps.provider,
                leftPath,
                leftContent,
                rightPath,
                rightContent,
              )
            } catch (error) {
              errors.push(
                `dedup LLM failed for ${leftPath} + ${rightPath}: ${error instanceof Error ? error.message : String(error)}`,
              )
              continue
            }

            switch (verdict) {
              case 'keep_first':
                await batch.delete(rightPath)
                deleted += 1
                ops.push({ kind: 'delete', path: rightPath, reason: 'keep_first' })
                break
              case 'keep_second':
                await batch.delete(leftPath)
                deleted += 1
                ops.push({ kind: 'delete', path: leftPath, reason: 'keep_second' })
                break
              case 'merge': {
                const mergePlan = planMerge(leftPath, leftContent, rightPath, rightContent)
                const combined = mergeBodies(
                  mergePlan.keeperContent,
                  mergePlan.donorContent,
                  mergePlan.donorPath,
                )
                await batch.write(mergePlan.keeperPath, combined)
                await batch.delete(mergePlan.donorPath)
                merged += 1
                ops.push({
                  kind: 'merge',
                  keeper: mergePlan.keeperPath,
                  donor: mergePlan.donorPath,
                })
                break
              }
              default:
                break
            }
          }
        }

        const maintenance = await runMaintenancePass(batch, {
          prefix,
          indexPath,
          now,
        })
        ops.push(...maintenance.ops)
        errors.push(...maintenance.errors)
      })
    } catch (error) {
      errors.push(
        `consolidate batch failed: ${error instanceof Error ? error.message : String(error)}`,
      )
    }

    return {
      merged,
      deleted,
      promoted: 0,
      ops,
      errors,
    }
  }
}

const candidatePairs = (topics: readonly Path[]): Array<[Path, Path]> => {
  const pairs: Array<[Path, Path]> = []
  for (let leftIndex = 0; leftIndex < topics.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < topics.length; rightIndex += 1) {
      const left = topics[leftIndex]
      const right = topics[rightIndex]
      if (left === undefined || right === undefined) continue
      if (
        jaccard(significantWords(lastSegment(left)), significantWords(lastSegment(right))) >=
        JACCARD_CUTOFF
      ) {
        pairs.push([left, right])
      }
    }
  }
  return pairs
}

const significantWords = (filename: string): Set<string> => {
  const base = filename.toLowerCase().replace(/\.md$/, '')
  return new Set(base.split(/[^a-z0-9]+/).filter((word) => word.length > 2))
}

const jaccard = (left: ReadonlySet<string>, right: ReadonlySet<string>): number => {
  if (left.size === 0 && right.size === 0) return 0
  let intersect = 0
  for (const value of left) {
    if (right.has(value)) intersect += 1
  }
  const union = left.size + right.size - intersect
  return union === 0 ? 0 : intersect / union
}

const safeRead = async (batch: Batch, path: Path): Promise<string | undefined> => {
  try {
    return await batch.read(path)
  } catch {
    return undefined
  }
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
  leftPath: Path,
  leftContent: string,
  rightPath: Path,
  rightContent: string,
): Promise<string> => {
  const response = await provider.complete({
    messages: [
      {
        role: 'user',
        content: `## File 1: ${lastSegment(leftPath)}\n\n${leftContent}\n\n---\n\n## File 2: ${lastSegment(rightPath)}\n\n${rightContent}`,
      },
    ],
    system: DEDUPLICATION_SYSTEM_PROMPT,
    maxTokens: DEDUP_MAX_TOKENS,
    temperature: DEDUP_TEMPERATURE,
  })
  const trimmed = response.content.trim()
  const start = trimmed.indexOf('{')
  const end = trimmed.lastIndexOf('}')
  if (start < 0 || end <= start) return 'distinct'
  try {
    return (
      (JSON.parse(trimmed.slice(start, end + 1)) as { readonly verdict?: string }).verdict ??
      'distinct'
    )
  } catch {
    return 'distinct'
  }
}

const planMerge = (
  leftPath: Path,
  leftContent: string,
  rightPath: Path,
  rightContent: string,
): {
  readonly keeperPath: Path
  readonly keeperContent: string
  readonly donorPath: Path
  readonly donorContent: string
} => {
  const leftObservedAt = observedAtFromContent(leftContent)
  const rightObservedAt = observedAtFromContent(rightContent)
  if (rightObservedAt.getTime() > leftObservedAt.getTime()) {
    return {
      keeperPath: rightPath,
      keeperContent: rightContent,
      donorPath: leftPath,
      donorContent: leftContent,
    }
  }
  return {
    keeperPath: leftPath,
    keeperContent: leftContent,
    donorPath: rightPath,
    donorContent: rightContent,
  }
}

const mergeBodies = (keeper: string, donor: string, donorPath: Path): string => {
  const { body } = parseFrontmatter(donor)
  const trimmedKeeper = keeper.trim()
  return `${trimmedKeeper}\n\n---\n\n*Merged from ${lastSegment(donorPath)}:*\n\n${body.trim()}\n`
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
    await batch.write(note.path, rewrite.content)
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
  } catch (error) {
    if (error instanceof ErrNotFound) return []
    errors.push(
      `listing ${prefix} for maintenance failed: ${error instanceof Error ? error.message : String(error)}`,
    )
    return []
  }

  const notes: NoteRecord[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    const filename = lastSegment(entry.path)
    if (!filename.endsWith('.md') || filename === 'MEMORY.md') continue
    try {
      const content = await reader.read(entry.path)
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
    } catch (error) {
      errors.push(
        `reading ${entry.path} for maintenance failed: ${error instanceof Error ? error.message : String(error)}`,
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

  let nextExtra = { ...note.frontmatter.extra }
  if (stale) {
    if (!nextExtra.stale_since) {
      nextExtra.stale_since = now.toISOString()
      changed = true
    }
  } else if (nextExtra.stale_since) {
    const { stale_since: _staleSince, ...remaining } = nextExtra
    nextExtra = remaining
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

  const { confidence: _confidence, tags: _tags, ...rest } = note.frontmatter
  const content = buildNoteContent(
    {
      ...rest,
      modified: note.frontmatter.modified?.trim() || note.observedAt.toISOString(),
      extra: nextExtra,
      ...(nextConfidence === undefined ? {} : { confidence: nextConfidence }),
      ...(nextTags.length === 0 ? {} : { tags: nextTags }),
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
  return `${notes.map((note) => buildIndexEntry(note)).join('\n')}\n`
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
    existing = await batch.read(path)
  } catch (error) {
    if (!(error instanceof ErrNotFound)) throw error
  }
  if (existing === content) return false
  await batch.write(path, content)
  return true
}

const compareNotesForIndex = (left: NoteRecord, right: NoteRecord): number => {
  const observedDelta = right.observedAt.getTime() - left.observedAt.getTime()
  if (observedDelta !== 0) return observedDelta
  return left.relativePath.localeCompare(right.relativePath)
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

const sameStringArray = (left: readonly string[], right: readonly string[]): boolean =>
  left.length === right.length && left.every((value, index) => value === right[index])

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
