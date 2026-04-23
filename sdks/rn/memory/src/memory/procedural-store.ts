import type { Message } from '../llm/types.js'
import { ErrNotFound, type Path, type Store, joinPath } from '../store/index.js'
import { parseFrontmatter } from './frontmatter.js'
import { detectProceduralRecords, formatProceduralRecord } from './procedural.js'
import type {
  DetectProceduralRecordsOptions,
  ProceduralOutcome,
  ProceduralRecord,
  ProceduralTier,
} from './types.js'

export const PROCEDURAL_RECORDS_PREFIX = 'memory/_procedural' as Path

const DEFAULT_PERSIST_REASON = 'procedural-records'
const DEFAULT_LIST_LIMIT = 100
const DEFAULT_QUERY_LIMIT = 20

const FNV_OFFSET_64 = 0xcbf29ce484222325n
const FNV_PRIME_64 = 0x100000001b3n
const FNV_MASK_64 = 0xffffffffffffffffn

export type PersistProceduralRecordsArgs = {
  readonly actorId: string
  readonly records: readonly ProceduralRecord[]
  readonly sessionId?: string
  readonly reason?: string
}

export type DetectAndPersistProceduralRecordsArgs = DetectProceduralRecordsOptions & {
  readonly actorId: string
  readonly messages: readonly Message[]
  readonly sessionId?: string
  readonly reason?: string
}

export type ListProceduralRecordsArgs = {
  readonly actorId?: string
  readonly sessionId?: string
  readonly tier?: ProceduralTier | readonly ProceduralTier[]
  readonly outcome?: ProceduralOutcome | readonly ProceduralOutcome[]
  readonly name?: string | readonly string[]
  readonly tags?: readonly string[]
  readonly since?: Date | string
  readonly until?: Date | string
  readonly limit?: number
  readonly sort?: 'asc' | 'desc'
}

export type QueryProceduralRecordsArgs = Omit<ListProceduralRecordsArgs, 'limit'> & {
  readonly text: string
  readonly limit?: number
}

export type StoredProceduralRecord = ProceduralRecord & {
  readonly path: Path
  readonly actorId: string
  readonly sessionId?: string
  readonly createdAt: string
  readonly modifiedAt: string
  readonly content: string
}

export type ProceduralQueryHit = StoredProceduralRecord & {
  readonly score: number
}

export type ProceduralStore = {
  persist(args: PersistProceduralRecordsArgs): Promise<readonly StoredProceduralRecord[]>
  detectAndPersist(
    args: DetectAndPersistProceduralRecordsArgs,
  ): Promise<readonly StoredProceduralRecord[]>
  list(args?: ListProceduralRecordsArgs): Promise<readonly StoredProceduralRecord[]>
  query(args: QueryProceduralRecordsArgs): Promise<readonly ProceduralQueryHit[]>
}

export const proceduralActorPrefix = (actorId: string): Path =>
  joinPath(PROCEDURAL_RECORDS_PREFIX, actorSegment(actorId))

export const proceduralSessionPrefix = (actorId: string, sessionId?: string): Path =>
  joinPath(proceduralActorPrefix(actorId), sessionSegment(sessionId))

export class StoreBackedProceduralStore implements ProceduralStore {
  constructor(private readonly store: Store) {}

  async persist(args: PersistProceduralRecordsArgs): Promise<readonly StoredProceduralRecord[]> {
    const actorId = requireText(args.actorId, 'actorId')
    if (args.records.length === 0) return []

    const sessionId = optionalText(args.sessionId)
    const writtenAt = new Date().toISOString()
    const collisions = new Map<string, number>()
    const stored = args.records.map((record) =>
      createStoredRecord({
        actorId,
        writtenAt,
        record: normaliseProceduralRecord(record),
        collisions,
        ...(sessionId === undefined ? {} : { sessionId }),
      }),
    )

    await this.store.batch({ reason: persistReason(args.reason) }, async (batch) => {
      for (const record of stored) {
        await batch.write(record.path, record.content)
      }
    })

    return stored
  }

  async detectAndPersist(
    args: DetectAndPersistProceduralRecordsArgs,
  ): Promise<readonly StoredProceduralRecord[]> {
    const records = detectProceduralRecords(args.messages, {
      ...(args.observedAt === undefined ? {} : { observedAt: args.observedAt }),
      ...(args.maxContextLength === undefined ? {} : { maxContextLength: args.maxContextLength }),
    })

    return await this.persist({
      actorId: args.actorId,
      records,
      ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
      ...(args.reason === undefined ? {} : { reason: args.reason }),
    })
  }

  async list(args: ListProceduralRecordsArgs = {}): Promise<readonly StoredProceduralRecord[]> {
    const sort = args.sort === 'asc' ? 'asc' : 'desc'
    const limit = sanitisePositiveInt(args.limit, DEFAULT_LIST_LIMIT)
    const records = await this.scan(args)
    records.sort((left, right) => compareStoredProceduralRecords(left, right, sort))
    return records.slice(0, limit)
  }

  async query(args: QueryProceduralRecordsArgs): Promise<readonly ProceduralQueryHit[]> {
    const text = collapseWhitespace(args.text)
    const limit = sanitisePositiveInt(args.limit, DEFAULT_QUERY_LIMIT)
    const sort = args.sort === 'asc' ? 'asc' : 'desc'
    const records = await this.scan(args)

    if (text === '') {
      return records
        .sort((left, right) => compareStoredProceduralRecords(left, right, sort))
        .slice(0, limit)
        .map((record) => ({ ...record, score: 0 }))
    }

    const hits = records
      .map((record) => ({
        ...record,
        score: scoreProceduralRecord(record, text),
      }))
      .filter((record) => record.score > 0)

    hits.sort((left, right) => {
      const scoreDelta = right.score - left.score
      if (scoreDelta !== 0) return scoreDelta
      return compareStoredProceduralRecords(left, right, sort)
    })

    return hits.slice(0, limit)
  }

  private async scan(
    args: Omit<ListProceduralRecordsArgs, 'limit' | 'sort'>,
  ): Promise<StoredProceduralRecord[]> {
    const prefix = selectListPrefix(args.actorId, args.sessionId)
    if (!(await this.prefixExists(prefix))) return []

    let entries: readonly { readonly path: Path; readonly isDir: boolean; readonly modTime: Date }[]
    try {
      entries = await this.store.list(prefix, { recursive: true })
    } catch (error) {
      if (error instanceof ErrNotFound) return []
      throw error
    }

    const records: StoredProceduralRecord[] = []
    for (const entry of entries) {
      if (entry.isDir || !entry.path.endsWith('.md')) continue
      const record = await this.readStoredRecord(entry.path, entry.modTime)
      if (record === undefined || !matchesProceduralFilters(record, args)) continue
      records.push(record)
    }
    return records
  }

  private async prefixExists(prefix: Path): Promise<boolean> {
    try {
      return await this.store.exists(prefix)
    } catch {
      return false
    }
  }

  private async readStoredRecord(
    path: Path,
    modTime: Date,
  ): Promise<StoredProceduralRecord | undefined> {
    let content: string
    try {
      content = await this.store.read(path)
    } catch (error) {
      if (error instanceof ErrNotFound) return undefined
      throw error
    }

    const { frontmatter, body } = parseFrontmatter(content)
    if ((frontmatter.type ?? '').trim() !== 'procedural') return undefined

    const tier = coerceTier(frontmatter.extra.tier)
    const outcome = coerceOutcome(frontmatter.extra.outcome)
    const name = optionalText(frontmatter.name) ?? ''
    if (tier === undefined || outcome === undefined || name === '') return undefined

    const actorId = optionalText(frontmatter.extra.actor_id) ?? actorIdFromPath(path)
    if (actorId === '') return undefined

    const parsedBody = parseProceduralBody(body)
    const observedAt = normaliseTimestamp(
      frontmatter.observed_on ??
        frontmatter.extra.observed ??
        frontmatter.modified ??
        frontmatter.created ??
        modTime.toISOString(),
    )
    const createdAt = normaliseTimestamp(frontmatter.created ?? modTime.toISOString())
    const modifiedAt = normaliseTimestamp(frontmatter.modified ?? createdAt)
    const tags = dedupeStrings([...normaliseStringList(frontmatter.tags), 'procedural', tier, name])

    return {
      tier,
      name,
      taskContext: parsedBody.taskContext,
      outcome,
      observedAt,
      toolCalls: parsedBody.toolCalls,
      tags,
      path,
      actorId,
      ...(frontmatter.session_id === undefined ? {} : { sessionId: frontmatter.session_id }),
      createdAt,
      modifiedAt,
      content,
    }
  }
}

export const createStoreBackedProceduralStore = (store: Store): ProceduralStore =>
  new StoreBackedProceduralStore(store)

export const createProceduralStore = createStoreBackedProceduralStore

const persistReason = (reason: string | undefined): string => {
  const trimmed = optionalText(reason)
  return trimmed ?? DEFAULT_PERSIST_REASON
}

const selectListPrefix = (actorId: string | undefined, sessionId: string | undefined): Path => {
  const actor = optionalText(actorId)
  if (actor === undefined) return PROCEDURAL_RECORDS_PREFIX
  const session = optionalText(sessionId)
  return session === undefined
    ? proceduralActorPrefix(actor)
    : proceduralSessionPrefix(actor, session)
}

const createStoredRecord = (args: {
  readonly actorId: string
  readonly sessionId?: string
  readonly writtenAt: string
  readonly record: ProceduralRecord
  readonly collisions: Map<string, number>
}): StoredProceduralRecord => {
  const key = proceduralRecordKey(args.actorId, args.sessionId, args.record)
  const collisionCount = (args.collisions.get(key) ?? 0) + 1
  args.collisions.set(key, collisionCount)

  const path = proceduralRecordPath(args.actorId, args.sessionId, args.record, collisionCount)
  const content = augmentProceduralNote(
    formatProceduralRecord(args.record),
    args.actorId,
    args.sessionId,
    args.writtenAt,
    args.record.observedAt,
  )

  return {
    ...args.record,
    path,
    actorId: args.actorId,
    ...(args.sessionId === undefined ? {} : { sessionId: args.sessionId }),
    createdAt: args.writtenAt,
    modifiedAt: args.writtenAt,
    content,
  }
}

const normaliseProceduralRecord = (record: ProceduralRecord): ProceduralRecord => {
  const tier = record.tier === 'agent' ? 'agent' : 'skill'
  const name = requireText(record.name, 'record.name')
  const tags = dedupeStrings([...normaliseStringList(record.tags), 'procedural', tier, name])

  return {
    tier,
    name,
    taskContext: optionalText(record.taskContext) ?? '',
    outcome: coerceOutcome(record.outcome) ?? 'partial',
    observedAt: normaliseTimestamp(record.observedAt),
    toolCalls: normaliseStringList(record.toolCalls),
    tags,
  }
}

const proceduralRecordPath = (
  actorId: string,
  sessionId: string | undefined,
  record: ProceduralRecord,
  collisionCount: number,
): Path => {
  const observedAt = normaliseTimestamp(record.observedAt)
  const filename = [
    compactTimestamp(observedAt),
    record.tier,
    slugify(record.name),
    proceduralRecordKey(actorId, sessionId, record).slice(0, 10),
    collisionCount > 1 ? String(collisionCount) : '',
  ]
    .filter((part) => part !== '')
    .join('-')

  return joinPath(
    proceduralSessionPrefix(actorId, sessionId),
    observedAt.slice(0, 4),
    observedAt.slice(5, 7),
    observedAt.slice(8, 10),
    `${filename}.md`,
  )
}

const proceduralRecordKey = (
  actorId: string,
  sessionId: string | undefined,
  record: ProceduralRecord,
): string =>
  stableHash(
    JSON.stringify({
      actorId,
      sessionId,
      tier: record.tier,
      name: record.name,
      taskContext: record.taskContext,
      outcome: record.outcome,
      observedAt: record.observedAt,
      toolCalls: record.toolCalls,
      tags: record.tags,
    }),
  )

const augmentProceduralNote = (
  note: string,
  actorId: string,
  sessionId: string | undefined,
  writtenAt: string,
  observedAt: string,
): string => {
  const lines = note.replace(/\r\n/g, '\n').split('\n')
  let closeIndex = -1
  for (let index = 1; index < lines.length; index += 1) {
    if ((lines[index] ?? '').trim() === '---') {
      closeIndex = index
      break
    }
  }
  if (closeIndex < 0) return note

  const extra = [
    `created: ${renderFrontmatterValue(writtenAt)}`,
    `modified: ${renderFrontmatterValue(writtenAt)}`,
    'source: procedural',
    `observed_on: ${renderFrontmatterValue(observedAt)}`,
    `actor_id: ${renderFrontmatterValue(actorId)}`,
    ...(sessionId === undefined ? [] : [`session_id: ${renderFrontmatterValue(sessionId)}`]),
  ]

  return `${[...lines.slice(0, closeIndex), ...extra, ...lines.slice(closeIndex)].join('\n').trimEnd()}\n`
}

const parseProceduralBody = (
  body: string,
): { readonly taskContext: string; readonly toolCalls: readonly string[] } => {
  const contextLines: string[] = []
  const toolLines: string[] = []
  let section: 'none' | 'context' | 'tools' = 'none'

  for (const line of body.split('\n')) {
    const trimmed = line.trim()
    if (trimmed === '## Context') {
      section = 'context'
      continue
    }
    if (trimmed === '## Tool sequence') {
      section = 'tools'
      continue
    }
    if (section === 'context') {
      contextLines.push(line)
      continue
    }
    if (section === 'tools') {
      toolLines.push(line)
    }
  }

  return {
    taskContext: contextLines.join('\n').trim(),
    toolCalls: collapseWhitespace(toolLines.join(' '))
      .split(/\s*->\s*/g)
      .map((value) => value.trim())
      .filter((value) => value !== ''),
  }
}

const compareStoredProceduralRecords = (
  left: StoredProceduralRecord,
  right: StoredProceduralRecord,
  sort: 'asc' | 'desc',
): number => {
  const diff = timestampValue(left.observedAt) - timestampValue(right.observedAt)
  if (diff !== 0) return sort === 'asc' ? diff : -diff
  return left.path.localeCompare(right.path)
}

const matchesProceduralFilters = (
  record: StoredProceduralRecord,
  args: Omit<ListProceduralRecordsArgs, 'limit' | 'sort'>,
): boolean => {
  const actorId = optionalText(args.actorId)
  if (actorId !== undefined && record.actorId !== actorId) return false

  const sessionId = optionalText(args.sessionId)
  if (sessionId !== undefined && record.sessionId !== sessionId) return false

  if (!matchesSelection(record.tier, args.tier)) return false
  if (!matchesSelection(record.outcome, args.outcome)) return false
  if (!matchesSelection(record.name, args.name)) return false

  const tags = normaliseStringList(args.tags)
  if (tags.length > 0) {
    const recordTags = new Set(record.tags.map(normaliseSearchText))
    for (const tag of tags) {
      if (!recordTags.has(normaliseSearchText(tag))) return false
    }
  }

  const since = parseTimestamp(args.since)
  if (since !== undefined && timestampValue(record.observedAt) < since) return false
  const until = parseTimestamp(args.until)
  if (until !== undefined && timestampValue(record.observedAt) > until) return false
  return true
}

const scoreProceduralRecord = (record: StoredProceduralRecord, query: string): number => {
  const phrase = normaliseSearchText(query)
  if (phrase === '') return 0

  const tokens = dedupeStrings(tokeniseSearchText(query))
  const name = normaliseSearchText(record.name)
  const context = normaliseSearchText(record.taskContext)
  const tags = record.tags.map(normaliseSearchText)
  const toolCalls = record.toolCalls.map(normaliseSearchText)
  let score = 0

  if (name.includes(phrase)) score += 12
  if (tags.some((tag) => tag.includes(phrase))) score += 10
  if (context.includes(phrase)) score += 8
  if (toolCalls.some((toolCall) => toolCall.includes(phrase))) score += 8
  if (record.tier === phrase || record.outcome === phrase) score += 4

  for (const token of tokens) {
    if (name.includes(token)) score += 5
    if (tags.some((tag) => tag.includes(token))) score += 4
    if (toolCalls.some((toolCall) => toolCall.includes(token))) score += 3
    if (context.includes(token)) score += 2
    if (record.tier === token || record.outcome === token) score += 2
  }

  return score
}

const matchesSelection = (
  value: string,
  selection: string | readonly string[] | undefined,
): boolean => {
  if (selection === undefined) return true
  const expected = new Set(
    (Array.isArray(selection) ? selection : [selection])
      .map((entry) => normaliseSearchText(entry))
      .filter((entry) => entry !== ''),
  )
  if (expected.size === 0) return true
  return expected.has(normaliseSearchText(value))
}

const actorIdFromPath = (path: Path): string => {
  for (const segment of path.split('/')) {
    if (segment.startsWith('actor_')) return segment.slice('actor_'.length)
  }
  return ''
}

const actorSegment = (actorId: string): string => `actor_${sanitisePathSegment(actorId, 'actorId')}`

const sessionSegment = (sessionId: string | undefined): string => {
  const normalised = optionalText(sessionId)
  return normalised === undefined
    ? 'session_none'
    : `session_${sanitisePathSegment(normalised, 'sessionId')}`
}

const sanitisePathSegment = (value: string, field: string): string => {
  const trimmed = requireText(value, field)
  const cleaned = trimmed.replace(/[^A-Za-z0-9._-]/g, '_')
  if (cleaned === '' || cleaned.startsWith('.')) {
    throw new Error(`procedural store: ${field} sanitises to an invalid path segment`)
  }
  return cleaned
}

const requireText = (value: string, field: string): string => {
  const trimmed = optionalText(value)
  if (trimmed === undefined) {
    throw new Error(`procedural store: ${field} must not be empty`)
  }
  return trimmed
}

const optionalText = (value: string | undefined): string | undefined => {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed === '' ? undefined : trimmed
}

const normaliseStringList = (values: readonly string[] | undefined): readonly string[] =>
  dedupeStrings(
    (values ?? [])
      .map((value) => stripWrappingQuotes(value.trim()))
      .filter((value) => value !== ''),
  )

const dedupeStrings = (values: readonly string[]): readonly string[] => {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    if (seen.has(value)) continue
    seen.add(value)
    result.push(value)
  }
  return result
}

const stripWrappingQuotes = (value: string): string => {
  if (value.length < 2) return value
  const first = value[0]
  const last = value[value.length - 1]
  if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
    return value.slice(1, -1)
  }
  return value
}

const tokeniseSearchText = (value: string): readonly string[] =>
  normaliseSearchText(value)
    .split(/[^a-z0-9._-]+/g)
    .filter((token) => token !== '')

const normaliseSearchText = (value: string): string => collapseWhitespace(value).toLowerCase()

const collapseWhitespace = (value: string): string => value.replace(/\s+/g, ' ').trim()

const parseTimestamp = (value: Date | string | undefined): number | undefined => {
  if (value instanceof Date) {
    return Number.isNaN(value.valueOf()) ? undefined : value.valueOf()
  }
  if (typeof value !== 'string' || value.trim() === '') return undefined
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

const timestampValue = (value: string): number => parseTimestamp(value) ?? 0

const normaliseTimestamp = (value: Date | string | undefined): string => {
  const parsed = parseTimestamp(value)
  return parsed === undefined ? new Date().toISOString() : new Date(parsed).toISOString()
}

const sanitisePositiveInt = (value: number | undefined, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback

const coerceTier = (value: unknown): ProceduralTier | undefined =>
  value === 'skill' || value === 'agent' ? value : undefined

const coerceOutcome = (value: unknown): ProceduralOutcome | undefined =>
  value === 'ok' || value === 'error' || value === 'partial' ? value : undefined

const slugify = (value: string): string => {
  const base = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return base === '' ? 'record' : base.slice(0, 48)
}

const compactTimestamp = (value: string): string =>
  value.replace(/[-:]/g, '').replace(/\.\d{3}Z$/, 'Z')

const renderFrontmatterValue = (value: string): string =>
  /^[A-Za-z0-9._:/+-]+$/.test(value) ? value : JSON.stringify(value)

const stableHash = (value: string): string => {
  let hash = FNV_OFFSET_64
  for (const char of value) {
    hash ^= BigInt(char.codePointAt(0) ?? 0)
    hash = (hash * FNV_PRIME_64) & FNV_MASK_64
  }
  return hash.toString(16).padStart(16, '0')
}
