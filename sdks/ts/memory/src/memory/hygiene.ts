// SPDX-License-Identifier: Apache-2.0

import { ErrNotFound } from '../store/errors.js'
import type { Batch, Store } from '../store/index.js'
import { type Path, lastSegment } from '../store/path.js'
import { type Frontmatter, buildFrontmatter, parseFrontmatter } from './frontmatter.js'
import { scopePrefix } from './paths.js'
import type { Scope } from './types.js'

export const DEFAULT_RETIRED_AGE_DAYS = 30

export type MemoryHygieneArgs = {
  readonly scope?: Scope
  readonly actorId?: string
  readonly retiredAgeDays?: number
  readonly apply?: boolean
  readonly now?: Date | string
}

export type RunMemoryHygieneOptions = MemoryHygieneArgs & {
  readonly store: Store
  readonly scope: Scope
  readonly actorId: string
}

export type MemoryHygieneTopic = {
  readonly name: string
  readonly description: string
  readonly type: string
  readonly path: Path
  readonly created?: string
  readonly modified?: string
  readonly tags: readonly string[]
  readonly confidence?: string
  readonly source?: string
  readonly scope: Scope
}

export type MemoryContradictionGroup = {
  readonly key: string
  readonly keyReason: 'claim_key' | 'name'
  readonly scope: Scope
  readonly actorId: string
  readonly members: readonly MemoryHygieneTopic[]
  readonly canonical?: Path
}

export type MemoryAgingRetirement = {
  readonly path: Path
  readonly ageMs: number
}

export type MemoryHygieneReport = {
  readonly contradictions: readonly MemoryContradictionGroup[]
  readonly agingRetired: readonly MemoryAgingRetirement[]
  readonly errors: readonly string[]
}

type TopicRecord = MemoryHygieneTopic & {
  readonly frontmatter: Frontmatter
  readonly content: string
  readonly body: string
}

export const runMemoryHygiene = async (
  opts: RunMemoryHygieneOptions,
): Promise<MemoryHygieneReport> => {
  const now = coerceDate(opts.now) ?? new Date()
  const retiredAgeDays =
    opts.retiredAgeDays === undefined || opts.retiredAgeDays <= 0
      ? DEFAULT_RETIRED_AGE_DAYS
      : opts.retiredAgeDays
  const maxAgeMs = retiredAgeDays * 24 * 60 * 60 * 1000
  const prefix = scopePrefix(opts.scope, opts.actorId)
  const errors: string[] = []
  const topics = await listTopicRecords(opts.store, prefix, opts.scope, errors)

  const contradictions = detectContradictions(topics, opts.scope, opts.actorId)
  const agingRetired = detectAgingRetirements(topics, now, maxAgeMs)

  if (opts.apply === true && (contradictions.length > 0 || agingRetired.length > 0)) {
    await applyHygiene(opts.store, {
      scope: opts.scope,
      actorId: opts.actorId,
      now,
      contradictions,
      agingRetired,
      errors,
    })
  }

  return {
    contradictions:
      opts.apply === true
        ? contradictions
        : contradictions.map(({ canonical: _canonical, ...group }) => group),
    agingRetired,
    errors,
  }
}

const detectContradictions = (
  topics: readonly TopicRecord[],
  scope: Scope,
  actorId: string,
): MemoryContradictionGroup[] => {
  const groupsByClaim = new Map<string, TopicRecord[]>()
  const groupsByName = new Map<string, TopicRecord[]>()

  for (const topic of topics) {
    if (topic.frontmatter.superseded_by || topic.frontmatter.retired === true) continue
    if (topic.frontmatter.claim_key) {
      appendGroup(groupsByClaim, topic.frontmatter.claim_key, topic)
    }
    if (topic.frontmatter.name) {
      appendGroup(groupsByName, topic.frontmatter.name.trim().toLowerCase(), topic)
    }
  }

  return [
    ...emitContradictionGroups(groupsByClaim, 'claim_key', scope, actorId),
    ...emitContradictionGroups(groupsByName, 'name', scope, actorId),
  ]
}

const appendGroup = (groups: Map<string, TopicRecord[]>, key: string, topic: TopicRecord): void => {
  const existing = groups.get(key)
  if (existing === undefined) {
    groups.set(key, [topic])
    return
  }
  existing.push(topic)
}

const emitContradictionGroups = (
  groups: ReadonlyMap<string, readonly TopicRecord[]>,
  keyReason: MemoryContradictionGroup['keyReason'],
  scope: Scope,
  actorId: string,
): MemoryContradictionGroup[] => {
  const out: MemoryContradictionGroup[] = []
  const keys = [...groups.keys()].sort()
  for (const key of keys) {
    const members = uniqueTopics(groups.get(key) ?? [])
    if (members.length < 2) continue
    out.push({
      key,
      keyReason,
      scope,
      actorId,
      members,
    })
  }
  return out
}

const detectAgingRetirements = (
  topics: readonly TopicRecord[],
  now: Date,
  maxAgeMs: number,
): MemoryAgingRetirement[] => {
  const out: MemoryAgingRetirement[] = []
  const threshold = now.getTime() - maxAgeMs
  for (const topic of topics) {
    if (!topic.frontmatter.superseded_by || topic.frontmatter.retired === true) continue
    const modified = parseDate(topic.frontmatter.modified)
    if (modified === undefined || modified.getTime() >= threshold) continue
    out.push({
      path: topic.path,
      ageMs: now.getTime() - modified.getTime(),
    })
  }
  return out
}

const applyHygiene = async (
  store: Store,
  input: {
    readonly scope: Scope
    readonly actorId: string
    readonly now: Date
    readonly contradictions: MemoryContradictionGroup[]
    readonly agingRetired: readonly MemoryAgingRetirement[]
    readonly errors: string[]
  },
): Promise<void> => {
  try {
    await store.batch(
      { reason: `memory:hygiene ${input.scope}${projectSuffix(input)}` },
      async (b) => {
        for (let index = 0; index < input.contradictions.length; index++) {
          const group = input.contradictions[index]
          if (group === undefined) continue
          const canonical = pickCanonical(group.members)
          if (canonical === undefined) continue
          input.contradictions[index] = { ...group, canonical: canonical.path }
          const canonicalFile = lastSegment(canonical.path)
          for (const member of group.members) {
            if (member.path === canonical.path) continue
            try {
              await stampSupersededBy(b, member.path, canonicalFile)
            } catch (err) {
              input.errors.push(
                `stamp ${member.path}: ${err instanceof Error ? err.message : String(err)}`,
              )
            }
          }
        }

        for (const retirement of input.agingRetired) {
          try {
            const existing = (await b.read(retirement.path)).toString('utf8')
            const reason = `auto-retired by hygiene: superseded for ${formatAge(retirement.ageMs)}`
            const stamped = retireFrontmatterInPlace(existing, input.now, reason)
            await b.write(retirement.path, Buffer.from(stamped, 'utf8'))
          } catch (err) {
            input.errors.push(
              `retire ${retirement.path}: ${err instanceof Error ? err.message : String(err)}`,
            )
          }
        }
      },
    )
  } catch (err) {
    input.errors.push(`hygiene batch failed: ${err instanceof Error ? err.message : String(err)}`)
  }
}

const listTopicRecords = async (
  store: Store,
  prefix: Path,
  scope: Scope,
  errors: string[],
): Promise<TopicRecord[]> => {
  let entries: Awaited<ReturnType<Store['list']>>
  try {
    entries = await store.list(prefix, { recursive: true, includeGenerated: true })
  } catch (err) {
    if (err instanceof ErrNotFound) return []
    errors.push(`listing ${prefix}: ${err instanceof Error ? err.message : String(err)}`)
    return []
  }

  const topics: TopicRecord[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    const filename = lastSegment(entry.path)
    if (!filename.endsWith('.md') || filename === 'MEMORY.md') continue
    try {
      const content = (await store.read(entry.path)).toString('utf8')
      const parsed = parseFrontmatter(content)
      const topicName = parsed.frontmatter.name ?? filename.replace(/\.md$/, '')
      topics.push({
        name: topicName,
        description: parsed.frontmatter.description ?? '',
        type: parsed.frontmatter.type ?? '',
        path: entry.path,
        tags: parsed.frontmatter.tags ?? [],
        scope,
        frontmatter: parsed.frontmatter,
        content,
        body: parsed.body,
        ...(parsed.frontmatter.created !== undefined
          ? { created: parsed.frontmatter.created }
          : {}),
        ...(parsed.frontmatter.modified !== undefined
          ? { modified: parsed.frontmatter.modified }
          : {}),
        ...(parsed.frontmatter.confidence !== undefined
          ? { confidence: parsed.frontmatter.confidence }
          : {}),
        ...(parsed.frontmatter.source !== undefined ? { source: parsed.frontmatter.source } : {}),
      })
    } catch (err) {
      errors.push(`reading ${entry.path}: ${err instanceof Error ? err.message : String(err)}`)
    }
  }

  return topics
}

const pickCanonical = (members: readonly MemoryHygieneTopic[]): MemoryHygieneTopic | undefined => {
  return [...members].sort((left, right) => {
    const confidenceDelta = confidenceRank(right.confidence) - confidenceRank(left.confidence)
    if (confidenceDelta !== 0) return confidenceDelta
    const leftModified = parseDate(left.modified)?.getTime() ?? 0
    const rightModified = parseDate(right.modified)?.getTime() ?? 0
    if (leftModified !== rightModified) return rightModified - leftModified
    return left.path.localeCompare(right.path)
  })[0]
}

const stampSupersededBy = async (batch: Batch, path: Path, newFile: string): Promise<void> => {
  const content = (await batch.read(path)).toString('utf8')
  if (!hasFrontmatter(content)) return
  const parsed = parseFrontmatter(content)
  const next = buildNoteContent(
    {
      ...parsed.frontmatter,
      superseded_by: newFile,
    },
    parsed.body,
  )
  await batch.write(path, Buffer.from(next, 'utf8'))
}

const retireFrontmatterInPlace = (content: string, now: Date, reason: string): string => {
  if (!hasFrontmatter(content)) return content
  const parsed = parseFrontmatter(content)
  if (parsed.frontmatter.retired === true) return content
  return buildNoteContent(
    {
      ...parsed.frontmatter,
      retired: true,
      retired_on: formatDateOnly(now),
      ...(reason !== '' ? { retired_reason: reason } : {}),
    },
    parsed.body,
  )
}

const buildNoteContent = (frontmatter: Frontmatter, body: string): string => {
  const built = buildFrontmatter(frontmatter)
  const trimmedBody = body.trim()
  return trimmedBody === '' ? `${built}\n` : `${built}\n${trimmedBody}\n`
}

const uniqueTopics = (topics: readonly TopicRecord[]): MemoryHygieneTopic[] => {
  const out: MemoryHygieneTopic[] = []
  const seen = new Set<string>()
  for (const topic of topics) {
    if (seen.has(topic.path)) continue
    seen.add(topic.path)
    out.push(topic)
  }
  return out
}

const confidenceRank = (value: string | undefined): number => {
  switch (value?.trim().toLowerCase()) {
    case 'high':
      return 3
    case 'medium':
      return 2
    case 'low':
      return 1
    default:
      return 0
  }
}

const parseDate = (value: string | undefined): Date | undefined => {
  if (value === undefined || value.trim() === '') return undefined
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? undefined : parsed
}

const coerceDate = (value: Date | string | undefined): Date | undefined => {
  if (value === undefined) return undefined
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? undefined : value
  return parseDate(value)
}

const hasFrontmatter = (content: string): boolean => content.split('\n')[0]?.trim() === '---'

const formatDateOnly = (date: Date): string => date.toISOString().slice(0, 10)

const formatAge = (ageMs: number): string => {
  const days = Math.round(ageMs / (24 * 60 * 60 * 1000))
  return `${days * 24}h0m0s`
}

const projectSuffix = (input: { readonly scope: Scope; readonly actorId: string }): string =>
  input.scope === 'project' ? `/${input.actorId}` : ''
