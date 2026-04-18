// SPDX-License-Identifier: Apache-2.0

/**
 * Reflect stage. End-of-session pass that asks the provider to extract
 * generalisable heuristics + a short summary, then persists a reflection
 * file at `reflections/<sessionId>.md` with structured frontmatter + open
 * questions. Writes flow through a single Store batch so partial
 * reflections cannot leak out on error.
 */

import type { Logger, Message, Provider } from '../llm/index.js'
import type { Store } from '../store/index.js'
import { buildFrontmatter } from './frontmatter.js'
import { parseFrontmatter } from './frontmatter.js'
import { reflectionPath, scopeIndex, scopeTopic } from './paths.js'
import { fireReflectionEnd, fireReflectionStart } from './plugins.js'
import { REFLECTION_SYSTEM_PROMPT } from './prompts.js'
import type {
  Heuristic,
  Plugin,
  ReflectArgs,
  ReflectionResult,
  Scope,
} from './types.js'

const REFLECT_MAX_TOKENS = 4096
const REFLECT_TEMPERATURE = 0.3
const REFLECT_MAX_RECENT = 60
const HEURISTIC_TAG = 'heuristic'
const HEURISTIC_ENTRY_LIMIT = 140

export type ReflectDeps = {
  readonly store: Store
  readonly provider: Provider
  readonly logger: Logger
  readonly plugins: readonly Plugin[]
  readonly defaultScope: Scope
  readonly defaultActorId: string
}

export const createReflect = (deps: ReflectDeps) => {
  return async (args: ReflectArgs): Promise<ReflectionResult | undefined> => {
    const actorId = args.actorId ?? deps.defaultActorId
    const scope = args.scope ?? deps.defaultScope
    const messages = args.messages

    await fireReflectionStart(
      deps.plugins,
      { actorId, scope, messages },
      deps.logger,
    )

    const recent =
      messages.length > REFLECT_MAX_RECENT
        ? messages.slice(-REFLECT_MAX_RECENT)
        : messages
    const userPrompt = buildReflectionPrompt(recent)

    let raw: string
    try {
      const resp = await deps.provider.complete({
        messages: [{ role: 'user', content: userPrompt }],
        system: REFLECTION_SYSTEM_PROMPT,
        maxTokens: REFLECT_MAX_TOKENS,
        temperature: REFLECT_TEMPERATURE,
      })
      raw = resp.content
    } catch (err) {
      deps.logger.warn('memory: reflect provider call failed', {
        err: err instanceof Error ? err.message : String(err),
      })
      return undefined
    }

    const parsed = parseReflectionJson(raw)
    if (!parsed.outcome) {
      await fireReflectionEnd(
        deps.plugins,
        { actorId, scope, messages },
        deps.logger,
      )
      return undefined
    }

    const path = reflectionPath(args.sessionId)
    const fileContent = buildReflectionFile(parsed, args.sessionId, scope)

    try {
      await deps.store.batch({ reason: 'reflect' }, async (b) => {
        await b.write(path, Buffer.from(fileContent, 'utf8'))
        await persistHeuristicsInBatch(b, {
          actorId,
          sessionId: args.sessionId,
          heuristics: parsed.heuristics,
        })
      })
    } catch (err) {
      deps.logger.warn('memory: reflect persist failed', {
        err: err instanceof Error ? err.message : String(err),
      })
      return undefined
    }

    const result: ReflectionResult = { ...parsed, path }
    await fireReflectionEnd(
      deps.plugins,
      { actorId, scope, messages, result },
      deps.logger,
    )
    return result
  }
}

type ParsedReflection = {
  outcome: 'success' | 'partial' | 'failure' | 'unknown'
  summary: string
  retryFeedback: string
  shouldRecordEpisode: boolean
  openQuestions: readonly string[]
  heuristics: readonly Heuristic[]
}

export const parseReflectionJson = (content: string): ParsedReflection => {
  const trimmed = content.trim()
  const first = trimmed.indexOf('{')
  const last = trimmed.lastIndexOf('}')
  if (first < 0 || last <= first) {
    return {
      outcome: 'unknown',
      summary: '',
      retryFeedback: '',
      shouldRecordEpisode: false,
      openQuestions: [],
      heuristics: [],
    }
  }
  const slice = trimmed.slice(first, last + 1)
  try {
    const parsed = JSON.parse(slice) as Record<string, unknown>
    const outcome =
      parsed.outcome === 'success' ||
      parsed.outcome === 'partial' ||
      parsed.outcome === 'failure'
        ? parsed.outcome
        : 'unknown'
    const summary = typeof parsed.summary === 'string' ? parsed.summary : ''
    const retryFeedback =
      typeof parsed.retry_feedback === 'string'
        ? parsed.retry_feedback
        : typeof parsed.retryFeedback === 'string'
          ? parsed.retryFeedback
          : ''
    const shouldRecordEpisode =
      typeof parsed.should_record_episode === 'boolean'
        ? parsed.should_record_episode
        : typeof parsed.shouldRecordEpisode === 'boolean'
          ? parsed.shouldRecordEpisode
          : false
    const openQuestionsRaw = Array.isArray(parsed.open_questions)
      ? parsed.open_questions
      : Array.isArray(parsed.openQuestions)
        ? parsed.openQuestions
        : []
    const openQuestions = openQuestionsRaw.filter(
      (s): s is string => typeof s === 'string',
    )
    const heuristicsRaw = Array.isArray(parsed.heuristics) ? parsed.heuristics : []
    const heuristics: Heuristic[] = []
    for (const h of heuristicsRaw) {
      if (typeof h !== 'object' || h === null) continue
      const r = h as Record<string, unknown>
      heuristics.push({
        rule: typeof r.rule === 'string' ? r.rule : '',
        context: typeof r.context === 'string' ? r.context : '',
        confidence:
          r.confidence === 'high' || r.confidence === 'medium' || r.confidence === 'low'
            ? r.confidence
            : 'low',
        category: typeof r.category === 'string' ? r.category : '',
        scope:
          r.scope === 'global' || r.scope === 'project' || r.scope === 'agent'
            ? r.scope
            : 'project',
        antiPattern:
          r.anti_pattern === true || r.antiPattern === true ? true : false,
      })
    }
    return {
      outcome,
      summary,
      retryFeedback,
      shouldRecordEpisode,
      openQuestions,
      heuristics,
    }
  } catch {
    return {
      outcome: 'unknown',
      summary: '',
      retryFeedback: '',
      shouldRecordEpisode: false,
      openQuestions: [],
      heuristics: [],
    }
  }
}

const buildReflectionPrompt = (messages: readonly Message[]): string => {
  const parts: string[] = ['## Session transcript', '']
  for (const m of messages) {
    let content = m.content ?? ''
    if (content.length > 1000) content = `${content.slice(0, 1000)}\n[...truncated]`
    if (m.role === 'tool') {
      parts.push(`[tool ${m.name ?? ''}]: ${content}`, '')
      continue
    }
    parts.push(`[${m.role}]: ${content}`, '')
  }
  return parts.join('\n')
}

const buildReflectionFile = (
  r: ParsedReflection,
  sessionId: string,
  scope: Scope,
): string => {
  const now = new Date().toISOString()
  const fm = buildFrontmatter({
    extra: {
      should_record_episode: r.shouldRecordEpisode ? 'true' : 'false',
    },
    type: 'reflection',
    scope,
    created: now,
    modified: now,
    source: 'reflection',
    session_id: sessionId,
  })
  const body: string[] = []
  body.push(`# Session ${sessionId}`)
  body.push('')
  body.push(`**Outcome:** ${r.outcome}`)
  body.push(`**Record episode:** ${r.shouldRecordEpisode ? 'yes' : 'no'}`)
  body.push('')
  body.push('## Summary', '', r.summary || '_no summary_', '')
  body.push('## Retry feedback', '', r.retryFeedback || '_none_', '')
  if (r.openQuestions.length > 0) {
    body.push('## Open questions', '')
    for (const q of r.openQuestions) body.push(`- ${q}`)
    body.push('')
  }
  if (r.heuristics.length > 0) {
    body.push('## Heuristics', '')
    for (const h of r.heuristics) {
      const marker = h.antiPattern ? '[anti-pattern]' : '[pattern]'
      body.push(
        `- ${marker} ${h.rule} _(context: ${h.context}; confidence: ${h.confidence}; category: ${h.category}; scope: ${h.scope})_`,
      )
    }
    body.push('')
  }
  return `${fm}\n${body.join('\n')}`
}

const persistHeuristicsInBatch = async (
  batch: import('../store/index.js').Batch,
  input: {
    readonly actorId: string
    readonly sessionId: string
    readonly heuristics: readonly Heuristic[]
  },
): Promise<void> => {
  if (input.heuristics.length === 0) return

  const groupedEntries = new Map<Scope, string[]>()
  const now = new Date().toISOString()
  const deduped = new Map<string, Heuristic>()
  for (const heuristic of input.heuristics) {
    deduped.set(heuristicStoragePath(heuristic, input.actorId), heuristic)
  }

  for (const [path, heuristic] of deduped) {
    const existing = await batch.read(path as import('../store/path.js').Path).catch(() => undefined)
    const parsedExisting =
      existing === undefined ? undefined : parseFrontmatter(existing.toString('utf8')).frontmatter
    await batch.write(
      path as import('../store/path.js').Path,
      Buffer.from(
        buildHeuristicFile({
          heuristic,
          sessionId: input.sessionId,
          created: parsedExisting?.created,
          now,
        }),
        'utf8',
      ),
    )
    const entries = groupedEntries.get(heuristic.scope) ?? []
    entries.push(heuristicIndexEntry(path, heuristic))
    groupedEntries.set(heuristic.scope, entries)
  }

  for (const [scope, entries] of groupedEntries) {
    await appendIndexEntries(batch, scopeIndex(scope, input.actorId), entries)
  }
}

const appendIndexEntries = async (
  batch: import('../store/index.js').Batch,
  indexPath: import('../store/path.js').Path,
  entries: readonly string[],
): Promise<void> => {
  let existing = ''
  try {
    existing = (await batch.read(indexPath)).toString('utf8').trim()
  } catch {
    // no index yet
  }
  let content = existing
  for (const raw of entries) {
    const entry = raw.trim()
    if (entry === '' || content.includes(entry)) continue
    content = content === '' ? entry : `${content}\n${entry}`
  }
  await batch.write(indexPath, Buffer.from(`${content}\n`, 'utf8'))
}

const heuristicStoragePath = (heuristic: Heuristic, actorId: string): string =>
  scopeTopic(heuristic.scope, actorId, heuristicFilename(heuristic))

const heuristicFilename = (heuristic: Heuristic): string => {
  const prefix = heuristic.antiPattern ? 'anti-pattern' : 'heuristic'
  return `${prefix}-${slugify(heuristic.category)}-${slugify(heuristic.rule)}.md`
}

const heuristicIndexEntry = (path: string, heuristic: Heuristic): string =>
  truncateOneLine(
    `- ${path.split('/').pop() ?? path}: ${(heuristic.antiPattern ? 'anti-pattern' : 'heuristic')} ${heuristic.rule}`,
    HEURISTIC_ENTRY_LIMIT,
  )

const buildHeuristicFile = (input: {
  readonly heuristic: Heuristic
  readonly sessionId: string
  readonly created?: string | undefined
  readonly now: string
}): string => {
  const { heuristic } = input
  const fm = buildFrontmatter({
    extra: {},
    name: heuristic.antiPattern ? `Anti-pattern: ${heuristic.rule}` : `Heuristic: ${heuristic.rule}`,
    description: heuristicDescription(heuristic),
    type: heuristic.scope === 'project' ? 'project' : 'feedback',
    scope: heuristic.scope,
    created: input.created ?? input.now,
    modified: input.now,
    confidence: heuristic.confidence,
    source: 'reflection',
    session_id: input.sessionId,
    tags: heuristicTags(heuristic),
  })
  const body = heuristic.antiPattern
    ? [
        `Anti-pattern: ${heuristic.rule}`,
        `Context: ${heuristicContext(heuristic)}`,
        'Why: This caused friction in a previous reflected session.',
        `How to apply: Avoid this when ${heuristicContext(heuristic)}.`,
      ]
    : [
        `Rule: ${heuristic.rule}`,
        `Context: ${heuristicContext(heuristic)}`,
        'Why: This pattern helped in a previous reflected session.',
        `How to apply: Use this when ${heuristicContext(heuristic)}.`,
      ]
  return `${fm}\n${body.join('\n\n')}\n`
}

const heuristicDescription = (heuristic: Heuristic): string =>
  truncateOneLine(
    `${heuristic.antiPattern ? 'Anti-pattern' : 'Heuristic'} for ${heuristicContext(heuristic)}`,
    140,
  )

const heuristicTags = (heuristic: Heuristic): readonly string[] =>
  [
    HEURISTIC_TAG,
    heuristic.category.trim().toLowerCase(),
    heuristic.confidence.trim().toLowerCase(),
    ...(heuristic.antiPattern ? ['anti-pattern'] : ['pattern']),
  ].filter((tag, index, tags) => tag !== '' && tags.indexOf(tag) === index)

const heuristicContext = (heuristic: Heuristic): string =>
  heuristic.context.trim() !== '' ? heuristic.context.trim() : 'the same type of work'

const slugify = (value: string): string => {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return slug === '' ? 'note' : slug.slice(0, 64).replace(/-+$/g, '')
}

const truncateOneLine = (value: string, limit: number): string => {
  const collapsed = value.replace(/\s+/g, ' ').trim()
  if (collapsed.length <= limit) return collapsed
  return `${collapsed.slice(0, Math.max(limit - 3, 1)).trimEnd()}...`
}
