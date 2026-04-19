// SPDX-License-Identifier: Apache-2.0

/**
 * Extract stage. Given a batch of conversation messages, ask the provider
 * to distil durable knowledge, then persist each note via the Store using
 * a single `Reason: 'extract'` batch.
 *
 * No in-memory cursor state. The actor or session cursor is read from the
 * injected `CursorStore` on entry and written back once extraction completes.
 */

import type { Logger, Message, Provider } from '../llm/index.js'
import { expandTemporal } from '../query/temporal.js'
import { lastSegment } from '../store/path.js'
import type { Store } from '../store/index.js'
import { CONTEXTUAL_PREFIX_MARKER } from './prompts.js'
import { EXTRACTION_SYSTEM_PROMPT } from './prompts.js'
import { buildFrontmatter } from './frontmatter.js'
import { parseFrontmatter } from './frontmatter.js'
import { ensureMarkdown, scopeIndex, scopePrefix, scopeTopic } from './paths.js'
import { fireExtractionEnd, fireExtractionStart } from './plugins.js'
import type {
  ContextualPrefixBuilder,
  CursorStore,
  ExtractArgs,
  ExtractedMemory,
  Plugin,
  Scope,
} from './types.js'

const EXTRACT_MAX_TOKENS = 4096
const EXTRACT_TEMPERATURE = 0
const DEFAULT_MIN_MESSAGES = 2
const DEFAULT_MAX_RECENT = 80
const EXISTING_MEMORY_LIMIT = 24
const EXISTING_MEMORY_PREVIEW_LIMIT = 400
const TRAILING_COMMA_RE = /,(\s*[}\]])/g
const DATE_TAG_RE = /\b\d{4}[-/]\d{2}[-/]\d{2}\b/g
const WEEKDAY_TAG_RE =
  /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/gi
const QUANTITY_TAG_RE = /\b\d{1,6}(?:\.\d+)?\b/g
const PROPER_NOUN_TAG_RE = /\b[A-Z][a-zA-Z]+\b/g
const MONEY_TAG_RE = /[\$£€]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?/g
const UNIT_QUANTITY_TAG_RE =
  /\b(\d{1,6}(?:\.\d+)?)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?|km|kilometres?|miles?|metres?|meters?|kg|kilograms?|pounds?|lbs?|grams?|percent|%|kbps|mbps|gbps|tbps|mb\/s|gb\/s|tb\/s)\b/gi
const WORD_UNIT_QUANTITY_TAG_RE =
  /\b(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(minutes?|mins?|hours?|hrs?|seconds?|secs?|days?|weeks?|months?|years?)\b/gi
const ORDINAL_QUANTITY_TAG_RE =
  /\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b/i
const MONTH_NAME_DATE_RE =
  /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b/i
const DATE_INPUT_RE =
  /^(\d{4})[/-](\d{2})[/-](\d{2})(?:\s+\([A-Za-z]{3}\))?(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?$/
const SYSTEM_SESSION_DATE_RE =
  /\b\d{4}[/-]\d{2}[/-]\d{2}(?:\s+\([A-Za-z]{3}\))?(?:\s+\d{2}:\d{2}(?::\d{2})?)?\b/
const HEURISTIC_USER_FACT_LIMIT = 2
const HEURISTIC_MILESTONE_FACT_LIMIT = 2
const HEURISTIC_PREFERENCE_FACT_LIMIT = 2
const HEURISTIC_PENDING_FACT_LIMIT = 3
const HEURISTIC_EVENT_FACT_LIMIT = 2
const FIRST_PERSON_FACT_RE = /\b(i|i'm|i’ve|i've|my|we|we're|we’ve|we've|our)\b/i
const HEURISTIC_WORD_RE = /[A-Za-z][A-Za-z-]{2,}/g
const HEURISTIC_CADENCE_FACT_RE =
  /\b(?:(?:every|each)\s+(?:day|morning|afternoon|evening|night|weekday|weekend|week|month|year|other\s+week|two\s+weeks?|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:once|twice)\s+(?:a|per)\s+(?:day|week|month|year)|\d+\s+times?\s+(?:a|per)\s+(?:day|week|month|year)|daily|weekly|monthly|yearly|annually|usually|normally|bi-?weekly|fortnightly)\b/i
const HEURISTIC_STORAGE_LOCATION_FACT_RE =
  /\b(?:i|i'm|i’ve|i've|i have)\s+(?:been\s+)?(?:keep(?:ing)?|kept|stor(?:e|ing|ed)|stash(?:ed|ing)?|leave|left|put|placed)\b[^.!?\n]*\b(?:under|inside|in|on|at|behind|beside|next to)\b/i
const HEURISTIC_QUESTION_LEAD_RE =
  /^\s*(?:what|when|where|why|how|which|who|can|could|should|would|do|did|does|is|are|am|have|has|had)\b/i
const SYSTEM_SESSION_ID_RE =
  /(?im)\bsession[_ ]id\s*[:=]\s*([A-Za-z0-9._-]+)\b/
const SENTENCE_SPLIT_PROTECTED_ABBREVIATIONS: ReadonlyArray<readonly [RegExp, string]> = [
  [/\bDr\./g, 'Dr__DOT__'],
  [/\bMr\./g, 'Mr__DOT__'],
  [/\bMrs\./g, 'Mrs__DOT__'],
  [/\bMs\./g, 'Ms__DOT__'],
  [/\bProf\./g, 'Prof__DOT__'],
] as const
const HEURISTIC_MILESTONE_EVENT_RE =
  /\b(?:(?:just|recently)\s+)?(?:completed|submitted|graduated|finished|started|joined|accepted|presented)\b/i
const HEURISTIC_MILESTONE_TIME_RE =
  /\b(?:today|yesterday|recently|just|last\s+(?:week|month|year|summer|spring|fall|autumn|winter))\b/i
const HEURISTIC_MILESTONE_TOPIC_RE =
  /\b(?:degree|thesis|dissertation|paper|research|conference|course|class|project|internship|job|role|group|club|community|network|forum|association|society|linkedin)\b/i
const HEURISTIC_PREFERENCE_BESIDES_LIKE_RE =
  /\bbesides\s+([^.!?]+?),\s*i\s+(?:also\s+)?like\s+([^.!?]+?)(?:[.!?]|$)/i
const HEURISTIC_PREFERENCE_LIKE_RE =
  /\bi\s+(?:also\s+)?(?:like|love|prefer|enjoy)\s+([^.!?]+?)(?:[.!?]|$)/i
const HEURISTIC_PREFERENCE_COMPATIBLE_RE =
  /\bcompatible with (?:my|the)\s+([^.!?,\n]+)/i
const HEURISTIC_PREFERENCE_DESIGNED_FOR_RE =
  /\bspecifically designed for\s+([^.!?,\n]+)/i
const HEURISTIC_PREFERENCE_AS_USER_RE =
  /\bas a[n]?\s+([^.!?,\n]+?)\s+user\b/i
const HEURISTIC_PREFERENCE_FIELD_RE =
  /\bfield of\s+([^.!?,\n]+)/i
const HEURISTIC_PREFERENCE_ADVANCED_RE =
  /\badvanced topics in\s+([^.!?,\n]+)/i
const HEURISTIC_PREFERENCE_SKIP_BASICS_RE = /\bskip the basics\b/i
const HEURISTIC_PREFERENCE_WORKING_IN_FIELD_RE =
  /\b(?:i am|i'm)\s+working in the field\b/i
const HEURISTIC_PENDING_ACTION_LEAD_RE =
  /^\s*(?:i(?:'ve)?(?:\s+still)?\s+(?:need|have)\s+to|i(?:'ve)?\s+got\s+to|i\s+must|i\s+should|i\s+need\s+to\s+remember\s+to|remember\s+to|don't\s+let\s+me\s+forget\s+to)\s+([^.!?]+)/i
const HEURISTIC_PENDING_ACTION_START_RE =
  /^(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|call|email|pay|renew|cancel|buy|send|post|fix|follow\s+up)\b/i
const HEURISTIC_APPOINTMENT_RE =
  /\b(?:appointment|check-?up|consultation|follow-?up|therapy session|scan|surgery|dentist|doctor|gp|dermatologist|orthodontist|hygienist|therapist|counsellor|counselor|psychiatrist|psychologist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian)\b/i
const HEURISTIC_MEDICAL_ENTITY_RE =
  /\b(?:gp|doctor|dentist|dermatologist|orthodontist|hygienist|therapist|counsellor|counselor|psychiatrist|psychologist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian)\b/i
const HEURISTIC_EVENT_RE =
  /\b(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar|service|mass|worship|prayer)\b/i
const HEURISTIC_EVENT_ATTENDANCE_RE =
  /\b(?:attend(?:ed|ing)?|went to|go(?:ing)? to|joined|join(?:ing)?|participat(?:ed|ing)|volunteer(?:ed|ing)|present(?:ed|ing)|watch(?:ed|ing)|listen(?:ed|ing)\s+to|got back from|completed)\b/i
const HEURISTIC_RELIGIOUS_SERVICE_RE =
  /\battend(?:ed|ing)?\s+([^,.!?]+?\s+service(?:\s+at\s+[^,.!?]+)?)\b/i
const HEURISTIC_EVENT_TITLE_RE =
  /\b([A-Z][A-Za-z0-9&'/-]*(?:\s+[A-Z][A-Za-z0-9&'/-]*){0,6}\s+(?:Workshop|Conference|Concert|Festival|Meetup|Show|Screening|Class|Course|Webinar|Lecture|Seminar))\b/
const HEURISTIC_WITH_PERSON_RE = /\bwith\s+(Dr\.?\s+[A-Z][a-zA-Z'-]+)\b/
const HEURISTIC_RELATIVE_DATE_RE =
  /\b(?:today|tomorrow|tonight|this morning|this afternoon|this evening|this weekend|next weekend|next week|next month|coming week|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|coming\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b/i
const HEURISTIC_CLOCK_TIME_RE =
  /\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s?(?:am|pm)|\d{1,2}:\d{2})\b/i
const HEURISTIC_DURATION_FACT_RE =
  /\b(?:\d{1,4}-day|[a-z]+-day|[a-z]+-week|[a-z]+-month|[a-z]+-year|week-long|month-long|year-long)\b/i
const HEURISTIC_RECOMMENDATION_REQUEST_RE =
  /\b(?:recommend|suggest|looking for|look for|what should i|which should i|where should i stay|what to watch|what to read|what to serve)\b/i
const HEURISTIC_RECOMMENDATION_UNDER_RE =
  /\bunder\s+(\d{1,4}(?:\.\d+)?\s*(?:minutes?|mins?|hours?|hrs?|pages?|£|€|\$))\b/i
const HEURISTIC_RECOMMENDATION_NOT_TOO_RE = /\b(?:nothing|not)\s+too\s+([^,.!?;\n]+)/i
const HEURISTIC_RECOMMENDATION_WITHOUT_RE =
  /\bwithout\s+([^,.!?;\n]+)/i
const HEURISTIC_RECOMMENDATION_FAMILY_RE =
  /\b(?:family-friendly|kid-friendly)\b/i
const HEURISTIC_RECOMMENDATION_LIGHT_RE =
  /\b(?:light-hearted|feel-good|cosy|cozy)\b/i
const RELATIVE_TEMPORAL_TAG_RE =
  /\b(?:today|tomorrow|tonight|this morning|this afternoon|this evening|this weekend|next weekend|next week|next month|coming week|next\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|this\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|coming\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b/gi
const CLOCK_TIME_TAG_RE =
  /\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b|\b\d{1,2}:\d{2}\b/gi
const PENDING_ACTION_TAG_RE =
  /\b(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|renew|cancel|follow\s+up)\b/gi
const MEDICAL_TAG_RE =
  /\b(?:appointment|check-?up|consultation|follow-?up|doctor|gp|dentist|dermatologist|orthodontist|hygienist|therapist|physio(?:therapist)?|optometrist|ophthalmologist|paediatrician|pediatrician|gynaecologist|gynecologist|cardiologist|neurologist|oncologist|surgeon|vet|veterinarian|clinic|hospital|prescription)\b/gi
const EVENT_TAG_RE =
  /\b(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar)\b/gi
const ENTERTAINMENT_TAG_RE =
  /\b(?:film|movie|show|series|book|novel|game|podcast|cinema)\b/gi
const MONTH_NAME_VALUES = [
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
] as const
const HEURISTIC_STOPWORDS = new Set([
  'been',
  'city',
  'definitely',
  'feels',
  'following',
  'getting',
  'have',
  'just',
  'last',
  'lately',
  'miles',
  'months',
  'really',
  'routine',
  'sticking',
  'their',
  'weeks',
])
const MEMORY_SCOPES: readonly Scope[] = ['global', 'project', 'agent']

export type ExtractDeps = {
  readonly store: Store
  readonly provider: Provider
  readonly cursorStore: CursorStore
  readonly logger: Logger
  readonly plugins: readonly Plugin[]
  readonly defaultScope: Scope
  readonly defaultActorId: string
  readonly minMessages: number
  readonly maxRecent: number
  readonly contextualPrefixBuilder?: ContextualPrefixBuilder
}

type ExtractRunOptions = {
  readonly persist: boolean
  readonly advanceCursor: boolean
}

type ExistingMemorySummary = {
  readonly path: string
  readonly scope: Scope
  readonly name: string
  readonly description: string
  readonly type: string
  readonly modified?: string
  readonly content: string
}

export const createExtract = (deps: ExtractDeps) => {
  return async (args: ExtractArgs): Promise<readonly ExtractedMemory[]> =>
    runExtract(deps, args, { persist: true, advanceCursor: true })
}

export const createPreviewExtract = (deps: ExtractDeps) => {
  return async (args: ExtractArgs): Promise<readonly ExtractedMemory[]> =>
    runExtract(deps, args, { persist: false, advanceCursor: false })
}

const runExtract = async (
  deps: ExtractDeps,
  args: ExtractArgs,
  opts: ExtractRunOptions,
): Promise<readonly ExtractedMemory[]> => {
  const actorId = args.actorId ?? deps.defaultActorId
  const scope = args.scope ?? deps.defaultScope
  const messages = args.messages
  const cursorScope = args.sessionId ? { sessionId: args.sessionId } : undefined

  const cursor = opts.advanceCursor
    ? Math.min(await deps.cursorStore.get(actorId, cursorScope), messages.length)
    : 0
  if (messages.length - cursor < deps.minMessages) {
    return []
  }

  const recent = messages.slice(cursor)
  const windowed = recent.length > deps.maxRecent ? recent.slice(-deps.maxRecent) : recent

  await fireExtractionStart(
    deps.plugins,
    { actorId, scope, messages: windowed, extracted: [] },
    deps.logger,
  )

  let existingMemories: readonly ExistingMemorySummary[] = []
  try {
    existingMemories = await listExistingMemories(deps.store, actorId, scope)
  } catch (err) {
    deps.logger.warn('memory: failed to load existing memories for extraction', {
      err: err instanceof Error ? err.message : String(err),
    })
  }

  const userPrompt = buildExtractUserPrompt(windowed, existingMemories)

  let raw: string
  try {
    const resp = await deps.provider.complete({
      messages: [{ role: 'user', content: userPrompt }],
      system: EXTRACTION_SYSTEM_PROMPT,
      maxTokens: EXTRACT_MAX_TOKENS,
      temperature: EXTRACT_TEMPERATURE,
    })
    raw = resp.content
  } catch (err) {
    deps.logger.warn('memory: extract provider call failed', {
      err: err instanceof Error ? err.message : String(err),
    })
    return []
  }

  const parsed = parseExtractionPayload(raw)
  if (!parsed.parsed) {
    deps.logger.warn('memory: extract response parse failed')
  }
  const normalised = parsed.memories.map((m) =>
    normaliseExtracted(m, scope, args.sessionId, args.sessionDate),
  )
  const extracted = await applyContextualPrefixes(
    deps.contextualPrefixBuilder,
    windowed,
    args.sessionId,
    postProcessSessionExtractions(
      windowed,
      [
        ...normalised,
        ...deriveHeuristicUserFacts(
          windowed,
          normalised,
          args.sessionId,
          args.sessionDate,
        ),
        ...deriveHeuristicPreferenceFacts(
          windowed,
          normalised,
          args.sessionId,
          args.sessionDate,
        ),
        ...deriveHeuristicPendingFacts(
          windowed,
          normalised,
          args.sessionId,
          args.sessionDate,
        ),
        ...deriveHeuristicEventFacts(
          windowed,
          normalised,
          args.sessionId,
          args.sessionDate,
        ),
        ...deriveHeuristicAssistantTableFacts(
          windowed,
          normalised,
          args.sessionId,
          args.sessionDate,
        ),
        ...deriveHeuristicMilestoneFacts(
          windowed,
          normalised,
          args.sessionId,
          args.sessionDate,
        ),
      ],
      args.sessionId,
      args.sessionDate,
    ),
  )

  if (opts.persist && extracted.length > 0) {
    try {
      await persistExtractions(deps.store, actorId, extracted)
    } catch (err) {
      deps.logger.warn('memory: extract persist failed', {
        err: err instanceof Error ? err.message : String(err),
      })
      return []
    }
  }

  if (opts.advanceCursor) {
    await deps.cursorStore.set(actorId, messages.length, cursorScope)
  }

  await fireExtractionEnd(
    deps.plugins,
    { actorId, scope, messages: windowed, extracted },
    deps.logger,
  )

  return extracted
}

export const defaultExtractConfig = (): { minMessages: number; maxRecent: number } => ({
  minMessages: DEFAULT_MIN_MESSAGES,
  maxRecent: DEFAULT_MAX_RECENT,
})

const buildExtractUserPrompt = (
  messages: readonly Message[],
  existingMemories: readonly ExistingMemorySummary[],
): string => {
  const parts: string[] = []
  if (existingMemories.length > 0) {
    parts.push('## Existing memories', '')
    for (const memory of existingMemories) {
      parts.push(`### [${memory.scope}] ${lastSegment(memory.path)}`)
      if (memory.name !== '') parts.push(`name: ${memory.name}`)
      if (memory.description !== '') parts.push(`description: ${memory.description}`)
      if (memory.type !== '') parts.push(`type: ${memory.type}`)
      if (memory.modified !== undefined) parts.push(`modified: ${memory.modified}`)
      if (memory.content !== '') parts.push(`content: ${memory.content}`)
      parts.push('')
    }
  }
  parts.push('## Recent conversation\n')
  for (const m of messages) {
    let content = m.content ?? ''
    if (content.length > 2000) content = `${content.slice(0, 2000)}\n[...truncated]`
    if (m.role === 'tool') {
      if (content.length > 300) content = `${content.slice(0, 300)}...`
      parts.push(`[tool (${m.name ?? ''})]: ${content}\n`)
      continue
    }
    parts.push(`[${m.role}]: ${content}\n`)
  }
  return parts.join('\n')
}

const listExistingMemories = async (
  store: Store,
  actorId: string,
  scope: Scope,
): Promise<readonly ExistingMemorySummary[]> => {
  const scopes: Scope[] = scope === 'global' ? ['global'] : ['global', scope]
  const summaries: ExistingMemorySummary[] = []

  for (const currentScope of scopes) {
    const prefix = scopePrefix(currentScope, actorId)
    const exists = await store.exists(prefix).catch(() => false)
    if (!exists) continue
    const entries = await store.list(prefix, { recursive: true })
    for (const entry of entries) {
      if (entry.isDir) continue
      const filename = lastSegment(entry.path)
      if (!filename.endsWith('.md') || filename === 'MEMORY.md') continue
      const raw = await store.read(entry.path).catch(() => undefined)
      if (raw === undefined) continue
      const parsed = parseFrontmatter(raw.toString('utf8'))
      const preview = truncatePromptContent(parsed.body)
      summaries.push({
        path: entry.path,
        scope: currentScope,
        name: parsed.frontmatter.name?.trim() ?? '',
        description: parsed.frontmatter.description?.trim() ?? '',
        type: parsed.frontmatter.type?.trim() ?? '',
        ...(parsed.frontmatter.modified !== undefined ? { modified: parsed.frontmatter.modified } : {}),
        content: preview,
      })
    }
  }

  summaries.sort((left, right) => {
    const modifiedDiff = memorySummaryTimestamp(right) - memorySummaryTimestamp(left)
    if (modifiedDiff !== 0) return modifiedDiff
    return left.path.localeCompare(right.path)
  })
  return summaries.slice(0, EXISTING_MEMORY_LIMIT)
}

const truncatePromptContent = (content: string): string => {
  const collapsed = content.replace(/\s+/g, ' ').trim()
  if (collapsed.length <= EXISTING_MEMORY_PREVIEW_LIMIT) return collapsed
  return `${collapsed.slice(0, EXISTING_MEMORY_PREVIEW_LIMIT)}...`
}

const memorySummaryTimestamp = (summary: ExistingMemorySummary): number => {
  if (summary.modified === undefined) return 0
  const parsed = Date.parse(summary.modified)
  return Number.isFinite(parsed) ? parsed : 0
}

type RawExtracted = Partial<Record<keyof ExtractedMemory, unknown>> & {
  index_entry?: unknown
  session_id?: unknown
  session_date?: unknown
  observed_on?: unknown
  context_prefix?: unknown
  modified_override?: unknown
}

type ParseExtractionPayload = {
  readonly memories: readonly RawExtracted[]
  readonly parsed: boolean
}

export const parseExtractionJson = (content: string): readonly RawExtracted[] => {
  return parseExtractionPayload(content).memories
}

const parseExtractionPayload = (content: string): ParseExtractionPayload => {
  for (const candidate of extractionJsonCandidates(content)) {
    const parsed = decodeExtractionJsonCandidate(candidate)
    if (parsed.parsed) return parsed
    const repaired = repairExtractionJsonCandidate(candidate)
    if (repaired === candidate) continue
    const repairedParsed = decodeExtractionJsonCandidate(repaired)
    if (repairedParsed.parsed) return repairedParsed
  }
  return { memories: [], parsed: false }
}

const extractionJsonCandidates = (content: string): readonly string[] => {
  const trimmed = content.trim()
  if (trimmed === '') return []
  const out: string[] = []
  const seen = new Set<string>()
  const add = (candidate: string) => {
    const value = candidate.trim()
    if (value === '' || seen.has(value)) return
    seen.add(value)
    out.push(value)
  }
  add(trimmed)
  add(bracketSlice(trimmed, '{', '}'))
  add(bracketSlice(trimmed, '[', ']'))
  return out
}

const bracketSlice = (content: string, open: string, close: string): string => {
  const start = content.indexOf(open)
  const end = content.lastIndexOf(close)
  return start >= 0 && end > start ? content.slice(start, end + 1) : ''
}

const repairExtractionJsonCandidate = (content: string): string => {
  let repaired = content
  while (true) {
    const next = repaired.replace(TRAILING_COMMA_RE, '$1')
    if (next === repaired) return repaired
    repaired = next
  }
}

const decodeExtractionJsonCandidate = (content: string): ParseExtractionPayload => {
  const trimmed = content.trim()
  if (trimmed === '') return { memories: [], parsed: false }
  try {
    const parsed = JSON.parse(trimmed) as unknown
    if (Array.isArray(parsed)) {
      return {
        memories: parsed.filter((m): m is RawExtracted => typeof m === 'object' && m !== null),
        parsed: true,
      }
    }
    if (!parsed || typeof parsed !== 'object') return { memories: [], parsed: false }

    const record = parsed as Record<string, unknown>
    const list = record.memories
    if (Array.isArray(list)) {
      return {
        memories: list.filter((m): m is RawExtracted => typeof m === 'object' && m !== null),
        parsed: true,
      }
    }
    const single = record.memory
    if (single && typeof single === 'object') {
      return { memories: [single as RawExtracted], parsed: true }
    }
    if (looksLikeRawExtracted(record)) {
      return { memories: [record as RawExtracted], parsed: true }
    }
    return { memories: [], parsed: true }
  } catch {
    return { memories: [], parsed: false }
  }
}

const looksLikeRawExtracted = (value: Record<string, unknown>): boolean =>
  typeof value.filename === 'string' ||
  typeof value.content === 'string' ||
  typeof value.name === 'string' ||
  typeof value.description === 'string' ||
  typeof value.indexEntry === 'string' ||
  typeof value.index_entry === 'string'

const normaliseExtracted = (
  raw: RawExtracted,
  defaultScope: Scope,
  sessionId?: string,
  sessionDate?: string,
): ExtractedMemory => {
  const rawSessionId =
    typeof raw.session_id === 'string' && raw.session_id.trim() !== ''
      ? raw.session_id.trim()
      : typeof raw.sessionId === 'string' && raw.sessionId.trim() !== ''
        ? raw.sessionId.trim()
        : sessionId?.trim() ?? ''
  const scopeRaw = typeof raw.scope === 'string' ? raw.scope : ''
  const scope: Scope =
    scopeRaw === 'global' || scopeRaw === 'project' || scopeRaw === 'agent'
      ? scopeRaw
      : defaultScope
  const action =
    raw.action === 'update' || raw.action === 'create' ? raw.action : 'create'
  const type =
    raw.type === 'user' ||
    raw.type === 'feedback' ||
    raw.type === 'project' ||
    raw.type === 'reference'
      ? raw.type
      : 'project'
  const tags = Array.isArray(raw.tags)
    ? raw.tags.filter((t): t is string => typeof t === 'string')
    : undefined
  const filename =
    typeof raw.filename === 'string'
      ? rewriteHeuristicFilenameForSession(raw.filename, rawSessionId)
      : ''
  const memory: ExtractedMemory = {
    action,
    filename,
    name: typeof raw.name === 'string' ? raw.name : '',
    description: typeof raw.description === 'string' ? raw.description : '',
    type,
    content: typeof raw.content === 'string' ? raw.content : '',
    indexEntry:
      typeof raw.indexEntry === 'string'
        ? raw.indexEntry
        : typeof raw.index_entry === 'string'
          ? raw.index_entry
          : '',
    scope,
    ...(typeof raw.supersedes === 'string' && raw.supersedes
      ? { supersedes: raw.supersedes }
      : {}),
    ...(tags && tags.length > 0 ? { tags } : {}),
    ...(rawSessionId !== '' ? { sessionId: rawSessionId } : {}),
    ...(typeof raw.observed_on === 'string' && raw.observed_on
      ? { observedOn: raw.observed_on }
      : typeof raw.observedOn === 'string' && raw.observedOn
        ? { observedOn: raw.observedOn }
        : {}),
    ...(typeof raw.session_date === 'string' && raw.session_date
      ? { sessionDate: raw.session_date }
      : typeof raw.sessionDate === 'string' && raw.sessionDate
        ? { sessionDate: raw.sessionDate }
        : sessionDate
          ? { sessionDate }
          : {}),
    ...(typeof raw.context_prefix === 'string' && raw.context_prefix
      ? { contextPrefix: raw.context_prefix }
      : typeof raw.contextPrefix === 'string' && raw.contextPrefix
        ? { contextPrefix: raw.contextPrefix }
        : {}),
    ...(typeof raw.modified_override === 'string' && raw.modified_override
      ? { modifiedOverride: raw.modified_override }
      : typeof raw.modifiedOverride === 'string' && raw.modifiedOverride
        ? { modifiedOverride: raw.modifiedOverride }
        : {}),
  }
  return memory
}

const persistExtractions = async (
  store: Store,
  actorId: string,
  extracted: readonly ExtractedMemory[],
): Promise<void> => {
  const indexEntriesByScope = new Map<Scope, string[]>()
  type Pending = { path: import('../store/path.js').Path; body: Buffer }
  const writes: Pending[] = []

  for (const em of extracted) {
    if (!em.filename || !em.content) continue
    const filename = ensureMarkdown(em.filename)
    const path = scopeTopic(em.scope, actorId, filename)
    const body = buildTopicFile(em)
    writes.push({ path, body })
    if (em.indexEntry) {
      const arr = indexEntriesByScope.get(em.scope) ?? []
      arr.push(em.indexEntry)
      indexEntriesByScope.set(em.scope, arr)
    }
  }

  if (writes.length === 0) return

  await store.batch({ reason: 'extract' }, async (b) => {
    for (const w of writes) await b.write(w.path, w.body)
    for (const em of extracted) {
      if (!em.filename || !em.content) continue
      if (!em.supersedes) continue
      const oldFile = ensureMarkdown(em.supersedes)
      const newFile = ensureMarkdown(em.filename)
      const oldPath = await resolveSupersededPath(b, actorId, em.scope, oldFile)
      if (oldPath === undefined) continue
      await stampSupersededBy(b, oldPath, newFile)
    }
    for (const [scope, entries] of indexEntriesByScope) {
      const indexPath = scopeIndex(scope, actorId)
      await appendIndexEntries(b, indexPath, entries)
    }
  })
}

const resolveSupersededPath = async (
  batch: import('../store/index.js').Batch,
  actorId: string,
  preferredScope: Scope,
  filename: string,
): Promise<import('../store/path.js').Path | undefined> => {
  const preferredPath = scopeTopic(preferredScope, actorId, filename)
  if (await batchPathExists(batch, preferredPath)) return preferredPath
  for (const scope of MEMORY_SCOPES) {
    if (scope === preferredScope) continue
    const candidatePath = scopeTopic(scope, actorId, filename)
    if (await batchPathExists(batch, candidatePath)) return candidatePath
  }
  return undefined
}

const batchPathExists = async (
  batch: import('../store/index.js').Batch,
  path: import('../store/path.js').Path,
): Promise<boolean> => {
  try {
    await batch.read(path)
    return true
  } catch {
    return false
  }
}

const stampSupersededBy = async (
  batch: import('../store/index.js').Batch,
  path: import('../store/path.js').Path,
  newFile: string,
): Promise<void> => {
  let raw: Buffer
  try {
    raw = await batch.read(path)
  } catch {
    return
  }
  const lines = raw.toString('utf8').split('\n')
  if (lines.length < 2 || (lines[0] ?? '').trim() !== '---') return
  let closeIdx = -1
  for (let i = 1; i < lines.length; i++) {
    if ((lines[i] ?? '').trim() === '---') {
      closeIdx = i
      break
    }
  }
  if (closeIdx < 0) return

  let replaced = false
  for (let i = 1; i < closeIdx; i++) {
    if ((lines[i] ?? '').trim().startsWith('superseded_by:')) {
      lines[i] = `superseded_by: ${newFile}`
      replaced = true
      break
    }
  }
  if (!replaced) {
    lines.splice(closeIdx, 0, `superseded_by: ${newFile}`)
  }
  await batch.write(path, Buffer.from(lines.join('\n'), 'utf8'))
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
    if (!entry) continue
    if (content.includes(entry)) continue
    content = content === '' ? entry : `${content}\n${entry}`
  }
  await batch.write(indexPath, Buffer.from(`${content}\n`, 'utf8'))
}

const buildTopicFile = (em: ExtractedMemory): Buffer => {
  const now = new Date().toISOString()
  const modified = em.modifiedOverride ?? now
  const created = em.modifiedOverride ?? now
  const fm = buildFrontmatter({
    extra: {},
    ...(em.name ? { name: em.name } : {}),
    ...(em.description ? { description: em.description } : {}),
    ...(em.type ? { type: em.type } : {}),
    scope: em.scope,
    ...(em.action === 'create' ? { created } : {}),
    modified,
    source: 'session',
    ...(em.supersedes ? { supersedes: em.supersedes } : {}),
    ...(em.sessionId ? { session_id: em.sessionId } : {}),
    ...(em.observedOn ? { observed_on: em.observedOn } : {}),
    ...(em.sessionDate ? { session_date: em.sessionDate } : {}),
    ...(em.tags && em.tags.length > 0 ? { tags: em.tags } : {}),
  })
  const prefix = em.contextPrefix?.trim()
  const body = prefix ? `${CONTEXTUAL_PREFIX_MARKER}${prefix}\n\n${em.content}` : em.content
  return Buffer.from(`${fm}\n${body}\n`, 'utf8')
}

const deriveHeuristicUserFacts = (
  messages: readonly Message[],
  existing: readonly ExtractedMemory[],
  sessionId?: string,
  sessionDate?: string,
): readonly ExtractedMemory[] => {
  const out: ExtractedMemory[] = []
  const seenSentences = buildExistingMemoryTextSet(existing)
  const resolvedSessionDate = resolveHeuristicSessionDate(messages, sessionDate)
  const stamp = parseSessionDateRfc3339(resolvedSessionDate)
  const iso = shortIsoDate(stamp)

  for (const message of messages) {
    if (message.role !== 'user') continue
    const sentences = heuristicUserFactCandidates(message.content ?? '')
    for (const sentence of sentences) {
      const canonical = sentence.toLowerCase()
      if (!FIRST_PERSON_FACT_RE.test(sentence)) continue
      if (!hasHeuristicUserFactSignal(sentence)) continue
      if (seenSentences.has(canonical)) continue
      const slug = heuristicFactSlug(sentence)
      if (slug === '') continue
      let observedOn = resolveHeuristicObservedOn(sentence, resolvedSessionDate)
      if (observedOn === '' && stamp !== '') observedOn = stamp
      out.push({
        action: 'create',
        filename: buildHeuristicUserFactFilename({
          iso,
          slug,
          ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        }),
        name: `User Fact: ${toTitleCase(slug.replaceAll('-', ' '))}`,
        description: truncateOneLine(sentence, 140),
        type: 'user',
        scope: 'global',
        content: withObservedDatePrefix(sentence, observedOn),
        indexEntry: truncateOneLine(sentence, 140),
        ...(observedOn !== '' ? { observedOn } : {}),
        ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        ...(resolvedSessionDate !== undefined && resolvedSessionDate !== ''
          ? { sessionDate: resolvedSessionDate }
          : {}),
      })
      seenSentences.add(canonical)
      if (out.length >= HEURISTIC_USER_FACT_LIMIT) return out
    }
  }
  return out
}

const buildHeuristicFilename = (
  prefix: string,
  iso: string,
  slug: string,
): string => {
  const parts = [prefix]
  if (iso !== '') parts.push(iso)
  parts.push(slug)
  return `${parts.join('-')}.md`
}

const buildHeuristicSessionFilename = (
  prefix: string,
  iso: string,
  sessionId: string | undefined,
  slug: string,
): string =>
  rewriteHeuristicFilenameForSession(
    buildHeuristicFilename(prefix, iso, slug),
    sessionId,
  )

const buildHeuristicUserFactFilename = (args: {
  readonly iso: string
  readonly sessionId?: string
  readonly slug: string
}): string =>
  buildHeuristicSessionFilename('user-fact', args.iso, args.sessionId, args.slug)

const deriveHeuristicMilestoneFacts = (
  messages: readonly Message[],
  existing: readonly ExtractedMemory[],
  sessionId?: string,
  sessionDate?: string,
): readonly ExtractedMemory[] => {
  const out: ExtractedMemory[] = []
  const seenSentences = buildExistingMemoryTextSet(existing)
  const resolvedSessionDate = resolveHeuristicSessionDate(messages, sessionDate)
  const stamp = parseSessionDateRfc3339(resolvedSessionDate)
  const iso = shortIsoDate(stamp)

  for (const message of messages) {
    if (message.role !== 'user') continue
    const sentences = splitIntoFactSentences(message.content ?? '')
    for (const sentence of sentences) {
      const canonical = sentence.toLowerCase()
      if (!hasMilestoneFact(sentence)) continue
      if (seenSentences.has(canonical)) continue
      const slug = heuristicFactSlug(sentence)
      if (slug === '') continue
      let observedOn = resolveHeuristicObservedOn(sentence, resolvedSessionDate)
      if (observedOn === '' && stamp !== '') observedOn = stamp
      out.push({
        action: 'create',
        filename: buildHeuristicUserFactFilename({
          iso,
          slug: `milestone-${slug}`,
          ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        }),
        name: `User Fact: ${toTitleCase(slug.replaceAll('-', ' '))}`,
        description: truncateOneLine(sentence, 140),
        type: 'user',
        scope: 'global',
        content: withObservedDatePrefix(sentence, observedOn),
        indexEntry: truncateOneLine(sentence, 140),
        ...(observedOn !== '' ? { observedOn } : {}),
        ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        ...(resolvedSessionDate !== undefined && resolvedSessionDate !== ''
          ? { sessionDate: resolvedSessionDate }
          : {}),
      })
      seenSentences.add(canonical)
      if (out.length >= HEURISTIC_MILESTONE_FACT_LIMIT) return out
    }
  }
  return out
}

type HeuristicPreferenceCandidate = {
  readonly summary: string
  readonly evidence: string
}

const deriveHeuristicPreferenceFacts = (
  messages: readonly Message[],
  existing: readonly ExtractedMemory[],
  sessionId?: string,
  sessionDate?: string,
): readonly ExtractedMemory[] => {
  const out: ExtractedMemory[] = []
  const seenSummaries = new Set(
    [
      ...buildExistingMemoryTextSet(existing),
      ...existing
        .filter((memory) => isHeuristicPreferenceFact(memory))
        .map((memory) => heuristicPreferenceSummary(memory.content).toLowerCase()),
    ]
      .filter((value) => value !== ''),
  )
  const resolvedSessionDate = resolveHeuristicSessionDate(messages, sessionDate)
  const stamp = parseSessionDateRfc3339(resolvedSessionDate)
  const iso = shortIsoDate(stamp)

  for (const message of messages) {
    if (message.role !== 'user') continue
    for (const candidate of buildHeuristicPreferenceCandidates(message.content ?? '')) {
      const canonical = candidate.summary.toLowerCase()
      if (seenSummaries.has(canonical)) continue
      const slug = heuristicFactSlug(candidate.summary)
      if (slug === '') continue
      out.push({
        action: 'create',
        filename: buildHeuristicPreferenceFilename({
          iso,
          slug,
          ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        }),
        name: `User Preference: ${toTitleCase(slug.replaceAll('-', ' '))}`,
        description: truncateOneLine(candidate.summary, 140),
        type: 'user',
        scope: 'global',
        content: buildHeuristicPreferenceContent(candidate),
        indexEntry: truncateOneLine(candidate.summary, 140),
        ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        ...(resolvedSessionDate !== undefined && resolvedSessionDate !== ''
          ? { sessionDate: resolvedSessionDate }
          : {}),
      })
      seenSummaries.add(canonical)
      if (out.length >= HEURISTIC_PREFERENCE_FACT_LIMIT) return out
    }
  }

  return out
}

const deriveHeuristicPendingFacts = (
  messages: readonly Message[],
  existing: readonly ExtractedMemory[],
  sessionId?: string,
  sessionDate?: string,
): readonly ExtractedMemory[] => {
  const out: ExtractedMemory[] = []
  const seen = buildExistingMemoryTextSet(existing)
  const resolvedSessionDate = resolveHeuristicSessionDate(messages, sessionDate)
  const stamp = parseSessionDateRfc3339(resolvedSessionDate)
  const iso = shortIsoDate(stamp)

  for (const message of messages) {
    if (message.role !== 'user') continue
    const sentences = splitIntoFactSentences(message.content ?? '')
    for (const sentence of sentences) {
      const pendingActions = extractPendingActions(sentence)
      for (const action of pendingActions) {
        const summary = buildPendingTaskSummary(action)
        const canonical = normaliseMemoryText(summary)
        if (seen.has(canonical)) continue
        const slug = heuristicFactSlug(summary)
        if (slug === '') continue
        out.push({
          action: 'create',
          filename: buildHeuristicUserFactFilename({
            iso,
            slug: `task-${slug}`,
            ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
          }),
          name: `User Task: ${toTitleCase(action)}`,
          description: truncateOneLine(summary, 140),
          type: 'user',
          scope: 'global',
          content: `${summary}\n\nEvidence: ${sentence}`,
          indexEntry: truncateOneLine(summary, 140),
          ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
          ...(resolvedSessionDate !== undefined && resolvedSessionDate !== ''
            ? { sessionDate: resolvedSessionDate }
            : {}),
        })
        seen.add(canonical)
        if (out.length >= HEURISTIC_PENDING_FACT_LIMIT) return out
      }
    }
  }

  return out
}

const deriveHeuristicEventFacts = (
  messages: readonly Message[],
  existing: readonly ExtractedMemory[],
  sessionId?: string,
  sessionDate?: string,
): readonly ExtractedMemory[] => {
  const out: ExtractedMemory[] = []
  const seen = buildExistingMemoryTextSet(existing)
  const resolvedSessionDate = resolveHeuristicSessionDate(messages, sessionDate)
  const stamp = parseSessionDateRfc3339(resolvedSessionDate)
  const iso = shortIsoDate(stamp)

  for (const message of messages) {
    if (message.role !== 'user') continue
    const sentences = splitIntoFactSentences(message.content ?? '')
    for (const sentence of sentences) {
      const summary =
        inferAppointmentSummary(sentence) ??
        inferEventSummary(sentence)
      if (summary === undefined) continue
      const canonical = normaliseMemoryText(summary)
      if (seen.has(canonical)) continue
      const slug = heuristicFactSlug(summary)
      if (slug === '') continue
      let observedOn = resolveHeuristicObservedOn(sentence, resolvedSessionDate)
      if (observedOn === '' && stamp !== '') observedOn = stamp
      out.push({
        action: 'create',
        filename: buildHeuristicUserFactFilename({
          iso,
          slug: `event-${slug}`,
          ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        }),
        name: `User Event: ${toTitleCase(slug.replaceAll('-', ' '))}`,
        description: truncateOneLine(summary, 140),
        type: 'user',
        scope: 'global',
        content: withObservedDatePrefix(`${summary}\n\nEvidence: ${sentence}`, observedOn),
        indexEntry: truncateOneLine(summary, 140),
        ...(observedOn !== '' ? { observedOn } : {}),
        ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
        ...(resolvedSessionDate !== undefined && resolvedSessionDate !== ''
          ? { sessionDate: resolvedSessionDate }
          : {}),
      })
      seen.add(canonical)
      if (out.length >= HEURISTIC_EVENT_FACT_LIMIT) return out
    }
  }

  return out
}

type AssistantTableCell = {
  readonly header: string
  readonly value: string
}

type AssistantMarkdownTable = {
  readonly headers: readonly string[]
  readonly rows: readonly string[][]
}

const deriveHeuristicAssistantTableFacts = (
  messages: readonly Message[],
  existing: readonly ExtractedMemory[],
  sessionId: string | undefined,
  sessionDate: string | undefined,
): readonly ExtractedMemory[] => {
  const out: ExtractedMemory[] = []
  const seen = buildExistingMemoryTextSet(existing)
  const resolvedSessionDate = resolveHeuristicSessionDate(messages, sessionDate)
  const stamp = parseSessionDateRfc3339(resolvedSessionDate)
  const iso = shortIsoDate(stamp)

  for (const message of messages) {
    if (message.role !== 'assistant') continue
    for (const table of extractMarkdownTables(message.content ?? '')) {
      for (const row of table.rows) {
        const weekday = normaliseWeekdayLabel(row[0] ?? '')
        if (weekday === undefined) continue

        const entries = table.headers
          .slice(1)
          .map((header, index) => ({
            header: header.trim(),
            value: (row[index + 1] ?? '').trim(),
          }))
          .filter(
            (entry): entry is AssistantTableCell =>
              entry.header !== '' && entry.value !== '' && entry.value !== '-',
          )

        if (entries.length === 0) continue

        const summary = buildAssistantTableRowSummary(weekday, entries)
        const canonical = normaliseMemoryText(summary)
        if (seen.has(canonical)) continue

        const slug = heuristicFactSlug(summary)
        if (slug === '') continue

        out.push({
          action: 'create',
          filename: buildAssistantTableFilename({ iso, slug }),
          name: `Reference: ${toTitleCase(slug.replaceAll('-', ' '))}`,
          description: truncateOneLine(summary, 140),
          type: 'reference',
          scope: 'project',
          content: withObservedDatePrefix(summary, stamp),
          indexEntry: truncateOneLine(summary, 140),
          ...(stamp !== '' ? { observedOn: stamp } : {}),
          ...(sessionId !== undefined && sessionId !== '' ? { sessionId } : {}),
          ...(resolvedSessionDate !== undefined && resolvedSessionDate !== ''
            ? { sessionDate: resolvedSessionDate }
            : {}),
        })
        seen.add(canonical)
      }
    }
  }

  return out
}

const buildAssistantTableRowSummary = (
  weekday: string,
  entries: readonly AssistantTableCell[],
): string =>
  ensureTrailingFullStop(
    `${weekday} roster: ${entries.map((entry) => `${entry.value}, ${entry.header}`).join('; ')}`,
  )

const buildAssistantTableFilename = (args: {
  readonly iso: string
  readonly slug: string
}): string => buildHeuristicFilename('reference', args.iso, args.slug)

const extractMarkdownTables = (content: string): readonly AssistantMarkdownTable[] => {
  const lines = content.split(/\r?\n/)
  const tables: AssistantMarkdownTable[] = []

  let index = 0
  while (index < lines.length - 1) {
    const headerLine = lines[index] ?? ''
    const separatorLine = lines[index + 1] ?? ''
    if (!isMarkdownTableRow(headerLine) || !isMarkdownTableSeparator(separatorLine)) {
      index++
      continue
    }

    const headers = parseMarkdownTableCells(headerLine)
    if (headers.length < 2) {
      index++
      continue
    }

    const rows: string[][] = []
    let rowIndex = index + 2
    while (rowIndex < lines.length) {
      const rowLine = lines[rowIndex] ?? ''
      if (!isMarkdownTableRow(rowLine)) break
      const cells = parseMarkdownTableCells(rowLine)
      if (cells.length >= 2) rows.push(cells)
      rowIndex++
    }

    if (rows.length > 0) {
      tables.push({ headers, rows })
    }

    index = rowIndex
  }

  return tables
}

const isMarkdownTableRow = (line: string): boolean => {
  const trimmed = line.trim()
  if (trimmed === '' || !trimmed.includes('|')) return false
  return trimmed.includes('|')
}

const isMarkdownTableSeparator = (line: string): boolean => {
  const cells = parseMarkdownTableCells(line)
  return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()))
}

const parseMarkdownTableCells = (line: string): string[] =>
  line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())

const normaliseWeekdayLabel = (value: string): string | undefined => {
  const trimmed = value.trim().toLowerCase()
  if (trimmed === '') return undefined
  const match = WEEKDAY_NAMES.find((weekday) => weekday.toLowerCase() === trimmed)
  return match
}

const splitIntoFactSentences = (content: string): readonly string[] =>
  protectSentenceSplitAbbreviations(content)
    .split(/[\n]+|(?<=[.!?])\s+/)
    .map((part) => restoreSentenceSplitAbbreviations(part).trim())
    .filter((part) => part !== '')

const protectSentenceSplitAbbreviations = (content: string): string => {
  let protectedContent = content
  for (const [pattern, replacement] of SENTENCE_SPLIT_PROTECTED_ABBREVIATIONS) {
    protectedContent = protectedContent.replace(pattern, replacement)
  }
  return protectedContent
}

const restoreSentenceSplitAbbreviations = (content: string): string =>
  content.replaceAll('__DOT__', '.')

const hasQuantifiedFact = (sentence: string): boolean => {
  UNIT_QUANTITY_TAG_RE.lastIndex = 0
  if (UNIT_QUANTITY_TAG_RE.test(sentence)) return true
  WORD_UNIT_QUANTITY_TAG_RE.lastIndex = 0
  if (WORD_UNIT_QUANTITY_TAG_RE.test(sentence)) return true
  DATE_TAG_RE.lastIndex = 0
  if (DATE_TAG_RE.test(sentence)) return true
  MONTH_NAME_DATE_RE.lastIndex = 0
  if (MONTH_NAME_DATE_RE.test(sentence)) return true
  if (HEURISTIC_DURATION_FACT_RE.test(sentence)) return true
  if (ORDINAL_QUANTITY_TAG_RE.test(sentence)) return true
  return /\b\d{1,4}\b/.test(sentence)
}

const hasHeuristicUserFactSignal = (sentence: string): boolean => {
  if (hasQuantifiedFact(sentence)) return true
  if (isLikelyQuestionSentence(sentence)) return false
  return hasCadenceFact(sentence) || hasStorageLocationFact(sentence)
}

const heuristicUserFactCandidates = (content: string): readonly string[] => {
  const candidates = splitIntoFactSentences(content)
  const trimmed = content.trim()
  if (
    trimmed === '' ||
    (!hasCadenceFact(trimmed) && !hasStorageLocationFact(trimmed))
  ) {
    return candidates
  }
  const canonical = trimmed.toLowerCase()
  for (const candidate of candidates) {
    if (candidate.trim().toLowerCase() === canonical) return candidates
  }
  return [...candidates, trimmed]
}

const hasCadenceFact = (sentence: string): boolean =>
  HEURISTIC_CADENCE_FACT_RE.test(sentence)

const hasStorageLocationFact = (sentence: string): boolean =>
  HEURISTIC_STORAGE_LOCATION_FACT_RE.test(sentence)

const isLikelyQuestionSentence = (sentence: string): boolean => {
  const trimmed = sentence.trim()
  if (trimmed === '') return false
  return trimmed.endsWith('?') || HEURISTIC_QUESTION_LEAD_RE.test(trimmed)
}

const hasMilestoneFact = (sentence: string): boolean => {
  if (!FIRST_PERSON_FACT_RE.test(sentence)) return false
  if (hasQuantifiedFact(sentence)) return false
  if (!HEURISTIC_MILESTONE_TOPIC_RE.test(sentence)) return false
  return (
    HEURISTIC_MILESTONE_EVENT_RE.test(sentence) ||
    HEURISTIC_MILESTONE_TIME_RE.test(sentence)
  )
}

const heuristicFactSlug = (sentence: string): string => {
  const words = sentence.toLowerCase().match(HEURISTIC_WORD_RE) ?? []
  const kept = words.filter((word) => !HEURISTIC_STOPWORDS.has(word)).slice(0, 5)
  return kept.join('-')
}

const buildHeuristicPreferenceCandidates = (
  content: string,
): readonly HeuristicPreferenceCandidate[] => {
  const seen = new Set<string>()
  const out: HeuristicPreferenceCandidate[] = []
  const candidates = [normalisePreferenceText(content), ...splitIntoFactSentences(content)]
  for (const candidate of candidates) {
    if (candidate === '') continue
    const inferred = inferHeuristicPreference(candidate)
    if (inferred === undefined) continue
    const canonical = inferred.summary.toLowerCase()
    if (seen.has(canonical)) continue
    seen.add(canonical)
    out.push(inferred)
  }
  return out
}

const inferHeuristicPreference = (
  content: string,
): HeuristicPreferenceCandidate | undefined => {
  const text = normalisePreferenceText(content)
  if (text === '') return undefined

  const explicit = inferExplicitPreference(text)
  if (explicit !== undefined) return explicit

  const compatibility = inferCompatibilityPreference(text)
  if (compatibility !== undefined) return compatibility

  const constrained = inferConstraintPreference(text)
  if (constrained !== undefined) return constrained

  const advanced = inferAdvancedPreference(text)
  if (advanced !== undefined) return advanced

  return undefined
}

const inferExplicitPreference = (
  text: string,
): HeuristicPreferenceCandidate | undefined => {
  const besides = HEURISTIC_PREFERENCE_BESIDES_LIKE_RE.exec(text)
  if (besides !== null) {
    const first = cleanPreferenceFragment(besides[1] ?? '')
    const second = cleanPreferenceFragment(besides[2] ?? '')
    if (first !== '' && second !== '') {
      const hotelMatch = /^hotels?\s+with\s+(.+)$/i.exec(second)
      const summary =
        hotelMatch !== null
          ? `The user prefers hotels with ${first} and ${cleanPreferenceFragment(hotelMatch[1] ?? '')}.`
          : `The user prefers ${first} and ${second}.`
      return { summary, evidence: text }
    }
  }

  const simple = HEURISTIC_PREFERENCE_LIKE_RE.exec(text)
  if (simple !== null) {
    const fragment = cleanPreferenceFragment(simple[1] ?? '')
    if (fragment !== '') {
      return { summary: `The user prefers ${fragment}.`, evidence: text }
    }
  }

  return undefined
}

const inferCompatibilityPreference = (
  text: string,
): HeuristicPreferenceCandidate | undefined => {
  const rawSubject =
    captureGroup(HEURISTIC_PREFERENCE_COMPATIBLE_RE, text) ??
    captureGroup(HEURISTIC_PREFERENCE_DESIGNED_FOR_RE, text) ??
    captureGroup(HEURISTIC_PREFERENCE_AS_USER_RE, text)
  if (rawSubject === undefined) return undefined
  const subject = cleanPreferenceFragment(rawSubject)
  if (subject === '') return undefined
  const category = inferPreferenceCategory(text)
  return {
    summary: `The user prefers ${category} compatible with their ${subject}.`,
    evidence: text,
  }
}

const inferConstraintPreference = (
  text: string,
): HeuristicPreferenceCandidate | undefined => {
  const category = inferRecommendationCategory(text)
  if (category === undefined) return undefined
  if (!HEURISTIC_RECOMMENDATION_REQUEST_RE.test(text)) return undefined

  const constraints = collectRecommendationConstraints(text)
  if (constraints.length === 0) return undefined

  return {
    summary: `The user prefers ${category} with these constraints: ${constraints.join('; ')}.`,
    evidence: text,
  }
}

const inferAdvancedPreference = (
  text: string,
): HeuristicPreferenceCandidate | undefined => {
  const topic =
    captureGroup(HEURISTIC_PREFERENCE_FIELD_RE, text) ??
    captureGroup(HEURISTIC_PREFERENCE_ADVANCED_RE, text)
  if (topic === undefined) return undefined
  if (
    !HEURISTIC_PREFERENCE_SKIP_BASICS_RE.test(text) &&
    !HEURISTIC_PREFERENCE_WORKING_IN_FIELD_RE.test(text) &&
    !/\badvanced\b/i.test(text)
  ) {
    return undefined
  }
  const cleanedTopic = cleanPreferenceFragment(topic)
  if (cleanedTopic === '') return undefined
  return {
    summary: `The user prefers advanced publications, papers, and conferences on ${cleanedTopic} rather than introductory material.`,
    evidence: text,
  }
}

const captureGroup = (pattern: RegExp, text: string): string | undefined => {
  const matched = cloneRegex(pattern).exec(text)
  const group = matched?.[1]
  const cleaned = group?.trim()
  return cleaned !== undefined && cleaned !== '' ? cleaned : undefined
}

const captureWholeMatch = (pattern: RegExp, text: string): string | undefined => {
  const matched = cloneRegex(pattern).exec(text)?.[0]?.trim()
  return matched !== undefined && matched !== '' ? matched : undefined
}

const cloneRegex = (pattern: RegExp): RegExp =>
  new RegExp(pattern.source, pattern.flags.replaceAll('g', ''))

const inferPreferenceCategory = (text: string): string => {
  const lower = text.toLowerCase()
  if (
    lower.includes('camera') ||
    lower.includes('photography') ||
    lower.includes('lens') ||
    lower.includes('flash') ||
    lower.includes('tripod') ||
    lower.includes('camera bag') ||
    lower.includes('gear')
  ) {
    return 'photography accessories and gear'
  }
  if (
    lower.includes('phone') ||
    lower.includes('iphone') ||
    lower.includes('screen protector') ||
    lower.includes('power bank')
  ) {
    return 'phone accessories'
  }
  const recommendationCategory = inferRecommendationCategory(text)
  if (recommendationCategory !== undefined) {
    return recommendationCategory
  }
  return 'accessories and options'
}

const inferRecommendationCategory = (text: string): string | undefined => {
  const lower = text.toLowerCase()
  if (/\b(?:film|movie|cinema)\b/i.test(lower)) {
    return 'films'
  }
  if (/\b(?:show|series|tv)\b/i.test(lower)) {
    return 'shows'
  }
  if (/\b(?:book|books|novel|novels|read|reading)\b/i.test(lower)) {
    return 'books'
  }
  if (/\b(?:hotel|hotels|accommodation|stay)\b/i.test(lower)) {
    return 'hotels'
  }
  if (/\b(?:restaurant|restaurants|dinner|lunch)\b/i.test(lower)) {
    return 'restaurants'
  }
  if (/\b(?:game|games)\b/i.test(lower)) {
    return 'games'
  }
  if (/\b(?:podcast|podcasts)\b/i.test(lower)) {
    return 'podcasts'
  }
  return undefined
}

const collectRecommendationConstraints = (text: string): readonly string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  const add = (value: string): void => {
    const cleaned = cleanPreferenceFragment(value)
    const canonical = cleaned.toLowerCase()
    if (cleaned === '' || seen.has(canonical)) return
    seen.add(canonical)
    out.push(cleaned)
  }

  if (HEURISTIC_RECOMMENDATION_FAMILY_RE.test(text)) {
    add('family-friendly')
  }
  const light = captureWholeMatch(HEURISTIC_RECOMMENDATION_LIGHT_RE, text)
  if (light !== undefined) add(light)

  const notToo = captureGroup(HEURISTIC_RECOMMENDATION_NOT_TOO_RE, text)
  if (notToo !== undefined) add(`not too ${notToo}`)

  const under = captureGroup(HEURISTIC_RECOMMENDATION_UNDER_RE, text)
  if (under !== undefined) add(`under ${under}`)

  const without = captureGroup(HEURISTIC_RECOMMENDATION_WITHOUT_RE, text)
  if (without !== undefined) add(`without ${without}`)

  return out
}

const normalisePreferenceText = (value: string): string =>
  value.replace(/\s+/g, ' ').trim()

const cleanPreferenceFragment = (value: string): string =>
  value
    .trim()
    .replace(/^[,:;\s]+/, '')
    .replace(/[,:;\s]+$/, '')
    .replace(/\s+/g, ' ')

const buildHeuristicPreferenceFilename = (args: {
  readonly iso: string
  readonly sessionId?: string
  readonly slug: string
}): string =>
  buildHeuristicSessionFilename('user-preference', args.iso, args.sessionId, args.slug)

const buildHeuristicPreferenceContent = (
  candidate: HeuristicPreferenceCandidate,
): string => `${candidate.summary}\n\nEvidence: ${candidate.evidence}`

const heuristicPreferenceSummary = (content: string): string => {
  const marker = '\n\nEvidence:'
  const split = content.indexOf(marker)
  const summary = split >= 0 ? content.slice(0, split) : content
  return summary.trim()
}

const isHeuristicPreferenceFact = (memory: ExtractedMemory): boolean =>
  memory.scope === 'global' &&
  memory.type === 'user' &&
  memory.filename.startsWith('user-preference-')

const sanitiseHeuristicFileSegment = (value: string): string =>
  value.replace(/[^A-Za-z0-9._-]/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '')

const truncateOneLine = (value: string, max: number): string =>
  value.length <= max ? value : `${value.slice(0, Math.max(1, max - 3)).trimEnd()}...`

const extractSessionSummary = (messages: readonly Message[]): string => {
  for (const message of messages) {
    if (message.role === 'system' && (message.content ?? '').trim() !== '') {
      return truncateOneLine(oneLine(message.content ?? ''), 240)
    }
  }
  for (const message of messages) {
    if (message.role === 'user' && (message.content ?? '').trim() !== '') {
      return truncateOneLine(oneLine(message.content ?? ''), 240)
    }
  }
  return ''
}

const oneLine = (value: string): string => value.replace(/\s+/g, ' ').trim()

const applyContextualPrefixes = async (
  builder: ContextualPrefixBuilder | undefined,
  messages: readonly Message[],
  sessionId: string | undefined,
  extracted: readonly ExtractedMemory[],
): Promise<readonly ExtractedMemory[]> => {
  if (builder === undefined || !builder.enabled() || extracted.length === 0) {
    return extracted
  }
  const sessionSummary = extractSessionSummary(messages)
  return Promise.all(
    extracted.map(async (memory) => {
      if (memory.contextPrefix !== undefined && memory.contextPrefix.trim() !== '') {
        return memory
      }
      const prefix = await builder
        .buildPrefix({
          sessionSummary,
          factBody: memory.content,
          ...(sessionId !== undefined ? { sessionId } : {}),
        })
        .catch(() => '')
      return prefix.trim() !== '' ? { ...memory, contextPrefix: prefix.trim() } : memory
    }),
  )
}

const toTitleCase = (value: string): string =>
  value
    .split(/\s+/)
    .filter((part) => part !== '')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')

const postProcessSessionExtractions = (
  messages: readonly Message[],
  extracted: readonly ExtractedMemory[],
  sessionId?: string,
  sessionDate?: string,
): readonly ExtractedMemory[] => {
  if (extracted.length === 0) return extracted

  const { sessionRaw, modifiedOverride } = resolveHeuristicSessionMetadata(
    messages,
    sessionDate,
  )
  const resolvedSessionId = resolveHeuristicSessionId(messages, extracted, sessionId)
  const sessionDateIso = shortIsoDate(modifiedOverride)
  const dateTokens = buildDateTokens(modifiedOverride)

  return extracted.map((memory) => {
    const shaped = shapeExtractedMemory(memory)
    const currentSessionId = shaped.sessionId?.trim() ?? ''
    const nextSessionId = currentSessionId !== '' ? currentSessionId : resolvedSessionId
    const content =
      sessionRaw !== '' &&
      shaped.content.trim() !== '' &&
      !shaped.content.startsWith('[Date:')
        ? `${dateTokens}[Observed on ${sessionRaw}]\n\n${shaped.content}`
        : shaped.content
    const tags = mergeTags(shaped.tags, autoFactTags(content))
    return {
      ...shaped,
      filename: rewriteHeuristicFilenameForSession(shaped.filename, nextSessionId),
      content,
      ...(nextSessionId !== '' ? { sessionId: nextSessionId } : {}),
      ...(modifiedOverride !== '' &&
      (shaped.modifiedOverride === undefined || shaped.modifiedOverride === '')
        ? { modifiedOverride }
        : {}),
      ...(modifiedOverride !== '' &&
      (shaped.observedOn === undefined || shaped.observedOn === '')
        ? { observedOn: modifiedOverride }
        : {}),
      ...(sessionDateIso !== '' &&
      (shaped.sessionDate === undefined || shaped.sessionDate === '')
        ? { sessionDate: sessionDateIso }
        : {}),
      ...(tags.length > 0 ? { tags } : {}),
    }
  })
}

const resolveHeuristicSessionId = (
  messages: readonly Message[],
  extracted: readonly ExtractedMemory[],
  sessionId?: string,
): string => {
  const direct = sessionId?.trim() ?? ''
  if (direct !== '') return direct
  for (const memory of extracted) {
    const fromMemory = memory.sessionId?.trim() ?? ''
    if (fromMemory !== '') return fromMemory
  }
  for (const message of messages) {
    if (message.role !== 'system') continue
    const matched = SYSTEM_SESSION_ID_RE.exec(message.content ?? '')?.[1]?.trim()
    if (matched !== undefined && matched !== '') return matched
  }
  return ''
}

const resolveHeuristicSessionMetadata = (
  messages: readonly Message[],
  sessionDate?: string,
): {
  readonly sessionRaw: string
  readonly modifiedOverride: string
} => {
  const direct = sessionDate?.trim() ?? ''
  if (direct !== '') {
    const parsed = parseDateInput(direct)
    if (parsed !== undefined) {
      return { sessionRaw: direct, modifiedOverride: parsed.toISOString() }
    }
  }
  for (const message of messages) {
    if (message.role !== 'system') continue
    const matched = SYSTEM_SESSION_DATE_RE.exec(message.content ?? '')?.[0]?.trim()
    if (matched === undefined || matched === '') continue
    const parsed = parseDateInput(matched)
    if (parsed !== undefined) {
      return { sessionRaw: matched, modifiedOverride: parsed.toISOString() }
    }
  }
  return { sessionRaw: '', modifiedOverride: '' }
}

const rewriteHeuristicFilenameForSession = (
  filename: string,
  sessionId: string | undefined,
): string => {
  const sessionSegment = sanitiseHeuristicFileSegment(sessionId ?? '')
  if (sessionSegment === '') return filename

  const dated = HEURISTIC_FILENAME_DATED_RE.exec(filename)
  if (dated !== null) {
    const prefix = dated[1] ?? ''
    const iso = dated[2] ?? ''
    const rest = dated[3] ?? ''
    if (rest === sessionSegment || rest.startsWith(`${sessionSegment}-`)) {
      return filename
    }
    return `${prefix}-${iso}-${sessionSegment}-${rest}.md`
  }

  const plain = HEURISTIC_FILENAME_RE.exec(filename)
  if (plain !== null) {
    const prefix = plain[1] ?? ''
    const rest = plain[2] ?? ''
    if (rest === sessionSegment || rest.startsWith(`${sessionSegment}-`)) {
      return filename
    }
    return `${prefix}-${sessionSegment}-${rest}.md`
  }

  return filename
}

const shapeExtractedMemory = (memory: ExtractedMemory): ExtractedMemory => {
  if (memory.content.trim() === '') return memory
  const summary = inferSearchableSummary(memory.content)
  if (summary === '') return memory
  const description = chooseMoreSpecificSummary(memory.description, summary)
  const indexEntry = chooseMoreSpecificIndexEntry(memory.indexEntry, summary)
  return {
    ...memory,
    description,
    indexEntry,
  }
}

const inferSearchableSummary = (content: string): string => {
  const text = stripSearchPrefixes(content)
  if (text === '') return ''
  return (
    inferPendingTaskSummary(text) ??
    inferAppointmentSummary(text) ??
    inferEventSummary(text) ??
    inferHeuristicPreference(text)?.summary ??
    ''
  )
}

const stripSearchPrefixes = (content: string): string =>
  content
    .replace(/^\[Date:[^\]]+\]\n\n/i, '')
    .replace(/^\[Observed on [^\]]+\]\n\n/i, '')
    .trim()

const chooseMoreSpecificSummary = (current: string, derived: string): string => {
  const cleanedCurrent = current.trim()
  const cleanedDerived = truncateOneLine(derived, 140)
  if (cleanedCurrent === '') return cleanedDerived
  if (!isLessSpecificSummary(cleanedCurrent, derived)) return cleanedCurrent
  return cleanedDerived
}

const chooseMoreSpecificIndexEntry = (current: string, derived: string): string => {
  const cleanedDerived = truncateOneLine(derived, 140)
  const cleanedCurrent = current.trim()
  if (cleanedCurrent === '') return cleanedDerived
  if (!isLessSpecificSummary(cleanedCurrent, derived)) return cleanedCurrent
  const colonIndex = cleanedCurrent.indexOf(':')
  if (cleanedCurrent.startsWith('-') && colonIndex > 0) {
    return truncateOneLine(
      `${cleanedCurrent.slice(0, colonIndex + 1)} ${stripTrailingFullStop(derived)}`,
      140,
    )
  }
  return cleanedDerived
}

const isLessSpecificSummary = (current: string, derived: string): boolean => {
  const currentTokens = informativeSummaryTokens(current)
  const derivedTokens = informativeSummaryTokens(derived)
  if (derivedTokens.size === 0) return false
  if (currentTokens.size === 0) return true
  let missing = 0
  for (const token of derivedTokens) {
    if (!currentTokens.has(token)) missing++
  }
  return missing >= Math.max(2, Math.ceil(derivedTokens.size / 2))
}

const informativeSummaryTokens = (value: string): ReadonlySet<string> => {
  const out = new Set<string>()
  for (const token of value.toLowerCase().match(/[a-z0-9]+/g) ?? []) {
    if (SUMMARY_STOPWORDS.has(token)) continue
    out.add(token)
  }
  return out
}

const mergeTags = (
  existing: readonly string[] | undefined,
  inferred: readonly string[],
): readonly string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of existing ?? []) {
    const tag = raw.trim()
    if (tag === '' || seen.has(tag)) continue
    seen.add(tag)
    out.push(tag)
  }
  for (const raw of inferred) {
    const tag = raw.trim()
    if (tag === '' || seen.has(tag)) continue
    seen.add(tag)
    out.push(tag)
  }
  return out
}

const buildDateTokens = (rfc3339: string): string => {
  const parsed = parseRfc3339(rfc3339)
  if (parsed === undefined) return ''
  const iso = shortIsoDate(rfc3339)
  if (iso === '') return ''
  return `[Date: ${iso} ${weekdayName(parsed)} ${monthName(parsed)} ${String(
    parsed.getUTCFullYear(),
  )}]\n\n`
}

const shortIsoDate = (rfc3339: string): string => {
  const parsed = parseRfc3339(rfc3339)
  if (parsed === undefined) return ''
  return `${String(parsed.getUTCFullYear()).padStart(4, '0')}-${String(
    parsed.getUTCMonth() + 1,
  ).padStart(2, '0')}-${String(parsed.getUTCDate()).padStart(2, '0')}`
}

const parseSessionDateRfc3339 = (value: string | undefined): string => {
  const trimmed = value?.trim() ?? ''
  if (trimmed === '') return ''
  const parsed = parseDateInput(trimmed)
  return parsed === undefined ? '' : parsed.toISOString()
}

const resolveHeuristicSessionDate = (
  messages: readonly Message[],
  sessionDate: string | undefined,
): string | undefined => {
  const direct = sessionDate?.trim()
  if (direct !== undefined && direct !== '' && parseDateInput(direct) !== undefined) {
    return direct
  }
  for (const message of messages) {
    if (message.role !== 'system') continue
    const matched = SYSTEM_SESSION_DATE_RE.exec(message.content ?? '')?.[0]?.trim()
    if (matched !== undefined && matched !== '' && parseDateInput(matched) !== undefined) {
      return matched
    }
  }
  return sessionDate
}

const parseDateInput = (value: string): Date | undefined => {
  const matched = DATE_INPUT_RE.exec(value)
  if (matched !== null) {
    const yearRaw = matched[1] ?? '0'
    const monthRaw = matched[2] ?? '1'
    const dayRaw = matched[3] ?? '1'
    const hourRaw = matched[4] ?? '0'
    const minuteRaw = matched[5] ?? '0'
    const secondRaw = matched[6] ?? '0'
    return new Date(
      Date.UTC(
        Number.parseInt(yearRaw, 10),
        Number.parseInt(monthRaw, 10) - 1,
        Number.parseInt(dayRaw, 10),
        Number.parseInt(hourRaw ?? '0', 10),
        Number.parseInt(minuteRaw ?? '0', 10),
        Number.parseInt(secondRaw ?? '0', 10),
      ),
    )
  }
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? undefined : parsed
}

const resolveHeuristicObservedOn = (
  sentence: string,
  sessionDate: string | undefined,
): string => {
  const anchor = parseDateInput(sessionDate?.trim() ?? '')
  if (anchor === undefined) return ''

  const explicitIso = captureWholeMatch(DATE_TAG_RE, sentence)
  if (explicitIso !== undefined) {
    const parsed = parseDateInput(explicitIso.replaceAll('-', '/'))
    return parsed?.toISOString() ?? ''
  }

  const monthNameDate = parseMonthNameDate(sentence, anchor)
  if (monthNameDate !== undefined) return monthNameDate.toISOString()

  const expansion = expandTemporal(sentence, sessionDate)
  const hint = expansion.dateHints[0]
  if (hint === undefined) return ''
  const parsed = parseDateInput(hint)
  return parsed?.toISOString() ?? ''
}

const parseMonthNameDate = (text: string, anchor: Date): Date | undefined => {
  const match = cloneRegex(MONTH_NAME_DATE_RE).exec(text)
  if (match === null) return undefined

  const monthName = match[1] ?? ''
  const dayPart = match[0].replace(monthName, '').trim()
  const dayMatch = /^(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?$/i.exec(dayPart)
  if (dayMatch === null) return undefined

  const month = MONTH_NAME_VALUES.findIndex(
    (candidate) => candidate.toLowerCase() === monthName.toLowerCase(),
  )
  if (month < 0) return undefined
  const day = Number.parseInt(dayMatch[1] ?? '0', 10)
  const year =
    dayMatch[2] !== undefined && dayMatch[2] !== ''
      ? Number.parseInt(dayMatch[2], 10)
      : anchor.getUTCFullYear()
  const resolved = new Date(Date.UTC(year, month, day))
  if (Number.isNaN(resolved.getTime())) return undefined

  if ((dayMatch[2] ?? '') === '' && resolved.getTime() > anchor.getTime() + 86_400_000) {
    resolved.setUTCFullYear(resolved.getUTCFullYear() - 1)
  }
  return resolved
}

const withObservedDatePrefix = (content: string, observedOn: string): string => {
  const trimmed = content.trim()
  if (trimmed === '' || observedOn === '') return trimmed
  const prefix = buildDateTokens(observedOn)
  return prefix === '' ? trimmed : `${prefix}${trimmed}`
}

const parseRfc3339 = (value: string): Date | undefined => {
  if (value.trim() === '') return undefined
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? undefined : parsed
}

const autoFactTags = (content: string): string[] => {
  if (content === '') return []
  const body = content.length > 4096 ? content.slice(0, 4096) : content
  const seen = new Set<string>()
  const add = (value: string): void => {
    const tag = value.trim()
    if (tag === '' || seen.has(tag)) return
    seen.add(tag)
  }

  for (const match of body.match(DATE_TAG_RE) ?? []) {
    add(match)
    const parsed = parseDateInput(match.replaceAll('-', '/'))
    if (parsed !== undefined) {
      add(weekdayName(parsed))
      add(monthName(parsed))
    }
  }
  for (const match of body.match(WEEKDAY_TAG_RE) ?? []) {
    add(match.charAt(0).toUpperCase() + match.slice(1).toLowerCase())
  }
  for (const match of body.match(RELATIVE_TEMPORAL_TAG_RE) ?? []) {
    add(match.toLowerCase())
  }
  for (const match of body.match(CLOCK_TIME_TAG_RE) ?? []) {
    add(match.toLowerCase())
  }
  for (const match of body.match(MONEY_TAG_RE) ?? []) {
    add(match)
  }
  for (const capture of body.matchAll(UNIT_QUANTITY_TAG_RE)) {
    const quantity = capture[1]
    const unit = capture[2]
    if (quantity !== undefined && unit !== undefined) {
      add(`${quantity} ${unit}`)
    }
  }
  for (const match of body.match(QUANTITY_TAG_RE) ?? []) {
    add(match)
  }
  for (const match of body.match(PROPER_NOUN_TAG_RE) ?? []) {
    if (match.length < 3) continue
    const lower = match.toLowerCase()
    if (AUTO_TAG_STOP_NOUNS.has(lower)) continue
    add(match)
  }
  for (const match of body.match(PENDING_ACTION_TAG_RE) ?? []) {
    add(match.toLowerCase())
  }
  for (const match of body.match(MEDICAL_TAG_RE) ?? []) {
    add(match.toLowerCase())
  }
  for (const match of body.match(EVENT_TAG_RE) ?? []) {
    add(match.toLowerCase())
  }
  for (const match of body.match(ENTERTAINMENT_TAG_RE) ?? []) {
    add(match.toLowerCase())
  }
  if (
    HEURISTIC_PENDING_ACTION_LEAD_RE.test(body) ||
    /\b(?:pick\s+up|drop\s+off|return|exchange|collect|book|schedule|renew|cancel|follow\s+up)\b/i.test(
      body,
    )
  ) {
    add('pending')
    add('task')
  }
  if (HEURISTIC_APPOINTMENT_RE.test(body)) {
    add('appointment')
  }
  if (HEURISTIC_MEDICAL_ENTITY_RE.test(body) || /\b(?:clinic|hospital|prescription)\b/i.test(body)) {
    add('medical')
  }
  if (HEURISTIC_EVENT_RE.test(body)) {
    add('event')
  }
  if (HEURISTIC_RECOMMENDATION_REQUEST_RE.test(body)) {
    add('recommendation')
  }
  if (/\b(?:film|movie|show|series|book|novel|game|podcast|cinema)\b/i.test(body)) {
    add('entertainment')
  }

  return [...seen]
}

const buildExistingMemoryTextSet = (
  existing: readonly ExtractedMemory[],
): Set<string> =>
  new Set(
    existing
      .flatMap((memory) => [
        memory.content,
        memory.description,
        memory.indexEntry,
        inferSearchableSummary(memory.content),
      ])
      .map(normaliseMemoryText)
      .filter((value) => value !== ''),
  )

const normaliseMemoryText = (value: string): string =>
  value.replace(/\s+/g, ' ').trim().toLowerCase()

const extractPendingActions = (sentence: string): readonly string[] => {
  if (sentence.trim().endsWith('?')) return []
  const matched = HEURISTIC_PENDING_ACTION_LEAD_RE.exec(sentence)
  const fragment = matched?.[1]?.trim()
  if (fragment === undefined || fragment === '') return []

  const parts = fragment
    .split(/\s*(?:,|;|\bthen\b|\band\b)\s*/i)
    .map((part) => part.trim())
    .filter((part) => part !== '')
  const out: string[] = []
  let current = ''

  for (const part of parts) {
    if (HEURISTIC_PENDING_ACTION_START_RE.test(part)) {
      if (current !== '') out.push(current)
      current = part
      continue
    }
    if (current !== '') {
      current = `${current} ${part}`.trim()
    }
  }

  if (current !== '') out.push(current)
  if (out.length > 0) return out.map(cleanPendingActionClause).filter((part) => part !== '')

  const cleaned = cleanPendingActionClause(fragment)
  return cleaned === '' ? [] : [cleaned]
}

const cleanPendingActionClause = (value: string): string =>
  value
    .trim()
    .replace(/^[,:;\s]+/, '')
    .replace(/[,:;\s]+$/, '')
    .replace(/\s+/g, ' ')

const buildPendingTaskSummary = (action: string): string =>
  ensureTrailingFullStop(`The user still needs to ${stripTrailingFullStop(action)}`)

const inferPendingTaskSummary = (text: string): string | undefined => {
  const sentences = splitIntoFactSentences(text)
  for (const sentence of sentences) {
    const action = extractPendingActions(sentence)[0]
    if (action !== undefined) return buildPendingTaskSummary(action)
  }
  return undefined
}

const inferAppointmentSummary = (text: string): string | undefined => {
  const sentences = splitIntoFactSentences(text)
  for (const sentence of sentences) {
    if (!HEURISTIC_APPOINTMENT_RE.test(sentence)) continue
    if (HEURISTIC_PENDING_ACTION_LEAD_RE.test(sentence)) continue
    if (sentence.trim().endsWith('?')) continue

    const medicalEntity = captureWholeMatch(HEURISTIC_MEDICAL_ENTITY_RE, sentence)
    const withPerson = captureGroup(HEURISTIC_WITH_PERSON_RE, sentence)
    const temporal = extractTemporalAnchor(sentence)
    const subject =
      medicalEntity !== undefined
        ? `${withIndefiniteArticle(medicalEntity)} ${medicalEntity} appointment`
        : 'a medical appointment'
    const summary = [
      `The user has ${subject}`,
      withPerson !== undefined ? `with ${withPerson}` : undefined,
      temporal !== '' ? temporal : undefined,
    ]
      .filter((part): part is string => part !== undefined && part !== '')
      .join(' ')
    return ensureTrailingFullStop(summary)
  }
  return undefined
}

const inferEventSummary = (text: string): string | undefined => {
  const sentences = splitIntoFactSentences(text)
  for (const sentence of sentences) {
    const religiousService = inferReligiousServiceSummary(sentence)
    if (religiousService !== undefined) return religiousService

    if (!FIRST_PERSON_FACT_RE.test(sentence)) continue
    if (!HEURISTIC_EVENT_RE.test(sentence)) continue
    if (HEURISTIC_PENDING_ACTION_LEAD_RE.test(sentence)) continue
    if (!HEURISTIC_EVENT_ATTENDANCE_RE.test(sentence)) continue
    if (sentence.trim().endsWith('?')) continue

    const title = captureGroup(HEURISTIC_EVENT_TITLE_RE, sentence)
    const temporal = extractTemporalAnchor(sentence)
    const eventPhrase = title ?? extractLooseEventPhrase(sentence)
    if (eventPhrase === undefined) continue
    const summary = [
      `The user attended ${prefixEventPhrase(eventPhrase)}`,
      temporal !== '' ? temporal : undefined,
    ]
      .filter((part): part is string => part !== undefined && part !== '')
      .join(' ')
    return ensureTrailingFullStop(summary)
  }
  return undefined
}

const inferReligiousServiceSummary = (sentence: string): string | undefined => {
  if (!FIRST_PERSON_FACT_RE.test(sentence)) return undefined
  if (HEURISTIC_PENDING_ACTION_LEAD_RE.test(sentence)) return undefined
  if (sentence.trim().endsWith('?')) return undefined

  const service = captureGroup(HEURISTIC_RELIGIOUS_SERVICE_RE, sentence)
  if (service === undefined) return undefined
  return ensureTrailingFullStop(`The user attended ${prefixEventPhrase(service)}`)
}

const extractLooseEventPhrase = (sentence: string): string | undefined => {
  const matched = sentence.match(
    /\b(?:a|an|the)\s+([^,.!?]+?\s+(?:workshop|conference|concert|gig|show|screening|play|musical|exhibition|festival|meetup|class|course|webinar|lecture|seminar))\b/i,
  )?.[0]
  const cleaned = matched?.trim()
  return cleaned !== undefined && cleaned !== '' ? cleaned : undefined
}

const prefixEventPhrase = (value: string): string => {
  const trimmed = value.trim()
  if (/^(?:a|an|the)\b/i.test(trimmed)) return trimmed
  return `the ${trimmed}`
}

const extractTemporalAnchor = (text: string): string => {
  const dateAnchor =
    captureWholeMatch(DATE_TAG_RE, text) ??
    captureWholeMatch(HEURISTIC_RELATIVE_DATE_RE, text)
  const timeAnchor = captureGroup(HEURISTIC_CLOCK_TIME_RE, text)
  const parts: string[] = []
  if (dateAnchor !== undefined) parts.push(`on ${dateAnchor}`)
  if (timeAnchor !== undefined) parts.push(`at ${timeAnchor}`)
  return parts.join(' ')
}

const ensureTrailingFullStop = (value: string): string =>
  /[.!?]$/.test(value) ? value : `${value}.`

const stripTrailingFullStop = (value: string): string =>
  value.replace(/[.!?]+$/, '')

const withIndefiniteArticle = (value: string): string =>
  /^[aeiou]/i.test(value) ? 'an' : 'a'

const weekdayName = (date: Date): string =>
  ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][
    date.getUTCDay()
  ] ?? 'Unknown'

const WEEKDAY_NAMES = [
  'Sunday',
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
] as const
const HEURISTIC_FILENAME_DATED_RE =
  /^(user-(?:fact|preference))-(\d{4}-\d{2}-\d{2})-(.+)\.md$/
const HEURISTIC_FILENAME_RE = /^(user-(?:fact|preference))-(.+)\.md$/

const monthName = (date: Date): string =>
  [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December',
  ][date.getUTCMonth()] ?? 'Unknown'

const AUTO_TAG_STOP_NOUNS = new Set([
  'the',
  'this',
  'that',
  'these',
  'those',
  'when',
  'where',
  'what',
  'who',
  'why',
  'how',
  'observed',
  'date',
  'mon',
  'tue',
  'wed',
  'thu',
  'fri',
  'sat',
  'sun',
  'user',
  'assistant',
])

const SUMMARY_STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'appointment',
  'event',
  'has',
  'is',
  'task',
  'the',
  'this',
  'to',
  'user',
  'with',
])
