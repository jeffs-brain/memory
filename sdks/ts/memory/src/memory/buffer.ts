// SPDX-License-Identifier: Apache-2.0

import type { Message, ToolCall } from '../llm/index.js'
import type {
  L0BufferAppendResult,
  L0BufferCompactionResult,
  L0BufferConfig,
  L0BufferSnapshot,
  L0Intent,
  L0Observation,
  L0Outcome,
  ObserveMessagesOptions,
  Semver,
} from './types.js'

const DEFAULT_L0_BUFFER_CONFIG: L0BufferConfig = {
  tokenBudget: 8192,
  compactThresholdPercent: 100,
  keepRecentPercent: 50,
  maxObservationLength: 160,
  maxEntities: 5,
  reminderHeading: 'Recent session observations:',
}

const PATH_KEYS = new Set([
  'path',
  'paths',
  'file',
  'files',
  'filepath',
  'file_path',
  'targetpath',
  'target_path',
  'sourcepath',
  'source_path',
  'destinationpath',
  'destination_path',
])

const EDIT_TOOL_NAMES = new Set(['edit', 'multi_edit', 'write', 'bash', 'patch'])
const READ_TOOL_NAMES = new Set(['read', 'grep', 'glob', 'search', 'kb_search'])
const PLAN_TOOL_NAMES = new Set(['skill', 'agent'])

export const L0_BUFFER_SNAPSHOT_FORMAT = 'l0-buffer-snapshot' as const
export const L0_BUFFER_SNAPSHOT_VERSION: Semver = '1.0.0'

export const defaultL0BufferConfig = (overrides: Partial<L0BufferConfig> = {}): L0BufferConfig => ({
  tokenBudget: sanitisePositiveInt(overrides.tokenBudget, DEFAULT_L0_BUFFER_CONFIG.tokenBudget),
  compactThresholdPercent: sanitisePercentage(
    overrides.compactThresholdPercent,
    DEFAULT_L0_BUFFER_CONFIG.compactThresholdPercent,
  ),
  keepRecentPercent: sanitisePercentage(
    overrides.keepRecentPercent,
    DEFAULT_L0_BUFFER_CONFIG.keepRecentPercent,
  ),
  maxObservationLength: sanitisePositiveInt(
    overrides.maxObservationLength,
    DEFAULT_L0_BUFFER_CONFIG.maxObservationLength,
  ),
  maxEntities: sanitisePositiveInt(overrides.maxEntities, DEFAULT_L0_BUFFER_CONFIG.maxEntities),
  reminderHeading:
    typeof overrides.reminderHeading === 'string' && overrides.reminderHeading.trim() !== ''
      ? overrides.reminderHeading.trim()
      : DEFAULT_L0_BUFFER_CONFIG.reminderHeading,
})

export const observeMessages = (
  messages: readonly Message[],
  opts: ObserveMessagesOptions = {},
): L0Observation | undefined => {
  if (messages.length === 0) {
    return undefined
  }

  const maxObservationLength = sanitisePositiveInt(
    opts.maxObservationLength,
    DEFAULT_L0_BUFFER_CONFIG.maxObservationLength,
  )
  const maxEntities = sanitisePositiveInt(opts.maxEntities, DEFAULT_L0_BUFFER_CONFIG.maxEntities)
  const turnStart = findTurnStart(messages)
  const turnMessages = messages.slice(turnStart)
  const summary = buildTurnSummary(messages, maxObservationLength)
  if (summary === '') {
    return undefined
  }

  return {
    at: normaliseTimestamp(opts.observedAt),
    intent: inferIntent(turnMessages, messages),
    outcome: inferOutcome(turnMessages),
    summary,
    entities: extractMentionedEntities(turnMessages, maxEntities),
  }
}

export const formatL0Observation = (observation: L0Observation): string => {
  const parts = [`- [${formatClock(observation.at)}]`]
  parts.push(`(${observation.intent})`)
  if (observation.outcome !== 'ok') {
    parts.push(`[${observation.outcome}]`)
  }

  let line = `${parts.join(' ')} ${collapseWhitespace(observation.summary)}`.trim()
  if (observation.entities.length > 0) {
    line += ` {${observation.entities.join(', ')}}`
  }
  return line
}

export const renderL0Buffer = (observations: readonly L0Observation[]): string =>
  observations.map(formatL0Observation).join('\n')

export const renderL0Reminder = (
  observations: readonly L0Observation[],
  overrides: Partial<L0BufferConfig> = {},
): string => {
  const content = renderL0Buffer(observations)
  if (content === '') {
    return ''
  }

  const config = defaultL0BufferConfig(overrides)
  return ['<system-reminder>', config.reminderHeading, '', content, '</system-reminder>'].join('\n')
}

export const estimateL0BufferTokens = (
  observations: readonly L0Observation[],
  overrides: Partial<L0BufferConfig> = {},
): number => Math.ceil(renderL0Reminder(observations, overrides).length / 4)

export const needsL0BufferCompaction = (
  observations: readonly L0Observation[],
  overrides: Partial<L0BufferConfig> = {},
): boolean => {
  const config = defaultL0BufferConfig(overrides)
  const threshold = Math.floor((config.tokenBudget * config.compactThresholdPercent) / 100)
  return estimateL0BufferTokens(observations, config) >= threshold
}

export const compactL0Buffer = (
  observations: readonly L0Observation[],
  overrides: Partial<L0BufferConfig> = {},
): L0BufferCompactionResult => {
  if (observations.length <= 1) {
    return { observations: [...observations], removed: 0 }
  }

  const config = defaultL0BufferConfig(overrides)
  let keepCount = Math.max(Math.floor((observations.length * config.keepRecentPercent) / 100), 1)
  let kept = observations.slice(-keepCount)

  while (kept.length > 1 && estimateL0BufferTokens(kept, config) > config.tokenBudget) {
    keepCount--
    kept = observations.slice(-keepCount)
  }

  return {
    observations: [...kept],
    removed: observations.length - kept.length,
  }
}

export const appendL0Observation = (
  observations: readonly L0Observation[],
  observation: L0Observation,
  overrides: Partial<L0BufferConfig> = {},
): L0BufferAppendResult => {
  const config = defaultL0BufferConfig(overrides)
  const next = [...observations, sanitiseObservation(observation, config)]
  if (!needsL0BufferCompaction(next, config)) {
    return { observations: next, removed: 0, compacted: false }
  }

  const compacted = compactL0Buffer(next, config)
  return {
    observations: compacted.observations,
    removed: compacted.removed,
    compacted: compacted.removed > 0,
  }
}

export const exportL0BufferSnapshot = (
  observations: readonly L0Observation[],
  opts: { readonly createdAt?: Date | string } = {},
): L0BufferSnapshot => ({
  metadata: {
    format: L0_BUFFER_SNAPSHOT_FORMAT,
    version: L0_BUFFER_SNAPSHOT_VERSION,
    createdAt: normaliseTimestamp(opts.createdAt),
    observationCount: observations.length,
  },
  observations: observations.map(cloneL0Observation),
})

export const restoreL0BufferSnapshot = (snapshot: unknown): readonly L0Observation[] => {
  const parsed = parseL0BufferSnapshot(snapshot)
  if (parsed.metadata.version.split('.')[0] !== '1') {
    throw new Error(`Invalid L0 buffer snapshot version: ${parsed.metadata.version}`)
  }
  if (parsed.metadata.observationCount !== parsed.observations.length) {
    throw new Error(
      `Invalid L0 buffer snapshot: expected ${parsed.metadata.observationCount} observations, got ${parsed.observations.length}`,
    )
  }

  return parsed.observations.map(cloneL0Observation)
}

const sanitiseObservation = (
  observation: L0Observation,
  config: L0BufferConfig,
): L0Observation => ({
  ...observation,
  at: normaliseTimestamp(observation.at),
  summary: truncateText(observation.summary, config.maxObservationLength),
  entities: dedupeStrings(observation.entities).slice(0, config.maxEntities),
})

const cloneL0Observation = (observation: L0Observation): L0Observation => ({
  ...observation,
  entities: [...observation.entities],
})

type ParsedL0BufferSnapshot = {
  readonly metadata: {
    readonly format: string
    readonly version: string
    readonly createdAt: string
    readonly observationCount: number
  }
  readonly observations: readonly L0Observation[]
}

const parseL0BufferSnapshot = (snapshot: unknown): ParsedL0BufferSnapshot => {
  if (!isRecord(snapshot)) {
    throw new Error('Invalid L0 buffer snapshot: expected an object')
  }

  if (!isRecord(snapshot.metadata)) {
    throw new Error('Invalid L0 buffer snapshot: missing metadata')
  }

  const format = snapshot.metadata.format
  const version = snapshot.metadata.version
  const createdAt = snapshot.metadata.createdAt
  const observationCount = snapshot.metadata.observationCount

  if (format !== L0_BUFFER_SNAPSHOT_FORMAT) {
    throw new Error(`Invalid L0 buffer snapshot format: ${String(format)}`)
  }
  if (typeof version !== 'string' || version.trim() === '' || !isSemver(version)) {
    throw new Error(`Invalid L0 buffer snapshot version: ${String(version)}`)
  }
  if (typeof createdAt !== 'string' || createdAt.trim() === '') {
    throw new Error('Invalid L0 buffer snapshot: missing metadata.createdAt')
  }
  if (!isNonNegativeInteger(observationCount)) {
    throw new Error(
      'Invalid L0 buffer snapshot: metadata.observationCount must be a non-negative integer',
    )
  }
  if (!Array.isArray(snapshot.observations)) {
    throw new Error('Invalid L0 buffer snapshot: observations must be an array')
  }

  return {
    metadata: {
      format,
      version,
      createdAt,
      observationCount,
    },
    observations: snapshot.observations.map(parseL0Observation),
  }
}

const parseL0Observation = (value: unknown): L0Observation => {
  if (!isRecord(value)) {
    throw new Error('Invalid L0 buffer snapshot: observation must be an object')
  }

  const at = parseTimestamp(value.at)
  const intent = value.intent
  const outcome = value.outcome
  const summary = value.summary
  const entities = value.entities

  if (at === undefined) {
    throw new Error('Invalid L0 buffer snapshot: observation.at must be a valid timestamp')
  }
  if (!isL0Intent(intent)) {
    throw new Error(`Invalid L0 buffer snapshot: unsupported observation.intent ${String(intent)}`)
  }
  if (!isL0Outcome(outcome)) {
    throw new Error(
      `Invalid L0 buffer snapshot: unsupported observation.outcome ${String(outcome)}`,
    )
  }
  if (typeof summary !== 'string') {
    throw new Error('Invalid L0 buffer snapshot: observation.summary must be a string')
  }
  if (!Array.isArray(entities) || entities.some((entity) => typeof entity !== 'string')) {
    throw new Error('Invalid L0 buffer snapshot: observation.entities must be a string array')
  }

  return {
    at,
    intent,
    outcome,
    summary,
    entities: [...entities],
  }
}

const findTurnStart = (messages: readonly Message[]): number => {
  for (let index = messages.length - 1; index >= 0; index--) {
    if (messages[index]?.role === 'user') {
      return index
    }
  }
  return 0
}

const inferIntent = (turnMessages: readonly Message[], messages: readonly Message[]): L0Intent => {
  const toolNames = turnMessages.flatMap((message) =>
    message.role === 'assistant'
      ? (message.toolCalls ?? []).map((toolCall) => normaliseToolName(toolCall))
      : [],
  )

  if (toolNames.some((name) => EDIT_TOOL_NAMES.has(name))) {
    return 'edit'
  }
  if (toolNames.some((name) => READ_TOOL_NAMES.has(name))) {
    return 'read'
  }
  if (toolNames.some((name) => PLAN_TOOL_NAMES.has(name))) {
    return 'plan'
  }

  const lastUser = [...messages].reverse().find((message) => message.role === 'user')
  const userText = stripSystemReminders(readMessageText(lastUser))
  if (userText.includes('?')) {
    return 'ask'
  }

  return 'chat'
}

const inferOutcome = (turnMessages: readonly Message[]): L0Outcome => {
  const toolCalls = turnMessages.flatMap((message, messageIndex) =>
    message.role === 'assistant'
      ? (message.toolCalls ?? []).map((toolCall, toolIndex) => ({
          id: toolCall.id,
          name: toolCall.name,
          key: toolCall.id || `${messageIndex}:${toolIndex}:${toolCall.name}`,
        }))
      : [],
  )

  if (toolCalls.length === 0) {
    return 'ok'
  }

  const matched = new Set<string>()
  for (const message of turnMessages) {
    for (const result of readToolResults(message)) {
      const match = toolCalls.find((toolCall) => {
        if (matched.has(toolCall.key)) {
          return false
        }
        if (toolCall.id !== '') {
          return result.toolCallId === toolCall.id
        }
        return result.toolName !== undefined && result.toolName === toolCall.name
      })

      if (!match) {
        continue
      }

      matched.add(match.key)
      if (isErrorResult(result.isError, result.content)) {
        return 'error'
      }
    }
  }

  return matched.size < toolCalls.length ? 'partial' : 'ok'
}

const buildTurnSummary = (messages: readonly Message[], maxObservationLength: number): string => {
  const lastUser = [...messages].reverse().find((message) => message.role === 'user')
  const content = stripSystemReminders(readMessageText(lastUser))
  return truncateText(content, maxObservationLength)
}

const extractMentionedEntities = (
  turnMessages: readonly Message[],
  maxEntities: number,
): readonly string[] => {
  const entities: string[] = []
  const seen = new Set<string>()

  for (const message of turnMessages) {
    if (message.role !== 'assistant') {
      continue
    }

    for (const toolCall of message.toolCalls ?? []) {
      const parsed = parseJson(toolCall.arguments)
      if (parsed === undefined) {
        continue
      }

      for (const entity of extractPathValues(parsed)) {
        if (seen.has(entity)) {
          continue
        }
        seen.add(entity)
        entities.push(entity)
        if (entities.length >= maxEntities) {
          return entities
        }
      }
    }
  }

  return entities
}

type ToolResultRecord = {
  readonly toolCallId: string
  readonly toolName?: string
  readonly content: string
  readonly isError?: boolean
}

const readToolResults = (message: Message): readonly ToolResultRecord[] => {
  const fromBlocks =
    message.blocks
      ?.filter((block) => block.type === 'tool_result' && block.toolResult !== undefined)
      .map((block) => ({
        toolCallId: block.toolResult?.toolCallId ?? '',
        content: block.toolResult?.content ?? '',
        ...(block.toolResult?.isError !== undefined ? { isError: block.toolResult.isError } : {}),
      })) ?? []

  if (fromBlocks.length > 0) {
    return fromBlocks
  }

  if (message.role !== 'tool') {
    return []
  }

  return [
    {
      toolCallId: message.toolCallId ?? '',
      ...(message.name !== undefined ? { toolName: message.name } : {}),
      content: readMessageText(message),
    },
  ]
}

const readMessageText = (message: Message | undefined): string => {
  if (!message) {
    return ''
  }

  const blocks = message.blocks
    ?.flatMap((block) => {
      if (block.type === 'text' && block.text !== undefined) {
        return [block.text]
      }
      if (block.type === 'tool_result' && block.toolResult?.content !== undefined) {
        return [block.toolResult.content]
      }
      return []
    })
    .filter((value) => value.trim() !== '')

  if (blocks && blocks.length > 0) {
    return blocks.join('\n')
  }

  return message.content?.trim() ?? ''
}

const normaliseToolName = (toolCall: ToolCall): string => toolCall.name.trim().toLowerCase()

const stripSystemReminders = (value: string): string =>
  collapseWhitespace(value.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/gi, ' '))

const extractPathValues = (value: unknown, parentKey = ''): readonly string[] => {
  if (typeof value === 'string') {
    return PATH_KEYS.has(parentKey) ? [value.trim()].filter(Boolean) : []
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => {
      if (typeof item === 'string') {
        return PATH_KEYS.has(parentKey) && item.trim() !== '' ? [item.trim()] : []
      }
      return extractPathValues(item, '')
    })
  }

  if (!isRecord(value)) {
    return []
  }

  return Object.entries(value).flatMap(([key, child]) =>
    extractPathValues(child, key.toLowerCase()),
  )
}

const parseJson = (value: string): Record<string, unknown> | undefined => {
  if (value.trim() === '') {
    return undefined
  }

  try {
    const parsed: unknown = JSON.parse(value)
    return isRecord(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

const isErrorResult = (flag: boolean | undefined, content: string): boolean => {
  if (flag === true) {
    return true
  }

  const normalised = content.toLowerCase()
  return (
    normalised.includes('error') ||
    normalised.includes('failed') ||
    normalised.includes('exception')
  )
}

const normaliseTimestamp = (value: Date | string | undefined): string => {
  if (value instanceof Date) {
    return value.toISOString()
  }
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = new Date(value)
    if (!Number.isNaN(parsed.valueOf())) {
      return parsed.toISOString()
    }
  }
  return new Date().toISOString()
}

const formatClock = (value: string): string => {
  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) {
    return '00:00:00'
  }
  return parsed.toISOString().slice(11, 19)
}

const truncateText = (value: string, limit: number): string => {
  const trimmed = collapseWhitespace(value)
  if (trimmed === '') {
    return ''
  }

  const chars = Array.from(trimmed)
  if (chars.length <= limit) {
    return trimmed
  }

  if (limit <= 3) {
    return chars.slice(0, limit).join('')
  }

  return `${chars.slice(0, limit - 3).join('')}...`
}

const collapseWhitespace = (value: string): string => value.replace(/\s+/g, ' ').trim()

const dedupeStrings = (values: readonly string[]): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const value of values) {
    const trimmed = value.trim()
    if (trimmed === '' || seen.has(trimmed)) {
      continue
    }
    seen.add(trimmed)
    out.push(trimmed)
  }
  return out
}

const sanitisePositiveInt = (value: number | undefined, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : fallback

const sanitisePercentage = (value: number | undefined, fallback: number): number => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }
  return Math.min(Math.max(Math.floor(value), 1), 100)
}

const parseTimestamp = (value: unknown): string | undefined => {
  if (typeof value !== 'string' || value.trim() === '') {
    return undefined
  }

  const parsed = new Date(value)
  if (Number.isNaN(parsed.valueOf())) {
    return undefined
  }

  return parsed.toISOString()
}

const isSemver = (value: string): boolean => /^(\d+)\.(\d+)\.(\d+)$/.test(value.trim())

const isNonNegativeInteger = (value: unknown): value is number =>
  typeof value === 'number' && Number.isInteger(value) && value >= 0

const isL0Intent = (value: unknown): value is L0Intent =>
  value === 'ask' || value === 'edit' || value === 'read' || value === 'plan' || value === 'chat'

const isL0Outcome = (value: unknown): value is L0Outcome =>
  value === 'ok' || value === 'error' || value === 'partial'

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)
