// SPDX-License-Identifier: Apache-2.0

import type { Message } from '../llm/index.js'
import type {
  DetectProceduralRecordsOptions,
  ProceduralOutcome,
  ProceduralRecord,
} from './types.js'

const DEFAULT_MAX_CONTEXT_LENGTH = 160

type Invocation = {
  readonly tier: ProceduralRecord['tier']
  readonly name: string
  readonly taskContext: string
  readonly outcome: ProceduralOutcome
  readonly toolCalls: readonly string[]
}

export const detectProceduralRecords = (
  messages: readonly Message[],
  opts: DetectProceduralRecordsOptions = {},
): readonly ProceduralRecord[] => {
  const observedAt = normaliseTimestamp(opts.observedAt)
  const maxContextLength = sanitisePositiveInt(
    opts.maxContextLength,
    DEFAULT_MAX_CONTEXT_LENGTH,
  )

  return [
    ...detectSkillInvocations(messages, maxContextLength),
    ...detectAgentInvocations(messages, maxContextLength),
  ].map((invocation) => ({
    ...invocation,
    observedAt,
    tags: ['procedural', invocation.tier, invocation.name],
  }))
}

export const formatProceduralRecord = (record: ProceduralRecord): string => {
  const lines = [
    '---',
    `name: ${quoteYaml(record.name)}`,
    'type: procedural',
    `tier: ${record.tier}`,
    `outcome: ${record.outcome}`,
    'tags:',
    ...record.tags.map((tag) => `  - ${quoteYaml(tag)}`),
    `observed: ${record.observedAt}`,
    '---',
    '',
  ]

  if (record.taskContext !== '') {
    lines.push('## Context', '', record.taskContext, '')
  }

  if (record.toolCalls.length > 0) {
    lines.push('## Tool sequence', '', record.toolCalls.join(' -> '), '')
  }

  return `${lines.join('\n').trimEnd()}\n`
}

const detectSkillInvocations = (
  messages: readonly Message[],
  maxContextLength: number,
): readonly Invocation[] => {
  const results: Invocation[] = []

  for (const [index, message] of messages.entries()) {
    if (message.role !== 'assistant' || !message.toolCalls) {
      continue
    }

    for (const toolCall of message.toolCalls) {
      if (toolCall.name !== 'skill') {
        continue
      }

      const args = parseJson(toolCall.arguments)
      const name = readStringField(args, 'skill')
      if (name === '') {
        continue
      }

      const taskContext =
        inferProceduralContext(messages, index, maxContextLength) ||
        truncateText(readStringField(args, 'args'), maxContextLength)

      results.push({
        tier: 'skill',
        name,
        taskContext,
        outcome: inferToolCallOutcome(messages, index, toolCall.id, toolCall.name),
        toolCalls: [toolCall.name],
      })
    }
  }

  return results
}

const detectAgentInvocations = (
  messages: readonly Message[],
  maxContextLength: number,
): readonly Invocation[] => {
  const results: Invocation[] = []

  for (const [index, message] of messages.entries()) {
    if (message.role !== 'assistant' || !message.toolCalls) {
      continue
    }

    for (const toolCall of message.toolCalls) {
      if (toolCall.name !== 'agent') {
        continue
      }

      const args = parseJson(toolCall.arguments)
      const name = readStringField(args, 'type')
      if (name === '') {
        continue
      }

      const prompt = truncateText(readStringField(args, 'prompt'), maxContextLength)
      results.push({
        tier: 'agent',
        name,
        taskContext:
          prompt || inferProceduralContext(messages, index, maxContextLength),
        outcome: inferToolCallOutcome(messages, index, toolCall.id, toolCall.name),
        toolCalls: [toolCall.name],
      })
    }
  }

  return results
}

const inferProceduralContext = (
  messages: readonly Message[],
  beforeIndex: number,
  maxContextLength: number,
): string => {
  for (let index = beforeIndex - 1; index >= 0; index--) {
    const message = messages[index]
    if (message?.role !== 'user') {
      continue
    }

    return truncateText(stripSystemReminders(readMessageText(message)), maxContextLength)
  }

  return ''
}

const inferToolCallOutcome = (
  messages: readonly Message[],
  afterIndex: number,
  toolCallId: string,
  toolName: string,
): ProceduralOutcome => {
  for (let index = afterIndex + 1; index < messages.length; index++) {
    const message = messages[index]
    if (!message) {
      continue
    }

    for (const result of readToolResults(message)) {
      const idMatches = toolCallId !== '' && result.toolCallId === toolCallId
      const nameMatches =
        toolCallId === '' &&
        result.toolName !== undefined &&
        result.toolName === toolName
      if (!idMatches && !nameMatches) {
        continue
      }

      return isErrorResult(result.isError, result.content) ? 'error' : 'ok'
    }
  }

  return 'partial'
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
        ...(block.toolResult?.isError !== undefined
          ? { isError: block.toolResult.isError }
          : {}),
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

const readMessageText = (message: Message): string => {
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

const readStringField = (value: Record<string, unknown> | undefined, key: string): string => {
  const candidate = value?.[key]
  return typeof candidate === 'string' ? candidate.trim() : ''
}

const stripSystemReminders = (value: string): string =>
  collapseWhitespace(value.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/gi, ' '))

const collapseWhitespace = (value: string): string => value.replace(/\s+/g, ' ').trim()

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

const sanitisePositiveInt = (value: number | undefined, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : fallback

const quoteYaml = (value: string): string => JSON.stringify(value)

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)
