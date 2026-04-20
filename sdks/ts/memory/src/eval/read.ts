// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 reader. For each LME example, call the injected retrieval
 * function to gather context, then pipe that context through the reader
 * LLM to produce a final predicted answer. Retrieval + read latency are
 * measured separately so downstream reports can attribute slow runs.
 */

import {
  READER_AUGMENTED_MAX_TOKENS,
  READER_AUGMENTED_TEMPERATURE,
  buildAugmentedReaderPrompt,
} from '../augmented-reader/prompt.js'
import { resolveDeterministicAugmentedAnswer } from '../augmented-reader/resolver.js'
import type { Provider } from '../llm/index.js'
import type { LMEExample, ReaderFn, RetrievalFn, RetrievalResult } from './types.js'
export type ReadOutcome = {
  readonly id: string
  readonly predicted: string
  readonly retrievalMs: number
  readonly readMs: number
  readonly passages: RetrievalResult['passages']
  readonly error?: string
}

export type ReadDeps = {
  readonly retrieval: RetrievalFn
  readonly reader: ReaderFn
}

export const runRead = async (deps: ReadDeps, example: LMEExample): Promise<ReadOutcome> => {
  const retrievalStart = Date.now()
  let retrieval: RetrievalResult
  try {
    retrieval = await deps.retrieval({
      question: example.question,
      ...(example.questionDate !== undefined ? { questionDate: example.questionDate } : {}),
      example,
    })
  } catch (err) {
    return {
      id: example.id,
      predicted: '',
      retrievalMs: Date.now() - retrievalStart,
      readMs: 0,
      passages: [],
      error: `retrieval: ${errText(err)}`,
    }
  }
  const retrievalMs = Date.now() - retrievalStart

  if (retrieval.rendered.trim() === '') {
    return {
      id: example.id,
      predicted: '',
      retrievalMs,
      readMs: 0,
      passages: retrieval.passages,
    }
  }

  const readStart = Date.now()
  const resolved = resolveDeterministicAugmentedAnswer({
    question: example.question,
    rendered: retrieval.rendered,
  })
  if (resolved !== undefined) {
    return {
      id: example.id,
      predicted: resolved.answer.trim(),
      retrievalMs,
      readMs: Date.now() - readStart,
      passages: retrieval.passages,
    }
  }
  let predicted = ''
  try {
    predicted = await deps.reader({
      question: example.question,
      ...(example.questionDate !== undefined ? { questionDate: example.questionDate } : {}),
      context: retrieval.rendered,
    })
  } catch (err) {
    return {
      id: example.id,
      predicted: '',
      retrievalMs,
      readMs: Date.now() - readStart,
      passages: retrieval.passages,
      error: `read: ${errText(err)}`,
    }
  }
  const readMs = Date.now() - readStart

  return {
    id: example.id,
    predicted: predicted.trim(),
    retrievalMs,
    readMs,
    passages: retrieval.passages,
  }
}

const READER_MAX_TOKENS = READER_AUGMENTED_MAX_TOKENS
const READER_TEMPERATURE = READER_AUGMENTED_TEMPERATURE

/** Build the default provider-backed reader. Mirrors `ReadAnswer` in Go. */
export const createProviderReader = (
  provider: Provider,
  opts: { readonly model?: string; readonly budgetChars?: number } = {},
): ReaderFn => {
  const budget = opts.budgetChars ?? 200_000
  return async ({ question, questionDate, context }) => {
    const trimmed = truncateForQuestion(context, budget, question)
    const prompt = buildAugmentedReaderPrompt(question, trimmed, questionDate)
    try {
      const resp = await provider.complete({
        ...(opts.model !== undefined ? { model: opts.model } : {}),
        messages: [{ role: 'user', content: prompt }],
        maxTokens: READER_MAX_TOKENS,
        temperature: READER_TEMPERATURE,
      })
      const content = resp.content.trim()
      return content === '' ? context.trim() : content
    } catch {
      return context.trim()
    }
  }
}

/**
 * Smart head+tail truncator. Mirrors the Go `truncateSmartly`: keep 70%
 * head / 30% tail so the reader sees both establishing and concluding
 * context when it cannot have everything.
 */
export const truncateSmartly = (content: string, budget: number): string => {
  if (budget <= 0) return ''
  if (content.length <= budget) return content
  const sections = splitSessions(content)
  if (sections.length <= 1) {
    return headTailTruncate(content, budget)
  }

  const totalLength = sections.reduce((sum, section) => sum + section.length, 0)
  const parts: string[] = []
  let remainingBudget = budget

  for (const [index, section] of sections.entries()) {
    if (remainingBudget <= 0) break
    let allocation = Math.max(Math.floor((section.length * budget) / totalLength), 500)
    allocation = Math.min(allocation, remainingBudget)
    if (index === sections.length - 1) allocation = remainingBudget
    parts.push(section.length <= allocation ? section : headTailTruncate(section, allocation))
    remainingBudget -= Math.min(section.length, allocation)
  }

  return parts.join('\n\n---\n\n')
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))

const questionTokens = (question: string): readonly string[] => {
  if (question === '') return []
  const stopWords = new Set([
    'the',
    'and',
    'for',
    'with',
    'what',
    'who',
    'when',
    'where',
    'why',
    'how',
    'did',
    'does',
    'was',
    'were',
    'are',
    'you',
    'your',
    'about',
    'this',
    'that',
    'have',
    'has',
    'had',
    'from',
    'into',
    'than',
    'then',
    'them',
    'they',
    'their',
  ])
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of question.toLowerCase().split(/\s+/)) {
    const token = raw.replace(/^[^a-z0-9]+|[^a-z0-9]+$/g, '')
    if (token.length < 3 || stopWords.has(token) || seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

const scoreChunkRelevance = (chunk: string, tokens: readonly string[]): number => {
  const lower = chunk.toLowerCase()
  let score = 0
  for (const token of tokens) {
    if (lower.includes(token)) score++
  }
  return score
}

const splitSessions = (content: string): readonly string[] => {
  const bySessionId = content.split('\n\n---\nsession_id:')
  if (bySessionId.length > 1) {
    return bySessionId.map((part, index) => (index === 0 ? part : `---\nsession_id:${part}`))
  }
  const byDivider = content.split('\n\n---\n\n')
  if (byDivider.length > 1) return byDivider
  const byFrontmatter = content.split('\n\n---\n')
  if (byFrontmatter.length > 1) {
    return byFrontmatter.map((part, index) => (index === 0 ? part : `---\n${part}`))
  }
  return [content]
}

const truncateForQuestion = (content: string, budget: number, question: string): string => {
  if (budget <= 0) return ''
  if (content.length <= budget) return content
  if (question.trim() === '') return truncateSmartly(content, budget)

  const sections = splitSessions(content)
  if (sections.length <= 1) {
    return relevantSnippetForQuestion(content, question, budget)
  }

  const parts: string[] = []
  let remaining = budget
  for (const section of sections) {
    const separator = parts.length > 0 ? '\n\n---\n\n' : ''
    if (remaining <= separator.length) break
    const allocation = remaining - separator.length
    if (section.length <= allocation) {
      parts.push(section)
      remaining -= separator.length + section.length
      continue
    }
    const snippet = relevantSnippetForQuestion(section, question, allocation)
    if (snippet === '') break
    parts.push(snippet)
    break
  }
  return parts.length > 0
    ? parts.join('\n\n---\n\n')
    : relevantSnippetForQuestion(content, question, budget)
}

const relevantSnippetForQuestion = (content: string, question: string, budget: number): string => {
  if (budget <= 0) return ''
  if (content.length <= budget) return content
  const tokens = questionTokens(question)
  if (tokens.length === 0) return headTailTruncate(content, budget)

  const lines = content.split('\n')
  let prefixEnd = 0
  while (
    prefixEnd < lines.length &&
    !lines[prefixEnd]?.startsWith('[user]:') &&
    !lines[prefixEnd]?.startsWith('[assistant]:')
  ) {
    prefixEnd++
  }

  const selected = Array<boolean>(lines.length).fill(false)
  for (let index = 0; index < prefixEnd; index++) {
    selected[index] = true
  }

  let matched = false
  for (let index = prefixEnd; index < lines.length; index++) {
    if (scoreChunkRelevance(lines[index] ?? '', tokens) === 0) continue
    matched = true
    const from = Math.max(0, index - 2)
    const to = Math.min(lines.length, index + 3)
    for (let cursor = from; cursor < to; cursor++) {
      selected[cursor] = true
    }
  }

  if (!matched) return headTailTruncate(content, budget)

  const rendered: string[] = []
  let omitted = false
  for (const [index, line] of lines.entries()) {
    if (selected[index]) {
      if (omitted) {
        rendered.push('[...omitted irrelevant lines...]')
        omitted = false
      }
      rendered.push(line)
      continue
    }
    if (rendered.length > 0) omitted = true
  }
  return headTailTruncate(rendered.join('\n').trim(), budget)
}

const headTailTruncate = (content: string, budget: number): string => {
  if (budget <= 0) return ''
  if (content.length <= budget) return content
  const marker = '\n[...truncated...]\n'
  if (budget <= marker.length) return content.slice(0, budget)
  const head = Math.floor((budget * 70) / 100)
  const tail = Math.max(budget - head - marker.length, 0)
  return `${content.slice(0, head)}${marker}${content.slice(content.length - tail)}`
}
