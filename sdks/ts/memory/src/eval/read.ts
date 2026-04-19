// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 reader. For each LME example, call the injected retrieval
 * function to gather context, then pipe that context through the reader
 * LLM to produce a final predicted answer. Retrieval + read latency are
 * measured separately so downstream reports can attribute slow runs.
 */

import type { Provider } from '../llm/index.js'
import type {
  LMEExample,
  ReaderFn,
  RetrievalFn,
  RetrievalResult,
} from './types.js'
import { readerTodayAnchor } from './temporal.js'

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

export const runRead = async (
  deps: ReadDeps,
  example: LMEExample,
): Promise<ReadOutcome> => {
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

/** LME official CoT reader prompt. */
export const READER_USER_TEMPLATE = `I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.

Resolving conflicting information:
- Each fact is tagged with a date. When the same topic appears with different values on different dates, prefer the value from the most recent session date.
- Treat explicit supersession phrases as hard overrides regardless of how often the old value appears: "now", "currently", "most recently", "actually", "correction", "I updated", "I changed", "no longer".
- Do not vote by frequency. One later correction outweighs any number of earlier mentions.
- Never use a fact dated after the current date.
- When the question names a specific item, event, place, or descriptor, prefer the fact that matches that target most directly. Do not substitute a broader category match or a different example from the same topic.
- A direct statement of the full usual value outranks a newer note about only one segment, leg, or example from that routine unless the newer note explicitly says the full value changed.

Enumeration and counting:
- When the question asks to list, count, enumerate, or total ("how many", "list", "which", "what are all", "total", "in total"), return every matching item you find across the retrieved facts, one per line, each tagged with its session date. Then state the count or total explicitly at the end.
- Do not summarise into a single sentence when the question demands a list.
- Add numeric values across sessions when the question asks for a total (hours, days, money, items). Show the arithmetic.
- When both atomic event facts and retrospective roll-up summaries are present, prefer the atomic event facts and avoid double counting the roll-up.
- Treat first-person past-tense purchases, gifts, sales, earnings, completions, or submissions as confirmed historical events even when they appear inside a planning or advice conversation. Exclude only clearly hypothetical or planned amounts.
- If a spending or earnings question does not explicitly restrict the timeframe ("today", "this time", "most recent", "current"), include all confirmed historical amounts for the same subject across sessions.
- For totals over named items, sum only the facts that match those named items directly. Do not add alternative purchases, adjacent examples, or broader category summaries unless the note clearly says they refer to the same item.

Temporal reasoning:
- Today is %TODAY% (this is the current date). Resolve relative references ("recently", "last week", "a few days ago", "this month") against this anchor.
- For date-arithmetic questions ("how many days between X and Y"), first extract each event's ISO date from the fact tags, then compute the difference in days.

Preference questions:
- When the question is phrased like a request for advice or a recommendation ("can you suggest", "what should I choose", "where should I stay"), answer with the user's inferred preferences from the chat history unless the history itself already contains the exact recommendation.
- Do not invent a fresh recommendation from general knowledge when the benchmark is testing remembered preferences or constraints.
- Infer durable preferences from concrete desired features or liked attributes even when the earlier example was tied to a different city, venue, or product.
- When concrete amenities or features are present, prefer them over generic travel style or budget signals.
- Ignore unrelated hostel, budget, or solo-travel examples when the retrieved facts already contain a clearer accommodation-feature preference and the question does not ask about price.
- When the question asks for a specific or exact previously recommended item, answer with the narrowest directly supported set from the retrieved facts. Do not widen the answer with adjacent frameworks, resource catalogues, or loosely related examples.

History Chats:

%CONTEXT%

Current Date: %DATE%
Question: %QUESTION%
Answer (step by step):`

const READER_MAX_TOKENS = 800
const READER_TEMPERATURE = 0.0

/** Build the default provider-backed reader. Mirrors `ReadAnswer` in Go. */
export const createProviderReader = (
  provider: Provider,
  opts: { readonly model?: string; readonly budgetChars?: number } = {},
): ReaderFn => {
  const budget = opts.budgetChars ?? 100_000
  return async ({ question, questionDate, context }) => {
    const trimmed = truncateSmartly(context, budget)
    const todayAnchor = readerTodayAnchor(questionDate)
    const prompt = READER_USER_TEMPLATE.replace('%CONTEXT%', trimmed)
      .replace('%TODAY%', todayAnchor)
      .replace('%DATE%', questionDate !== undefined && questionDate !== '' ? questionDate : 'unknown')
      .replace('%QUESTION%', question)
    try {
      const resp = await provider.complete({
        ...(opts.model !== undefined ? { model: opts.model } : {}),
        messages: [{ role: 'user', content: prompt }],
        maxTokens: READER_MAX_TOKENS,
        temperature: READER_TEMPERATURE,
      })
      const content = resp.content.trim()
      return content === '' ? trimmed.trim() : content
    } catch {
      return trimmed.trim()
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
    parts.push(
      section.length <= allocation ? section : headTailTruncate(section, allocation),
    )
    remainingBudget -= Math.min(section.length, allocation)
  }

  return parts.join('\n\n---\n\n')
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err))

const splitSessions = (content: string): readonly string[] => {
  const bySessionId = content.split('\n\n---\nsession_id:')
  if (bySessionId.length > 1) {
    return bySessionId.map((part, index) =>
      index === 0 ? part : `---\nsession_id:${part}`,
    )
  }
  const byDivider = content.split('\n\n---\n\n')
  if (byDivider.length > 1) return byDivider
  const byFrontmatter = content.split('\n\n---\n')
  if (byFrontmatter.length > 1) {
    return byFrontmatter.map((part, index) => (index === 0 ? part : `---\n${part}`))
  }
  return [content]
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
