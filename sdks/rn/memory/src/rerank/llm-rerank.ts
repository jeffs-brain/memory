import type { CompletionRequest, Provider } from '../llm/types.js'
import { runBatches } from './batch.js'
import { runWithSharedRerankConcurrency } from './concurrency.js'
import type { RerankRequest, RerankResult, Reranker } from './index.js'

export const DEFAULT_RERANK_BATCH_SIZE = 5
export const DEFAULT_RERANK_PARALLELISM = 4
export const DEFAULT_UNANIMITY_AGREE_MIN = 2

const RERANK_SYSTEM_PROMPT = `You are scoring wiki articles against a user's question.
Return ONLY a JSON array of objects matching the input order:
[{"id": 0, "score": 8.5}, {"id": 1, "score": 2.0}, ...]
Score 0 means irrelevant. Score 10 means perfectly answers the
question. Use British English.`

const RERANK_SYSTEM_PROMPT_STRICT = `You are scoring wiki articles against a user's question.
Return ONLY a raw JSON array and NOTHING ELSE. No prose, no markdown,
no backticks, no commentary. The array must have one object per input
article in the same order:
[{"id": 0, "score": 8.5}, {"id": 1, "score": 2.0}, ...]
Each score is a number between 0 and 10.
Use British English.`

const RERANK_MAX_TOKENS = 2048
const RERANK_TEMPERATURE = 0.0
const RERANK_SNIPPET_LIMIT = 1200

export type LLMRerankerConfig = {
  readonly provider: Provider
  readonly batchSize?: number
  readonly parallelism?: number
  readonly label?: string
  readonly concurrencyCap?: number
}

export type UnanimityCandidate = {
  readonly id: string
}

export type UnanimityShortcut = {
  readonly ids: readonly string[]
  readonly agreements: number
}

export const unanimityShortcut = (
  bm25: readonly UnanimityCandidate[],
  vector: readonly UnanimityCandidate[],
  agreeMin: number = DEFAULT_UNANIMITY_AGREE_MIN,
): UnanimityShortcut | undefined => {
  const window = 3
  if (bm25.length < window || vector.length < window) return undefined

  let agreements = 0
  for (let index = 0; index < window; index += 1) {
    const left = bm25[index]
    const right = vector[index]
    if (left === undefined || right === undefined) continue
    if (left.id === right.id) agreements += 1
  }

  if (agreements < agreeMin) return undefined

  const ids: string[] = []
  for (let index = 0; index < window; index += 1) {
    const item = bm25[index]
    if (item !== undefined) ids.push(item.id)
  }

  return { ids, agreements }
}

export class LLMReranker implements Reranker {
  private readonly provider: Provider
  private readonly batchSize: number
  private readonly parallelism: number
  private readonly label: string
  private readonly concurrencyCap: number | undefined

  constructor(config: LLMRerankerConfig) {
    this.provider = config.provider
    this.batchSize = config.batchSize ?? DEFAULT_RERANK_BATCH_SIZE
    this.parallelism = config.parallelism ?? DEFAULT_RERANK_PARALLELISM
    this.label = config.label ?? 'llm-rerank'
    this.concurrencyCap = config.concurrencyCap
    if (this.batchSize <= 0) throw new Error('LLMReranker: batchSize must be > 0')
    if (this.parallelism <= 0) throw new Error('LLMReranker: parallelism must be > 0')
  }

  name(): string {
    return this.label
  }

  async isAvailable(): Promise<boolean> {
    return true
  }

  async rerank(request: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]> {
    if (request.documents.length === 0) return []

    const batches: Array<{
      readonly offset: number
      readonly docs: readonly {
        readonly local: number
        readonly text: string
      }[]
    }> = []

    for (let index = 0; index < request.documents.length; index += this.batchSize) {
      const slice = request.documents.slice(index, index + this.batchSize)
      batches.push({
        offset: index,
        docs: slice.map((document, local) => ({ local, text: document.text })),
      })
    }

    const scores = new Array<number>(request.documents.length).fill(0)
    await runBatches({
      batches,
      parallelism: this.parallelism,
      worker: async (batch) => {
        const batchScores = await callRerank(
          this.provider,
          request.query,
          batch.docs,
          signal,
          this.concurrencyCap,
        )
        for (let local = 0; local < batchScores.length; local += 1) {
          const global = batch.offset + local
          if (global >= scores.length) break
          const score = batchScores[local]
          if (score !== undefined) {
            scores[global] = score
          }
        }
      },
    })

    const results = request.documents.map((document, index) => ({
      index,
      id: document.id,
      score: scores[index] ?? 0,
    }))
    results.sort((left, right) => {
      if (left.score !== right.score) return right.score - left.score
      return left.index - right.index
    })
    return results
  }
}

const callRerank = async (
  provider: Provider,
  query: string,
  candidates: readonly {
    readonly local: number
    readonly text: string
  }[],
  signal: AbortSignal | undefined,
  concurrencyCap: number | undefined,
): Promise<readonly number[]> => {
  const userPrompt = renderUserPrompt(query, candidates)
  const request: CompletionRequest = {
    system: RERANK_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: userPrompt }],
    maxTokens: RERANK_MAX_TOKENS,
    temperature: RERANK_TEMPERATURE,
  }

  try {
    const response = await runWithSharedRerankConcurrency(
      () => provider.complete(request, signal),
      concurrencyCap,
    )
    const parsed = parseRerankResponse(response.content, candidates.length)
    if (parsed !== undefined) return parsed
  } catch {
    // Fall through to the strict prompt retry.
  }

  const response = await runWithSharedRerankConcurrency(
    () =>
      provider.complete(
        {
          ...request,
          system: RERANK_SYSTEM_PROMPT_STRICT,
        },
        signal,
      ),
    concurrencyCap,
  )
  const parsed = parseRerankResponse(response.content, candidates.length)
  return parsed ?? new Array<number>(candidates.length).fill(0)
}

const renderUserPrompt = (
  query: string,
  candidates: readonly {
    readonly local: number
    readonly text: string
  }[],
): string => {
  const lines = ['## Question', query.trim(), '', '## Articles']
  for (const candidate of candidates) {
    lines.push(`[${candidate.local}] ${candidate.text.trim()}`, '')
  }
  return lines.join('\n')
}

export const composeLLMRerankDocument = (args: {
  readonly id: number
  readonly path: string
  readonly title: string
  readonly summary: string
  readonly content: string
}): string => {
  const body = args.content.replace(/\s+/g, ' ').trim()
  const snippet =
    body === ''
      ? '(no body excerpt available)'
      : body.length <= RERANK_SNIPPET_LIMIT
        ? body
        : `${body.slice(0, RERANK_SNIPPET_LIMIT)}...`

  return [
    `[${args.id}] title: ${args.title.trim() !== '' ? args.title.trim() : '(untitled)'}`,
    `    path: ${args.path}`,
    `    summary: ${args.summary.trim() !== '' ? args.summary.trim() : '(no summary available)'}`,
    '',
    `    content: ${snippet}`,
  ].join('\n')
}

const parseRerankResponse = (raw: string, expected: number): readonly number[] | undefined => {
  const payload = extractJSONArray(raw)
  if (payload === undefined) return undefined

  try {
    const parsed = JSON.parse(payload) as unknown
    if (!Array.isArray(parsed) || parsed.length === 0) return undefined

    if (parsed.every((item) => typeof item === 'object' && item !== null && !Array.isArray(item))) {
      const scores = new Array<number>(expected).fill(0)
      for (let index = 0; index < parsed.length; index += 1) {
        const item = parsed[index] as { id?: unknown; score?: unknown }
        if (typeof item.score !== 'number' || !Number.isFinite(item.score)) continue
        let target = index
        if (typeof item.id === 'number' && Number.isInteger(item.id)) {
          target = item.id
        }
        if (target < 0 || target >= expected) continue
        scores[target] = item.score
      }
      return scores
    }

    if (parsed.every((item) => typeof item === 'number')) {
      const scores = new Array<number>(expected).fill(0)
      for (let index = 0; index < Math.min(parsed.length, expected); index += 1) {
        const value = parsed[index]
        if (typeof value === 'number' && Number.isFinite(value)) {
          scores[index] = value
        }
      }
      return scores
    }

    return undefined
  } catch {
    return undefined
  }
}

const extractJSONArray = (raw: string): string | undefined => {
  const start = raw.indexOf('[')
  const end = raw.lastIndexOf(']')
  if (start < 0 || end <= start) return undefined
  return raw.slice(start, end + 1)
}
