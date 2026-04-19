// SPDX-License-Identifier: Apache-2.0

/**
 * LLMReranker drives an LLM provider over the retrieval candidates to
 * score each one 0-10. The fused ranking is sliced into fixed-size
 * batches (default 5) and fanned out in parallel (default 4) so the
 * wall-clock cost is bounded by a single provider round trip plus
 * marshalling overhead. Malformed or missing scores default to 0.
 *
 * A unanimity shortcut is exposed as a pure helper so the retrieval
 * pipeline can skip the LLM entirely when BM25 and vector top-3 agree
 * on enough positions.
 */

import type { CompletionRequest, Provider } from '../llm/types.js'
import { runWithSharedRerankConcurrency } from './concurrency.js'
import { runBatches } from './batch.js'
import type { Reranker, RerankRequest, RerankResult } from './index.js'

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
Each score is a number between 0 (irrelevant) and 10 (perfect match).
Use British English.`

const RERANK_MAX_TOKENS = 2048
const RERANK_TEMPERATURE = 0.0
const RERANK_SNIPPET_LIMIT = 1200

export type LLMRerankerConfig = {
  provider: Provider
  batchSize?: number
  parallelism?: number
  /** Optional label surfaced via name() for traces. */
  label?: string
  /** Shared concurrency cap for outbound provider calls. */
  concurrencyCap?: number
}

/**
 * Minimal candidate shape accepted by the unanimity helper. The
 * retrieval pipeline passes SearchResult-like records in; all we need
 * is a stable identifier to compare positions by.
 */
export type UnanimityCandidate = {
  id: string
}

export type UnanimityShortcut = {
  /** The candidate ids that will be returned as the shortcut output. */
  ids: readonly string[]
  /** How many positions the two rankings agreed on within the window. */
  agreements: number
}

/**
 * unanimityShortcut reports whether BM25 and vector top-3 rankings
 * agree on enough positions to skip the rerank pass entirely. Returns
 * the shared top-3 (ordered by BM25) when the agreement count reaches
 * the threshold; otherwise undefined. Inputs with fewer than 3 entries
 * are treated as "no shortcut" because there is not enough signal to
 * trust the fused ranking.
 */
export function unanimityShortcut(
  bm25: readonly UnanimityCandidate[],
  vector: readonly UnanimityCandidate[],
  agreeMin: number = DEFAULT_UNANIMITY_AGREE_MIN,
): UnanimityShortcut | undefined {
  const window = 3
  if (bm25.length < window || vector.length < window) return undefined

  let agreements = 0
  for (let i = 0; i < window; i++) {
    const b = bm25[i]
    const v = vector[i]
    if (b === undefined || v === undefined) continue
    if (b.id === v.id) agreements++
  }
  if (agreements < agreeMin) return undefined

  const ids: string[] = []
  for (let i = 0; i < window; i++) {
    const entry = bm25[i]
    if (entry !== undefined) ids.push(entry.id)
  }
  return { ids, agreements }
}

export class LLMReranker implements Reranker {
  private readonly provider: Provider
  private readonly batchSize: number
  private readonly parallelism: number
  private readonly label: string
  private readonly concurrencyCap: number | undefined

  constructor(cfg: LLMRerankerConfig) {
    this.provider = cfg.provider
    this.batchSize = cfg.batchSize ?? DEFAULT_RERANK_BATCH_SIZE
    this.parallelism = cfg.parallelism ?? DEFAULT_RERANK_PARALLELISM
    this.label = cfg.label ?? 'llm-rerank'
    this.concurrencyCap = cfg.concurrencyCap
    if (this.batchSize <= 0) throw new Error('LLMReranker: batchSize must be > 0')
    if (this.parallelism <= 0) throw new Error('LLMReranker: parallelism must be > 0')
  }

  name(): string {
    return this.label
  }

  async isAvailable(): Promise<boolean> {
    return true
  }

  async rerank(req: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]> {
    if (req.documents.length === 0) return []

    const batches: { offset: number; docs: readonly { local: number; text: string }[] }[] = []
    for (let i = 0; i < req.documents.length; i += this.batchSize) {
      const slice = req.documents.slice(i, i + this.batchSize)
      const docs = slice.map((d, li) => ({ local: li, text: d.text }))
      batches.push({ offset: i, docs })
    }

    const scores = new Array<number>(req.documents.length).fill(0)
    await runBatches({
      batches,
      parallelism: this.parallelism,
      worker: async (batch) => {
        const batchScores = await callRerank(
          this.provider,
          req.query,
          batch.docs,
          signal,
          this.concurrencyCap,
        )
        for (let li = 0; li < batchScores.length; li++) {
          const gi = batch.offset + li
          if (gi >= scores.length) break
          const s = batchScores[li]
          if (s !== undefined) scores[gi] = s
        }
      },
    })

    const seeded: RerankResult[] = req.documents.map((doc, i) => ({
      index: i,
      id: doc.id,
      score: scores[i] ?? 0,
    }))
    seeded.sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score
      return a.index - b.index
    })
    return seeded
  }
}

async function callRerank(
  provider: Provider,
  query: string,
  candidates: readonly { local: number; text: string }[],
  signal: AbortSignal | undefined,
  concurrencyCap: number | undefined,
): Promise<readonly number[]> {
  const user = renderUserPrompt(query, candidates)
  const baseReq: CompletionRequest = {
    system: RERANK_SYSTEM_PROMPT,
    messages: [{ role: 'user', content: user }],
    maxTokens: RERANK_MAX_TOKENS,
    temperature: RERANK_TEMPERATURE,
  }

  try {
    const resp = await runWithSharedRerankConcurrency(
      () => provider.complete(baseReq, signal),
      concurrencyCap,
    )
    const parsed = parseRerankResponse(resp.content, candidates.length)
    if (parsed !== undefined) return parsed
  } catch {
    // Fall through to the strict-prompt retry.
  }

  const strictReq: CompletionRequest = { ...baseReq, system: RERANK_SYSTEM_PROMPT_STRICT }
  const resp = await runWithSharedRerankConcurrency(
    () => provider.complete(strictReq, signal),
    concurrencyCap,
  )
  const parsed = parseRerankResponse(resp.content, candidates.length)
  if (parsed === undefined) {
    // Malformed -> all zeros is the documented fallback.
    return new Array<number>(candidates.length).fill(0)
  }
  return parsed
}

function renderUserPrompt(
  query: string,
  candidates: readonly { local: number; text: string }[],
): string {
  const lines: string[] = ['## Question', query.trim(), '', '## Articles']
  for (const c of candidates) {
    lines.push(`[${c.local}] ${c.text.trim()}`)
    lines.push('')
  }
  return lines.join('\n')
}

export function composeLLMRerankDocument(args: {
  readonly id: number
  readonly path: string
  readonly title: string
  readonly summary: string
  readonly content: string
}): string {
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

/**
 * parseRerankResponse extracts an ordered score array from the LLM's
 * response. Accepts both object form (`[{id, score}]`) and bare numeric
 * form (`[8.5, 2]`). Returns undefined when no JSON array is
 * recoverable; missing scores in object form default to 0.
 */
function parseRerankResponse(raw: string, expected: number): readonly number[] | undefined {
  const payload = extractJSONArray(raw)
  if (payload === undefined) return undefined

  try {
    const objForm = JSON.parse(payload) as unknown
    if (Array.isArray(objForm) && objForm.length > 0) {
      // Object form first.
      if (objForm.every((e) => typeof e === 'object' && e !== null && !Array.isArray(e))) {
        const scores = new Array<number>(expected).fill(0)
        for (let i = 0; i < objForm.length; i++) {
          const entry = objForm[i] as { id?: unknown; score?: unknown }
          if (typeof entry.score !== 'number' || !Number.isFinite(entry.score)) continue
          let idx = i
          if (typeof entry.id === 'number' && Number.isInteger(entry.id)) idx = entry.id
          if (idx < 0 || idx >= expected) continue
          scores[idx] = entry.score
        }
        return scores
      }
      // Bare numeric form.
      if (objForm.every((e) => typeof e === 'number')) {
        const scores = new Array<number>(expected).fill(0)
        for (let i = 0; i < Math.min(objForm.length, expected); i++) {
          const v = objForm[i]
          if (typeof v === 'number' && Number.isFinite(v)) scores[i] = v
        }
        return scores
      }
    }
    return undefined
  } catch {
    return undefined
  }
}

function extractJSONArray(raw: string): string | undefined {
  const start = raw.indexOf('[')
  const end = raw.lastIndexOf(']')
  if (start < 0 || end <= start) return undefined
  return raw.slice(start, end + 1)
}
