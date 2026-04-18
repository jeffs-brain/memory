// SPDX-License-Identifier: Apache-2.0

/**
 * Hugging Face Text Embeddings Inference (TEI) client. Provides both an
 * Embedder and a Reranker over the stock TEI HTTP endpoints.
 *
 * Endpoints:
 *   POST /embed    { inputs: string[] }  -> number[][]
 *   POST /rerank   { query, texts: string[] } -> [{ index, score }]
 */

import { createHash } from 'node:crypto'
import { ProviderError } from './errors.js'
import { type HttpClient, defaultHttpClient, postForText } from './http.js'
import { LRUCache, SingleFlight } from './lru.js'
import type { Embedder, Logger, RerankScore, Reranker } from './types.js'
import { noopLogger } from './types.js'

const DEFAULT_CACHE_SIZE = 10_000

export type TEIEmbedderConfig = {
  baseURL: string
  model?: string
  cacheSize?: number
  logger?: Logger
  http?: HttpClient
}

export class TEIEmbedder implements Embedder {
  private readonly baseURL: string
  private readonly modelName: string
  private readonly http: HttpClient
  private readonly logger: Logger
  private readonly cache: LRUCache<string, number[]> | null
  private readonly singleFlight = new SingleFlight<string, number[]>()
  private dim = 0

  constructor(cfg: TEIEmbedderConfig) {
    if (cfg.baseURL === '') throw new ProviderError('tei: baseURL required', 0)
    this.baseURL = cfg.baseURL.replace(/\/+$/, '')
    this.modelName = cfg.model ?? 'tei'
    this.http = cfg.http ?? defaultHttpClient
    this.logger = cfg.logger ?? noopLogger
    const size = cfg.cacheSize ?? DEFAULT_CACHE_SIZE
    this.cache = size > 0 ? new LRUCache<string, number[]>(size) : null
  }

  name(): string {
    return 'tei-embed'
  }

  model(): string {
    return this.modelName
  }

  dimension(): number {
    return this.dim
  }

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return []
    const out: (number[] | undefined)[] = new Array(texts.length)
    const misses: { idx: number; text: string; key: string }[] = []
    for (let i = 0; i < texts.length; i++) {
      const t = texts[i] ?? ''
      const key = cacheKey(this.modelName, t)
      if (this.cache !== null) {
        const hit = this.cache.get(key)
        if (hit !== undefined) {
          out[i] = [...hit]
          continue
        }
      }
      misses.push({ idx: i, text: t, key })
    }

    if (misses.length > 0) {
      const byKey = new Map<string, { text: string; idxs: number[] }>()
      for (const m of misses) {
        const entry = byKey.get(m.key)
        if (entry !== undefined) {
          entry.idxs.push(m.idx)
        } else {
          byKey.set(m.key, { text: m.text, idxs: [m.idx] })
        }
      }
      await Promise.all(
        Array.from(byKey.entries()).map(async ([key, { text, idxs }]) => {
          const vec = await this.singleFlight.do(key, () => this.fetchOne(text, signal))
          if (this.cache !== null) this.cache.set(key, vec)
          for (const idx of idxs) out[idx] = [...vec]
        }),
      )
    }

    for (let i = 0; i < out.length; i++) {
      if (out[i] === undefined) {
        out[i] = this.dim > 0 ? new Array(this.dim).fill(0) : []
      }
    }
    return out as number[][]
  }

  private async fetchOne(text: string, signal: AbortSignal | undefined): Promise<number[]> {
    const { response, text: body } = await postForText(
      this.http,
      `${this.baseURL}/embed`,
      { inputs: [text] },
      { ...(signal ? { signal } : {}) },
    )
    if (!response.ok) {
      throw new ProviderError(
        `tei: /embed failed with status ${response.status}`,
        response.status,
        body,
      )
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(body)
    } catch (err) {
      throw new ProviderError('tei: /embed returned invalid JSON', response.status, body, err)
    }
    if (!Array.isArray(parsed) || parsed.length === 0) {
      throw new ProviderError('tei: /embed returned empty response', response.status, body)
    }
    const first = parsed[0]
    if (!Array.isArray(first) || first.some((v) => typeof v !== 'number')) {
      throw new ProviderError('tei: /embed returned unexpected shape', response.status, body)
    }
    const vec = first as number[]
    if (this.dim === 0) this.dim = vec.length
    this.logger.debug('tei: embed succeeded', { dim: vec.length })
    return vec
  }
}

export type TEIRerankerConfig = {
  baseURL: string
  logger?: Logger
  http?: HttpClient
}

export class TEIReranker implements Reranker {
  private readonly baseURL: string
  private readonly http: HttpClient
  private readonly logger: Logger

  constructor(cfg: TEIRerankerConfig) {
    if (cfg.baseURL === '') throw new ProviderError('tei: baseURL required', 0)
    this.baseURL = cfg.baseURL.replace(/\/+$/, '')
    this.http = cfg.http ?? defaultHttpClient
    this.logger = cfg.logger ?? noopLogger
  }

  name(): string {
    return 'tei-rerank'
  }

  async rerank(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<readonly RerankScore[]> {
    if (documents.length === 0) return []
    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/rerank`,
      { query, texts: documents, raw_scores: true },
      { ...(signal ? { signal } : {}) },
    )
    if (!response.ok) {
      throw new ProviderError(
        `tei: /rerank failed with status ${response.status}`,
        response.status,
        text,
      )
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(text)
    } catch (err) {
      throw new ProviderError('tei: /rerank returned invalid JSON', response.status, text, err)
    }
    if (!Array.isArray(parsed)) {
      throw new ProviderError('tei: /rerank returned unexpected shape', response.status, text)
    }
    const scores: RerankScore[] = []
    for (const raw of parsed) {
      if (
        typeof raw === 'object' &&
        raw !== null &&
        typeof (raw as { index?: unknown }).index === 'number' &&
        typeof (raw as { score?: unknown }).score === 'number'
      ) {
        scores.push({
          index: (raw as { index: number }).index,
          score: (raw as { score: number }).score,
        })
      }
    }
    this.logger.debug('tei: rerank succeeded', { n: scores.length })
    return scores
  }
}

function cacheKey(model: string, text: string): string {
  const hash = createHash('sha256').update(text).digest('hex').slice(0, 16)
  return `${model}\x1f${hash}`
}
