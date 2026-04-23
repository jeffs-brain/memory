import { createCacheKey } from './cache-key.js'
import { ProviderError } from './errors.js'
import { type HttpClient, defaultHttpClient, postForText } from './http.js'
import { LRUCache, SingleFlight } from './lru.js'
import type { Embedder, Logger, RerankScore, Reranker } from './types.js'
import { noopLogger } from './types.js'

const DEFAULT_CACHE_SIZE = 10_000
const DEFAULT_TEI_PROBE_TTL_MS = 30_000

export type TEIEmbedderConfig = {
  readonly baseURL: string
  readonly model?: string
  readonly cacheSize?: number
  readonly logger?: Logger
  readonly http?: HttpClient
}

export class TEIEmbedder implements Embedder {
  private readonly baseURL: string
  private readonly modelNameValue: string
  private readonly http: HttpClient
  private readonly logger: Logger
  private readonly cache: LRUCache<string, number[]> | null
  private readonly singleFlight = new SingleFlight<string, number[]>()
  private dimensionValue = 0

  constructor(config: TEIEmbedderConfig) {
    if (config.baseURL === '') throw new ProviderError('tei: baseURL required', 0)
    this.baseURL = config.baseURL.replace(/\/+$/, '')
    this.modelNameValue = config.model ?? 'tei'
    this.http = config.http ?? defaultHttpClient
    this.logger = config.logger ?? noopLogger
    const size = config.cacheSize ?? DEFAULT_CACHE_SIZE
    this.cache = size > 0 ? new LRUCache<string, number[]>(size) : null
  }

  name(): string {
    return 'tei-embed'
  }

  model(): string {
    return this.modelNameValue
  }

  dimension(): number {
    return this.dimensionValue
  }

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return []

    const out: (number[] | undefined)[] = new Array(texts.length)
    const misses: Array<{ idx: number; text: string; key: string }> = []

    for (let index = 0; index < texts.length; index += 1) {
      const text = texts[index] ?? ''
      const key = createCacheKey(this.modelNameValue, text)
      if (this.cache !== null) {
        const cached = this.cache.get(key)
        if (cached !== undefined) {
          out[index] = [...cached]
          continue
        }
      }
      misses.push({ idx: index, text, key })
    }

    if (misses.length > 0) {
      const byKey = new Map<string, { text: string; indexes: number[] }>()
      for (const miss of misses) {
        const existing = byKey.get(miss.key)
        if (existing !== undefined) {
          existing.indexes.push(miss.idx)
          continue
        }
        byKey.set(miss.key, { text: miss.text, indexes: [miss.idx] })
      }

      await Promise.all(
        Array.from(byKey.entries()).map(async ([key, entry]) => {
          const vector = await this.singleFlight.do(key, () => this.fetchOne(entry.text, signal))
          if (this.cache !== null) {
            this.cache.set(key, vector)
          }
          for (const index of entry.indexes) {
            out[index] = [...vector]
          }
        }),
      )
    }

    for (let index = 0; index < out.length; index += 1) {
      if (out[index] === undefined) {
        out[index] = this.dimensionValue > 0 ? new Array(this.dimensionValue).fill(0) : []
      }
    }

    return out as number[][]
  }

  private async fetchOne(text: string, signal: AbortSignal | undefined): Promise<number[]> {
    const { response, text: body } = await postForText(
      this.http,
      `${this.baseURL}/embed`,
      { inputs: [text] },
      { ...(signal === undefined ? {} : { signal }) },
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
    } catch (error) {
      throw new ProviderError('tei: /embed returned invalid JSON', response.status, body, error)
    }

    if (!Array.isArray(parsed) || parsed.length === 0) {
      throw new ProviderError('tei: /embed returned empty response', response.status, body)
    }

    const first = parsed[0]
    if (!Array.isArray(first) || first.some((value) => typeof value !== 'number')) {
      throw new ProviderError('tei: /embed returned unexpected shape', response.status, body)
    }

    const vector = first as number[]
    if (this.dimensionValue === 0) {
      this.dimensionValue = vector.length
    }
    this.logger.debug('tei: embed succeeded', { dim: vector.length })
    return vector
  }
}

export type TEIRerankerConfig = {
  readonly baseURL: string
  readonly logger?: Logger
  readonly http?: HttpClient
  readonly probeTtlMs?: number
}

export class TEIReranker implements Reranker {
  private readonly baseURL: string
  private readonly http: HttpClient
  private readonly logger: Logger
  private readonly probeTtlMs: number
  private availabilityMemo:
    | {
        value: boolean
        checkedAt: number
      }
    | undefined
  private availabilityProbe: Promise<boolean> | undefined

  constructor(config: TEIRerankerConfig) {
    if (config.baseURL === '') throw new ProviderError('tei: baseURL required', 0)
    this.baseURL = config.baseURL.replace(/\/+$/, '')
    this.http = config.http ?? defaultHttpClient
    this.logger = config.logger ?? noopLogger
    this.probeTtlMs = config.probeTtlMs ?? DEFAULT_TEI_PROBE_TTL_MS
  }

  name(): string {
    return 'tei-rerank'
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    const cached = this.availabilityMemo
    if (cached !== undefined && Date.now() - cached.checkedAt < this.probeTtlMs) {
      return cached.value
    }
    if (this.availabilityProbe !== undefined) {
      return this.availabilityProbe
    }

    const probe = this.probeAvailability(signal)
      .then((value) => {
        this.setAvailability(value)
        return value
      })
      .finally(() => {
        this.availabilityProbe = undefined
      })

    this.availabilityProbe = probe
    return probe
  }

  async rerank(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<readonly RerankScore[]> {
    if (documents.length === 0) return []

    try {
      const { response, text } = await postForText(
        this.http,
        `${this.baseURL}/rerank`,
        { query, texts: documents, raw_scores: true },
        { ...(signal === undefined ? {} : { signal }) },
      )

      if (!response.ok) {
        this.setAvailability(response.status < 500)
        throw new ProviderError(
          `tei: /rerank failed with status ${response.status}`,
          response.status,
          text,
        )
      }

      let parsed: unknown
      try {
        parsed = JSON.parse(text)
      } catch (error) {
        this.setAvailability(false)
        throw new ProviderError('tei: /rerank returned invalid JSON', response.status, text, error)
      }

      if (!Array.isArray(parsed)) {
        this.setAvailability(false)
        throw new ProviderError('tei: /rerank returned unexpected shape', response.status, text)
      }

      const scores: RerankScore[] = []
      for (const entry of parsed) {
        if (
          typeof entry === 'object' &&
          entry !== null &&
          typeof (entry as { index?: unknown }).index === 'number' &&
          typeof (entry as { score?: unknown }).score === 'number'
        ) {
          scores.push({
            index: (entry as { index: number }).index,
            score: (entry as { score: number }).score,
          })
        }
      }

      this.setAvailability(true)
      this.logger.debug('tei: rerank succeeded', { n: scores.length })
      return scores
    } catch (error) {
      if (error instanceof ProviderError) {
        throw error
      }
      this.setAvailability(false)
      throw error
    }
  }

  private setAvailability(value: boolean): void {
    this.availabilityMemo = {
      value,
      checkedAt: Date.now(),
    }
  }

  private async probeAvailability(signal?: AbortSignal): Promise<boolean> {
    for (const path of ['/health', '/info']) {
      try {
        const response = await this.http.fetch(`${this.baseURL}${path}`, {
          method: 'GET',
          headers: { accept: 'application/json' },
          ...(signal === undefined ? {} : { signal }),
        })
        void response.text().catch(() => undefined)
        if (response.ok) {
          return true
        }
      } catch {
        // Try the next path.
      }
    }
    return false
  }
}
