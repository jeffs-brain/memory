import type { RerankRequest, RerankResult, Reranker } from './index.js'

export type AutoRerankerConfig = {
  readonly primary: Reranker
  readonly fallback?: Reranker
  readonly label?: string
}

const rerankerAvailable = async (
  reranker: Reranker | undefined,
  signal?: AbortSignal,
): Promise<boolean> => {
  if (reranker === undefined) return false
  if (reranker.isAvailable === undefined) return true
  return await reranker.isAvailable(signal)
}

export class AutoReranker implements Reranker {
  private readonly primary: Reranker
  private readonly fallback: Reranker | undefined
  private readonly label: string

  constructor(config: AutoRerankerConfig) {
    this.primary = config.primary
    this.fallback = config.fallback
    this.label = config.label ?? 'auto-rerank'
  }

  name(): string {
    return this.label
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    if (await rerankerAvailable(this.primary, signal)) return true
    return await rerankerAvailable(this.fallback, signal)
  }

  async rerank(request: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]> {
    const primaryAvailable = await rerankerAvailable(this.primary, signal).catch(() => false)
    if (primaryAvailable) {
      try {
        return await this.primary.rerank(request, signal)
      } catch (error) {
        if (this.fallback === undefined) throw error
      }
    }

    if (this.fallback !== undefined) {
      return await this.fallback.rerank(request, signal)
    }

    throw new Error('auto-rerank: no available reranker')
  }
}
