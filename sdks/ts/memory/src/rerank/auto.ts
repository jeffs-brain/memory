// SPDX-License-Identifier: Apache-2.0

/**
 * AutoReranker prefers a primary backend when it reports healthy and
 * falls back to a secondary reranker when the primary is unavailable
 * or the call itself fails.
 */

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
  if (typeof reranker.isAvailable !== 'function') return true
  return reranker.isAvailable(signal)
}

export class AutoReranker implements Reranker {
  private readonly primary: Reranker
  private readonly fallback: Reranker | undefined
  private readonly label: string

  constructor(cfg: AutoRerankerConfig) {
    this.primary = cfg.primary
    this.fallback = cfg.fallback
    this.label = cfg.label ?? 'auto-rerank'
  }

  name(): string {
    return this.label
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    if (await rerankerAvailable(this.primary, signal)) return true
    return rerankerAvailable(this.fallback, signal)
  }

  async rerank(req: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]> {
    const primaryAvailable = await rerankerAvailable(this.primary, signal).catch(() => false)
    if (primaryAvailable) {
      try {
        return await this.primary.rerank(req, signal)
      } catch (err) {
        if (this.fallback === undefined) throw err
      }
    }

    if (this.fallback !== undefined) {
      return this.fallback.rerank(req, signal)
    }

    throw new Error('auto-rerank: no available reranker')
  }
}
