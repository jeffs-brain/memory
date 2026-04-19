// SPDX-License-Identifier: Apache-2.0

/**
 * CrossEncoderReranker adapts the llm/TEIReranker (a thin HTTP client
 * over TEI /rerank) into the retrieval-pipeline Reranker contract.
 * Scores are returned sorted descending by score; ties fall back to
 * original input order so callers get a deterministic ranking.
 */

import type { Reranker as TEIRerankerContract } from '../llm/types.js'
import { runWithSharedRerankConcurrency } from './concurrency.js'
import type { Reranker, RerankRequest, RerankResult } from './index.js'

export type CrossEncoderRerankerConfig = {
  /** Underlying TEI rerank client (typically TEIReranker from ../llm). */
  client: TEIRerankerContract
  /** Optional label surfaced via name() for traces. */
  label?: string
  /** Shared concurrency cap for outbound rerank requests. */
  concurrencyCap?: number
}

export class CrossEncoderReranker implements Reranker {
  private readonly client: TEIRerankerContract
  private readonly label: string
  private readonly concurrencyCap: number | undefined

  constructor(cfg: CrossEncoderRerankerConfig) {
    this.client = cfg.client
    this.label = cfg.label ?? 'cross-encoder'
    this.concurrencyCap = cfg.concurrencyCap
  }

  name(): string {
    return this.label
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    if (typeof this.client.isAvailable !== 'function') return true
    return this.client.isAvailable(signal)
  }

  async rerank(req: RerankRequest, signal?: AbortSignal): Promise<readonly RerankResult[]> {
    if (req.documents.length === 0) return []

    const texts = req.documents.map((d) => d.text)
    const raw = await runWithSharedRerankConcurrency(
      () => this.client.rerank(req.query, texts, signal),
      this.concurrencyCap,
    )

    // Seed every candidate so missing indices sink to the tail rather
    // than silently vanishing from the ranking. This mirrors the Go
    // implementation's behaviour when the backend skips a document.
    const seeded: RerankResult[] = req.documents.map((doc, i) => ({
      index: i,
      id: doc.id,
      score: Number.NEGATIVE_INFINITY,
    }))
    for (const hit of raw) {
      if (hit.index < 0 || hit.index >= seeded.length) continue
      const slot = seeded[hit.index]
      if (slot === undefined) continue
      slot.score = hit.score
    }

    const sorted = [...seeded].sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score
      return a.index - b.index
    })
    return sorted
  }
}
