// SPDX-License-Identifier: Apache-2.0

/**
 * Cost accounting for the LME eval harness. Amounts are stored as
 * microcents (USD * 1e8) using BigInt so sub-cent additions never
 * round to zero — mirroring the Go implementation in
 * `go/eval/lme/costs.go`.
 */

/**
 * Per-million-token input and output prices in USD.
 */
export type Pricing = {
  readonly inputPerMTok: number
  readonly outputPerMTok: number
}

/**
 * Hard-coded price table for models the eval harness routinely invokes.
 * Prices are USD per million tokens and reflect public list prices.
 */
export const DEFAULT_PRICING: Readonly<Record<string, Pricing>> = {
  'gpt-4o': { inputPerMTok: 2.5, outputPerMTok: 10.0 },
  'gpt-4o-2024-08-06': { inputPerMTok: 2.5, outputPerMTok: 10.0 },
  'gpt-4o-mini': { inputPerMTok: 0.15, outputPerMTok: 0.6 },
  'claude-sonnet-4': { inputPerMTok: 3.0, outputPerMTok: 15.0 },
  'claude-sonnet-4-6': { inputPerMTok: 3.0, outputPerMTok: 15.0 },
  'claude-sonnet-4-7': { inputPerMTok: 3.0, outputPerMTok: 15.0 },
  'claude-opus-4': { inputPerMTok: 15.0, outputPerMTok: 75.0 },
  'claude-opus-4-6': { inputPerMTok: 15.0, outputPerMTok: 75.0 },
  'claude-opus-4-7': { inputPerMTok: 15.0, outputPerMTok: 75.0 },
  'claude-haiku-4-5': { inputPerMTok: 1.0, outputPerMTok: 5.0 },
  'gemma-4-31B-it': { inputPerMTok: 0.0, outputPerMTok: 0.0 },
  'bge-m3': { inputPerMTok: 0.0, outputPerMTok: 0.0 },
}

/**
 * Token-level accounting shape threaded through provider calls.
 * Mirrors the Go `Usage` struct: input/output tokens plus cache
 * buckets for providers that break them out.
 */
export type Usage = {
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheRead: number
  readonly cacheCreate: number
}

/**
 * Returns an approximate USD cost for a single LLM call given a model
 * name and token usage. Unknown models price at zero.
 */
export const estimateUSD = (model: string, usage: Usage): number => {
  const price = DEFAULT_PRICING[model]
  if (price === undefined) return 0
  const input = ((usage.inputTokens + usage.cacheCreate) / 1_000_000) * price.inputPerMTok
  const cache = (usage.cacheRead / 1_000_000) * price.inputPerMTok * 0.1
  const output = (usage.outputTokens / 1_000_000) * price.outputPerMTok
  return input + cache + output
}

/**
 * Breaks down run cost by stage so expensive components are visible
 * per-run rather than hidden in a single total.
 */
export type CostAccounting = {
  readonly ingestUSD: number
  readonly agentUSD: number
  readonly judgeUSD: number
  readonly totalUSD: number
}

/**
 * Microcent scale factor: 1e8 keeps eight decimal places of precision
 * so sub-cent additions do not round to zero.
 */
const MICRO_SCALE = 100_000_000n

const usdToMicro = (usd: number): bigint => BigInt(Math.round(usd * Number(MICRO_SCALE)))

const microToUSD = (micro: bigint): number => Number(micro) / Number(MICRO_SCALE)

/**
 * Accumulates token costs across concurrent work without float drift.
 * All amounts are stored internally as microcents (USD * 1e8) using
 * BigInt arithmetic. Call `snapshot()` for the final USD breakdown.
 *
 * Note: JavaScript is single-threaded so no mutex is required, but the
 * accumulator API mirrors the Go CostAccumulator for familiarity.
 */
export class CostAccumulator {
  private ingestMicro = 0n
  private agentMicro = 0n
  private judgeMicro = 0n

  addIngest(usd: number): void {
    this.ingestMicro += usdToMicro(usd)
  }

  addAgent(usd: number): void {
    this.agentMicro += usdToMicro(usd)
  }

  addJudge(usd: number): void {
    this.judgeMicro += usdToMicro(usd)
  }

  /**
   * Returns a CostAccounting snapshot whose `totalUSD` is exactly the
   * sum of the three buckets, free of float rounding drift.
   */
  snapshot(): CostAccounting {
    const ingest = this.ingestMicro
    const agent = this.agentMicro
    const judge = this.judgeMicro
    return {
      ingestUSD: microToUSD(ingest),
      agentUSD: microToUSD(agent),
      judgeUSD: microToUSD(judge),
      totalUSD: microToUSD(ingest + agent + judge),
    }
  }
}

/**
 * Returns a zero-value CostAccounting (all fields zero).
 */
export const zeroCostAccounting = (): CostAccounting => ({
  ingestUSD: 0,
  agentUSD: 0,
  judgeUSD: 0,
  totalUSD: 0,
})

/**
 * Adds two CostAccounting structs together. Uses microcent conversion
 * internally to avoid float drift on the total.
 */
export const addCostAccounting = (a: CostAccounting, b: CostAccounting): CostAccounting => {
  const ingestMicro = usdToMicro(a.ingestUSD) + usdToMicro(b.ingestUSD)
  const agentMicro = usdToMicro(a.agentUSD) + usdToMicro(b.agentUSD)
  const judgeMicro = usdToMicro(a.judgeUSD) + usdToMicro(b.judgeUSD)
  return {
    ingestUSD: microToUSD(ingestMicro),
    agentUSD: microToUSD(agentMicro),
    judgeUSD: microToUSD(judgeMicro),
    totalUSD: microToUSD(ingestMicro + agentMicro + judgeMicro),
  }
}

/**
 * Constructs a Usage from raw token counts, defaulting cache fields to
 * zero. Convenience helper matching Go's `usageFromResponse`.
 */
export const usageFromTokens = (inputTokens: number, outputTokens: number): Usage => ({
  inputTokens,
  outputTokens,
  cacheRead: 0,
  cacheCreate: 0,
})
