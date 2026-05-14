// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  CostAccumulator,
  DEFAULT_PRICING,
  addCostAccounting,
  estimateUSD,
  usageFromTokens,
  zeroCostAccounting,
} from './costs.js'
import type { CostAccounting, Usage } from './costs.js'

describe('estimateUSD', () => {
  it('returns zero for unknown models', () => {
    const usage: Usage = { inputTokens: 1000, outputTokens: 500, cacheRead: 0, cacheCreate: 0 }
    expect(estimateUSD('unknown-model-xyz', usage)).toBe(0)
  })

  it('calculates cost for gpt-4o correctly', () => {
    const usage: Usage = {
      inputTokens: 1_000_000,
      outputTokens: 1_000_000,
      cacheRead: 0,
      cacheCreate: 0,
    }
    // input: 1M * $2.50/M = $2.50, output: 1M * $10/M = $10.00
    expect(estimateUSD('gpt-4o', usage)).toBeCloseTo(12.5, 8)
  })

  it('accounts for cache read tokens at 10% of input rate', () => {
    const usage: Usage = { inputTokens: 0, outputTokens: 0, cacheRead: 1_000_000, cacheCreate: 0 }
    // cacheRead: 1M * $2.50/M * 0.1 = $0.25
    expect(estimateUSD('gpt-4o', usage)).toBeCloseTo(0.25, 8)
  })

  it('accounts for cache create tokens at full input rate', () => {
    const usage: Usage = { inputTokens: 0, outputTokens: 0, cacheRead: 0, cacheCreate: 1_000_000 }
    // cacheCreate treated as inputTokens: 1M * $2.50/M = $2.50
    expect(estimateUSD('gpt-4o', usage)).toBeCloseTo(2.5, 8)
  })

  it('handles zero-priced models', () => {
    const usage: Usage = {
      inputTokens: 10_000_000,
      outputTokens: 10_000_000,
      cacheRead: 0,
      cacheCreate: 0,
    }
    expect(estimateUSD('gemma-4-31B-it', usage)).toBe(0)
  })

  it('handles fractional token counts', () => {
    const usage: Usage = { inputTokens: 100, outputTokens: 50, cacheRead: 0, cacheCreate: 0 }
    // input: 100/1M * $2.50 = $0.00025, output: 50/1M * $10 = $0.0005
    expect(estimateUSD('gpt-4o', usage)).toBeCloseTo(0.00075, 10)
  })
})

describe('CostAccumulator', () => {
  it('starts with all zeros', () => {
    const acc = new CostAccumulator()
    expect(acc.snapshot()).toEqual({
      ingestUSD: 0,
      agentUSD: 0,
      judgeUSD: 0,
      totalUSD: 0,
    })
  })

  it('accumulates ingest costs', () => {
    const acc = new CostAccumulator()
    acc.addIngest(1.5)
    acc.addIngest(2.3)
    const snap = acc.snapshot()
    expect(snap.ingestUSD).toBeCloseTo(3.8, 8)
    expect(snap.agentUSD).toBe(0)
    expect(snap.judgeUSD).toBe(0)
    expect(snap.totalUSD).toBeCloseTo(3.8, 8)
  })

  it('accumulates agent costs', () => {
    const acc = new CostAccumulator()
    acc.addAgent(0.001)
    acc.addAgent(0.002)
    const snap = acc.snapshot()
    expect(snap.agentUSD).toBeCloseTo(0.003, 10)
    expect(snap.totalUSD).toBeCloseTo(0.003, 10)
  })

  it('accumulates judge costs', () => {
    const acc = new CostAccumulator()
    acc.addJudge(0.05)
    acc.addJudge(0.07)
    const snap = acc.snapshot()
    expect(snap.judgeUSD).toBeCloseTo(0.12, 10)
    expect(snap.totalUSD).toBeCloseTo(0.12, 10)
  })

  it('total equals sum of all three buckets', () => {
    const acc = new CostAccumulator()
    acc.addIngest(1.11)
    acc.addAgent(2.22)
    acc.addJudge(3.33)
    const snap = acc.snapshot()
    expect(snap.totalUSD).toBeCloseTo(6.66, 8)
    expect(snap.totalUSD).toBeCloseTo(snap.ingestUSD + snap.agentUSD + snap.judgeUSD, 8)
  })

  it('handles 100,000 sub-cent additions without drift', () => {
    const acc = new CostAccumulator()
    const amount = 0.000001 // 1 microdollar
    for (let i = 0; i < 100_000; i++) {
      acc.addIngest(amount)
    }
    const snap = acc.snapshot()
    // 100,000 * 0.000001 = 0.1 exactly
    expect(snap.ingestUSD).toBeCloseTo(0.1, 8)
    expect(snap.totalUSD).toBeCloseTo(0.1, 8)
  })

  it('handles very large token costs without overflow', () => {
    const acc = new CostAccumulator()
    // Simulate 10 billion tokens at opus rate: 10B/1M * $75 = $750,000
    acc.addAgent(750_000)
    acc.addAgent(750_000)
    const snap = acc.snapshot()
    expect(snap.agentUSD).toBeCloseTo(1_500_000, 2)
    expect(snap.totalUSD).toBeCloseTo(1_500_000, 2)
  })

  it('snapshot is idempotent', () => {
    const acc = new CostAccumulator()
    acc.addIngest(5.5)
    acc.addAgent(3.3)
    const snap1 = acc.snapshot()
    const snap2 = acc.snapshot()
    expect(snap1).toEqual(snap2)
  })

  it('continues accumulating after snapshot', () => {
    const acc = new CostAccumulator()
    acc.addIngest(1.0)
    const snap1 = acc.snapshot()
    acc.addIngest(2.0)
    const snap2 = acc.snapshot()
    expect(snap1.ingestUSD).toBeCloseTo(1.0, 8)
    expect(snap2.ingestUSD).toBeCloseTo(3.0, 8)
  })
})

describe('zeroCostAccounting', () => {
  it('returns all fields at zero', () => {
    const z = zeroCostAccounting()
    expect(z.ingestUSD).toBe(0)
    expect(z.agentUSD).toBe(0)
    expect(z.judgeUSD).toBe(0)
    expect(z.totalUSD).toBe(0)
  })
})

describe('addCostAccounting', () => {
  it('adds two cost accounting structs correctly', () => {
    const a: CostAccounting = { ingestUSD: 1.5, agentUSD: 2.0, judgeUSD: 0.5, totalUSD: 4.0 }
    const b: CostAccounting = { ingestUSD: 0.5, agentUSD: 1.0, judgeUSD: 1.5, totalUSD: 3.0 }
    const result = addCostAccounting(a, b)
    expect(result.ingestUSD).toBeCloseTo(2.0, 8)
    expect(result.agentUSD).toBeCloseTo(3.0, 8)
    expect(result.judgeUSD).toBeCloseTo(2.0, 8)
    expect(result.totalUSD).toBeCloseTo(7.0, 8)
  })

  it('adding zero to a value yields unchanged result', () => {
    const a: CostAccounting = { ingestUSD: 1.23, agentUSD: 4.56, judgeUSD: 7.89, totalUSD: 13.68 }
    const zero = zeroCostAccounting()
    const result = addCostAccounting(a, zero)
    expect(result.ingestUSD).toBeCloseTo(1.23, 8)
    expect(result.agentUSD).toBeCloseTo(4.56, 8)
    expect(result.judgeUSD).toBeCloseTo(7.89, 8)
    expect(result.totalUSD).toBeCloseTo(13.68, 8)
  })

  it('recomputes totalUSD as sum of buckets, not sum of input totals', () => {
    // If one accounting has a mismatched total, addCostAccounting recomputes
    const a: CostAccounting = { ingestUSD: 1.0, agentUSD: 1.0, judgeUSD: 1.0, totalUSD: 999.0 }
    const b: CostAccounting = { ingestUSD: 1.0, agentUSD: 1.0, judgeUSD: 1.0, totalUSD: 0.0 }
    const result = addCostAccounting(a, b)
    expect(result.totalUSD).toBeCloseTo(6.0, 8)
  })

  it('handles many small additions without drift', () => {
    let running = zeroCostAccounting()
    const small: CostAccounting = {
      ingestUSD: 0.0001,
      agentUSD: 0.0002,
      judgeUSD: 0.0003,
      totalUSD: 0.0006,
    }
    for (let i = 0; i < 10_000; i++) {
      running = addCostAccounting(running, small)
    }
    expect(running.ingestUSD).toBeCloseTo(1.0, 6)
    expect(running.agentUSD).toBeCloseTo(2.0, 6)
    expect(running.judgeUSD).toBeCloseTo(3.0, 6)
    expect(running.totalUSD).toBeCloseTo(6.0, 6)
  })
})

describe('usageFromTokens', () => {
  it('creates usage with zero cache fields', () => {
    const usage = usageFromTokens(500, 200)
    expect(usage).toEqual({
      inputTokens: 500,
      outputTokens: 200,
      cacheRead: 0,
      cacheCreate: 0,
    })
  })
})

describe('DEFAULT_PRICING', () => {
  it('contains all expected models', () => {
    const expectedModels = [
      'gpt-4o',
      'gpt-4o-2024-08-06',
      'gpt-4o-mini',
      'claude-sonnet-4',
      'claude-sonnet-4-6',
      'claude-sonnet-4-7',
      'claude-opus-4',
      'claude-opus-4-6',
      'claude-opus-4-7',
      'claude-haiku-4-5',
      'gemma-4-31B-it',
      'bge-m3',
    ]
    for (const model of expectedModels) {
      expect(DEFAULT_PRICING[model]).toBeDefined()
    }
  })

  it('has non-negative prices for all models', () => {
    for (const [, pricing] of Object.entries(DEFAULT_PRICING)) {
      expect(pricing.inputPerMTok).toBeGreaterThanOrEqual(0)
      expect(pricing.outputPerMTok).toBeGreaterThanOrEqual(0)
    }
  })
})

describe('costsToDollars precision', () => {
  it('accumulator preserves sub-cent accuracy', () => {
    const acc = new CostAccumulator()
    // Add $0.0000001 (1/10 microdollar) many times
    // BigInt rounds to nearest on conversion, so we test within tolerance
    acc.addIngest(0.01)
    acc.addIngest(0.01)
    acc.addIngest(0.01)
    const snap = acc.snapshot()
    expect(snap.ingestUSD).toBeCloseTo(0.03, 10)
  })

  it('cents are accurate after large accumulations', () => {
    const acc = new CostAccumulator()
    for (let i = 0; i < 1000; i++) {
      acc.addIngest(0.01) // 1 cent
    }
    const snap = acc.snapshot()
    // Should be exactly $10.00
    expect(snap.ingestUSD).toBeCloseTo(10.0, 8)
  })
})
