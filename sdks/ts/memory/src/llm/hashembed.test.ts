// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { HashEmbedder, createHashEmbedder } from './hashembed.js'

function cosineSim(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) throw new Error('vector length mismatch')
  let dot = 0
  let aa = 0
  let bb = 0
  for (let i = 0; i < a.length; i++) {
    const av = a[i] ?? 0
    const bv = b[i] ?? 0
    dot += av * bv
    aa += av * av
    bb += bv * bv
  }
  const denom = Math.sqrt(aa) * Math.sqrt(bb)
  return denom === 0 ? 0 : dot / denom
}

function l2Norm(v: readonly number[]): number {
  let sum = 0
  for (let i = 0; i < v.length; i++) {
    const x = v[i] ?? 0
    sum += x * x
  }
  return Math.sqrt(sum)
}

describe('HashEmbedder', () => {
  it('returns identical vectors for identical input (determinism)', async () => {
    const embedder = createHashEmbedder()
    const [a] = await embedder.embed(['the quick brown fox jumps over the lazy dog'])
    const [b] = await embedder.embed(['the quick brown fox jumps over the lazy dog'])
    expect(a).toBeDefined()
    expect(b).toBeDefined()
    expect(a).toEqual(b)
  })

  it('produces L2-normalised vectors within 1e-6 for non-empty input', async () => {
    const embedder = createHashEmbedder()
    const inputs = [
      'hello world',
      'a',
      'Mary had a little lamb, its fleece was white as snow.',
      'unicode: résumé naïve café — straße',
    ]
    const vectors = await embedder.embed(inputs)
    for (const v of vectors) {
      expect(Math.abs(l2Norm(v) - 1)).toBeLessThan(1e-6)
    }
  })

  it('gives different vectors for unrelated inputs (cosine sim < 0.99)', async () => {
    const embedder = createHashEmbedder()
    const [a, b] = await embedder.embed([
      'the quick brown fox jumps over the lazy dog',
      'artificial intelligence is transforming modern software engineering',
    ])
    expect(a).toBeDefined()
    expect(b).toBeDefined()
    const sim = cosineSim(a ?? [], b ?? [])
    expect(sim).toBeLessThan(0.99)
  })

  it('returns a zero vector for empty string input (documented behaviour)', async () => {
    const embedder = createHashEmbedder({ dim: 16 })
    const [v] = await embedder.embed([''])
    expect(v).toBeDefined()
    expect(v).toHaveLength(16)
    expect(v?.every((x) => x === 0)).toBe(true)
  })

  it('respects a custom dim', async () => {
    const embedder = createHashEmbedder({ dim: 64 })
    expect(embedder.dimension()).toBe(64)
    const [v] = await embedder.embed(['hello world'])
    expect(v).toBeDefined()
    expect(v).toHaveLength(64)
    expect(Math.abs(l2Norm(v ?? []) - 1)).toBeLessThan(1e-6)
  })

  it('single-flights concurrent calls for the same input (same array instance)', async () => {
    const embedder = new HashEmbedder()
    const [resA, resB] = await Promise.all([
      embedder.embed(['shared token stream']),
      embedder.embed(['shared token stream']),
    ])
    expect(resA[0]).toBeDefined()
    expect(resB[0]).toBeDefined()
    expect(resA[0]).toBe(resB[0])
  })

  it('reports name, model, and dimension', () => {
    const e = createHashEmbedder({ dim: 128 })
    expect(e.name()).toBe('hash')
    expect(e.model()).toBe('hash-128')
    expect(e.dimension()).toBe(128)
  })

  it('rejects invalid dims', () => {
    expect(() => new HashEmbedder({ dim: 0 })).toThrow()
    expect(() => new HashEmbedder({ dim: -5 })).toThrow()
    expect(() => new HashEmbedder({ dim: 1.5 })).toThrow()
  })

  it('returns empty array for empty batch', async () => {
    const embedder = createHashEmbedder()
    const out = await embedder.embed([])
    expect(out).toEqual([])
  })
})
