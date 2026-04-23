import { describe, expect, it } from 'vitest'

import { HashEmbedder, createHashEmbedder } from './hashembed.js'

const cosineSim = (left: readonly number[], right: readonly number[]): number => {
  if (left.length !== right.length) throw new Error('vector length mismatch')
  let dot = 0
  let aa = 0
  let bb = 0
  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0
    const rightValue = right[index] ?? 0
    dot += leftValue * rightValue
    aa += leftValue * leftValue
    bb += rightValue * rightValue
  }
  const denom = Math.sqrt(aa) * Math.sqrt(bb)
  return denom === 0 ? 0 : dot / denom
}

const l2Norm = (vector: readonly number[]): number => {
  let sum = 0
  for (let index = 0; index < vector.length; index += 1) {
    const value = vector[index] ?? 0
    sum += value * value
  }
  return Math.sqrt(sum)
}

describe('HashEmbedder', () => {
  it('returns identical vectors for identical input', async () => {
    const embedder = createHashEmbedder()
    const [left] = await embedder.embed(['the quick brown fox jumps over the lazy dog'])
    const [right] = await embedder.embed(['the quick brown fox jumps over the lazy dog'])

    expect(left).toEqual(right)
  })

  it('produces L2-normalised vectors for non-empty input', async () => {
    const embedder = createHashEmbedder()
    const vectors = await embedder.embed([
      'hello world',
      'a',
      'Mary had a little lamb, its fleece was white as snow.',
      'plain ascii text only',
    ])

    for (const vector of vectors) {
      expect(Math.abs(l2Norm(vector) - 1)).toBeLessThan(1e-6)
    }
  })

  it('gives different vectors for unrelated inputs', async () => {
    const embedder = createHashEmbedder()
    const [left, right] = await embedder.embed([
      'the quick brown fox jumps over the lazy dog',
      'artificial intelligence is transforming modern software engineering',
    ])

    expect(cosineSim(left ?? [], right ?? [])).toBeLessThan(0.99)
  })

  it('returns a zero vector for empty string input', async () => {
    const embedder = createHashEmbedder({ dim: 16 })
    const [vector] = await embedder.embed([''])

    expect(vector).toHaveLength(16)
    expect(vector?.every((value) => value === 0)).toBe(true)
  })

  it('respects a custom dim', async () => {
    const embedder = createHashEmbedder({ dim: 64 })
    const [vector] = await embedder.embed(['hello world'])

    expect(embedder.dimension()).toBe(64)
    expect(vector).toHaveLength(64)
    expect(Math.abs(l2Norm(vector ?? []) - 1)).toBeLessThan(1e-6)
  })

  it('single-flights concurrent calls for the same input', async () => {
    const embedder = new HashEmbedder()
    const [left, right] = await Promise.all([
      embedder.embed(['shared token stream']),
      embedder.embed(['shared token stream']),
    ])

    expect(left[0]).toBe(right[0])
  })

  it('reports name, model, and dimension', () => {
    const embedder = createHashEmbedder({ dim: 128 })

    expect(embedder.name()).toBe('hash')
    expect(embedder.model()).toBe('hash-128')
    expect(embedder.dimension()).toBe(128)
  })

  it('rejects invalid dims', () => {
    expect(() => new HashEmbedder({ dim: 0 })).toThrow()
    expect(() => new HashEmbedder({ dim: -5 })).toThrow()
    expect(() => new HashEmbedder({ dim: 1.5 })).toThrow()
  })

  it('returns an empty array for an empty batch', async () => {
    const embedder = createHashEmbedder()

    await expect(embedder.embed([])).resolves.toEqual([])
  })
})
