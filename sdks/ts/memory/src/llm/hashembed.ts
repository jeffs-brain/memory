// SPDX-License-Identifier: Apache-2.0

/**
 * Deterministic dev-only fallback embedder. Tokenises input on whitespace
 * and Unicode punctuation boundaries, hashes each token with FNV-1a 64-bit,
 * folds into a fixed-size Float32 bucket vector with signed contributions,
 * then L2-normalises and returns a plain number[].
 *
 * Zero network, stable across runs, trivially cheap. Retrieval quality is
 * awful: this is intended for local smoke tests, CI bootstrap, and offline
 * demos. Do NOT use in production.
 */

import { SingleFlight } from './lru.js'
import type { Embedder } from './types.js'

const DEFAULT_DIM = 1024

/** FNV-1a 64-bit offset basis and prime, held as BigInt for portability. */
const FNV_OFFSET = 0xcbf29ce484222325n
const FNV_PRIME = 0x100000001b3n
const MASK_64 = 0xffffffffffffffffn

/** Tokeniser: splits on whitespace and any Unicode punctuation, lowercases. */
function tokenise(text: string): string[] {
  if (text === '') return []
  const out: string[] = []
  const parts = text.toLowerCase().split(/[\s\p{P}]+/u)
  for (const part of parts) {
    if (part.length > 0) out.push(part)
  }
  return out
}

/** FNV-1a 64-bit over the UTF-8 bytes of a string. */
function fnv1a64(input: string): bigint {
  let hash = FNV_OFFSET
  const bytes = new TextEncoder().encode(input)
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i] ?? 0
    hash ^= BigInt(byte)
    hash = (hash * FNV_PRIME) & MASK_64
  }
  return hash
}

export type HashEmbedderOptions = {
  /** Output dimension. Default 1024. Must be > 0. */
  dim?: number
}

export class HashEmbedder implements Embedder {
  private readonly dim: number
  private readonly singleFlight = new SingleFlight<string, number[]>()
  private readonly cache = new Map<string, number[]>()

  constructor(opts: HashEmbedderOptions = {}) {
    const requested = opts.dim ?? DEFAULT_DIM
    if (!Number.isFinite(requested) || requested <= 0 || !Number.isInteger(requested)) {
      throw new Error(`HashEmbedder dim must be a positive integer, got ${String(requested)}`)
    }
    this.dim = requested
  }

  name(): string {
    return 'hash'
  }

  model(): string {
    return `hash-${this.dim}`
  }

  dimension(): number {
    return this.dim
  }

  async embed(texts: readonly string[]): Promise<number[][]> {
    if (texts.length === 0) return []
    const out: number[][] = new Array(texts.length)
    await Promise.all(
      texts.map(async (text, idx) => {
        const key = text
        const cached = this.cache.get(key)
        if (cached !== undefined) {
          out[idx] = cached
          return
        }
        const vec = await this.singleFlight.do(key, async () => {
          const existing = this.cache.get(key)
          if (existing !== undefined) return existing
          const fresh = this.embedOne(text)
          this.cache.set(key, fresh)
          return fresh
        })
        out[idx] = vec
      }),
    )
    return out
  }

  private embedOne(text: string): number[] {
    const acc = new Float32Array(this.dim)
    const tokens = tokenise(text)
    if (tokens.length === 0) {
      return Array.from(acc)
    }
    const dimBig = BigInt(this.dim)
    for (const token of tokens) {
      const hash = fnv1a64(token)
      const sign = (hash & 1n) === 1n ? 1 : -1
      const bucket = Number((hash >> 1n) % dimBig)
      acc[bucket] = (acc[bucket] ?? 0) + sign
    }
    let sumSq = 0
    for (let i = 0; i < acc.length; i++) {
      const v = acc[i] ?? 0
      sumSq += v * v
    }
    const norm = Math.sqrt(sumSq)
    const result: number[] = new Array(this.dim)
    if (norm === 0) {
      for (let i = 0; i < this.dim; i++) result[i] = 0
      return result
    }
    for (let i = 0; i < this.dim; i++) {
      result[i] = (acc[i] ?? 0) / norm
    }
    return result
  }
}

/** Convenience factory that mirrors the other embedder factories. */
export function createHashEmbedder(opts: HashEmbedderOptions = {}): HashEmbedder {
  return new HashEmbedder(opts)
}
