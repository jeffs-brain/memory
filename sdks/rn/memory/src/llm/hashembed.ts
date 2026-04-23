import { SingleFlight } from './lru.js'
import type { Embedder } from './types.js'

const DEFAULT_DIM = 1024
const FNV_OFFSET = 0xcbf29ce484222325n
const FNV_PRIME = 0x100000001b3n
const MASK_64 = 0xffffffffffffffffn

const tokenise = (text: string): string[] => {
  if (text === '') return []
  const out: string[] = []
  const parts = text.toLowerCase().split(/[\s\p{P}]+/u)
  for (const part of parts) {
    if (part.length > 0) out.push(part)
  }
  return out
}

const fnv1a64 = (input: string): bigint => {
  let hash = FNV_OFFSET
  const bytes = new TextEncoder().encode(input)
  for (let index = 0; index < bytes.length; index += 1) {
    hash ^= BigInt(bytes[index] ?? 0)
    hash = (hash * FNV_PRIME) & MASK_64
  }
  return hash
}

export type HashEmbedderOptions = {
  readonly dim?: number
}

export class HashEmbedder implements Embedder {
  private readonly dimValue: number
  private readonly singleFlight = new SingleFlight<string, number[]>()
  private readonly cache = new Map<string, number[]>()

  constructor(options: HashEmbedderOptions = {}) {
    const requested = options.dim ?? DEFAULT_DIM
    if (!Number.isFinite(requested) || requested <= 0 || !Number.isInteger(requested)) {
      throw new Error(`HashEmbedder dim must be a positive integer, got ${String(requested)}`)
    }
    this.dimValue = requested
  }

  name(): string {
    return 'hash'
  }

  model(): string {
    return `hash-${this.dimValue}`
  }

  dimension(): number {
    return this.dimValue
  }

  async embed(texts: readonly string[]): Promise<number[][]> {
    if (texts.length === 0) return []
    const out: number[][] = new Array(texts.length)
    await Promise.all(
      texts.map(async (text, index) => {
        const cached = this.cache.get(text)
        if (cached !== undefined) {
          out[index] = cached
          return
        }
        const vector = await this.singleFlight.do(text, async () => {
          const existing = this.cache.get(text)
          if (existing !== undefined) return existing
          const fresh = this.embedOne(text)
          this.cache.set(text, fresh)
          return fresh
        })
        out[index] = vector
      }),
    )
    return out
  }

  private embedOne(text: string): number[] {
    const acc = new Float32Array(this.dimValue)
    const tokens = tokenise(text)
    if (tokens.length === 0) {
      return Array.from(acc)
    }

    const dim = BigInt(this.dimValue)
    for (const token of tokens) {
      const hash = fnv1a64(token)
      const sign = (hash & 1n) === 1n ? 1 : -1
      const bucket = Number((hash >> 1n) % dim)
      acc[bucket] = (acc[bucket] ?? 0) + sign
    }

    let sumSq = 0
    for (let index = 0; index < acc.length; index += 1) {
      const value = acc[index] ?? 0
      sumSq += value * value
    }

    const norm = Math.sqrt(sumSq)
    const result: number[] = new Array(this.dimValue)
    if (norm === 0) {
      for (let index = 0; index < this.dimValue; index += 1) {
        result[index] = 0
      }
      return result
    }

    for (let index = 0; index < this.dimValue; index += 1) {
      result[index] = (acc[index] ?? 0) / norm
    }
    return result
  }
}

export const createHashEmbedder = (options: HashEmbedderOptions = {}): HashEmbedder => {
  return new HashEmbedder(options)
}
