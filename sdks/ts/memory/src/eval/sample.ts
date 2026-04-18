// SPDX-License-Identifier: Apache-2.0

import type { LMEExample } from './types.js'

export type SampleExamplesArgs = {
  readonly examples: readonly LMEExample[]
  readonly size: number
  readonly seed: number
}

export const sampleExamples = (
  args: SampleExamplesArgs,
): readonly LMEExample[] => {
  if (args.size <= 0 || args.size >= args.examples.length) {
    return [...args.examples]
  }

  const byCategory = new Map<string, LMEExample[]>()
  for (const example of args.examples) {
    const bucket = byCategory.get(example.category)
    if (bucket !== undefined) {
      bucket.push(example)
      continue
    }
    byCategory.set(example.category, [example])
  }

  const categories = [...byCategory.keys()].sort((left, right) =>
    left.localeCompare(right),
  )
  const allocations = new Map<string, number>()
  let assigned = 0

  for (const category of categories) {
    const bucket = byCategory.get(category) ?? []
    const allocation = Math.floor((bucket.length * args.size) / args.examples.length)
    allocations.set(category, allocation)
    assigned += allocation
  }

  let remaining = args.size - assigned
  for (const category of categories) {
    if (remaining <= 0) break
    const bucket = byCategory.get(category) ?? []
    const current = allocations.get(category) ?? 0
    if (current >= bucket.length) continue
    allocations.set(category, current + 1)
    remaining--
  }

  const out: LMEExample[] = []
  for (const category of categories) {
    const bucket = byCategory.get(category) ?? []
    const allocation = Math.min(allocations.get(category) ?? 0, bucket.length)
    if (allocation <= 0) continue
    out.push(...deterministicShuffle(bucket, args.seed).slice(0, allocation))
  }
  return out
}

const deterministicShuffle = <T>(
  input: readonly T[],
  seed: number,
): T[] => {
  const out = [...input]
  let state = BigInt.asUintN(64, BigInt(seed))
  for (let index = out.length - 1; index > 0; index--) {
    state =
      BigInt.asUintN(
        64,
        state * 6364136223846793005n + 1442695040888963407n,
      )
    const swapIndex = Number((state >> 33n) % BigInt(index + 1))
    const current = out[index]
    out[index] = out[swapIndex] as T
    out[swapIndex] = current as T
  }
  return out
}
