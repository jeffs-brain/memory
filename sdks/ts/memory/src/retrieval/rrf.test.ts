// SPDX-License-Identifier: Apache-2.0

/**
 * RRF fusion tests. Scope is the pure algorithm: the expected score
 * formula must hold for deterministic hand-computed inputs, and the
 * first-list metadata preservation must kick in when a later list
 * references the same doc with sparser fields.
 */

import { describe, expect, it } from 'vitest'
import { RRF_DEFAULT_K, reciprocalRankFusion } from './rrf.js'

const expectDefined = <T>(value: T | undefined, message: string): T => {
  if (value === undefined) throw new Error(message)
  return value
}

describe('reciprocalRankFusion', () => {
  it('produces scores matching the hand-computed RRF formula', () => {
    // bm25: a, b, c   vector: b, a, d
    // k = 60
    // a: 1/(60+1) + 1/(60+2) = 0.03278688524590164 + 0.016129032258064516 wait
    // Actually: a at bm25 rank 1 -> 1/61, a at vec rank 2 -> 1/62
    // b at bm25 rank 2 -> 1/62, b at vec rank 1 -> 1/61
    // c at bm25 rank 3 -> 1/63
    // d at vec rank 3 -> 1/63
    const bm25 = [
      { id: 'a', path: 'p/a.md', title: 'A', summary: 'sa' },
      { id: 'b', path: 'p/b.md', title: 'B', summary: 'sb' },
      { id: 'c', path: 'p/c.md', title: 'C', summary: 'sc' },
    ]
    const vec = [
      { id: 'b', path: 'p/b.md' },
      { id: 'a', path: 'p/a.md' },
      { id: 'd', path: 'p/d.md' },
    ]

    const fused = reciprocalRankFusion([bm25, vec], RRF_DEFAULT_K)

    expect(fused.map((r) => r.id)).toEqual(['a', 'b', 'c', 'd'])

    const expectA = 1 / 61 + 1 / 62
    const expectB = 1 / 62 + 1 / 61
    const expectC = 1 / 63
    const expectD = 1 / 63

    const byId = new Map(fused.map((r) => [r.id, r]))
    expect(byId.get('a')?.score).toBeCloseTo(expectA, 12)
    expect(byId.get('b')?.score).toBeCloseTo(expectB, 12)
    expect(byId.get('c')?.score).toBeCloseTo(expectC, 12)
    expect(byId.get('d')?.score).toBeCloseTo(expectD, 12)
  })

  it('preserves title and summary from the first list that provided them', () => {
    // `a` is in both lists but the vector list carries empty
    // title/summary (the typical shape of a vector-only hit). The
    // fused row must keep BM25's richer metadata.
    const bm25 = [
      { id: 'a', path: 'p/a.md', title: 'Rich title', summary: 'Rich summary' },
      { id: 'b', path: 'p/b.md', title: 'B title', summary: 'B summary' },
    ]
    const vec = [
      { id: 'a', path: 'p/a.md', title: '', summary: '' },
      { id: 'c', path: 'p/c.md', title: '', summary: '' },
    ]

    const fused = reciprocalRankFusion([bm25, vec], RRF_DEFAULT_K)
    const a = expectDefined(
      fused.find((r) => r.id === 'a'),
      'expected fused result for a',
    )
    expect(a.title).toBe('Rich title')
    expect(a.summary).toBe('Rich summary')

    // And a vector-only doc with no title survives with empty strings
    // rather than throwing.
    const c = expectDefined(
      fused.find((r) => r.id === 'c'),
      'expected fused result for c',
    )
    expect(c.title).toBe('')
    expect(c.summary).toBe('')
  })

  it('fills in missing metadata when a later list carries richer fields', () => {
    // Inverse of the previous case: first list had empty title, the
    // second list has the real one. The merge must promote the richer
    // field.
    const first = [{ id: 'a', path: 'p/a.md', title: '', summary: '' }]
    const second = [{ id: 'a', path: 'p/a.md', title: 'Late title', summary: 'Late summary' }]

    const fused = reciprocalRankFusion([first, second], RRF_DEFAULT_K)
    expect(fused[0]?.title).toBe('Late title')
    expect(fused[0]?.summary).toBe('Late summary')
  })

  it('breaks score ties by path ascending for deterministic output', () => {
    // Two docs at the same rank across a single list: identical scores,
    // deterministic path-asc ordering.
    const list = [
      { id: 'z', path: 'z/z.md' },
      { id: 'a', path: 'a/a.md' },
    ]
    const fused = reciprocalRankFusion([list], RRF_DEFAULT_K)
    expect(fused.map((r) => r.id)).toEqual(['z', 'a']) // z at rank 0 wins score
    // and when both are at rank 0 in separate lists, path wins.
    const l1 = [{ id: 'z', path: 'z/z.md' }]
    const l2 = [{ id: 'a', path: 'a/a.md' }]
    const fused2 = reciprocalRankFusion([l1, l2], RRF_DEFAULT_K)
    expect(fused2.map((r) => r.id)).toEqual(['a', 'z'])
  })
})
