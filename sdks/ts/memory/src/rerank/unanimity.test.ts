import { describe, expect, it } from 'vitest'
import { unanimityShortcut } from './llm-rerank.js'

const mk = (ids: readonly string[]) => ids.map((id) => ({ id }))

describe('unanimityShortcut', () => {
  it('returns the shortcut when the top-3 agree on two or more positions', () => {
    const bm25 = mk(['a', 'b', 'c', 'd'])
    const vec = mk(['a', 'b', 'x', 'y'])
    const out = unanimityShortcut(bm25, vec)
    expect(out?.ids).toEqual(['a', 'b', 'c'])
    expect(out?.agreements).toBe(2)
  })

  it('returns undefined when the agreement falls below the threshold', () => {
    const bm25 = mk(['a', 'b', 'c'])
    const vec = mk(['x', 'y', 'c'])
    expect(unanimityShortcut(bm25, vec)).toBeUndefined()
  })

  it('returns undefined when either ranking has fewer than three entries', () => {
    const bm25 = mk(['a', 'b'])
    const vec = mk(['a', 'b', 'c'])
    expect(unanimityShortcut(bm25, vec)).toBeUndefined()
    expect(unanimityShortcut(mk(['a', 'b', 'c']), mk(['a']))).toBeUndefined()
    expect(unanimityShortcut([], [])).toBeUndefined()
  })

  it('accepts a custom agreement threshold', () => {
    const bm25 = mk(['a', 'b', 'c'])
    const vec = mk(['a', 'b', 'c'])
    expect(unanimityShortcut(bm25, vec, 3)?.agreements).toBe(3)
  })
})
