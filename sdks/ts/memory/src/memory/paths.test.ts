import { describe, expect, it } from 'vitest'
import { scopePrefix } from './paths.js'

describe('scopePrefix', () => {
  it('rejects actor ids that are not a single path segment', () => {
    expect(() => scopePrefix('project', '../global')).toThrow(/path segment/)
    expect(() => scopePrefix('agent', '../../sessions')).toThrow(/path segment/)
  })
})
