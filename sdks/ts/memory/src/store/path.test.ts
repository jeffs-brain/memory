// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { ErrInvalidPath } from './errors.js'
import { isGenerated, joinPath, matchGlob, toPath, validatePath } from './path.js'

describe('validatePath', () => {
  const validCases = ['memory/a.md', 'wiki/topic/article.md', 'kb-schema.md']
  for (const p of validCases) {
    it(`accepts ${p}`, () => {
      expect(() => validatePath(p)).not.toThrow()
    })
  }

  const invalidCases: Array<[string, string]> = [
    ['', 'empty'],
    ['..', 'traversal'],
    ['a/../b', 'traversal in middle'],
    ['/absolute', 'leading slash'],
    ['memory/', 'trailing slash'],
    ['a\\b', 'backslash'],
    ['a//b', 'empty segment'],
    ['./foo', 'dot segment'],
    ['a\0b', 'null byte'],
    ['a/./b', 'interior dot'],
  ]

  for (const [input, why] of invalidCases) {
    it(`rejects ${JSON.stringify(input)} (${why})`, () => {
      expect(() => validatePath(input)).toThrow(ErrInvalidPath)
    })
  }
})

describe('toPath', () => {
  it('returns the branded value on success', () => {
    const p = toPath('memory/a.md')
    expect(p).toBe('memory/a.md')
  })

  it('throws on invalid input', () => {
    expect(() => toPath('../escape')).toThrow(ErrInvalidPath)
  })
})

describe('isGenerated', () => {
  it('detects underscore-prefixed base names', () => {
    expect(isGenerated('wiki/_index.md')).toBe(true)
    expect(isGenerated('wiki/topic/_log.md')).toBe(true)
    expect(isGenerated('wiki/article.md')).toBe(false)
  })
})

describe('joinPath', () => {
  it('joins and validates the result', () => {
    expect(joinPath('memory', 'global', 'a.md')).toBe('memory/global/a.md')
  })

  it('drops empty segments', () => {
    expect(joinPath('memory', '', 'a.md')).toBe('memory/a.md')
  })

  it('rejects a ".." escape that breaks out of the root', () => {
    expect(() => joinPath('..', 'escape')).toThrow(ErrInvalidPath)
  })
})

describe('matchGlob', () => {
  it('matches *.md', () => {
    expect(matchGlob('*.md', 'a.md')).toBe(true)
    expect(matchGlob('*.md', 'a.txt')).toBe(false)
  })

  it('matches ?', () => {
    expect(matchGlob('a?.md', 'ab.md')).toBe(true)
    expect(matchGlob('a?.md', 'abc.md')).toBe(false)
  })

  it('matches [abc]', () => {
    expect(matchGlob('[ab].md', 'a.md')).toBe(true)
    expect(matchGlob('[ab].md', 'c.md')).toBe(false)
  })

  it('matches ranges [a-c]', () => {
    expect(matchGlob('[a-c].md', 'b.md')).toBe(true)
    expect(matchGlob('[a-c].md', 'd.md')).toBe(false)
  })
})
