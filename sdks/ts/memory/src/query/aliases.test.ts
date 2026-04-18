import { describe, expect, it } from 'vitest'
import { expand } from './aliases.js'
import { parseQuery } from './parser.js'

describe('expand (alias expansion)', () => {
  it('is a no-op when the alias table is empty', () => {
    const ast = parseQuery('kubernetes deployment')
    const out = expand(ast, new Map())
    expect(out.tokens).toEqual(ast.tokens)
    expect(out).toBe(ast)
  })

  it('expands a single-alternative alias into one token (pass-through)', () => {
    const ast = parseQuery('bosch')
    const table = new Map<string, readonly string[]>([['bosch', ['bosch']]])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([{ kind: 'term', text: 'bosch' }])
  })

  it('expands a single-alternative alias into a replacement token', () => {
    const ast = parseQuery('dude')
    const table = new Map<string, readonly string[]>([['dude', ['oude']]])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([{ kind: 'term', text: 'oude' }])
  })

  it('expands multi-target aliases into phrase tokens for hyphenated values', () => {
    const ast = parseQuery('aware production')
    const table = new Map<string, readonly string[]>([
      ['aware', ['royal-aware', 'a-ware', 'aware']],
    ])
    const out = expand(ast, table)
    const surface = out.tokens.map((t) => `${t.kind}:${t.text}`).sort()
    expect(surface).toContain('phrase:royal aware')
    expect(surface).toContain('phrase:a ware')
    expect(surface).toContain('term:aware')
    expect(surface).toContain('term:production')
  })

  it('matches aliases case-insensitively', () => {
    const ast = parseQuery('BOSCH')
    const table = new Map<string, readonly string[]>([['bosch', ['bosch', 'robert-bosch']]])
    const out = expand(ast, table)
    const surface = out.tokens.map((t) => `${t.kind}:${t.text}`).sort()
    expect(surface).toEqual(['phrase:robert bosch', 'term:bosch'])
  })

  it('carries the leading operator onto the first expanded token only', () => {
    const ast = parseQuery('foo AND bar')
    const table = new Map<string, readonly string[]>([['bar', ['barone', 'bartwo']]])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([
      { kind: 'term', text: 'foo' },
      { kind: 'term', text: 'barone', operator: 'AND' },
      { kind: 'term', text: 'bartwo' },
    ])
  })

  it('leaves phrase and prefix tokens untouched', () => {
    const ast = parseQuery('"bosch" kube*')
    const table = new Map<string, readonly string[]>([
      ['bosch', ['bosch', 'robert-bosch']],
      ['kube', ['k8s', 'kubernetes']],
    ])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([
      { kind: 'phrase', text: 'bosch' },
      { kind: 'prefix', text: 'kube' },
    ])
  })
})
