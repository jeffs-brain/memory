// SPDX-License-Identifier: Apache-2.0

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
    const ast = parseQuery('zenco')
    const table = new Map<string, readonly string[]>([['zenco', ['zenco']]])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([{ kind: 'term', text: 'zenco' }])
  })

  it('expands a single-alternative alias into a replacement token', () => {
    const ast = parseQuery('hill')
    const table = new Map<string, readonly string[]>([['hill', ['mill']]])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([{ kind: 'term', text: 'mill' }])
  })

  it('expands multi-target aliases into phrase tokens for hyphenated values', () => {
    const ast = parseQuery('evolt production')
    const table = new Map<string, readonly string[]>([
      ['evolt', ['nova-evolt', 'e-volt', 'evolt']],
    ])
    const out = expand(ast, table)
    const surface = out.tokens.map((t) => `${t.kind}:${t.text}`).sort()
    expect(surface).toContain('phrase:nova evolt')
    expect(surface).toContain('phrase:e volt')
    expect(surface).toContain('term:evolt')
    expect(surface).toContain('term:production')
  })

  it('matches aliases case-insensitively', () => {
    const ast = parseQuery('ZENCO')
    const table = new Map<string, readonly string[]>([['zenco', ['zenco', 'zenco-group']]])
    const out = expand(ast, table)
    const surface = out.tokens.map((t) => `${t.kind}:${t.text}`).sort()
    expect(surface).toEqual(['phrase:zenco group', 'term:zenco'])
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
    const ast = parseQuery('"zenco" kube*')
    const table = new Map<string, readonly string[]>([
      ['zenco', ['zenco', 'zenco-group']],
      ['kube', ['k8s', 'kubernetes']],
    ])
    const out = expand(ast, table)
    expect(out.tokens).toEqual([
      { kind: 'phrase', text: 'zenco' },
      { kind: 'prefix', text: 'kube' },
    ])
  })
})
