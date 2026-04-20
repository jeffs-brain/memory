// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { compileToFTS } from './index.js'
import { parseQuery } from './parser.js'

describe('parseQuery', () => {
  it('returns an empty AST for empty input', () => {
    const ast = parseQuery('')
    expect(ast.tokens).toEqual([])
    expect(ast.hasOperators).toBe(false)
  })

  it('lowercases bare terms and drops stop words', () => {
    const ast = parseQuery('The Kubernetes Deployment')
    const texts = ast.tokens.map((t) => t.text)
    expect(texts).toEqual(['kubernetes', 'deployment'])
    expect(ast.hasOperators).toBe(false)
  })

  it('preserves quoted phrases exactly including stop words', () => {
    const ast = parseQuery('"the quick brown fox" jumps')
    expect(ast.tokens).toEqual([
      { kind: 'phrase', text: 'the quick brown fox' },
      { kind: 'term', text: 'jumps' },
    ])
    expect(ast.hasOperators).toBe(true)
  })

  it('honours explicit AND/OR/NOT operators without filtering stop words', () => {
    const ast = parseQuery('foo AND the OR bar NOT baz')
    expect(ast.hasOperators).toBe(true)
    expect(ast.tokens).toEqual([
      { kind: 'term', text: 'foo' },
      { kind: 'term', text: 'the', operator: 'AND' },
      { kind: 'term', text: 'bar', operator: 'OR' },
      { kind: 'term', text: 'baz', operator: 'NOT' },
    ])
  })

  it('supports trailing prefix wildcards', () => {
    const ast = parseQuery('kube* cluster')
    expect(ast.tokens).toEqual([
      { kind: 'prefix', text: 'kube' },
      { kind: 'term', text: 'cluster' },
    ])
  })

  it('normalises to NFC and strips zero-width punctuation', () => {
    // 'é' as composed (U+00E9) vs decomposed (e + U+0301). After NFC,
    // the decomposed form becomes the composed form.
    const composed = parseQuery('caf\u00e9')
    const decomposed = parseQuery('cafe\u0301')
    expect(decomposed.tokens).toEqual(composed.tokens)
    expect(decomposed.tokens[0]?.text.length).toBe(4)

    const zeroWidth = parseQuery('hello\u200bworld')
    expect(zeroWidth.tokens).toEqual([{ kind: 'term', text: 'helloworld' }])
  })

  it('strips short stop-word fragments on bare queries but keeps them inside quotes', () => {
    // `to`, `do` fall out via the 2-char check; `list` is in the EN
    // noise word list. Nothing survives the bare-word filter.
    const bare = parseQuery('to do list')
    expect(bare.tokens).toEqual([])

    const quoted = parseQuery('"to do list"')
    expect(quoted.tokens).toEqual([{ kind: 'phrase', text: 'to do list' }])
  })

  it('drops lowercase boolean words from bare natural-language queries', () => {
    const ast = parseQuery('not cocktail and snacks or dessert')
    expect(ast.tokens).toEqual([
      { kind: 'term', text: 'cocktail' },
      { kind: 'term', text: 'snacks' },
      { kind: 'term', text: 'dessert' },
    ])
  })

  it('drops recommendation filler words from long natural-language queries', () => {
    const ast = parseQuery(
      "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?",
    )
    expect(ast.tokens).toEqual([
      { kind: 'term', text: 'cocktail' },
      { kind: 'term', text: 'gettogether' },
      { kind: 'term', text: 'one' },
      { kind: 'term', text: 'choose' },
    ])
  })

  it('strips generic question scaffolding from memory-style queries', () => {
    const pickup = parseQuery(
      'How many items of clothing do I need to pick up or return from a store?',
    )
    expect(pickup.tokens).toEqual([
      { kind: 'term', text: 'clothing' },
      { kind: 'term', text: 'pick' },
      { kind: 'term', text: 'return' },
      { kind: 'term', text: 'store' },
    ])

    const movie = parseQuery('Can you recommend a show or movie for me to watch tonight?')
    expect(movie.tokens).toEqual([{ kind: 'term', text: 'movie' }])

    const battery = parseQuery(
      'I have been having trouble with the battery life on my phone lately. Any tips?',
    )
    expect(battery.tokens).toEqual([
      { kind: 'term', text: 'battery' },
      { kind: 'term', text: 'life' },
      { kind: 'term', text: 'phone' },
    ])

    const doctors = parseQuery('What types of doctors have I seen?')
    expect(doctors.tokens).toEqual([{ kind: 'term', text: 'doctors' }])
  })
})

describe('compileToFTS', () => {
  it('ORs bare terms by default', () => {
    const ast = parseQuery('kubernetes deployment')
    expect(compileToFTS(ast)).toBe('kubernetes OR deployment')
  })

  it('rewrites NOT into AND NOT after the first token', () => {
    const ast = parseQuery('foo AND bar NOT baz')
    expect(compileToFTS(ast)).toBe('foo AND bar AND NOT baz')
  })

  it('drops a leading NOT because FTS5 cannot start with NOT', () => {
    const ast = parseQuery('NOT alpha bravo')
    // Leading NOT is stripped, bravo defaults to OR.
    expect(compileToFTS(ast)).toBe('alpha OR bravo')
  })

  it('wraps phrases and preserves prefix wildcards', () => {
    const ast = parseQuery('"hello world" kube*')
    expect(compileToFTS(ast)).toBe('"hello world" OR kube*')
  })
})
