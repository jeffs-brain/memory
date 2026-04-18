// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { queryTokens, strongestTerm } from './retry.js'

describe('retry token normalisation', () => {
  it('prefers the domain term over recommendation filler words', () => {
    const question =
      "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?"

    expect(strongestTerm(question)).toBe('cocktail')
  })

  it('drops punctuation noise and filler tokens from trigram inputs', () => {
    const question =
      "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?"

    expect(queryTokens(question)).toContain('cocktail')
    expect(queryTokens(question)).not.toContain('together')
    expect(queryTokens(question)).not.toContain('suggestions')
    expect(queryTokens(question)).not.toContain('ive')
  })

  it('drops generic question scaffolding from recall-style prompts', () => {
    expect(
      queryTokens(
        'Can you recommend a show or movie for me to watch tonight?',
      ),
    ).toEqual(['movie'])

    expect(
      queryTokens(
        'How many items of clothing do I need to pick up or return from a store?',
      ),
    ).toEqual(['clothing', 'pick', 'return', 'store'])

    expect(queryTokens('What types of doctors have I seen?')).toEqual([
      'doctors',
    ])
  })
})
