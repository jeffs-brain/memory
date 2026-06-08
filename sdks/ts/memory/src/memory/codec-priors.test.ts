// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  CODEC_PRIORS_MAX_ITEMS_PER_LIST,
  CODEC_PRIORS_MAX_ITEM_LENGTH,
  CodecPriorsError,
  applyCodecPriors,
  buildCodecPriorsBlock,
} from './codec-priors.js'
import { EXTRACTION_SYSTEM_PROMPT } from './prompts.js'
import type { CodecPriors } from './types.js'

describe('buildCodecPriorsBlock — positive', () => {
  it('renders the canonical entities, relations, and domain terms', () => {
    const priors: CodecPriors = {
      entities: ['Person', 'Organisation', 'Product'],
      relations: ['worksAt', 'owns', 'dependsOn'],
      domainTerms: ['sprint', 'deployment', 'incident'],
    }
    const block = buildCodecPriorsBlock(priors)
    expect(block).toContain('## Project codec priors')
    expect(block).toContain('### Entities')
    expect(block).toContain('- Person')
    expect(block).toContain('### Relations')
    expect(block).toContain('- worksAt')
    expect(block).toContain('### Domain terms')
    expect(block).toContain('- sprint')
  })

  it('applyCodecPriors appends the block to the base prompt', () => {
    const priors: CodecPriors = { entities: ['Person'] }
    const composed = applyCodecPriors(EXTRACTION_SYSTEM_PROMPT, priors)
    expect(composed.startsWith(EXTRACTION_SYSTEM_PROMPT)).toBe(true)
    expect(composed).toContain('- Person')
    expect(composed.length).toBeGreaterThan(EXTRACTION_SYSTEM_PROMPT.length)
  })

  it('renders only the lists that are present', () => {
    const block = buildCodecPriorsBlock({ entities: ['Person'] })
    expect(block).toContain('### Entities')
    expect(block).not.toContain('### Relations')
    expect(block).not.toContain('### Domain terms')
  })
})

describe('buildCodecPriorsBlock — negative / backward compatibility', () => {
  it('returns empty string when priors are undefined', () => {
    expect(buildCodecPriorsBlock(undefined)).toBe('')
  })

  it('returns empty string for an empty object', () => {
    expect(buildCodecPriorsBlock({})).toBe('')
  })

  it('returns empty string when every list is empty', () => {
    expect(buildCodecPriorsBlock({ entities: [], relations: [], domainTerms: [] })).toBe('')
  })

  it('leaves the base prompt byte-identical when priors are absent or empty', () => {
    expect(applyCodecPriors(EXTRACTION_SYSTEM_PROMPT, undefined)).toBe(EXTRACTION_SYSTEM_PROMPT)
    expect(applyCodecPriors(EXTRACTION_SYSTEM_PROMPT, {})).toBe(EXTRACTION_SYSTEM_PROMPT)
    expect(applyCodecPriors(EXTRACTION_SYSTEM_PROMPT, { entities: ['   '] })).toBe(
      EXTRACTION_SYSTEM_PROMPT,
    )
  })

  it('throws a typed error for a non-array list', () => {
    // Simulate malformed runtime input that bypasses the type system.
    const malformed = { entities: 'Person' } as unknown as CodecPriors
    expect(() => buildCodecPriorsBlock(malformed)).toThrow(CodecPriorsError)
  })

  it('throws a typed error for a non-string item', () => {
    const malformed = { entities: ['Person', 42] } as unknown as CodecPriors
    expect(() => buildCodecPriorsBlock(malformed)).toThrow(CodecPriorsError)
  })

  it('throws a typed error for an item containing a line break', () => {
    expect(() => buildCodecPriorsBlock({ entities: ['Person\nInjected heading'] })).toThrow(
      CodecPriorsError,
    )
  })
})

// GO ↔ TS PARITY: this golden string is asserted byte-for-byte in the Go
// test TestBuildCodecPriorsBlock_GoldenParity (go/memory/codec_priors_test.go).
// Any change here MUST be mirrored there, and vice-versa.
const GOLDEN_PRIORS: CodecPriors = {
  entities: ['Person', 'person', 'Organisation'],
  relations: ['worksAt'],
  domainTerms: ['sprint', 'deployment'],
}
const GOLDEN_BLOCK = [
  '',
  '## Project codec priors',
  'The following are the project’s canonical entities, relations, and domain terms. Treat them as SOFT hints: when a saved fact references one of these, prefer the canonical name below. Do NOT invent facts to match them, and do NOT discard durable knowledge that falls outside this list.',
  '',
  '### Entities',
  '- Person',
  '- Organisation',
  '### Relations',
  '- worksAt',
  '### Domain terms',
  '- sprint',
  '- deployment',
].join('\n')

describe('buildCodecPriorsBlock — Go/TS parity', () => {
  it('renders the golden block byte-for-byte', () => {
    expect(buildCodecPriorsBlock(GOLDEN_PRIORS)).toBe(GOLDEN_BLOCK)
  })
})

describe('buildCodecPriorsBlock — edge', () => {
  it('deduplicates case-insensitively, first occurrence wins', () => {
    const block = buildCodecPriorsBlock({ entities: ['Person', 'person', 'PERSON', 'Org'] })
    const lines = block.split('\n').filter((line) => line.startsWith('- '))
    expect(lines).toEqual(['- Person', '- Org'])
  })

  it('caps the number of items per list', () => {
    const many = Array.from(
      { length: CODEC_PRIORS_MAX_ITEMS_PER_LIST + 50 },
      (_v, i) => `Entity${i}`,
    )
    const block = buildCodecPriorsBlock({ entities: many })
    const lines = block.split('\n').filter((line) => line.startsWith('- '))
    expect(lines.length).toBe(CODEC_PRIORS_MAX_ITEMS_PER_LIST)
  })

  it('truncates an oversized item to the per-item length cap', () => {
    const long = 'A'.repeat(CODEC_PRIORS_MAX_ITEM_LENGTH + 40)
    const block = buildCodecPriorsBlock({ entities: [long] })
    const line = block.split('\n').find((l) => l.startsWith('- ')) ?? ''
    expect(line).toBe(`- ${'A'.repeat(CODEC_PRIORS_MAX_ITEM_LENGTH)}`)
  })

  it('bounds total prompt growth even with pathological input', () => {
    const huge = Array.from({ length: 500 }, (_v, i) => 'X'.repeat(80) + i)
    const block = buildCodecPriorsBlock({
      entities: huge,
      relations: huge,
      domainTerms: huge,
    })
    // Body budget plus fixed framing; never unbounded.
    expect(block.length).toBeLessThan(6000)
  })

  it('drops blank and whitespace-only items', () => {
    const block = buildCodecPriorsBlock({ entities: ['', '   ', 'Person'] })
    const lines = block.split('\n').filter((line) => line.startsWith('- '))
    expect(lines).toEqual(['- Person'])
  })
})
