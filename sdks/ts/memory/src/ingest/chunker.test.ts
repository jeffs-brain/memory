// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  chunkAuto,
  chunkMarkdown,
  chunkPlainText,
  countTokens,
  looksLikeMarkdown,
} from './chunker.js'

describe('countTokens', () => {
  it('is monotonic in input length', () => {
    const a = countTokens('hello')
    const b = countTokens('hello world')
    const c = countTokens('hello world, how are you doing today?')
    expect(a).toBeLessThanOrEqual(b)
    expect(b).toBeLessThanOrEqual(c)
  })

  it('is zero for the empty string', () => {
    expect(countTokens('')).toBe(0)
  })
})

describe('chunkMarkdown', () => {
  it('splits at heading boundaries', () => {
    const text = [
      '# Alpha',
      'Alpha section body with enough words to exceed the minimum token threshold for standalone chunks in the output.',
      '',
      '# Beta',
      'Beta section body with enough words to exceed the minimum token threshold for standalone chunks in the output.',
      '',
      '# Gamma',
      'Gamma section body with enough words to exceed the minimum token threshold for standalone chunks in the output.',
    ].join('\n')
    const chunks = chunkMarkdown(text)
    expect(chunks).toHaveLength(3)
    expect(chunks[0]?.headingPath).toEqual(['Alpha'])
    expect(chunks[1]?.headingPath).toEqual(['Beta'])
    expect(chunks[2]?.headingPath).toEqual(['Gamma'])
    expect(chunks[0]?.content).toContain('Alpha')
    expect(chunks[1]?.content).toContain('Beta section body')
  })

  it('preserves nested heading paths', () => {
    const text = [
      '# A',
      'Introductory paragraph with enough content to exceed the minimum token threshold for standalone chunks in the output.',
      '',
      '## A1',
      'Sub body paragraph one with enough content to exceed the minimum token threshold for standalone chunks in the output.',
      '',
      '## A2',
      'Sub body paragraph two with enough content to exceed the minimum token threshold for standalone chunks in the output.',
    ].join('\n')
    const chunks = chunkMarkdown(text)
    expect(chunks.length).toBeGreaterThanOrEqual(3)
    expect(chunks[0]?.headingPath).toEqual(['A'])
    expect(chunks[1]?.headingPath).toEqual(['A', 'A1'])
    expect(chunks[2]?.headingPath).toEqual(['A', 'A2'])
  })

  it('breaks oversized sections into overlapping windows', () => {
    // Several paragraphs, each small enough on its own so the packer
    // batches them into windows rather than hard-splitting a single para.
    const paragraphs = Array.from(
      { length: 20 },
      (_, i) => `paragraph-${i} ${'token '.repeat(12).trim()}`,
    )
    const text = `# Big\n\n${paragraphs.join('\n\n')}`
    const chunks = chunkMarkdown(text, { maxTokens: 48, overlapTokens: 24 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) {
      expect(chunk.headingPath).toEqual(['Big'])
    }
    // Overlap: at least one paragraph from chunk N appears at the start of
    // chunk N+1.
    const first = chunks[0]?.content ?? ''
    const second = chunks[1]?.content ?? ''
    const firstParas = first.split(/\n{2,}/)
    const lastParaOfFirst = firstParas.at(-1) ?? ''
    expect(second).toContain(lastParaOfFirst)
  })

  it('returns empty list for blank input', () => {
    expect(chunkMarkdown('   \n\n  ')).toHaveLength(0)
  })

  it('assigns sequential ordinals', () => {
    const text = [
      '# A',
      'First section with enough words to exceed the minimum token threshold for standalone chunks in the final output of the chunker pipeline.',
      '',
      '# B',
      'Second section with enough words to exceed the minimum token threshold for standalone chunks in the final output of the chunker pipeline.',
      '',
      '# C',
      'Third section with enough words to exceed the minimum token threshold for standalone chunks in the final output of the chunker pipeline.',
    ].join('\n')
    const chunks = chunkMarkdown(text)
    expect(chunks.map((c) => c.ordinal)).toEqual([0, 1, 2])
  })
})

describe('chunkPlainText', () => {
  it('splits into windows when the text exceeds the token budget', () => {
    const text = 'x'.repeat(2048)
    const chunks = chunkPlainText(text, { maxTokens: 64, overlapTokens: 8 })
    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks[0]?.headingPath).toEqual([])
  })

  it('returns a single chunk for short text', () => {
    const chunks = chunkPlainText('hello world')
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.content).toBe('hello world')
  })
})

describe('chunkAuto + looksLikeMarkdown', () => {
  it('routes to markdown when headings are present', () => {
    const chunks = chunkAuto('# Title\n\nbody text')
    expect(chunks[0]?.headingPath).toEqual(['Title'])
  })

  it('routes to plain when no markdown signals', () => {
    const chunks = chunkAuto('plain text without any markers')
    expect(chunks[0]?.headingPath).toEqual([])
  })

  it('detects lists and blockquotes as markdown', () => {
    expect(looksLikeMarkdown('- item\n- item')).toBe(true)
    expect(looksLikeMarkdown('> quoted\n> more')).toBe(true)
    expect(looksLikeMarkdown('1. numbered\n2. list')).toBe(true)
    expect(looksLikeMarkdown('plain prose')).toBe(false)
  })
})

describe('chunkMarkdown minTokens', () => {
  it('merges chunks below minTokens threshold', () => {
    const text = [
      '# Big Section',
      '',
      'This section has enough content to remain standalone because it is well above the minimum token threshold configured for this test.',
      '',
      '# Tiny',
      '',
      'Hi.',
    ].join('\n')
    // "Hi." is ~1 token, well below minTokens=30.
    const chunks = chunkMarkdown(text, { minTokens: 30 })
    // The tiny chunk should be merged into the big section.
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.content).toContain('Hi.')
  })

  it('uses spec defaults when no options provided', () => {
    // With no options, minTokens defaults to 30. A section with ~2 tokens
    // should be merged into its predecessor.
    const text = [
      '# Main',
      '',
      'This main section contains enough words to be well above the minimum token threshold when using spec defaults.',
      '',
      '# Short',
      '',
      'Ok.',
    ].join('\n')
    const chunks = chunkMarkdown(text)
    // "Ok." (~1 token) is below the default minTokens=30 and should be merged.
    expect(chunks).toHaveLength(1)
    expect(chunks[0]?.content).toContain('Ok.')
    expect(chunks[0]?.content).toContain('Main')
  })

  it('keeps chunks at or above minTokens threshold', () => {
    const longEnough = 'word '.repeat(40).trim()
    const text = [
      '# Section A',
      '',
      longEnough,
      '',
      '# Section B',
      '',
      longEnough,
    ].join('\n')
    const chunks = chunkMarkdown(text, { minTokens: 10 })
    // Both sections are well above 10 tokens, so both survive.
    expect(chunks).toHaveLength(2)
  })
})

describe('conformance', () => {
  const fixturesDir = join(__dirname, '..', '..', '..', '..', '..', 'spec', 'fixtures', 'ingestion')

  it('matches conformance fixture chunk boundaries', () => {
    const md = readFileSync(join(fixturesDir, 'chunking-conformance.md'), 'utf-8')
    const expectedRaw = readFileSync(join(fixturesDir, 'chunking-expected.json'), 'utf-8')
    const expected = JSON.parse(expectedRaw) as {
      config: { maxTokens: number; overlapTokens: number; minTokens: number }
      chunkCount: number
      chunks: Array<{
        ordinal: number
        heading: string
        minTokens: number
        maxTokens: number
        contentPrefix: string
      }>
    }

    const chunks = chunkMarkdown(md, {
      maxTokens: expected.config.maxTokens,
      overlapTokens: expected.config.overlapTokens,
      minTokens: expected.config.minTokens,
    })

    expect(chunks).toHaveLength(expected.chunkCount)

    for (const ec of expected.chunks) {
      const chunk = chunks[ec.ordinal]
      expect(chunk).toBeDefined()
      if (chunk === undefined) continue

      expect(chunk.ordinal).toBe(ec.ordinal)

      if (ec.minTokens > 0) {
        expect(chunk.tokens).toBeGreaterThanOrEqual(ec.minTokens)
      }
      if (ec.maxTokens > 0) {
        expect(chunk.tokens).toBeLessThanOrEqual(ec.maxTokens)
      }
      if (ec.contentPrefix !== '') {
        expect(chunk.content).toContain(ec.contentPrefix)
      }
    }
  })
})
