// SPDX-License-Identifier: Apache-2.0

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
      'alpha body.',
      '',
      '# Beta',
      'beta body.',
      '',
      '# Gamma',
      'gamma body.',
    ].join('\n')
    const chunks = chunkMarkdown(text)
    expect(chunks).toHaveLength(3)
    expect(chunks[0]?.headingPath).toEqual(['Alpha'])
    expect(chunks[1]?.headingPath).toEqual(['Beta'])
    expect(chunks[2]?.headingPath).toEqual(['Gamma'])
    expect(chunks[0]?.content).toContain('Alpha')
    expect(chunks[1]?.content).toContain('beta body')
  })

  it('preserves nested heading paths', () => {
    const text = ['# A', 'intro', '', '## A1', 'sub body', '', '## A2', 'sub body 2'].join('\n')
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
    const text = '# A\nfoo\n\n# B\nbar\n\n# C\nbaz'
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
