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
    const paragraphs = Array.from(
      { length: 20 },
      (_, index) => `paragraph-${index} ${'token '.repeat(12).trim()}`,
    )
    const text = `# Big\n\n${paragraphs.join('\n\n')}`
    const chunks = chunkMarkdown(text, { maxTokens: 48, overlapTokens: 24 })
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) {
      expect(chunk.headingPath).toEqual(['Big'])
    }
    const first = chunks[0]?.content ?? ''
    const second = chunks[1]?.content ?? ''
    const firstParagraphs = first.split(/\n{2,}/)
    const lastParagraph = firstParagraphs.at(-1) ?? ''
    expect(second).toContain(lastParagraph)
  })
})

describe('chunkPlainText', () => {
  it('splits into windows when the text exceeds the token budget', () => {
    const text = 'x'.repeat(2048)
    const chunks = chunkPlainText(text, { maxTokens: 64, overlapTokens: 8 })
    expect(chunks.length).toBeGreaterThan(1)
    expect(chunks[0]?.headingPath).toEqual([])
  })
})

describe('chunkAuto + looksLikeMarkdown', () => {
  it('routes to markdown when headings are present', () => {
    const chunks = chunkAuto('# Title\n\nbody text')
    expect(chunks[0]?.headingPath).toEqual(['Title'])
  })

  it('detects markdown markers and ignores plain prose', () => {
    expect(looksLikeMarkdown('- item\n- item')).toBe(true)
    expect(looksLikeMarkdown('> quoted\n> more')).toBe(true)
    expect(looksLikeMarkdown('1. numbered\n2. list')).toBe(true)
    expect(looksLikeMarkdown('plain prose')).toBe(false)
  })
})
