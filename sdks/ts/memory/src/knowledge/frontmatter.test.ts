// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { parseFrontmatter, serialiseFrontmatter } from './frontmatter.js'

describe('frontmatter', () => {
  it('parses a YAML block with tags and sources', () => {
    const raw = [
      '---',
      'title: My Article',
      'summary: A thing about a thing',
      'tags:',
      '  - alpha',
      '  - beta',
      'sources:',
      '  - ingested/abc.md',
      'created: 2026-04-17T00:00:00Z',
      '---',
      '',
      'Body text here.',
      '',
    ].join('\n')
    const { frontmatter, body, present } = parseFrontmatter(raw)
    expect(present).toBe(true)
    expect(frontmatter.title).toBe('My Article')
    expect(frontmatter.summary).toBe('A thing about a thing')
    expect(frontmatter.tags).toEqual(['alpha', 'beta'])
    expect(frontmatter.sources).toEqual(['ingested/abc.md'])
    expect(frontmatter.created).toBe('2026-04-17T00:00:00Z')
    expect(body).toBe('Body text here.')
  })

  it('round-trips via serialiseFrontmatter', () => {
    const original = {
      title: 'Round Trip',
      summary: 'Back and forth',
      tags: ['one', 'two'],
      sources: ['ingested/deadbeef.md'],
      created: '2026-04-17T00:00:00Z',
    }
    const serialised = serialiseFrontmatter(original, 'Hello world.')
    const { frontmatter, body } = parseFrontmatter(serialised)
    expect(frontmatter.title).toBe(original.title)
    expect(frontmatter.summary).toBe(original.summary)
    expect(frontmatter.tags).toEqual(original.tags)
    expect(frontmatter.sources).toEqual(original.sources)
    expect(body).toBe('Hello world.')
  })

  it('returns present=false when no fence is present', () => {
    const { present, body } = parseFrontmatter('just a note')
    expect(present).toBe(false)
    expect(body).toBe('just a note')
  })
})
