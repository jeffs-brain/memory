// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { extractDocumentMetadata } from './metadata.js'

describe('extractDocumentMetadata', () => {
  it('returns an empty object when no frontmatter fence is present', () => {
    expect(extractDocumentMetadata('just a body, no frontmatter')).toEqual({})
  })

  it('returns an empty object for an unterminated fence', () => {
    expect(extractDocumentMetadata('---\ntitle: x\nbody without close')).toEqual({})
  })

  it('returns an empty object for an empty string', () => {
    expect(extractDocumentMetadata('')).toEqual({})
  })

  it('preserves ontology_type (snake_case) that the typed parser would drop', () => {
    const content = '---\nontology_type: customer\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ ontology_type: 'customer' })
  })

  it('preserves ontologyType (camelCase)', () => {
    const content = '---\nontologyType: customer\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ ontologyType: 'customer' })
  })

  it('preserves session_id and supersedes scalar edge inputs', () => {
    const content = '---\nsession_id: s1\nsupersedes: legacy-note\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({
      session_id: 's1',
      supersedes: 'legacy-note',
    })
  })

  it('parses inline-list tags into a string array', () => {
    const content = '---\ntags: [alpha, beta]\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ tags: ['alpha', 'beta'] })
  })

  it('keeps a bare comma-separated scalar as a string (no key-name list inference)', () => {
    // A generic extractor cannot know which arbitrary key is a list, so only
    // explicit `[...]` or block `- ` syntax yields an array. This is safe for
    // the edge queries: computeSharedTagEdges treats a non-array `tags` as `[]`.
    const content = '---\ntags: alpha, beta\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ tags: 'alpha, beta' })
  })

  it('parses block-list tags into a string array', () => {
    const content = '---\ntags:\n  - alpha\n  - beta\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ tags: ['alpha', 'beta'] })
  })

  it('records an empty array for an empty block list', () => {
    const content = '---\ntags:\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ tags: [] })
  })

  it('coerces boolean-ish scalars', () => {
    const content = '---\narchived: true\ndraft: no\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ archived: true, draft: false })
  })

  it('strips matching quotes from scalar values', () => {
    const content = '---\ntitle: "Hello, World"\nsummary: \'a: b\'\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({
      title: 'Hello, World',
      summary: 'a: b',
    })
  })

  it('keeps all keys together for a realistic typed-document frontmatter', () => {
    const content = [
      '---',
      'title: Acme Corp',
      'summary: A customer',
      'tags: [crm, account]',
      'ontology_type: customer',
      'session_id: sess-42',
      '---',
      '',
      'Acme is a customer.',
    ].join('\n')
    expect(extractDocumentMetadata(content)).toEqual({
      title: 'Acme Corp',
      summary: 'A customer',
      tags: ['crm', 'account'],
      ontology_type: 'customer',
      session_id: 'sess-42',
    })
  })

  it('ignores blank lines and non key-value lines inside the block', () => {
    const content = '---\nontology_type: customer\n\n# a comment line without colon\n---\n\nbody'
    expect(extractDocumentMetadata(content)).toEqual({ ontology_type: 'customer' })
  })
})
