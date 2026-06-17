// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  deriveOkfTypeFromPath,
  extractOkfLinks,
  normaliseOkfDocument,
  normaliseOkfLinkTarget,
  parseOkfDocument,
  validateOkfDocument,
} from './index.js'

describe('OKF helpers', () => {
  it('parses frontmatter and normalises OKF fields', () => {
    const doc = [
      '---',
      'type: Playbook',
      'title: Data freshness alert',
      'description: Steps to triage a data freshness alert.',
      'resource: https://example.com/runbook',
      'tags: [oncall, incident]',
      'timestamp: 2026-05-28T14:30:00Z',
      'custom: preserved',
      '---',
      '',
      '# Body',
      '',
      'Use the dashboard.',
    ].join('\n')

    const parsed = parseOkfDocument(doc)
    expect(parsed.present).toBe(true)
    expect(parsed.frontmatter.custom).toBe('preserved')

    expect(normaliseOkfDocument(doc, { path: 'playbooks/freshness.md' })).toMatchObject({
      conceptId: 'playbooks/freshness',
      type: 'Playbook',
      title: 'Data freshness alert',
      description: 'Steps to triage a data freshness alert.',
      resource: 'https://example.com/runbook',
      tags: ['oncall', 'incident'],
      timestamp: '2026-05-28T14:30:00Z',
    })
  })

  it('maps legacy Jeff fields without requiring a rewrite', () => {
    const doc = [
      '---',
      'summary: Legacy one-line summary',
      'modified: 2026-06-01',
      'tags:',
      '  - memory',
      '---',
      '',
      '# Legacy Article Heading',
      '',
      'Body.',
    ].join('\n')

    expect(
      normaliseOkfDocument(doc, {
        path: 'wiki/projects/jeffs-brain-memory.md',
        defaultType: deriveOkfTypeFromPath('wiki/projects/jeffs-brain-memory.md'),
      }),
    ).toMatchObject({
      type: 'Article',
      title: 'Legacy Article Heading',
      description: 'Legacy one-line summary',
      timestamp: '2026-06-01',
      tags: ['memory'],
    })
  })

  it('reports required OKF conformance issues', () => {
    expect(validateOkfDocument('plain body', { path: 'wiki/foo.md' })).toMatchObject({
      ok: false,
      issues: [{ code: 'missing_frontmatter' }],
    })
    expect(
      validateOkfDocument('---\ntitle: Missing type\n---\n\nBody', { path: 'wiki/foo.md' }),
    ).toMatchObject({
      ok: false,
      issues: [{ code: 'missing_type' }],
    })
    expect(
      validateOkfDocument('---\nokf_version: "0.1"\n---\n\n# Index', { path: 'index.md' }),
    ).toMatchObject({
      ok: true,
      issues: [],
    })
  })

  it('normalises bundle-relative and relative links', () => {
    expect(normaliseOkfLinkTarget('/tables/customers.md', 'tables/orders.md')).toBe(
      'tables/customers.md',
    )
    expect(normaliseOkfLinkTarget('./customers.md', 'tables/orders.md')).toBe('tables/customers.md')
    expect(normaliseOkfLinkTarget('../datasets/sales.md', 'tables/orders.md')).toBe(
      'datasets/sales.md',
    )
    expect(normaliseOkfLinkTarget('https://example.com/doc', 'tables/orders.md')).toBeUndefined()
  })

  it('extracts standard markdown links and wikilinks', () => {
    const links = extractOkfLinks(
      'See [customers](/tables/customers.md), [neighbour](./other.md), ![image](./image.png), and [[tables/orders|orders]].',
      'tables/current.md',
    )

    expect(links.map((link) => [link.kind, link.target, link.resolvedTarget])).toEqual([
      ['wikilink', 'tables/orders', 'tables/orders'],
      ['markdown', '/tables/customers.md', 'tables/customers.md'],
      ['markdown', './other.md', 'tables/other.md'],
    ])
  })
})
