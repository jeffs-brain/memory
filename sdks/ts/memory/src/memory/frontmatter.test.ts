// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { type Frontmatter, buildFrontmatter, parseFrontmatter } from './frontmatter.js'

const sample = (): Frontmatter => ({
  name: 'Run the test suite',
  description: 'Always run the project suite, never the package manager default.',
  type: 'project',
  scope: 'global',
  created: '2026-01-01T00:00:00.000Z',
  modified: '2026-02-01T00:00:00.000Z',
  source: 'session',
  session_id: 'sess-123',
  tags: ['testing', 'ci'],
  extra: {},
})

describe('buildFrontmatter — default profile', () => {
  it('is unchanged by an explicit default profile (byte-for-byte)', () => {
    const fm = sample()
    expect(buildFrontmatter(fm, { profile: 'default' })).toBe(buildFrontmatter(fm))
  })

  it('emits the native key set, not OKF keys', () => {
    const out = buildFrontmatter(sample())
    expect(out).toContain('name: Run the test suite')
    expect(out).toContain('modified: 2026-02-01T00:00:00.000Z')
    expect(out).toContain('created: 2026-01-01T00:00:00.000Z')
    expect(out).not.toContain('title:')
    expect(out).not.toContain('timestamp:')
  })
})

describe('buildFrontmatter — okf profile', () => {
  it('emits OKF-shaped frontmatter (type/title/description/tags/timestamp)', () => {
    const out = buildFrontmatter(sample(), { profile: 'okf' })
    expect(out).toContain('type: project')
    expect(out).toContain('title: Run the test suite')
    expect(out).toContain(
      'description: Always run the project suite, never the package manager default.',
    )
    expect(out).toContain('tags: [testing, ci]')
    // timestamp tracks `modified` (the recency signal recall ranks on).
    expect(out).toContain('timestamp: 2026-02-01T00:00:00.000Z')
    // OKF uses `title`/`timestamp` in place of the native `name`/`modified` keys.
    expect(out).not.toMatch(/^name:/m)
    expect(out).not.toMatch(/^modified:/m)
  })

  it('orders OKF core fields before the preserved extension keys', () => {
    const out = buildFrontmatter(sample(), { profile: 'okf' })
    expect(out.indexOf('type:')).toBeLessThan(out.indexOf('scope:'))
    expect(out.indexOf('timestamp:')).toBeLessThan(out.indexOf('source:'))
  })

  it('falls back to `created` for the timestamp when `modified` is absent', () => {
    const out = buildFrontmatter(
      { name: 'x', type: 'note', created: '2026-01-01T00:00:00.000Z', extra: {} },
      { profile: 'okf' },
    )
    expect(out).toContain('timestamp: 2026-01-01T00:00:00.000Z')
  })

  it('preserves arbitrary extension keys (e.g. episode metadata)', () => {
    const out = buildFrontmatter(
      {
        name: 'Episode',
        type: 'episode',
        modified: 't',
        extra: { actor_id: 'tom', outcome: 'success' },
      },
      { profile: 'okf' },
    )
    expect(out).toContain('actor_id: tom')
    expect(out).toContain('outcome: success')
  })
})

describe('parseFrontmatter — OKF reader tolerance', () => {
  it('reads a hand-authored OKF note (title→name, timestamp→modified)', () => {
    const note = [
      '---',
      'type: reference',
      'title: API base URL',
      'description: The production API base URL.',
      'tags: [api, ops]',
      'timestamp: 2026-03-03T00:00:00.000Z',
      '---',
      '',
      'The base URL is https://api.example.com.',
    ].join('\n')
    const { frontmatter, body } = parseFrontmatter(note)
    expect(frontmatter.name).toBe('API base URL')
    expect(frontmatter.modified).toBe('2026-03-03T00:00:00.000Z')
    expect(frontmatter.type).toBe('reference')
    expect(frontmatter.description).toBe('The production API base URL.')
    expect(frontmatter.tags).toEqual(['api', 'ops'])
    expect(body).toBe('The base URL is https://api.example.com.')
  })

  it('lets a native key win when both it and its OKF alias are present (either order)', () => {
    const aliasFirst = [
      '---',
      'title: Alias',
      'name: Canonical',
      'timestamp: 2026-01-01T00:00:00.000Z',
      'modified: 2026-05-05T00:00:00.000Z',
      '---',
      '',
      'x',
    ].join('\n')
    const nativeFirst = [
      '---',
      'name: Canonical',
      'title: Alias',
      'modified: 2026-05-05T00:00:00.000Z',
      'timestamp: 2026-01-01T00:00:00.000Z',
      '---',
      '',
      'x',
    ].join('\n')
    for (const raw of [aliasFirst, nativeFirst]) {
      const { frontmatter } = parseFrontmatter(raw)
      expect(frontmatter.name).toBe('Canonical')
      expect(frontmatter.modified).toBe('2026-05-05T00:00:00.000Z')
    }
  })
})

describe('round-trip + cross-format equivalence', () => {
  it('round-trips an OKF-written note back to equivalent frontmatter', () => {
    const fm = sample()
    const { frontmatter: parsed } = parseFrontmatter(
      `${buildFrontmatter(fm, { profile: 'okf' })}\nbody`,
    )
    expect(parsed.name).toBe(fm.name)
    expect(parsed.description).toBe(fm.description)
    expect(parsed.type).toBe(fm.type)
    expect(parsed.scope).toBe(fm.scope)
    expect(parsed.source).toBe(fm.source)
    expect(parsed.session_id).toBe(fm.session_id)
    expect(parsed.created).toBe(fm.created)
    expect(parsed.modified).toBe(fm.modified)
    expect(parsed.tags).toEqual(fm.tags)
  })

  it('parses default and OKF renderings of the same note to identical logical fields', () => {
    const fm = sample()
    const fromDefault = parseFrontmatter(`${buildFrontmatter(fm)}\nbody`).frontmatter
    const fromOkf = parseFrontmatter(
      `${buildFrontmatter(fm, { profile: 'okf' })}\nbody`,
    ).frontmatter
    const keys = [
      'name',
      'description',
      'type',
      'scope',
      'source',
      'session_id',
      'created',
      'modified',
    ] as const
    for (const key of keys) expect(fromOkf[key]).toBe(fromDefault[key])
    expect(fromOkf.tags).toEqual(fromDefault.tags)
  })
})
