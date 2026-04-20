// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { noopLogger } from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { createIngest } from './ingest.js'
import { DEFAULT_PATTERNS, applyPatterns, createScrub } from './scrub.js'

describe('scrub', () => {
  it('strips default email and phone matches', async () => {
    const input = 'Reach me at user@example.com or +31 6 12345678.'
    const { after, matches } = applyPatterns(input, DEFAULT_PATTERNS)
    expect(after).not.toContain('user@example.com')
    expect(after).toContain('[redacted:email]')
    expect(after).toContain('[redacted:phone]')
    expect(matches.find((m) => m.name === 'email')?.count).toBe(1)
    expect(matches.find((m) => m.name === 'phone')?.count).toBeGreaterThanOrEqual(1)
  })

  it('applies a custom pattern on top of defaults', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    await ingest('Hello user@example.com, token is TKN-ABC-123 and TKN-ZZZ-777.')

    const scrub = createScrub({ store, logger: noopLogger })
    const results = await scrub({
      patterns: [{ name: 'token', pattern: /TKN-[A-Z]+-\d+/g, replacement: '[redacted:token]' }],
    })

    expect(results).toHaveLength(1)
    const first = results[0]
    expect(first).toBeDefined()
    if (!first) return
    expect(first.after).not.toContain('user@example.com')
    expect(first.after).not.toContain('TKN-ABC-123')
    expect(first.after).not.toContain('TKN-ZZZ-777')
    expect(first.after).toContain('[redacted:email]')
    expect(first.after).toContain('[redacted:token]')
    expect(first.matches.find((m) => m.name === 'token')?.count).toBe(2)
  })

  it('supports dryRun without mutating the store', async () => {
    const store = createMemStore()
    const ingest = createIngest({ store, logger: noopLogger })
    const ingested = await ingest('Call me at user@example.com')

    const scrub = createScrub({ store, logger: noopLogger })
    const dry = await scrub({ dryRun: true })
    expect(dry).toHaveLength(1)

    const unchanged = (await store.read(ingested.path)).toString('utf8')
    expect(unchanged).toContain('user@example.com')
  })
})
