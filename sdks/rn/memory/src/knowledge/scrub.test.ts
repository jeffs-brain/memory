import { describe, expect, it } from 'vitest'

import { createIngest } from './ingest.js'
import { DEFAULT_PATTERNS, applyPatterns, createScrub } from './scrub.js'
import { createTestStore } from './test-helpers.js'

const logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
}

describe('scrub', () => {
  it('strips default email and phone matches', async () => {
    const input = 'Reach me at user@example.com or +31 6 12345678.'
    const { after, matches } = applyPatterns(input, DEFAULT_PATTERNS)
    expect(after).toContain('[redacted:email]')
    expect(after).toContain('[redacted:phone]')
    expect(matches.find((match) => match.name === 'email')?.count).toBe(1)
  })

  it('supports dryRun without mutating the store', async () => {
    const store = await createTestStore()
    const ingest = createIngest({ store, logger })
    const ingested = await ingest('Call me at user@example.com')

    const scrub = createScrub({ store, logger })
    const dryRun = await scrub({ dryRun: true })
    expect(dryRun).toHaveLength(1)
    expect(await store.read(ingested.path)).toContain('user@example.com')
  })
})
