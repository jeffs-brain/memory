// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { hashChunk, hashDocument, hashString } from './hash.js'

describe('hashDocument', () => {
  it('produces deterministic 64-char hex output', () => {
    const input = Buffer.from('the quick brown fox jumps over the lazy dog')
    const first = hashDocument(input)
    const second = hashDocument(input)
    expect(first).toBe(second)
    expect(first).toHaveLength(64)
    expect(first).toMatch(/^[0-9a-f]{64}$/)
  })

  it('produces different hashes for different inputs', () => {
    const a = hashDocument(Buffer.from('input alpha'))
    const b = hashDocument(Buffer.from('input beta'))
    expect(a).not.toBe(b)
  })

  it('handles empty input without throwing', () => {
    const hash = hashDocument(Buffer.alloc(0))
    expect(hash).toHaveLength(64)
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
  })

  it('handles large inputs', () => {
    const largeBuffer = Buffer.alloc(100 * 1024, 0x61)
    const hash = hashDocument(largeBuffer)
    expect(hash).toHaveLength(64)
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
  })
})

describe('hashChunk', () => {
  it('produces the same output as hashDocument for identical input', () => {
    const input = Buffer.from('chunk content for verification')
    expect(hashChunk(input)).toBe(hashDocument(input))
  })

  it('produces 64-char hex output', () => {
    const hash = hashChunk(Buffer.from('some chunk'))
    expect(hash).toHaveLength(64)
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
  })
})

describe('hashString', () => {
  it('produces the same output as hashDocument with Buffer.from(s, utf8)', () => {
    const s = 'string input for verification'
    const fromString = hashString(s)
    const fromBuffer = hashDocument(Buffer.from(s, 'utf8'))
    expect(fromString).toBe(fromBuffer)
  })

  it('produces 64-char hex output', () => {
    const hash = hashString('hello world')
    expect(hash).toHaveLength(64)
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
  })

  it('handles empty string', () => {
    const hash = hashString('')
    expect(hash).toHaveLength(64)
    expect(hash).toMatch(/^[0-9a-f]{64}$/)
  })
})

describe('cross-SDK conformance', () => {
  it('produces identical hash for the canonical test vector across Go and TS', () => {
    // This test vector MUST produce the same hash in both the Go and TS SDKs.
    // The Go test logs this value; if it ever changes, both tests must be
    // updated in lockstep.
    const input = "jeff's brain memory system"
    const hash = hashString(input)

    // BLAKE3("jeff's brain memory system") — verified against Go SDK output.
    // If this assertion fails after a dependency update, re-run the Go test
    // to obtain the correct expected value.
    expect(hash).toHaveLength(64)
    expect(hash).toMatch(/^[0-9a-f]{64}$/)

    // Hardcoded expected value from BLAKE3 reference implementation.
    // Computed via: blake3.Sum256([]byte("jeff's brain memory system"))
    expect(hash).toBe(
      'e311e54b56b26bfef4e5c8501f04c708f1e02233106022f58a7e94b728b7265c',
    )
  })
})
