// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import {
  DEFAULT_CHUNK_CONFIG,
  DEFAULT_MAX_TOKENS,
  DEFAULT_MIN_TOKENS,
  DEFAULT_OVERLAP_TOKENS,
  DEFAULT_SEPARATORS,
  DEFAULT_STRATEGY,
  estimateTokens,
  validateChunkConfig,
} from './chunk-config.js'
import type { ChunkConfig } from './chunk-config.js'

describe('DEFAULT_CHUNK_CONFIG', () => {
  it('has spec-mandated maxTokens of 512', () => {
    expect(DEFAULT_CHUNK_CONFIG.maxTokens).toBe(512)
    expect(DEFAULT_MAX_TOKENS).toBe(512)
  })

  it('has spec-mandated overlapTokens of 64', () => {
    expect(DEFAULT_CHUNK_CONFIG.overlapTokens).toBe(64)
    expect(DEFAULT_OVERLAP_TOKENS).toBe(64)
  })

  it('has spec-mandated minTokens of 30', () => {
    expect(DEFAULT_CHUNK_CONFIG.minTokens).toBe(30)
    expect(DEFAULT_MIN_TOKENS).toBe(30)
  })

  it('uses recursive strategy by default', () => {
    expect(DEFAULT_CHUNK_CONFIG.strategy).toBe('recursive')
    expect(DEFAULT_STRATEGY).toBe('recursive')
  })

  it('has the spec-mandated separator hierarchy', () => {
    expect(DEFAULT_CHUNK_CONFIG.separators).toEqual(['\n\n', '\n', '. ', ' ', ''])
    expect(DEFAULT_SEPARATORS).toEqual(['\n\n', '\n', '. ', ' ', ''])
  })

  it('passes its own validation', () => {
    expect(() => validateChunkConfig(DEFAULT_CHUNK_CONFIG)).not.toThrow()
  })
})

describe('validateChunkConfig', () => {
  const validBase: ChunkConfig = {
    maxTokens: 512,
    overlapTokens: 64,
    minTokens: 30,
    strategy: 'recursive',
    separators: ['\n\n', '\n', '. ', ' ', ''],
  }

  it('accepts spec defaults', () => {
    expect(() => validateChunkConfig(validBase)).not.toThrow()
  })

  it('accepts small chunks with no overlap', () => {
    expect(() =>
      validateChunkConfig({ ...validBase, maxTokens: 64, overlapTokens: 0, minTokens: 10 }),
    ).not.toThrow()
  })

  it('accepts markdown strategy', () => {
    expect(() =>
      validateChunkConfig({ ...validBase, maxTokens: 1024, overlapTokens: 128, strategy: 'markdown' }),
    ).not.toThrow()
  })

  it('accepts markdown strategy with empty separators', () => {
    expect(() =>
      validateChunkConfig({ ...validBase, strategy: 'markdown', separators: [] }),
    ).not.toThrow()
  })

  it('rejects zero maxTokens', () => {
    expect(() => validateChunkConfig({ ...validBase, maxTokens: 0 })).toThrow(
      'maxTokens must be > 0',
    )
  })

  it('rejects negative maxTokens', () => {
    expect(() => validateChunkConfig({ ...validBase, maxTokens: -1 })).toThrow(
      'maxTokens must be > 0',
    )
  })

  it('rejects overlap equal to max', () => {
    expect(() =>
      validateChunkConfig({ ...validBase, maxTokens: 100, overlapTokens: 100 }),
    ).toThrow('overlapTokens (100) must be < maxTokens (100)')
  })

  it('rejects overlap exceeding max', () => {
    expect(() =>
      validateChunkConfig({ ...validBase, maxTokens: 100, overlapTokens: 200 }),
    ).toThrow('overlapTokens (200) must be < maxTokens (100)')
  })

  it('rejects negative overlap', () => {
    expect(() => validateChunkConfig({ ...validBase, overlapTokens: -1 })).toThrow(
      'overlapTokens must be >= 0',
    )
  })

  it('rejects minTokens equal to max', () => {
    expect(() => validateChunkConfig({ ...validBase, minTokens: 512 })).toThrow(
      'minTokens (512) must be < maxTokens (512)',
    )
  })

  it('rejects negative minTokens', () => {
    expect(() => validateChunkConfig({ ...validBase, minTokens: -5 })).toThrow(
      'minTokens must be >= 0',
    )
  })

  it('rejects unknown strategy', () => {
    expect(() =>
      validateChunkConfig({ ...validBase, strategy: 'unknown' as ChunkConfig['strategy'] }),
    ).toThrow('unknown strategy')
  })

  it('rejects empty separators with recursive strategy', () => {
    expect(() => validateChunkConfig({ ...validBase, separators: [] })).toThrow(
      'separators must be non-empty',
    )
  })
})

describe('estimateTokens', () => {
  it('returns 0 for empty string', () => {
    expect(estimateTokens('')).toBe(0)
  })

  it('returns ceil(len/4) for non-empty strings', () => {
    expect(estimateTokens('a')).toBe(1)
    expect(estimateTokens('ab')).toBe(1)
    expect(estimateTokens('abc')).toBe(1)
    expect(estimateTokens('abcd')).toBe(1)
    expect(estimateTokens('abcde')).toBe(2)
  })

  it('matches conformance fixture values', () => {
    expect(estimateTokens('hello world')).toBe(3)
    expect(estimateTokens('The quick brown fox jumps over the lazy dog.')).toBe(11)
    expect(estimateTokens('a b c d e f g h')).toBe(4)
  })

  it('is monotonic in input length', () => {
    let prev = estimateTokens('')
    for (let i = 1; i <= 100; i++) {
      const cur = estimateTokens('x'.repeat(i))
      expect(cur).toBeGreaterThanOrEqual(prev)
      prev = cur
    }
  })

  it('produces same result as Go formula for boundary cases', () => {
    // Go formula: (len + 3) / 4 using integer division
    // TS formula: Math.ceil(len / 4)
    // Both should produce identical results
    const goFormula = (len: number): number => Math.trunc((len + 3) / 4)
    for (let i = 0; i <= 50; i++) {
      const tsResult = estimateTokens('x'.repeat(i))
      const goResult = i === 0 ? 0 : goFormula(i)
      expect(tsResult).toBe(goResult)
    }
  })
})
