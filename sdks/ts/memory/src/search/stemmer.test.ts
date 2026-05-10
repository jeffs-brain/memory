// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { containsCJK, isCJK, tokenizeCJK } from './trigram-cjk.js'
import {
  createStemmer,
  detectLanguage,
  SUPPORTED_LANGUAGES,
  UnsupportedLanguageError,
} from './stemmer.js'
import type { StemmerLanguage } from './stemmer.js'

describe('createStemmer', () => {
  it('stems English words correctly', () => {
    const stemmer = createStemmer('en')
    expect(stemmer.language).toBe('en')
    expect(stemmer.stem('running')).toBe('run')
    expect(stemmer.stem('dogs')).toBe('dog')
    expect(stemmer.stem('studies')).toBe('studi')
    expect(stemmer.stem('caresses')).toBe('caress')
    expect(stemmer.stem('ponies')).toBe('poni')
    expect(stemmer.stem('cats')).toBe('cat')
    expect(stemmer.stem('connection')).toBe('connect')
  })

  it('stems German words correctly', () => {
    const stemmer = createStemmer('de')
    expect(stemmer.language).toBe('de')
    expect(stemmer.stem('häuser')).toBe('haus')
    expect(stemmer.stem('kinder')).toBe('kind')
    expect(stemmer.stem('laufen')).toBe('lauf')
  })

  it('stems French words correctly', () => {
    const stemmer = createStemmer('fr')
    expect(stemmer.language).toBe('fr')
    expect(stemmer.stem('maisons')).toBe('maison')
    expect(stemmer.stem('manger')).toBe('mang')
  })

  it('stems Spanish words correctly', () => {
    const stemmer = createStemmer('es')
    expect(stemmer.language).toBe('es')
    expect(stemmer.stem('corriendo')).toBe('corr')
  })

  it('normalises uppercase input', () => {
    const stemmer = createStemmer('en')
    expect(stemmer.stem('Running')).toBe('run')
    expect(stemmer.stem('DOGS')).toBe('dog')
  })

  it('returns empty string for empty input', () => {
    const stemmer = createStemmer('en')
    expect(stemmer.stem('')).toBe('')
  })

  it('creates stemmers for all supported languages', () => {
    for (const lang of SUPPORTED_LANGUAGES) {
      const stemmer = createStemmer(lang)
      expect(stemmer.language).toBe(lang)
      const result = stemmer.stem('test')
      expect(result.length).toBeGreaterThan(0)
    }
  })

  it('throws UnsupportedLanguageError for unknown language', () => {
    expect(() => createStemmer('xx' as StemmerLanguage)).toThrow(UnsupportedLanguageError)
  })

  it('produces same English stems as Go SDK (cross-SDK conformance)', () => {
    const stemmer = createStemmer('en')
    // These expected values are verified against blevesearch/snowballstem in Go.
    const conformancePairs: ReadonlyArray<readonly [string, string]> = [
      ['running', 'run'],
      ['dogs', 'dog'],
      ['studies', 'studi'],
      ['caresses', 'caress'],
      ['ponies', 'poni'],
      ['cats', 'cat'],
      ['connection', 'connect'],
    ]
    for (const [input, expected] of conformancePairs) {
      expect(stemmer.stem(input)).toBe(expected)
    }
  })
})

describe('detectLanguage', () => {
  it('detects English with high confidence', () => {
    const text =
      'The quick brown fox jumps over the lazy dog and then runs across the green field towards the other side of the river where the fisherman was standing quietly'
    const result = detectLanguage(text)
    expect(result.language).toBe('en')
    expect(result.confidence).toBeGreaterThanOrEqual(0.5)
  })

  it('detects German text', () => {
    const text =
      'Die Bundesrepublik Deutschland ist ein demokratischer und sozialer Bundesstaat mit einer langen Geschichte und vielen verschiedenen Regionen die sich durch ihre Kultur unterscheiden'
    const result = detectLanguage(text)
    expect(result.language).toBe('de')
    expect(result.confidence).toBeGreaterThanOrEqual(0.5)
  })

  it('detects French text', () => {
    const text =
      'La République française est un pays dont la majeure partie du territoire se situe en Europe occidentale et qui possède de nombreuses régions outre-mer dans le monde entier'
    const result = detectLanguage(text)
    expect(result.language).toBe('fr')
    expect(result.confidence).toBeGreaterThanOrEqual(0.5)
  })

  it('detects Russian text', () => {
    const text =
      'Российская Федерация является демократическим федеративным правовым государством с республиканской формой правления и развитой системой управления'
    const result = detectLanguage(text)
    expect(result.language).toBe('ru')
    expect(result.confidence).toBeGreaterThanOrEqual(0.5)
  })

  it('returns English with low confidence for very short text', () => {
    const result = detectLanguage('hello')
    expect(result.language).toBe('en')
    expect(result.confidence).toBe(0.0)
  })

  it('returns English with zero confidence for empty text', () => {
    const result = detectLanguage('')
    expect(result.language).toBe('en')
    expect(result.confidence).toBe(0.0)
  })
})

describe('isCJK', () => {
  it('identifies Han characters', () => {
    expect(isCJK('机')).toBe(true)
    expect(isCJK('学')).toBe(true)
  })

  it('identifies Hiragana', () => {
    expect(isCJK('あ')).toBe(true)
    expect(isCJK('の')).toBe(true)
  })

  it('identifies Katakana including prolonged sound mark', () => {
    expect(isCJK('ア')).toBe(true)
    expect(isCJK('ー')).toBe(true)
  })

  it('identifies Hangul', () => {
    expect(isCJK('한')).toBe(true)
    expect(isCJK('글')).toBe(true)
  })

  it('rejects Latin letters', () => {
    expect(isCJK('A')).toBe(false)
    expect(isCJK('z')).toBe(false)
  })

  it('rejects digits and symbols', () => {
    expect(isCJK('1')).toBe(false)
    expect(isCJK(' ')).toBe(false)
  })
})

describe('containsCJK', () => {
  it('returns true for text with CJK characters', () => {
    expect(containsCJK('Hello 机器学习')).toBe(true)
  })

  it('returns false for pure Latin text', () => {
    expect(containsCJK('Hello world')).toBe(false)
  })

  it('returns false for empty string', () => {
    expect(containsCJK('')).toBe(false)
  })
})

describe('tokenizeCJK', () => {
  it('produces trigrams from Chinese text', () => {
    const tokens = tokenizeCJK('机器学习')
    expect(tokens).toEqual(['机器学', '器学习'])
  })

  it('handles Japanese text with prolonged sound mark', () => {
    const tokens = tokenizeCJK('東京タワー')
    expect(tokens).toEqual(['東京タ', '京タワ', 'タワー'])
  })

  it('handles Korean text', () => {
    const tokens = tokenizeCJK('안녕하세요')
    expect(tokens).toEqual(['안녕하', '녕하세', '하세요'])
  })

  it('handles mixed CJK and Latin text', () => {
    const tokens = tokenizeCJK('Hello 机器学习 world')
    expect(tokens).toEqual(['hello', '机器学', '器学习', 'world'])
  })

  it('returns short CJK runs as-is', () => {
    const tokens = tokenizeCJK('学习')
    expect(tokens).toEqual(['学习'])
  })

  it('returns empty array for empty string', () => {
    expect(tokenizeCJK('')).toEqual([])
  })

  it('returns empty array for whitespace-only string', () => {
    expect(tokenizeCJK('   ')).toEqual([])
  })
})
