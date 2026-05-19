// SPDX-License-Identifier: Apache-2.0

/**
 * CJK trigram tokenizer for full-text search on scripts without
 * explicit word boundaries.
 *
 * Chinese, Japanese (kanji/hiragana/katakana), and Korean (Hangul)
 * text is split into overlapping 3-character trigrams. Non-CJK
 * segments are returned as whitespace-separated word tokens.
 *
 * Time: O(N) where N = text length in codepoints.
 * Space: O(N) for the output array.
 */

/** Size of CJK trigram window. */
const CJK_TRIGRAM_SIZE = 3

/**
 * Reports whether a single character is in a CJK script range:
 * CJK Unified Ideographs, Hiragana, Katakana (including prolonged
 * sound mark and modifier characters), or Hangul Syllables.
 */
export function isCJK(char: string): boolean {
  const code = char.codePointAt(0)
  if (code === undefined) return false
  return isCJKCodePoint(code)
}

/**
 * Reports whether text contains any CJK codepoints, indicating that
 * trigram tokenization should be used alongside or instead of
 * whitespace-based splitting.
 */
export function containsCJK(text: string): boolean {
  for (const char of text) {
    const code = char.codePointAt(0)
    if (code !== undefined && isCJKCodePoint(code)) return true
  }
  return false
}

/**
 * Split text into tokens suitable for CJK full-text search. CJK runs
 * are converted into overlapping 3-character trigrams; non-CJK segments
 * are split on whitespace and returned as lowercased word tokens.
 *
 * Short CJK runs (fewer than 3 characters) are returned as-is rather
 * than being discarded, ensuring no content is lost.
 */
export function tokenizeCJK(text: string): readonly string[] {
  if (text === '') return []

  const chars = [...text]
  const tokens: string[] = []

  let cjkRun: string[] = []
  let latinRun = ''

  const flushCJK = (): void => {
    if (cjkRun.length === 0) return
    if (cjkRun.length < CJK_TRIGRAM_SIZE) {
      tokens.push(cjkRun.join(''))
    } else {
      for (let i = 0; i + CJK_TRIGRAM_SIZE <= cjkRun.length; i++) {
        tokens.push(cjkRun.slice(i, i + CJK_TRIGRAM_SIZE).join(''))
      }
    }
    cjkRun = []
  }

  const flushLatin = (): void => {
    if (latinRun.length === 0) return
    for (const word of latinRun.trim().split(/\s+/)) {
      if (word.length > 0) {
        tokens.push(word.toLowerCase())
      }
    }
    latinRun = ''
  }

  for (const char of chars) {
    if (isCJK(char)) {
      flushLatin()
      cjkRun.push(char)
    } else if (/[\s\p{P}]/u.test(char)) {
      flushCJK()
      if (latinRun.length > 0) {
        latinRun += char
      }
    } else {
      flushCJK()
      latinRun += char
    }
  }

  flushCJK()
  flushLatin()

  return tokens.length > 0 ? tokens : []
}

/**
 * Check whether a Unicode codepoint belongs to a CJK script range.
 *
 * Ranges covered:
 * - CJK Unified Ideographs (U+4E00..U+9FFF)
 * - CJK Unified Ideographs Extension A (U+3400..U+4DBF)
 * - CJK Unified Ideographs Extension B (U+20000..U+2A6DF)
 * - CJK Compatibility Ideographs (U+F900..U+FAFF)
 * - Hiragana (U+3040..U+309F)
 * - Katakana (U+30A0..U+30FF) — includes prolonged sound mark
 * - Katakana Phonetic Extensions (U+31F0..U+31FF)
 * - Hangul Syllables (U+AC00..U+D7AF)
 * - Hangul Jamo (U+1100..U+11FF)
 * - Hangul Compatibility Jamo (U+3130..U+318F)
 */
function isCJKCodePoint(code: number): boolean {
  return (
    (code >= 0x4e00 && code <= 0x9fff) ||
    (code >= 0x3400 && code <= 0x4dbf) ||
    (code >= 0x20000 && code <= 0x2a6df) ||
    (code >= 0xf900 && code <= 0xfaff) ||
    (code >= 0x3040 && code <= 0x309f) ||
    (code >= 0x30a0 && code <= 0x30ff) ||
    (code >= 0x31f0 && code <= 0x31ff) ||
    (code >= 0xac00 && code <= 0xd7af) ||
    (code >= 0x1100 && code <= 0x11ff) ||
    (code >= 0x3130 && code <= 0x318f)
  )
}
