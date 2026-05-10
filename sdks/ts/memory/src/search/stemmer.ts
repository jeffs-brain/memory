// SPDX-License-Identifier: Apache-2.0

/**
 * Multi-language Snowball stemmer integration for BM25 full-text search.
 *
 * Uses the `snowball-stemmers` package (pure JS port of the official
 * Snowball algorithms) to provide language-specific stemming. Language
 * detection uses character-level bigram frequency profiles with cosine
 * similarity.
 *
 * This module provides standalone stemming utilities. FTS5 integration
 * (custom tokenizer registration + schema migration) is deferred to a
 * follow-up ticket.
 */

import snowballStemmers from 'snowball-stemmers'

/**
 * Supported language codes for stemming. Subset of ISO 639-1 codes
 * that have Snowball implementations in both Go and TS SDKs.
 */
export type StemmerLanguage =
  | 'en'
  | 'de'
  | 'fr'
  | 'es'
  | 'nl'
  | 'it'
  | 'pt'
  | 'sv'
  | 'no'
  | 'da'
  | 'fi'
  | 'hu'
  | 'tr'
  | 'ro'
  | 'ru'

/** The set of language codes for which stemmers are available. */
export const SUPPORTED_LANGUAGES: ReadonlySet<StemmerLanguage> = new Set([
  'en',
  'de',
  'fr',
  'es',
  'nl',
  'it',
  'pt',
  'sv',
  'no',
  'da',
  'fi',
  'hu',
  'tr',
  'ro',
  'ru',
])

/** Language-specific stemmer that reduces a word to its root form. */
export type Stemmer = {
  readonly stem: (word: string) => string
  readonly language: StemmerLanguage
}

/** Maps ISO 639-1 codes to snowball-stemmers algorithm names. */
const LANGUAGE_TO_ALGORITHM: Readonly<Record<StemmerLanguage, string>> = {
  en: 'english',
  de: 'german',
  fr: 'french',
  es: 'spanish',
  nl: 'dutch',
  it: 'italian',
  pt: 'portuguese',
  sv: 'swedish',
  no: 'norwegian',
  da: 'danish',
  fi: 'finnish',
  hu: 'hungarian',
  tr: 'turkish',
  ro: 'romanian',
  ru: 'russian',
}

/**
 * Create a Snowball stemmer for the given language code.
 *
 * @throws {UnsupportedLanguageError} when the language is not supported
 */
export function createStemmer(lang: StemmerLanguage): Stemmer {
  if (!SUPPORTED_LANGUAGES.has(lang)) {
    throw new UnsupportedLanguageError(lang)
  }

  const algorithm = LANGUAGE_TO_ALGORITHM[lang]
  const snowball = snowballStemmers.newStemmer(algorithm)

  return {
    stem: (word: string): string => {
      if (word === '') return ''
      return snowball.stem(word.toLowerCase())
    },
    language: lang,
  }
}

/** Thrown when a stemmer is requested for an unsupported language. */
export class UnsupportedLanguageError extends Error {
  override readonly name = 'UnsupportedLanguageError'
  readonly lang: string

  constructor(lang: string) {
    super(`search: unsupported stemmer language: ${lang}`)
    this.lang = lang
  }
}

/** Result of language detection. */
export type DetectLanguageResult = {
  readonly language: StemmerLanguage
  readonly confidence: number
}

/**
 * Minimum confidence threshold. Detection results below this value
 * default to English.
 */
const LANGUAGE_CONFIDENCE_THRESHOLD = 0.5

/**
 * Minimum number of alphabetic characters required for language
 * detection. Below this, English is returned with zero confidence.
 */
const MIN_DETECTION_LENGTH = 20

/**
 * Detect the dominant language of text using character bigram
 * frequency profiles.
 *
 * Returns the detected language and a confidence score in [0, 1].
 * When confidence is below 0.5 or the text is too short (< 20
 * alphabetic characters), returns English as the safe default.
 *
 * Time: O(N) where N = text length.
 * Space: O(N) for the bigram frequency map.
 */
export function detectLanguage(text: string): DetectLanguageResult {
  const cleaned = extractAlphaRuns(text)
  if ([...cleaned].length < MIN_DETECTION_LENGTH) {
    return { language: 'en', confidence: 0.0 }
  }

  const bigrams = buildBigrams(cleaned)
  if (bigrams.size === 0) {
    return { language: 'en', confidence: 0.0 }
  }

  let bestLang: StemmerLanguage = 'en'
  let bestScore = 0

  const langs = Object.keys(LANGUAGE_PROFILES) as StemmerLanguage[]
  for (const lang of langs) {
    const profile = LANGUAGE_PROFILES[lang]
    const score = bigramCosineSimilarity(bigrams, profile)
    if (score > bestScore) {
      bestScore = score
      bestLang = lang
    }
  }

  const confidence = Math.min(1.0, bestScore * 2.0)

  if (confidence < LANGUAGE_CONFIDENCE_THRESHOLD) {
    return { language: 'en', confidence }
  }

  return { language: bestLang, confidence }
}

/**
 * Extract runs of letter characters from text, lowercased, with
 * non-letter characters collapsed into single spaces.
 */
function extractAlphaRuns(text: string): string {
  const chars: string[] = []
  let prevSpace = true
  for (const char of text) {
    if (isLetter(char)) {
      chars.push(char.toLowerCase())
      prevSpace = false
    } else if (!prevSpace) {
      chars.push(' ')
      prevSpace = true
    }
  }
  return chars.join('').trimEnd()
}

/** Pre-compiled regex for Unicode letter detection. */
const LETTER_RE = /\p{L}/u

/** Check if a character is a Unicode letter. */
const isLetter = (char: string): boolean => LETTER_RE.test(char)

/**
 * Build a normalised bigram frequency map from text. Each bigram's
 * value is its count divided by the total number of bigrams.
 */
function buildBigrams(text: string): ReadonlyMap<string, number> {
  const chars = [...text]
  if (chars.length < 2) return new Map()

  const counts = new Map<string, number>()
  let total = 0
  for (let i = 0; i + 1 < chars.length; i++) {
    const bigram = chars[i] + chars[i + 1]!
    counts.set(bigram, (counts.get(bigram) ?? 0) + 1)
    total++
  }

  if (total === 0) return new Map()

  const freqs = new Map<string, number>()
  for (const [bg, count] of counts) {
    freqs.set(bg, count / total)
  }
  return freqs
}

/**
 * Cosine similarity between a document bigram vector and a language
 * profile vector.
 */
function bigramCosineSimilarity(
  doc: ReadonlyMap<string, number>,
  profile: Readonly<Record<string, number>>,
): number {
  let dot = 0
  let normDoc = 0
  let normProfile = 0

  for (const [bg, freq] of doc) {
    normDoc += freq * freq
    const pFreq = profile[bg]
    if (pFreq !== undefined) {
      dot += freq * pFreq
    }
  }

  for (const pFreq of Object.values(profile)) {
    normProfile += pFreq * pFreq
  }

  if (normDoc === 0 || normProfile === 0) return 0
  return dot / (Math.sqrt(normDoc) * Math.sqrt(normProfile))
}

/**
 * Character bigram frequency profiles for language detection. Top-40
 * bigrams per language, derived from representative corpora.
 */
const LANGUAGE_PROFILES: Readonly<Record<StemmerLanguage, Readonly<Record<string, number>>>> = {
  en: {
    th: 0.037, he: 0.034, in: 0.029, er: 0.028, an: 0.026,
    re: 0.022, on: 0.021, en: 0.019, at: 0.018, nd: 0.018,
    ti: 0.017, es: 0.017, or: 0.016, te: 0.016, of: 0.015,
    ed: 0.015, is: 0.014, it: 0.014, al: 0.014, ar: 0.013,
    st: 0.013, to: 0.013, nt: 0.012, ng: 0.012, se: 0.011,
    ha: 0.011, as: 0.010, ou: 0.010, io: 0.010, le: 0.010,
    ve: 0.010, co: 0.009, me: 0.009, de: 0.009, hi: 0.009,
    ri: 0.009, ro: 0.009, ic: 0.008, ne: 0.008, ea: 0.008,
  },
  de: {
    en: 0.038, er: 0.036, ch: 0.028, de: 0.024, ei: 0.022,
    in: 0.021, te: 0.020, nd: 0.019, ie: 0.018, ge: 0.017,
    be: 0.016, di: 0.015, un: 0.015, re: 0.014, ic: 0.014,
    st: 0.013, an: 0.013, au: 0.012, es: 0.012, he: 0.011,
    ne: 0.011, da: 0.010, se: 0.010, le: 0.010, sc: 0.010,
    it: 0.009, al: 0.009, ng: 0.009, si: 0.009, ar: 0.008,
    is: 0.008, li: 0.008, ht: 0.008, mi: 0.008, el: 0.008,
    ni: 0.007, ra: 0.007, ve: 0.007, uf: 0.007, as: 0.007,
  },
  fr: {
    es: 0.032, le: 0.028, de: 0.027, en: 0.025, re: 0.023,
    on: 0.021, nt: 0.020, la: 0.019, ti: 0.018, er: 0.017,
    ou: 0.016, te: 0.016, an: 0.015, qu: 0.015, se: 0.014,
    ai: 0.014, io: 0.013, ne: 0.013, co: 0.012, me: 0.012,
    et: 0.012, ns: 0.011, is: 0.011, ur: 0.011, it: 0.010,
    li: 0.010, ra: 0.010, pa: 0.009, ar: 0.009, us: 0.009,
    ie: 0.009, ce: 0.009, al: 0.008, ue: 0.008, ma: 0.008,
    si: 0.008, da: 0.008, un: 0.007, in: 0.007, em: 0.007,
  },
  es: {
    de: 0.031, en: 0.028, es: 0.026, la: 0.022, ci: 0.020,
    on: 0.019, el: 0.018, re: 0.017, an: 0.017, nt: 0.016,
    er: 0.015, ar: 0.015, al: 0.014, os: 0.014, te: 0.013,
    io: 0.013, co: 0.013, as: 0.012, ta: 0.012, se: 0.011,
    ra: 0.011, ie: 0.010, do: 0.010, un: 0.010, st: 0.009,
    or: 0.009, ad: 0.009, ac: 0.009, to: 0.009, in: 0.009,
    da: 0.008, no: 0.008, pa: 0.008, ue: 0.008, qu: 0.008,
    ti: 0.007, lo: 0.007, ri: 0.007, le: 0.007, ca: 0.007,
  },
  nl: {
    en: 0.042, de: 0.030, an: 0.024, er: 0.023, in: 0.021,
    he: 0.019, et: 0.018, va: 0.017, te: 0.016, ge: 0.015,
    ve: 0.014, ee: 0.014, ij: 0.013, st: 0.013, nd: 0.012,
    re: 0.012, aa: 0.011, ie: 0.011, or: 0.010, el: 0.010,
    on: 0.010, al: 0.009, ng: 0.009, is: 0.009, ar: 0.009,
    be: 0.009, le: 0.008, da: 0.008, oo: 0.008, di: 0.008,
    ni: 0.007, ze: 0.007, me: 0.007, we: 0.007, at: 0.007,
    it: 0.007, ch: 0.007, ti: 0.006, se: 0.006, li: 0.006,
  },
  it: {
    di: 0.025, re: 0.023, la: 0.022, er: 0.021, in: 0.021,
    el: 0.020, to: 0.019, en: 0.018, on: 0.017, de: 0.017,
    ti: 0.016, an: 0.015, co: 0.015, le: 0.014, ta: 0.014,
    te: 0.013, al: 0.013, ne: 0.012, no: 0.012, io: 0.012,
    ri: 0.011, li: 0.011, si: 0.010, ra: 0.010, at: 0.010,
    ar: 0.009, ll: 0.009, or: 0.009, ni: 0.009, se: 0.008,
    pe: 0.008, es: 0.008, un: 0.008, st: 0.008, il: 0.007,
    na: 0.007, ch: 0.007, me: 0.007, is: 0.007, ci: 0.007,
  },
  pt: {
    de: 0.030, os: 0.024, es: 0.023, do: 0.021, en: 0.020,
    re: 0.019, da: 0.018, ao: 0.017, er: 0.016, an: 0.016,
    co: 0.015, ar: 0.015, se: 0.014, te: 0.014, ra: 0.013,
    as: 0.013, to: 0.012, al: 0.012, or: 0.012, ta: 0.011,
    in: 0.011, st: 0.011, nt: 0.010, on: 0.010, ad: 0.010,
    ca: 0.009, no: 0.009, la: 0.009, is: 0.009, el: 0.008,
    me: 0.008, ci: 0.008, ma: 0.008, ri: 0.008, io: 0.007,
    pa: 0.007, qu: 0.007, na: 0.007, em: 0.007, po: 0.007,
  },
  sv: {
    en: 0.038, ar: 0.026, er: 0.024, de: 0.021, an: 0.020,
    in: 0.019, et: 0.018, tt: 0.017, ng: 0.016, or: 0.015,
    st: 0.014, ra: 0.014, at: 0.013, te: 0.013, nd: 0.012,
    ti: 0.011, ll: 0.011, al: 0.010, om: 0.010, re: 0.010,
    le: 0.009, av: 0.009, ta: 0.009, ge: 0.009, ri: 0.008,
    ch: 0.008, ni: 0.008, on: 0.008, ka: 0.008, so: 0.007,
    la: 0.007, li: 0.007, ha: 0.007, me: 0.007, ig: 0.007,
    da: 0.006, is: 0.006, se: 0.006, fo: 0.006, va: 0.006,
  },
  no: {
    en: 0.040, er: 0.030, et: 0.022, de: 0.020, an: 0.019,
    re: 0.017, or: 0.016, ar: 0.015, in: 0.015, te: 0.014,
    st: 0.014, ti: 0.013, ng: 0.013, nd: 0.012, le: 0.011,
    ge: 0.011, me: 0.010, om: 0.010, se: 0.010, at: 0.010,
    al: 0.009, fo: 0.009, il: 0.009, ha: 0.009, el: 0.008,
    ra: 0.008, li: 0.008, so: 0.008, on: 0.008, ve: 0.007,
    ke: 0.007, sk: 0.007, ne: 0.007, be: 0.007, ri: 0.007,
    vi: 0.006, ko: 0.006, av: 0.006, ta: 0.006, da: 0.006,
  },
  da: {
    er: 0.038, en: 0.034, de: 0.026, et: 0.022, re: 0.019,
    an: 0.018, ge: 0.017, nd: 0.016, in: 0.016, or: 0.015,
    te: 0.014, ar: 0.014, ti: 0.013, le: 0.012, st: 0.012,
    ng: 0.012, af: 0.011, me: 0.011, se: 0.010, fo: 0.010,
    el: 0.009, li: 0.009, at: 0.009, al: 0.009, ig: 0.009,
    il: 0.008, ve: 0.008, ke: 0.008, be: 0.008, ha: 0.008,
    ne: 0.007, om: 0.007, sk: 0.007, ri: 0.007, so: 0.007,
    da: 0.006, vi: 0.006, si: 0.006, on: 0.006, la: 0.006,
  },
  fi: {
    en: 0.030, ta: 0.025, in: 0.024, an: 0.022, is: 0.020,
    tt: 0.019, st: 0.018, al: 0.017, aa: 0.016, on: 0.015,
    te: 0.015, se: 0.014, la: 0.014, ll: 0.013, ti: 0.013,
    ka: 0.012, el: 0.012, si: 0.011, it: 0.011, sa: 0.011,
    ri: 0.010, ai: 0.010, es: 0.010, va: 0.009, as: 0.009,
    li: 0.009, ol: 0.008, tu: 0.008, ni: 0.008, mi: 0.008,
    pa: 0.007, ei: 0.007, ki: 0.007, us: 0.007, le: 0.007,
    na: 0.007, ra: 0.006, ha: 0.006, ar: 0.006, ma: 0.006,
  },
  hu: {
    el: 0.028, en: 0.025, sz: 0.022, te: 0.020, et: 0.019,
    al: 0.018, an: 0.017, le: 0.016, re: 0.015, gy: 0.014,
    me: 0.014, tt: 0.013, er: 0.013, eg: 0.012, es: 0.012,
    ta: 0.011, ni: 0.011, at: 0.011, ne: 0.010, is: 0.010,
    on: 0.010, mi: 0.009, ke: 0.009, ra: 0.009, se: 0.009,
    be: 0.008, ol: 0.008, in: 0.008, ve: 0.008, ak: 0.008,
    ar: 0.007, em: 0.007, ti: 0.007, la: 0.007, ek: 0.007,
    ze: 0.006, ge: 0.006, to: 0.006, og: 0.006, ha: 0.006,
  },
  tr: {
    in: 0.028, la: 0.025, an: 0.024, ar: 0.022, en: 0.020,
    le: 0.018, er: 0.018, ir: 0.016, de: 0.015, da: 0.015,
    bi: 0.014, ri: 0.013, ya: 0.013, ak: 0.012, al: 0.012,
    nd: 0.011, ni: 0.011, li: 0.010, ne: 0.010, il: 0.010,
    ta: 0.010, ka: 0.009, si: 0.009, ra: 0.009, ol: 0.009,
    el: 0.008, is: 0.008, un: 0.008, ek: 0.008, ba: 0.008,
    se: 0.007, or: 0.007, ti: 0.007, me: 0.007, ge: 0.007,
    di: 0.006, on: 0.006, im: 0.006, as: 0.006, mi: 0.006,
  },
  ro: {
    re: 0.028, in: 0.026, de: 0.024, ul: 0.022, ar: 0.020,
    ea: 0.019, te: 0.018, la: 0.017, at: 0.016, en: 0.015,
    le: 0.015, ta: 0.014, ti: 0.013, es: 0.013, an: 0.012,
    ri: 0.012, al: 0.011, ca: 0.011, st: 0.010, pe: 0.010,
    ni: 0.010, or: 0.009, ra: 0.009, ii: 0.009, el: 0.009,
    un: 0.008, cu: 0.008, ne: 0.008, lu: 0.008, se: 0.008,
    co: 0.007, ce: 0.007, nd: 0.007, si: 0.007, il: 0.007,
    er: 0.006, is: 0.006, nu: 0.006, pr: 0.006, da: 0.006,
  },
  ru: {
    'ов': 0.022, 'но': 0.020, 'ен': 0.019,
    'на': 0.018, 'ст': 0.017, 'пр': 0.016,
    'ос': 0.015, 'по': 0.015, 'ни': 0.014,
    'то': 0.014, 'ра': 0.013, 'ор': 0.013,
    'не': 0.012, 'он': 0.012, 'ли': 0.011,
    'ер': 0.011, 'из': 0.010, 'ол': 0.010,
    'ел': 0.010, 'ре': 0.009, 'ко': 0.009,
    'ан': 0.009, 'те': 0.009, 'од': 0.008,
    'де': 0.008, 'ро': 0.008, 'ет': 0.008,
    'во': 0.007, 'ит': 0.007, 'ал': 0.007,
    'го': 0.007, 'от': 0.006, 'ва': 0.006,
    'ат': 0.006, 'ин': 0.006, 'ыл': 0.006,
    'нь': 0.005, 'ых': 0.005, 'ес': 0.005,
    'ис': 0.005,
  },
}
