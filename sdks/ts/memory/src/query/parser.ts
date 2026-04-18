// SPDX-License-Identifier: Apache-2.0

/**
 * Parser that walks a normalised query string into a token AST. Quoted
 * phrases, explicit AND/OR/NOT boolean operators, and trailing prefix
 * wildcards are preserved; bare terms run through the EN/NL stop-word
 * filter. Stop-word filtering is disabled when the raw input contains
 * any quoted phrase or explicit boolean operator so power-user queries
 * keep every token they typed.
 *
 * This is a port of apps/jeff/internal/search/query_parser.go adapted
 * to inject alias expansion via the compile step rather than a global.
 */

import { lowerToken, normalise } from './normalise.js'
import { isStopWord } from './stopwords.js'

export type TokenKind = 'term' | 'phrase' | 'prefix'

export type BooleanOp = 'AND' | 'OR' | 'NOT'

export type Token = {
  kind: TokenKind
  text: string
  /** The operator prefixing this token when the user typed one. */
  operator?: BooleanOp
}

/**
 * A parsed query. The original normalised text is retained so callers
 * can log it or feed it into downstream retrieval pipelines that still
 * want the raw form (e.g. vector similarity).
 */
export type QueryAST = {
  raw: string
  tokens: readonly Token[]
  /** True when the raw input contained quotes or explicit AND/OR/NOT. */
  hasOperators: boolean
}

/**
 * Characters stripped from bare terms before they reach FTS5. Quoted
 * phrase contents keep almost everything except the closing quote.
 */
const FTS_TERM_STRIP_CHARS = new Set<string>([
  '*',
  '(',
  ')',
  ':',
  '^',
  '+',
  '"',
  '-',
  '?',
  '!',
  '.',
  ',',
  ';',
  '/',
  '\\',
  '[',
  ']',
  '{',
  '}',
  '<',
  '>',
  '|',
  '&',
  "'",
  '$',
  '#',
  '@',
  '%',
  '=',
  '~',
  '`',
])

const BOOLEAN_OPERATORS: ReadonlySet<string> = new Set(['AND', 'OR', 'NOT'])

function stripFTSTermChars(word: string): string {
  let out = ''
  for (const ch of word) {
    if (!FTS_TERM_STRIP_CHARS.has(ch)) out += ch
  }
  return out.trim()
}

function stripPhraseChars(phrase: string): string {
  return phrase.replace(/"/g, '').trim()
}

function isWhitespace(ch: string): boolean {
  return /\s/.test(ch)
}

/**
 * parseQuery walks input and produces a query AST. Stop words are
 * dropped from bare-word inputs only; inputs containing quotes or
 * explicit boolean operators skip the stop-word filter so the user's
 * intent survives verbatim.
 */
export function parseQuery(input: string): QueryAST {
  const raw = normalise(input)
  if (raw === '') {
    return { raw, tokens: [], hasOperators: false }
  }

  // First pass: detect operator/quote usage so we can decide whether
  // to strip stop words. A naive substring probe is enough: explicit
  // operators must be uppercase and whitespace-bounded.
  const hasOperators = detectOperatorsOrQuotes(raw)

  const tokens: Token[] = []
  const chars = Array.from(raw)
  let i = 0
  let pendingOp: BooleanOp | undefined

  while (i < chars.length) {
    const ch = chars[i]
    if (ch === undefined) break

    if (isWhitespace(ch)) {
      i++
      continue
    }

    // Quoted phrase.
    if (ch === '"') {
      i++
      const start = i
      while (i < chars.length && chars[i] !== '"') i++
      const rawPhrase = chars.slice(start, i).join('')
      if (i < chars.length) i++
      const phrase = stripPhraseChars(rawPhrase).toLocaleLowerCase('en')
      if (phrase !== '') {
        const token: Token = { kind: 'phrase', text: phrase }
        if (pendingOp !== undefined) token.operator = pendingOp
        tokens.push(token)
        pendingOp = undefined
      }
      continue
    }

    // Bare word: run until whitespace or quote.
    const start = i
    while (i < chars.length) {
      const c = chars[i]
      if (c === undefined || isWhitespace(c) || c === '"') break
      i++
    }
    const word = chars.slice(start, i).join('')
    if (word === '') continue

    // Explicit boolean operator (case-sensitive uppercase).
    if (BOOLEAN_OPERATORS.has(word)) {
      pendingOp = word as BooleanOp
      continue
    }

    const isPrefix = word.endsWith('*')
    const trimmedPrefix = isPrefix ? word.slice(0, -1) : word
    const cleaned = stripFTSTermChars(trimmedPrefix)
    if (cleaned === '') {
      pendingOp = undefined
      continue
    }
    const lowered = lowerToken(cleaned)

    // Lowercase boolean words are usually natural-language filler in a
    // bare query, but SQLite FTS still treats them as operators. Drop
    // them unless the user explicitly opted into boolean syntax.
    if (!hasOperators && BOOLEAN_OPERATORS.has(cleaned.toUpperCase())) {
      pendingOp = undefined
      continue
    }

    // Stop-word filter only applies when the query is bare-word.
    if (!hasOperators && isStopWord(lowered)) {
      pendingOp = undefined
      continue
    }

    const kind: TokenKind = isPrefix ? 'prefix' : 'term'
    const token: Token = { kind, text: lowered }
    if (pendingOp !== undefined) token.operator = pendingOp
    tokens.push(token)
    pendingOp = undefined
  }

  return { raw, tokens, hasOperators }
}

/**
 * detectOperatorsOrQuotes returns true when the normalised raw input
 * contains a double quote or at least one whitespace-separated uppercase
 * AND/OR/NOT token. These markers switch off the stop-word filter for
 * the entire query.
 */
function detectOperatorsOrQuotes(raw: string): boolean {
  if (raw.includes('"')) return true
  for (const part of raw.split(/\s+/)) {
    if (BOOLEAN_OPERATORS.has(part)) return true
  }
  return false
}

/**
 * Render a single token back into its FTS5 surface form. Phrases are
 * wrapped in double quotes (the one place quoting is legitimate inside
 * an FTS5 MATCH expression); prefix tokens keep their trailing `*`.
 */
export function renderToken(token: Token): string {
  const text = token.text.trim()
  if (text === '') return ''
  switch (token.kind) {
    case 'phrase':
      return `"${text}"`
    case 'prefix':
      return `${text}*`
    default:
      return text
  }
}
