import { lowerToken, normalise } from './normalise.js'
import { isStopWord } from './stopwords.js'

export type TokenKind = 'term' | 'phrase' | 'prefix'

export type BooleanOp = 'AND' | 'OR' | 'NOT'

export type Token = {
  readonly kind: TokenKind
  readonly text: string
  readonly operator?: BooleanOp
}

export type QueryAST = {
  readonly raw: string
  readonly tokens: readonly Token[]
  readonly hasOperators: boolean
}

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

const stripFtsTermChars = (word: string): string => {
  let output = ''
  for (const character of word) {
    if (!FTS_TERM_STRIP_CHARS.has(character)) output += character
  }
  return output.trim()
}

const stripPhraseChars = (phrase: string): string => phrase.replace(/"/g, '').trim()

const isWhitespace = (character: string): boolean => /\s/.test(character)

const detectOperatorsOrQuotes = (raw: string): boolean => {
  if (raw.includes('"')) return true
  for (const part of raw.split(/\s+/)) {
    if (BOOLEAN_OPERATORS.has(part)) return true
  }
  return false
}

export const parseQuery = (input: string): QueryAST => {
  const raw = normalise(input)
  if (raw === '') {
    return { raw, tokens: [], hasOperators: false }
  }

  const hasOperators = detectOperatorsOrQuotes(raw)
  const tokens: Token[] = []
  const chars = Array.from(raw)
  let index = 0
  let pendingOp: BooleanOp | undefined

  while (index < chars.length) {
    const character = chars[index]
    if (character === undefined) break

    if (isWhitespace(character)) {
      index += 1
      continue
    }

    if (character === '"') {
      index += 1
      const start = index
      while (index < chars.length && chars[index] !== '"') index += 1
      const rawPhrase = chars.slice(start, index).join('')
      if (index < chars.length) index += 1
      const phrase = stripPhraseChars(rawPhrase).toLocaleLowerCase('en')
      if (phrase !== '') {
        const token: Token =
          pendingOp === undefined
            ? { kind: 'phrase', text: phrase }
            : { kind: 'phrase', text: phrase, operator: pendingOp }
        tokens.push(token)
        pendingOp = undefined
      }
      continue
    }

    const start = index
    while (index < chars.length) {
      const current = chars[index]
      if (current === undefined || isWhitespace(current) || current === '"') break
      index += 1
    }

    const word = chars.slice(start, index).join('')
    if (word === '') continue

    if (BOOLEAN_OPERATORS.has(word)) {
      pendingOp = word as BooleanOp
      continue
    }

    const isPrefix = word.endsWith('*')
    const trimmedPrefix = isPrefix ? word.slice(0, -1) : word
    const cleaned = stripFtsTermChars(trimmedPrefix)
    if (cleaned === '') {
      pendingOp = undefined
      continue
    }

    const lowered = lowerToken(cleaned)
    if (!hasOperators && BOOLEAN_OPERATORS.has(cleaned.toUpperCase())) {
      pendingOp = undefined
      continue
    }
    if (!hasOperators && isStopWord(lowered)) {
      pendingOp = undefined
      continue
    }

    const token: Token =
      pendingOp === undefined
        ? { kind: isPrefix ? 'prefix' : 'term', text: lowered }
        : { kind: isPrefix ? 'prefix' : 'term', text: lowered, operator: pendingOp }
    tokens.push(token)
    pendingOp = undefined
  }

  return { raw, tokens, hasOperators }
}

export const renderToken = (token: Token): string => {
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
