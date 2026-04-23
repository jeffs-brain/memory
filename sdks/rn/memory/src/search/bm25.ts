import type { SqlDb } from './sqlite-types.js'

export type BM25Row = {
  readonly chunk_id: string
  readonly rank: number
}

const FTS5_SPECIAL = /[*():^+?!,;$#@%=]/g
const BOOLEAN_OPS = new Set(['AND', 'OR', 'NOT'])

const tokenise = (query: string): string[] => {
  const tokens: string[] = []
  let buffer = ''
  let inPhrase = false
  for (const character of query) {
    if (character === '"') {
      if (inPhrase) {
        if (buffer.trim() !== '') tokens.push(`"${buffer.trim()}"`)
        buffer = ''
        inPhrase = false
      } else {
        if (buffer.trim() !== '') tokens.push(...buffer.trim().split(/\s+/))
        buffer = ''
        inPhrase = true
      }
      continue
    }
    if (!inPhrase && /\s/.test(character)) {
      if (buffer.trim() !== '') tokens.push(...buffer.trim().split(/\s+/))
      buffer = ''
      continue
    }
    buffer += character
  }
  if (buffer.trim() !== '') {
    if (inPhrase) tokens.push(`"${buffer.trim()}"`)
    else tokens.push(...buffer.trim().split(/\s+/))
  }
  return tokens.filter((token) => token.length > 0)
}

const scrubToken = (token: string): string => {
  if (token.startsWith('"') && token.endsWith('"')) return token
  return token.replace(FTS5_SPECIAL, '')
}

export const compileFts5Query = (query: string): string => {
  const trimmed = query.trim()
  if (trimmed === '') return ''
  const tokens = tokenise(trimmed)
  if (tokens.length === 0) return ''

  const out: string[] = []
  for (const raw of tokens) {
    const upper = raw.toLocaleUpperCase('en')
    if (BOOLEAN_OPS.has(upper)) {
      out.push(upper)
      continue
    }
    const scrubbed = scrubToken(raw)
    if (scrubbed === '' || scrubbed === '""') continue
    const previous = out[out.length - 1]
    if (previous !== undefined && !BOOLEAN_OPS.has(previous)) out.push('OR')
    out.push(scrubbed)
  }

  while (out.length > 0 && BOOLEAN_OPS.has(out[out.length - 1] as string)) {
    out.pop()
  }
  return out.join(' ')
}

const buildBm25Sql = (): string => {
  return `SELECT chunk_id, rank
          FROM knowledge_fts
          WHERE knowledge_fts MATCH ?
          ORDER BY rank
          LIMIT ?`
}

export const runBm25 = (db: SqlDb, expr: string, limit: number): BM25Row[] => {
  if (expr.trim() === '' || limit <= 0) return []
  return db.prepare(buildBm25Sql()).all(expr, limit) as BM25Row[]
}
