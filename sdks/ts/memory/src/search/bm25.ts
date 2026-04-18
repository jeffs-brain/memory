/**
 * FTS5 query compilation and BM25 ranking.
 *
 * Two pieces:
 *
 *   - {@link compileFts5Query} — converts a user-friendly query string into a
 *     valid FTS5 MATCH expression. Quoted phrases and boolean operators
 *     (AND/OR/NOT) are preserved; bare tokens are stripped of FTS5 special
 *     characters and joined with OR.
 *
 *   - {@link buildSearchSQL} — assembles the SELECT against knowledge_fts.
 *     `ORDER BY rank LIMIT N` relies on FTS5's top-N short-circuit; an
 *     explicit bm25() call in the ORDER BY would force a full sort.
 *
 * The BM25 weights themselves live on the virtual table's rank config row
 * (see schema.ts), so `rank` here is already our weighted score — we do
 * not need to call bm25() explicitly.
 */

import type { SqlDb } from './driver.js'

export type BM25Row = {
  readonly chunk_id: string
  readonly rank: number
}

/**
 * Characters that break the FTS5 parser when they appear in a bare token.
 * Double-quote is not stripped because we use it to preserve phrases; we
 * only scrub it inside bare tokens below.
 */
const FTS5_SPECIAL = /[*():^+?!,;$#@%=]/g

/**
 * Split a query into tokens, preserving "double quoted" phrases as a
 * single token. Whitespace outside phrases is the delimiter. Naive but
 * sufficient — the upstream query/ package will layer richer parsing.
 */
function tokenise(query: string): string[] {
  const tokens: string[] = []
  let buf = ''
  let inPhrase = false
  for (const ch of query) {
    if (ch === '"') {
      if (inPhrase) {
        if (buf.trim() !== '') {
          tokens.push(`"${buf.trim()}"`)
        }
        buf = ''
        inPhrase = false
      } else {
        if (buf.trim() !== '') {
          tokens.push(...buf.trim().split(/\s+/))
        }
        buf = ''
        inPhrase = true
      }
      continue
    }
    if (!inPhrase && /\s/.test(ch)) {
      if (buf.trim() !== '') {
        tokens.push(...buf.trim().split(/\s+/))
      }
      buf = ''
      continue
    }
    buf += ch
  }
  if (buf.trim() !== '') {
    if (inPhrase) {
      tokens.push(`"${buf.trim()}"`)
    } else {
      tokens.push(...buf.trim().split(/\s+/))
    }
  }
  return tokens.filter((t) => t.length > 0)
}

const BOOLEAN_OPS = new Set(['AND', 'OR', 'NOT'])

/**
 * Scrub a bare token of FTS5 specials. Phrase tokens (wrapped in quotes)
 * are left alone — the FTS5 parser handles them natively.
 */
function scrubToken(token: string): string {
  if (token.startsWith('"') && token.endsWith('"')) {
    return token
  }
  return token.replace(FTS5_SPECIAL, '')
}

/**
 * Convert a natural-language query into an FTS5 MATCH expression.
 *
 * Rules:
 *   - Quoted phrases preserved as FTS5 phrases.
 *   - Boolean operators (case-insensitive AND/OR/NOT) preserved verbatim.
 *   - Bare tokens joined with `OR` to maximise recall; callers that want
 *     strict AND semantics should pass the operators themselves.
 *
 * Returns an empty string for empty / whitespace input so the caller can
 * short-circuit without hitting SQLite.
 */
export function compileFts5Query(query: string): string {
  const trimmed = query.trim()
  if (trimmed === '') return ''

  const tokens = tokenise(trimmed)
  if (tokens.length === 0) return ''

  const out: string[] = []
  for (let i = 0; i < tokens.length; i += 1) {
    const raw = tokens[i] as string
    const upper = raw.toUpperCase()
    if (BOOLEAN_OPS.has(upper)) {
      out.push(upper)
      continue
    }
    const scrubbed = scrubToken(raw)
    if (scrubbed === '' || scrubbed === '""') continue
    // Insert an implicit OR between consecutive bare tokens; skip if the
    // previous emitted token was already a boolean operator.
    const prev = out[out.length - 1]
    if (prev !== undefined && !BOOLEAN_OPS.has(prev)) {
      out.push('OR')
    }
    out.push(scrubbed)
  }

  // Drop a trailing dangling operator.
  while (out.length > 0 && BOOLEAN_OPS.has(out[out.length - 1] as string)) {
    out.pop()
  }
  return out.join(' ')
}

/**
 * Build the ORDER BY rank LIMIT N SQL for knowledge_fts.
 *
 * FTS5 can short-circuit `ORDER BY rank LIMIT N` — it stops scanning once
 * it has N matches whose rank cannot be beaten. Calling bm25() explicitly
 * in ORDER BY would defeat that optimisation. The rank configuration row
 * set at CREATE time bakes in our column weights, so `rank` here is
 * already the weighted BM25 score.
 */
export function buildBm25SQL(limit: number): string {
  return `SELECT chunk_id, rank
          FROM knowledge_fts
          WHERE knowledge_fts MATCH ?
          ORDER BY rank
          LIMIT ?`
}

/**
 * Execute a compiled FTS5 MATCH expression and return chunk ids ordered
 * by weighted BM25 rank (ascending — lower is better in FTS5).
 */
export function runBm25(db: SqlDb, expr: string, limit: number): BM25Row[] {
  if (expr.trim() === '' || limit <= 0) return []
  const rows = db.prepare(buildBm25SQL(limit)).all(expr, limit) as BM25Row[]
  return rows
}
