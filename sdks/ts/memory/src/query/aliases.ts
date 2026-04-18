// SPDX-License-Identifier: Apache-2.0

/**
 * Entity alias expansion. Callers supply an AliasTable mapping a
 * surface token to one or more alternatives; the parser only expands
 * bare terms (phrase and prefix tokens are literal by design so we
 * never silently rewrite them).
 *
 * Multi-word or hyphenated alternatives are emitted as phrase tokens
 * so they match how the FTS5 porter+unicode61 tokenizer indexed the
 * source document. Single-word alternatives stay as bare terms so
 * they benefit from stemming.
 */

import type { QueryAST, Token } from './parser.js'

/**
 * Map of lowercase surface token -> alternative surface strings.
 * Alternatives are lowercased and normalised by the expander so the
 * caller can load the table from whatever canonical form they prefer.
 */
export type AliasTable = ReadonlyMap<string, readonly string[]>

/**
 * expand walks the AST tokens and substitutes each bare term that
 * appears as a key in aliasTable with the configured alternatives.
 * Phrase and prefix tokens are passed through unchanged. Deduplication
 * is keyed on (kind, text) so overlapping alternatives collapse to a
 * single emitted token.
 */
export function expand(ast: QueryAST, aliasTable: AliasTable): QueryAST {
  if (aliasTable.size === 0) return ast

  const tokens: Token[] = []
  for (const token of ast.tokens) {
    if (token.kind !== 'term') {
      tokens.push(token)
      continue
    }
    const alts = lookup(aliasTable, token.text)
    if (alts === undefined || alts.length === 0) {
      tokens.push(token)
      continue
    }

    const seen = new Set<string>()
    const expanded: Token[] = []
    for (const alt of alts) {
      const altToken = altToAliasToken(alt)
      if (altToken === undefined) continue
      const dedupeKey = `${altToken.kind}|${altToken.text}`
      if (seen.has(dedupeKey)) continue
      seen.add(dedupeKey)
      expanded.push(altToken)
    }
    if (expanded.length === 0) {
      tokens.push(token)
      continue
    }

    const first = expanded[0]
    if (first !== undefined && token.operator !== undefined) {
      const withOp: Token = { ...first, operator: token.operator }
      expanded[0] = withOp
    }
    for (const t of expanded) tokens.push(t)
  }

  return { raw: ast.raw, tokens, hasOperators: ast.hasOperators }
}

function lookup(table: AliasTable, key: string): readonly string[] | undefined {
  // Case-insensitive lookup. Tokens arrive lowercased already, but
  // accept the original surface form as a convenience in case a caller
  // passes in the pre-lower form.
  const direct = table.get(key)
  if (direct !== undefined) return direct
  const lowered = key.toLocaleLowerCase('en')
  if (lowered !== key) return table.get(lowered)
  return undefined
}

/**
 * Convert an alias alternative into a Token ready for FTS5. Multi-word
 * or punctuation-bearing alternatives become phrase tokens; single
 * alphanumeric words become bare terms. Returns undefined when the
 * alternative collapses to nothing after cleanup.
 */
function altToAliasToken(alt: string): Token | undefined {
  const lowered = alt.trim().toLocaleLowerCase('en')
  if (lowered === '') return undefined

  // Split on anything other than letters/digits to mirror the FTS5
  // porter+unicode61 tokenizer's own boundary rules.
  const fields = lowered.split(/[^\p{L}\p{N}]+/u).filter((p) => p.length > 0)
  if (fields.length === 0) return undefined
  if (fields.length === 1) {
    const only = fields[0]
    if (only === undefined) return undefined
    return { kind: 'term', text: only }
  }
  return { kind: 'phrase', text: fields.join(' ') }
}
