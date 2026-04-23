import type { QueryAST, Token } from './parser.js'

export type AliasTable = ReadonlyMap<string, readonly string[]>

export const expand = (ast: QueryAST, aliasTable: AliasTable): QueryAST => {
  if (aliasTable.size === 0) return ast
  const tokens: Token[] = []
  for (const token of ast.tokens) {
    if (token.kind !== 'term') {
      tokens.push(token)
      continue
    }

    const alternatives =
      aliasTable.get(token.text) ?? aliasTable.get(token.text.toLocaleLowerCase('en'))
    if (alternatives === undefined || alternatives.length === 0) {
      tokens.push(token)
      continue
    }

    const seen = new Set<string>()
    const expanded: Token[] = []
    for (const alternative of alternatives) {
      const next = altToAliasToken(alternative)
      if (next === undefined) continue
      const key = `${next.kind}|${next.text}`
      if (seen.has(key)) continue
      seen.add(key)
      expanded.push(next)
    }

    if (expanded.length === 0) {
      tokens.push(token)
      continue
    }

    const first = expanded[0]
    if (first !== undefined && token.operator !== undefined) {
      expanded[0] = { ...first, operator: token.operator }
    }
    tokens.push(...expanded)
  }
  return { raw: ast.raw, tokens, hasOperators: ast.hasOperators }
}

const altToAliasToken = (alternative: string): Token | undefined => {
  const lowered = alternative.trim().toLocaleLowerCase('en')
  if (lowered === '') return undefined
  const fields = lowered.split(/[^\p{L}\p{N}]+/u).filter((part) => part.length > 0)
  if (fields.length === 0) return undefined
  if (fields.length === 1) {
    const value = fields[0]
    if (value === undefined) return undefined
    return { kind: 'term', text: value }
  }
  return { kind: 'phrase', text: fields.join(' ') }
}
