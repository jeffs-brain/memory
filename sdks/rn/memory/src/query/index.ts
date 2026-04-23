import { type AliasTable, expand } from './aliases.js'
import { type QueryAST, type Token, parseQuery, renderToken } from './parser.js'

export type { AliasTable } from './aliases.js'
export type { BooleanOp, QueryAST, Token, TokenKind } from './parser.js'
export { parseQuery } from './parser.js'
export { normalise, lowerToken } from './normalise.js'
export { EN_STOP_WORDS, NL_STOP_WORDS, STOP_WORDS, isStopWord } from './stopwords.js'

export const compileToFts = (ast: QueryAST): string => {
  if (ast.tokens.length === 0) return ''
  const pieces: string[] = []
  for (let index = 0; index < ast.tokens.length; index += 1) {
    const token = ast.tokens[index]
    if (token === undefined) continue
    const rendered = renderToken(token)
    if (rendered === '') continue
    if (pieces.length === 0) {
      pieces.push(rendered)
      continue
    }
    let op: string = token.operator ?? 'OR'
    if (op === 'NOT') op = 'AND NOT'
    pieces.push(op, rendered)
  }
  return pieces.join(' ').trim()
}

export const expandAliases = (ast: QueryAST, table: AliasTable): QueryAST => expand(ast, table)

export const tokensForDebug = (ast: QueryAST): string =>
  ast.tokens
    .map((token: Token) => `${token.operator ?? ''}${token.kind}:${token.text}`.trim())
    .join(' ')
