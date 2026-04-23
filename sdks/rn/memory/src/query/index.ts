import { type AliasTable as QueryAliasTable, expand } from './aliases.js'
import { type QueryAST, type Token, parseQuery, renderToken } from './parser.js'

export type { AliasTable } from './aliases.js'
export type { Cache, CacheOptions } from './cache.js'
export type { Distiller, DistillerOptions } from './distill.js'
export type { BooleanOp, QueryAST, Token, TokenKind } from './parser.js'
export type { TemporalExpansion } from './temporal.js'

export { parseQuery } from './parser.js'
export { expand } from './aliases.js'
export {
  augmentQueryWithTemporal,
  dateSearchTokens,
  expandTemporal,
  parseQuestionDate,
  readerTodayAnchor,
  resolvedTemporalHintLine,
} from './temporal.js'
export { normalise, lowerToken } from './normalise.js'
export { EN_STOP_WORDS, NL_STOP_WORDS, STOP_WORDS, isStopWord } from './stopwords.js'
export { cacheKey, createCache } from './cache.js'
export { createDistiller } from './distill.js'
export {
  DISTILL_PROMPT_VERSION,
  DISTILL_SYSTEM_PROMPT,
  callDistillLLM,
} from './prompt.js'

export function compileToFTS(ast: QueryAST): string {
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

export const compileToFts = compileToFTS

export const expandAliases = (ast: QueryAST, table: QueryAliasTable): QueryAST => expand(ast, table)

export const tokensForDebug = (ast: QueryAST): string =>
  ast.tokens
    .map((token: Token) => `${token.operator ?? ''}${token.kind}:${token.text}`.trim())
    .join(' ')
