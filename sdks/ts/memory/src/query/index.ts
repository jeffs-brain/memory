// SPDX-License-Identifier: Apache-2.0

/**
 * Query parser entry point. Exposes the AST + compile helpers used by
 * the retrieval pipeline to turn a raw user query into an FTS5 MATCH
 * expression. The compiler understands quoted phrases, explicit AND /
 * OR / NOT operators, and prefix wildcards; stop-word filtering only
 * applies to bare-word queries.
 */

import { type AliasTable, expand } from './aliases.js'
import { type QueryAST, type Token, parseQuery, renderToken } from './parser.js'

export type { QueryAST, Token, TokenKind, BooleanOp } from './parser.js'
export type { AliasTable } from './aliases.js'
export type { TemporalExpansion } from './temporal.js'
export type { Cache, CacheOptions } from './cache.js'
export type { Distiller, DistillerOptions } from './distill.js'
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

/**
 * compileToFTS walks the AST and produces a valid FTS5 MATCH
 * expression. The default operator between consecutive tokens is OR so
 * the retriever casts a wide recall net; explicit operators on a token
 * override the default. Leading NOT tokens are dropped because FTS5
 * cannot start a MATCH expression with NOT; any other NOT is rewritten
 * into `AND NOT`, which is the only form FTS5 accepts.
 */
export function compileToFTS(ast: QueryAST): string {
  if (ast.tokens.length === 0) return ''

  const pieces: string[] = []
  for (let idx = 0; idx < ast.tokens.length; idx++) {
    const token = ast.tokens[idx]
    if (token === undefined) continue
    const rendered = renderToken(token)
    if (rendered === '') continue

    if (pieces.length === 0) {
      // First emitted token: a leading explicit NOT is illegal in FTS5,
      // so strip the operator and keep the bare piece.
      pieces.push(rendered)
      continue
    }

    let op: string = token.operator ?? 'OR'
    if (op === 'NOT') op = 'AND NOT'
    pieces.push(op, rendered)
  }

  return pieces.join(' ').trim()
}

/**
 * expandAliases is a convenience re-export so the retrieval pipeline
 * can go `parseQuery -> expandAliases -> compileToFTS` without pulling
 * from two modules.
 */
export function expandAliases(ast: QueryAST, table: AliasTable): QueryAST {
  return expand(ast, table)
}

/**
 * tokensForDebug returns a compact string representation of the AST
 * suitable for logging and trace output. Not part of the retrieval
 * contract; consumers can ignore it.
 */
export function tokensForDebug(ast: QueryAST): string {
  return ast.tokens.map((t: Token) => `${t.operator ?? ''}${t.kind}:${t.text}`.trim()).join(' ')
}
