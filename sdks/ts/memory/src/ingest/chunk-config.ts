// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical chunk configuration shared across Go and TypeScript SDKs.
 * Defines the ChunkConfig type, defaults, validation, and token
 * estimation. All values align with spec/INGESTION.md.
 */

/**
 * Strategy selects how the chunker identifies split points within a
 * document body.
 */
export type Strategy = 'recursive' | 'markdown' | 'code' | 'table' | 'conversation'

/** All valid strategy values for membership checks. */
const VALID_STRATEGIES: ReadonlySet<string> = new Set([
  'recursive',
  'markdown',
  'code',
  'table',
  'conversation',
])

/** Spec-mandated ceiling per chunk. */
export const DEFAULT_MAX_TOKENS = 512

/** Spec-mandated overlap between adjacent chunks. */
export const DEFAULT_OVERLAP_TOKENS = 64

/** Spec-mandated floor below which a chunk is merged into its neighbour. */
export const DEFAULT_MIN_TOKENS = 30

/** Default strategy for splitting. */
export const DEFAULT_STRATEGY: Strategy = 'recursive'

/**
 * Default separator hierarchy for the recursive strategy. Tried in
 * order until one produces chunks within the token budget.
 */
export const DEFAULT_SEPARATORS: readonly string[] = ['\n\n', '\n', '. ', ' ', '']

/**
 * ChunkConfig controls the segmentation strategy for ingested documents.
 * All fields are required — use [DEFAULT_CHUNK_CONFIG] to obtain the
 * spec defaults.
 */
export type ChunkConfig = {
  /** Target ceiling per chunk. Must be > 0. */
  readonly maxTokens: number
  /** Trailing tokens from previous chunk prepended to the next. Must be >= 0 and < maxTokens. */
  readonly overlapTokens: number
  /** Floor below which a chunk is merged into its neighbour. Must be >= 0 and < maxTokens. */
  readonly minTokens: number
  /** How split points are identified. */
  readonly strategy: Strategy
  /** Ordered list of split delimiters for the recursive strategy. */
  readonly separators: readonly string[]
}

/**
 * Spec-mandated default configuration as defined in spec/INGESTION.md.
 */
export const DEFAULT_CHUNK_CONFIG: ChunkConfig = {
  maxTokens: DEFAULT_MAX_TOKENS,
  overlapTokens: DEFAULT_OVERLAP_TOKENS,
  minTokens: DEFAULT_MIN_TOKENS,
  strategy: DEFAULT_STRATEGY,
  separators: DEFAULT_SEPARATORS,
}

/**
 * Validates a ChunkConfig against the constraints defined in
 * spec/INGESTION.md. Throws on the first violated rule with a
 * descriptive message.
 */
export const validateChunkConfig = (cfg: ChunkConfig): void => {
  if (cfg.maxTokens <= 0) {
    throw new Error(`ingest: maxTokens must be > 0, got ${String(cfg.maxTokens)}`)
  }
  if (cfg.overlapTokens < 0) {
    throw new Error(`ingest: overlapTokens must be >= 0, got ${String(cfg.overlapTokens)}`)
  }
  if (cfg.overlapTokens >= cfg.maxTokens) {
    throw new Error(
      `ingest: overlapTokens (${String(cfg.overlapTokens)}) must be < maxTokens (${String(cfg.maxTokens)})`,
    )
  }
  if (cfg.minTokens < 0) {
    throw new Error(`ingest: minTokens must be >= 0, got ${String(cfg.minTokens)}`)
  }
  if (cfg.minTokens >= cfg.maxTokens) {
    throw new Error(
      `ingest: minTokens (${String(cfg.minTokens)}) must be < maxTokens (${String(cfg.maxTokens)})`,
    )
  }
  if (!VALID_STRATEGIES.has(cfg.strategy)) {
    throw new Error(`ingest: unknown strategy "${cfg.strategy}"`)
  }
  if (cfg.strategy === 'recursive' && cfg.separators.length === 0) {
    throw new Error(`ingest: separators must be non-empty when strategy is "recursive"`)
  }
}

/**
 * Coarse token estimation using the spec formula: ceil(len / 4).
 * Produces identical results to the Go implementation for any given
 * string.
 *
 * Time: O(1). Space: O(1).
 */
export const estimateTokens = (text: string): number =>
  text.length === 0 ? 0 : Math.ceil(text.length / 4)

export class ChunkConfigError extends Error {
  override readonly name = 'ChunkConfigError'
  constructor(message: string) {
    super(message)
  }
}

/**
 * Creates a validated ChunkConfig. Applies defaults for zero/negative
 * values. Throws ChunkConfigError when invariants are violated:
 * minTokens must be less than maxTokens; overlapTokens must be less
 * than maxTokens.
 */
export const createChunkConfig = (
  maxTokens?: number,
  overlapTokens?: number,
  minTokens?: number,
): ChunkConfig => {
  const max = maxTokens !== undefined && maxTokens > 0 ? maxTokens : DEFAULT_MAX_TOKENS
  const overlap =
    overlapTokens !== undefined && overlapTokens >= 0 ? overlapTokens : DEFAULT_OVERLAP_TOKENS
  const min = minTokens !== undefined && minTokens >= 0 ? minTokens : DEFAULT_MIN_TOKENS

  if (min >= max) {
    throw new ChunkConfigError(
      `minTokens (${min}) must be less than maxTokens (${max})`,
    )
  }
  if (overlap >= max) {
    throw new ChunkConfigError(
      `overlapTokens (${overlap}) must be less than maxTokens (${max})`,
    )
  }
  return { maxTokens: max, overlapTokens: overlap, minTokens: min, strategy: DEFAULT_STRATEGY, separators: [...DEFAULT_SEPARATORS] }
}

/** Returns a ChunkConfig with the package defaults. */
export const defaultChunkConfig = (): ChunkConfig => ({
  maxTokens: DEFAULT_MAX_TOKENS,
  overlapTokens: DEFAULT_OVERLAP_TOKENS,
  minTokens: DEFAULT_MIN_TOKENS,
  strategy: DEFAULT_STRATEGY,
  separators: [...DEFAULT_SEPARATORS],
})
