/**
 * Public entry for the hybrid retrieval pipeline. Re-exports the
 * factory, types, and pure RRF helper so downstream packages can build
 * custom fusion layers without pulling in the whole pipeline.
 */

export {
  createRetrieval,
  type CreateRetrievalOptions,
  type Retrieval,
} from './hybrid.js'
export { reciprocalRankFusion, RRF_DEFAULT_K, type RRFCandidate } from './rrf.js'
export {
  buildTrigramIndex,
  computeTrigrams,
  forceRefreshIndex,
  queryTokens,
  sanitiseQuery,
  slugTextFor,
  strongestTerm,
  type TrigramHit,
  type TrigramIndex,
  type TrigramSourceChunk,
} from './retry.js'
export type { AliasTable } from '../query/index.js'
export type {
  HybridMode,
  HybridTrace,
  RetrievalRequest,
  RetrievalResponse,
  RetrievalResult,
  RetryAttempt,
} from './types.js'
