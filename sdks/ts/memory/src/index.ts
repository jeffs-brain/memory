export const MEMORY_PACKAGE = '@jeffs-brain/memory' as const

export * from './store/index.js'
export * from './llm/index.js'
export * from './query/index.js'
// The rerank module defines its own Reranker + RerankResult types that
// collide with the llm/types re-exports above. Consumers should import
// it explicitly from '@jeffs-brain/memory/rerank' or via the module
// namespace.
export * as rerank from './rerank/index.js'
export * as retrieval from './retrieval/index.js'
// Public re-exports of the retrieval factory + the SQLite search index
// so consumers can wire the pipeline without digging into subpaths.
export {
  createRetrieval,
  reciprocalRankFusion,
  RRF_DEFAULT_K,
  type CreateRetrievalOptions,
  type AliasTable,
  type HybridMode,
  type HybridTrace,
  type Retrieval,
  type RetrievalRequest,
  type RetrievalResponse,
  type RetrievalResult,
  type RetryAttempt,
} from './retrieval/index.js'
// The SQLite SearchIndex is re-exported under a distinct name because
// `memory/types.ts` defines its own `SearchIndex` contract used by the
// memory stages. Consumers that want the SQLite index should use the
// aliased names or import from `@jeffs-brain/memory/search/index.js`.
export {
  createSearchIndex,
  type CreateSearchIndexOptions,
  type SearchIndex as SqliteSearchIndex,
  type Chunk as SqliteSearchIndexChunk,
  type BM25Result,
  type VectorResult,
} from './search/index.js'
export * from './knowledge/index.js'
export * from './memory/index.js'
export * from './ingest/index.js'

// Access-control primitives (RBAC, OpenFGA adapter, Store wrapper).
export * from './acl/index.js'

export {
  createHttpSearchIndex,
  type HttpSearchIndex,
  type HttpSearchIndexOptions,
  type HttpSearchResult,
} from './search/http.js'
