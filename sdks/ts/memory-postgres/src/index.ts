// SPDX-License-Identifier: Apache-2.0

export const MEMORY_POSTGRES_PACKAGE = '@jeffs-brain/memory-postgres' as const

export {
  PostgresStore,
  createPostgresStore,
  type PgPendingQuery,
  type PgSql,
  type PostgresStoreOptions,
} from './store.js'

export {
  PostgresSearchIndex,
  createPostgresSearchIndex,
  type PostgresBM25Result,
  type PostgresChunk,
  type PostgresHybridResult,
  type PostgresSearchIndexOptions,
  type PostgresVectorResult,
} from './search.js'

export {
  createPostgresRetriever,
  type CreatePostgresRetrieverOptions,
  type PostgresEmbedderFactory,
  type PostgresRerankerFactory,
  type PostgresRetrievalAttempt,
  type PostgresRetrievalFilter,
  type PostgresRetrievalFilterOperator,
  type PostgresRetrievalFilters,
  type PostgresRetrievalMode,
  type PostgresRetrievalPrimitive,
  type PostgresRetrievalRequest,
  type PostgresRetrievalResponse,
  type PostgresRetrievalScoreMap,
  type PostgresRetrievalSort,
  type PostgresRetrievalTrace,
  type PostgresRetrievedChunk,
  type PostgresRetriever,
  type PostgresRetrieverFactory,
  type PostgresRetrieverFactoryInput,
  type PostgresSearchLike,
} from './retrieval.js'
