// SPDX-License-Identifier: Apache-2.0

/**
 * Barrel export for the ingest pipeline + source adapters.
 */

export {
  chunkMarkdown,
  chunkPlainText,
  chunkAuto,
  countTokens,
  looksLikeMarkdown,
  type Chunk as IngestChunk,
  type ChunkOptions,
} from './chunker.js'

export {
  ingestDocument,
  type IngestPipelineDeps,
  type IngestPipelineInput,
  type IngestPipelineResult,
  type IngestProgress,
  type IngestProgressStage,
} from './pipeline.js'

export * from './sources/index.js'

export {
  createSafetyScanner,
  preprocessText,
  wrapInIsolation,
  buildSafetyMetadata,
  type SafetyScanResult,
  type SafetyScannerConfig,
  type SafetyMetadata,
  type IsolatedContent,
} from './safety.js'

export {
  FilePipelineStateStore,
  type FilePipelineStateStoreOptions,
  type PipelineStage,
  type PipelineStateEntry,
  type PipelineStateStore,
} from './state-store.js'

export {
  PostgresPipelineStateStore,
  type PostgresPipelineStateStoreOptions,
  type PgSql as PipelineStatePgSql,
} from './state-store-pg.js'
