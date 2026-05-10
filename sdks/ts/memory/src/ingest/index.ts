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

export { hashChunk, hashDocument, hashString } from './hash.js'

export {
  DEFAULT_CHUNK_CONFIG,
  DEFAULT_MAX_TOKENS,
  DEFAULT_MIN_TOKENS,
  DEFAULT_OVERLAP_TOKENS,
  DEFAULT_SEPARATORS,
  DEFAULT_STRATEGY,
  estimateTokens,
  validateChunkConfig,
  type ChunkConfig,
  type Strategy,
} from './chunk-config.js'

export {
  ingestDocument,
  type IngestPipelineDeps,
  type IngestPipelineInput,
  type IngestPipelineResult,
  type IngestProgress,
  type IngestProgressStage,
} from './pipeline.js'

export {
  createPipelineStateMachine,
  isValidTransition,
  migrateFromV1,
  STAGE_ORDER,
  type PipelineStage,
  type PipelineStateEntry,
  type PipelineStateStore,
  type PipelineStateMachineConfig,
  type TransitionCallback,
  type V1PipelineStateEntry,
} from './state-machine.js'

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
