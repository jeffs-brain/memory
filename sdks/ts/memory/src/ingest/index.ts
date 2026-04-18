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
