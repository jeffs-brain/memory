// SPDX-License-Identifier: Apache-2.0

/**
 * Public entry-point for the LLM provider + embedder layer. Re-exports
 * the types and the three provider implementations plus the factory.
 */

export type {
  CompletionRequest,
  CompletionResponse,
  ContentBlock,
  Embedder,
  ImageBlock,
  ImageSource,
  Logger,
  Message,
  Provider,
  Reranker,
  RerankScore,
  ResponseFormat,
  Role,
  StopReason,
  StreamEvent,
  StreamEventType,
  StructuredRequest,
  ToolCall,
  ToolDefinition,
  ToolResult,
  Usage,
} from './types.js'
export { noopLogger } from './types.js'

export {
  LLMError,
  PromptTooLongError,
  ProviderError,
  SchemaValidationError,
  StreamIdleTimeoutError,
  TransportError,
} from './errors.js'

export { LRUCache, SingleFlight } from './lru.js'
export { SSEParser, iterateSSE, type SSEEvent } from './sse.js'
export { extractJSON, runStructured, validateAgainstSchema } from './structured.js'

export { AnthropicProvider, type AnthropicConfig } from './anthropic.js'
export { OpenAIProvider, type OpenAIConfig } from './openai.js'
export { OllamaProvider, OllamaEmbedder, type OllamaConfig, type OllamaEmbedderConfig } from './ollama.js'
export { TEIEmbedder, TEIReranker, type TEIEmbedderConfig, type TEIRerankerConfig } from './tei.js'

export {
  HashEmbedder,
  createHashEmbedder,
  type HashEmbedderOptions,
} from './hashembed.js'

export {
  createProvider,
  createEmbedder,
  type ProviderConfig,
  type EmbedderConfig,
} from './factory.js'
