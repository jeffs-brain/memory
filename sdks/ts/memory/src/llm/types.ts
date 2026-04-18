/**
 * Public types for the LLM provider / embedder layer. Shared across
 * Anthropic, OpenAI, Ollama, and TEI clients. Ported from
 * apps/jeff/internal/llm/{provider,message,structured}.go.
 */

export type Role = 'system' | 'user' | 'assistant' | 'tool'

export type BlockType = 'text' | 'tool_use' | 'tool_result' | 'image'

export type ImageSource = {
  type: 'base64'
  mediaType: string
  data: string
}

export type ImageBlock = {
  source: ImageSource
}

export type ToolCall = {
  id: string
  name: string
  /** Arguments as a JSON-encoded string. */
  arguments: string
}

export type ToolResult = {
  toolCallId: string
  content: string
  isError?: boolean
}

export type ContentBlock = {
  type: BlockType
  text?: string
  toolUse?: ToolCall
  toolResult?: ToolResult
  image?: ImageBlock
}

export type Message = {
  role: Role
  content?: string
  toolCalls?: readonly ToolCall[]
  toolCallId?: string
  name?: string
  blocks?: readonly ContentBlock[]
}

export type ToolDefinition = {
  name: string
  description: string
  /** JSON Schema object, serialised as JSON string for provider portability. */
  inputSchema: string
}

export type Usage = {
  inputTokens: number
  outputTokens: number
  cacheReadTokens?: number
  cacheCreateTokens?: number
}

export type StopReason = 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence' | ''

export type StreamEventType =
  | 'text_delta'
  | 'thinking_delta'
  | 'tool_call_start'
  | 'tool_call_delta'
  | 'tool_call_end'
  | 'done'
  | 'error'

export type StreamEvent =
  | { type: 'text_delta'; text: string }
  | { type: 'thinking_delta'; text: string }
  | { type: 'tool_call_start'; toolCall: ToolCall }
  | { type: 'tool_call_delta'; toolCall: ToolCall; text: string }
  | { type: 'tool_call_end'; toolCall: ToolCall }
  | { type: 'done'; usage?: Usage; stopReason: StopReason }
  | { type: 'error'; error: Error }

export type ResponseFormatJsonObject = { type: 'json_object' }
export type ResponseFormatJsonSchema = {
  type: 'json_schema'
  jsonSchema: {
    name: string
    strict?: boolean
    /** JSON Schema serialised as string. */
    schema: string
  }
}
export type ResponseFormat = ResponseFormatJsonObject | ResponseFormatJsonSchema

export type CompletionRequest = {
  model?: string
  messages: readonly Message[]
  tools?: readonly ToolDefinition[]
  /** Full system prompt (used when SystemStatic/SystemDynamic are empty). */
  system?: string
  /** Cache-friendly static portion. */
  systemStatic?: string
  /** Per-turn dynamic portion. */
  systemDynamic?: string
  maxTokens?: number
  temperature?: number
  jsonMode?: boolean
  responseFormat?: ResponseFormat
  reasoningEffort?: 'low' | 'medium' | 'high'
  /** Opaque extras merged into the provider request body. */
  extraBody?: Record<string, unknown>
  /** Fires once per tool_use block as soon as its arguments JSON is fully assembled. */
  onToolUseReady?: (id: string, name: string, input: string) => void
}

export type CompletionResponse = {
  content: string
  toolCalls: readonly ToolCall[]
  usage: Usage
  stopReason: StopReason
}

export type StructuredRequest = Omit<CompletionRequest, 'responseFormat' | 'jsonMode'> & {
  /** JSON Schema serialised as string. Used to constrain the response. */
  schema: string
  /** Human-readable schema name for providers that surface it. */
  schemaName?: string
  /** Cap retries on schema validation failure. Default 5. */
  maxRetries?: number
}

export type Provider = {
  name(): string
  modelName(): string
  stream(req: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent>
  complete(req: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse>
  /** True when the provider constrains generation against a JSON schema natively. */
  supportsStructuredDecoding(): boolean
  /** Request a JSON-schema-constrained completion. Validates + retries on failure. */
  structured(req: StructuredRequest, signal?: AbortSignal): Promise<string>
}

export type Embedder = {
  name(): string
  model(): string
  /** Dimension of the returned vectors; zero until the first successful embed. */
  dimension(): number
  embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]>
}

export type Reranker = {
  name(): string
  rerank(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<readonly RerankScore[]>
}

export type RerankScore = {
  index: number
  score: number
}

export type Logger = {
  debug(msg: string, ctx?: Record<string, unknown>): void
  info(msg: string, ctx?: Record<string, unknown>): void
  warn(msg: string, ctx?: Record<string, unknown>): void
  error(msg: string, ctx?: Record<string, unknown>): void
}

export const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
}
