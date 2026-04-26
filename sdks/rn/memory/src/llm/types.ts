export type Role = 'system' | 'user' | 'assistant' | 'tool'

export type BlockType = 'text' | 'tool_use' | 'tool_result' | 'image'

export type Base64ImageSource = {
  readonly type: 'base64'
  readonly mediaType: string
  readonly data: string
}

export type UrlImageSource = {
  readonly type: 'url'
  readonly url: string
}

export type ImageSource = Base64ImageSource | UrlImageSource

export type ImageBlock = {
  readonly source: ImageSource
}

export type ToolCall = {
  readonly id: string
  readonly name: string
  readonly arguments: string
}

export type ToolResult = {
  readonly toolCallId: string
  readonly content: string
  readonly isError?: boolean
}

export type ContentBlock = {
  readonly type: BlockType
  readonly text?: string
  readonly toolUse?: ToolCall
  readonly toolResult?: ToolResult
  readonly image?: ImageBlock
}

export type Message = {
  readonly role: Role
  readonly content?: string
  readonly toolCalls?: readonly ToolCall[]
  readonly toolCallId?: string
  readonly name?: string
  readonly blocks?: readonly ContentBlock[]
}

export type ToolDefinition = {
  readonly name: string
  readonly description: string
  readonly inputSchema: string
}

export type Usage = {
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheReadTokens?: number
  readonly cacheCreateTokens?: number
}

export type StopReason = 'end_turn' | 'tool_use' | 'max_tokens' | 'stop_sequence' | ''

export type StreamEventType =
  | 'text_delta'
  | 'thinking_delta'
  | 'memory_retrieval_start'
  | 'memory_retrieval_end'
  | 'tool_call'
  | 'tool_call_start'
  | 'tool_call_delta'
  | 'tool_call_end'
  | 'done'
  | 'error'

export type ResponseFormatJsonObject = { readonly type: 'json_object' }

export type ResponseFormatJsonSchema = {
  readonly type: 'json_schema'
  readonly jsonSchema: {
    readonly name: string
    readonly strict?: boolean
    readonly schema: string
  }
}

export type ResponseFormat = ResponseFormatJsonObject | ResponseFormatJsonSchema

export type CompletionRequest = {
  readonly model?: string
  readonly messages: readonly Message[]
  readonly tools?: readonly ToolDefinition[]
  readonly system?: string
  readonly systemStatic?: string
  readonly systemDynamic?: string
  readonly maxTokens?: number
  readonly temperature?: number
  readonly jsonMode?: boolean
  readonly responseFormat?: ResponseFormat
  readonly reasoningEffort?: 'low' | 'medium' | 'high'
  readonly extraBody?: Record<string, unknown>
  readonly onToolUseReady?: (id: string, name: string, input: string) => void
  readonly taskType?: string
}

export type CompletionResponse = {
  readonly content: string
  readonly toolCalls: readonly ToolCall[]
  readonly usage: Usage
  readonly stopReason: StopReason
}

export type StructuredRequest = Omit<CompletionRequest, 'responseFormat' | 'jsonMode'> & {
  readonly schema: string
  readonly schemaName?: string
  readonly maxRetries?: number
}

export type StreamEvent =
  | { readonly type: 'text_delta'; readonly text: string }
  | { readonly type: 'thinking_delta'; readonly text: string }
  | { readonly type: 'memory_retrieval_start'; readonly query: string }
  | {
      readonly type: 'memory_retrieval_end'
      readonly query: string
      readonly hitCount: number
      readonly elapsedMs: number
      readonly paths: readonly string[]
      readonly pathMode: 'fast-recall' | 'tool-enabled'
    }
  | { readonly type: 'tool_call'; readonly toolCall: ToolCall }
  | { readonly type: 'tool_call_start'; readonly toolCall: ToolCall }
  | { readonly type: 'tool_call_delta'; readonly toolCall: ToolCall; readonly text: string }
  | { readonly type: 'tool_call_end'; readonly toolCall: ToolCall }
  | { readonly type: 'done'; readonly usage?: Usage; readonly stopReason: StopReason }
  | { readonly type: 'error'; readonly error: Error }

export type Provider = {
  name(): string
  modelName(): string
  supportsStructuredDecoding(): boolean
  complete(req: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse>
  structured(req: StructuredRequest, signal?: AbortSignal): Promise<string>
  stream?(req: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent>
}

export type Embedder = {
  name(): string
  model(): string
  dimension(): number
  embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]>
}

export type RerankScore = {
  readonly index: number
  readonly score: number
}

export type Reranker = {
  name(): string
  isAvailable?(signal?: AbortSignal): Promise<boolean>
  rerank(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<readonly RerankScore[]>
}

export type Logger = {
  debug(message: string, context?: Record<string, unknown>): void
  info(message: string, context?: Record<string, unknown>): void
  warn(message: string, context?: Record<string, unknown>): void
  error(message: string, context?: Record<string, unknown>): void
}

export const noopLogger: Logger = {
  debug: () => {},
  info: () => {},
  warn: () => {},
  error: () => {},
}
