export type Role = 'system' | 'user' | 'assistant' | 'tool'

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

export type Message = {
  readonly role: Role
  readonly content?: string
  readonly toolCalls?: readonly ToolCall[]
  readonly toolCallId?: string
  readonly name?: string
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
  readonly system?: string
  readonly systemStatic?: string
  readonly systemDynamic?: string
  readonly maxTokens?: number
  readonly temperature?: number
  readonly jsonMode?: boolean
  readonly responseFormat?: ResponseFormat
  readonly tools?: readonly ToolDefinition[]
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
  | { readonly type: 'tool_call'; readonly toolCall: ToolCall }
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
