import type {
  Message,
  ResponseFormat,
  StopReason,
  ToolCall,
  ToolDefinition,
  Usage,
} from '../llm/types.js'

export type InferenceBackend = 'litert' | 'mediapipe' | 'llama_cpp'
export type EmbeddingBackend = 'onnx' | 'coreml' | 'litert'

export type ModelConfig = {
  readonly modelPath: string
  readonly multimodalProjectorPath?: string
  readonly backend: InferenceBackend
  readonly maxTokens: number
  readonly contextLength: number
  readonly quantisation: '2bit' | '4bit' | '8bit'
  readonly threads: number
  readonly useGpu: boolean
  readonly useNpu?: boolean
}

export type GenerateParams = {
  readonly messages: readonly Message[]
  readonly tools?: readonly ToolDefinition[]
  readonly toolChoice?: 'auto' | 'none' | 'required'
  readonly temperature?: number
  readonly maxTokens?: number
  readonly stopSequences?: readonly string[]
  readonly responseFormat?: 'text' | 'json' | ResponseFormat
  readonly enableThinking?: boolean
  readonly thinkingBudgetTokens?: number
  readonly thinkingBudgetMessage?: string
  readonly reasoningFormat?: 'none' | 'auto' | 'deepseek'
  readonly jinja?: boolean
  readonly chatTemplate?: string
  readonly parallelToolCalls?: boolean
}

export type GenerateResult = {
  readonly content: string
  readonly toolCalls?: readonly ToolCall[]
  readonly usage?: Usage
  readonly stopReason?: StopReason
}

export type StreamChunk =
  | { readonly type: 'text_delta'; readonly text: string }
  | { readonly type: 'thinking_delta'; readonly text: string }
  | { readonly type: 'tool_call_start'; readonly toolCall: ToolCall }
  | { readonly type: 'tool_call_delta'; readonly toolCall: ToolCall; readonly text: string }
  | { readonly type: 'tool_call_end'; readonly toolCall: ToolCall }
  | { readonly type: 'done'; readonly usage?: Usage; readonly stopReason?: StopReason }
  | { readonly type: 'error'; readonly error: string }

export type ModelInfo = {
  readonly backend: InferenceBackend
  readonly modelPath: string
  readonly contextLength: number
  readonly quantisation: string
}

export type InferenceModule = {
  loadModel(config: ModelConfig): Promise<void>
  unloadModel(): Promise<void>
  isModelLoaded(): boolean
  generate(params: GenerateParams): Promise<GenerateResult>
  generateStream?(params: GenerateParams): AsyncIterable<StreamChunk>
  getModelInfo(): ModelInfo | null
}

export type EmbeddingModelConfig = {
  readonly modelPath: string
  readonly backend: EmbeddingBackend
  readonly dimension: number
}

export type EmbeddingModule = {
  loadModel(config: EmbeddingModelConfig): Promise<void>
  unloadModel(): Promise<void>
  embed(texts: readonly string[]): Promise<number[][]>
  dimension(): number
}
