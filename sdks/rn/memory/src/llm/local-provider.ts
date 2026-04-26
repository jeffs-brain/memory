import type { InferenceBridge } from '../native/inference-bridge.js'
import type { GenerateParams } from '../native/types.js'
import { runStructured } from './structured.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Logger,
  Message,
  Provider,
  StreamEvent,
  StructuredRequest,
} from './types.js'
import { noopLogger } from './types.js'

const DEFAULT_LOCAL_MAX_TOKENS = 512

const estimateTokens = (messages: readonly Message[]): number => {
  return messages.reduce(
    (total, message) => total + Math.ceil((message.content ?? '').length / 4),
    0,
  )
}

const buildPromptMessages = (
  request: CompletionRequest,
): readonly Message[] => {
  const messages: Message[] = []
  const system = [request.systemStatic, request.systemDynamic, request.system].filter(
    (value): value is string => typeof value === 'string' && value.trim() !== '',
  )
  if (system.length > 0) {
    messages.push({ role: 'system', content: system.join('\n\n') })
  }
  for (const message of request.messages) {
    messages.push(message)
  }
  return messages
}

const withStructuredInstructions = (request: StructuredRequest): StructuredRequest => {
  const schemaLabel =
    request.schemaName === undefined
      ? 'the JSON schema below'
      : `the "${request.schemaName}" JSON schema below`
  const instruction = [
    'Return only valid JSON. Do not use markdown fences, prose, or explanatory text.',
    `The response must match ${schemaLabel}.`,
    request.schema,
  ].join('\n\n')

  return {
    ...request,
    system: [request.system, instruction]
      .filter((value): value is string => typeof value === 'string' && value.trim() !== '')
      .join('\n\n'),
  }
}

const booleanExtra = (
  request: CompletionRequest,
  key: string,
): boolean | undefined => {
  const value = request.extraBody?.[key]
  return typeof value === 'boolean' ? value : undefined
}

const positiveIntegerExtra = (
  request: CompletionRequest,
  key: string,
): number | undefined => {
  const value = request.extraBody?.[key]
  return typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined
}

const stringExtra = (
  request: CompletionRequest,
  key: string,
): string | undefined => {
  const value = request.extraBody?.[key]
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : undefined
}

const reasoningFormatExtra = (
  request: CompletionRequest,
): 'none' | 'auto' | 'deepseek' | undefined => {
  const value = request.extraBody?.reasoningFormat
  return value === 'none' || value === 'auto' || value === 'deepseek' ? value : undefined
}

const localGenerationOptions = (
  request: CompletionRequest,
): {
  readonly enableThinking?: boolean
  readonly thinkingBudgetTokens?: number
  readonly thinkingBudgetMessage?: string
  readonly reasoningFormat?: 'none' | 'auto' | 'deepseek'
  readonly jinja?: boolean
  readonly chatTemplate?: string
  readonly parallelToolCalls?: boolean
} => {
  const enableThinking = booleanExtra(request, 'enableThinking')
  const thinkingBudgetTokens = positiveIntegerExtra(request, 'thinkingBudgetTokens')
  const thinkingBudgetMessage = stringExtra(request, 'thinkingBudgetMessage')
  const reasoningFormat = reasoningFormatExtra(request)
  const jinja = booleanExtra(request, 'jinja')
  const chatTemplate = stringExtra(request, 'chatTemplate')
  const parallelToolCalls = booleanExtra(request, 'parallelToolCalls')
  return {
    ...(enableThinking === undefined ? {} : { enableThinking }),
    ...(thinkingBudgetTokens === undefined ? {} : { thinkingBudgetTokens }),
    ...(thinkingBudgetMessage === undefined ? {} : { thinkingBudgetMessage }),
    ...(reasoningFormat === undefined ? {} : { reasoningFormat }),
    ...(jinja === undefined ? {} : { jinja }),
    ...(chatTemplate === undefined ? {} : { chatTemplate }),
    ...(parallelToolCalls === undefined ? {} : { parallelToolCalls }),
  }
}

const localResponseFormat = (
  request: CompletionRequest,
): 'json' | CompletionRequest['responseFormat'] | undefined => {
  if (request.responseFormat !== undefined) return request.responseFormat
  return request.jsonMode === true ? 'json' : undefined
}

const buildGenerateParams = (
  request: CompletionRequest,
  defaultMaxTokens: number,
): GenerateParams => {
  const responseFormat = localResponseFormat(request)

  return {
    messages: buildPromptMessages(request),
    ...(request.tools === undefined ? {} : { tools: request.tools, toolChoice: 'auto' }),
    ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
    maxTokens: request.maxTokens ?? defaultMaxTokens,
    ...(responseFormat === undefined ? {} : { responseFormat }),
    ...localGenerationOptions(request),
  }
}

export class LocalProvider implements Provider {
  private readonly logger: Logger
  private readonly defaultMaxTokens: number

  constructor(
    private readonly bridge: InferenceBridge,
    private readonly model: string,
    logger?: Logger,
    defaultMaxTokens = DEFAULT_LOCAL_MAX_TOKENS,
  ) {
    this.logger = logger ?? noopLogger
    this.defaultMaxTokens = defaultMaxTokens
  }

  name(): string {
    return 'local'
  }

  modelName(): string {
    return this.model
  }

  supportsStructuredDecoding(): boolean {
    return true
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const response = await this.bridge.generate(
      buildGenerateParams(request, this.defaultMaxTokens),
    )

    return {
      content: response.content,
      toolCalls: response.toolCalls ?? [],
      usage:
        response.usage ??
        ({
          inputTokens: estimateTokens(buildPromptMessages(request)),
          outputTokens: Math.ceil(response.content.length / 4),
        } as const),
      stopReason: response.stopReason ?? 'end_turn',
    }
  }

  async structured(request: StructuredRequest): Promise<string> {
    const instructedRequest = withStructuredInstructions(request)

    return await runStructured(
      {
        call: async (messages) => {
          const response = await this.complete({
            ...instructedRequest,
            messages: messages.map((message) => ({
              role: message.role as Message['role'],
              content: message.content,
            })),
            responseFormat: {
              type: 'json_schema',
              jsonSchema: {
                name: request.schemaName ?? 'structured_response',
                strict: true,
                schema: request.schema,
              },
            },
            extraBody: {
              ...(instructedRequest.extraBody ?? {}),
              enableThinking: false,
            },
          })
          return response.content
        },
      },
      request.messages.map((message) => ({
        role: String(message.role),
        content: message.content ?? '',
      })),
      request.schema,
      request.maxRetries ?? 3,
    )
  }

  async *stream(request: CompletionRequest): AsyncIterable<StreamEvent> {
    if (this.bridge.generateStream === undefined) {
      const response = await this.complete(request)
      if (response.content !== '') {
        yield { type: 'text_delta', text: response.content }
      }
      yield {
        type: 'done',
        ...(response.usage === undefined ? {} : { usage: response.usage }),
        stopReason: response.stopReason,
      }
      return
    }

    try {
      for await (const chunk of this.bridge.generateStream(
        buildGenerateParams(request, this.defaultMaxTokens),
      )) {
        if (chunk.type === 'text_delta') {
          yield { type: 'text_delta', text: chunk.text }
          continue
        }
        if (chunk.type === 'thinking_delta') {
          yield { type: 'thinking_delta', text: chunk.text }
          continue
        }
        if (chunk.type === 'tool_call_start') {
          yield { type: 'tool_call_start', toolCall: chunk.toolCall }
          continue
        }
        if (chunk.type === 'tool_call_delta') {
          yield { type: 'tool_call_delta', toolCall: chunk.toolCall, text: chunk.text }
          continue
        }
        if (chunk.type === 'tool_call_end') {
          yield { type: 'tool_call_end', toolCall: chunk.toolCall }
          continue
        }
        if (chunk.type === 'done') {
          yield {
            type: 'done',
            ...(chunk.usage === undefined ? {} : { usage: chunk.usage }),
            stopReason: chunk.stopReason ?? 'end_turn',
          }
          continue
        }
        const error = new Error(chunk.error)
        this.logger.error('local provider stream error', {
          model: this.model,
          error: error.message,
        })
        yield { type: 'error', error }
      }
    } catch (error) {
      const resolved = error instanceof Error ? error : new Error(String(error))
      this.logger.error('local provider stream failure', {
        model: this.model,
        error: resolved.message,
      })
      yield { type: 'error', error: resolved }
    }
  }
}
