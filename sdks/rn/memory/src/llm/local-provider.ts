import type { InferenceBridge } from '../native/inference-bridge.js'
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

const estimateTokens = (messages: readonly Message[]): number => {
  return messages.reduce(
    (total, message) => total + Math.ceil((message.content ?? '').length / 4),
    0,
  )
}

const buildPromptMessages = (
  request: CompletionRequest,
): readonly Pick<Message, 'role' | 'content'>[] => {
  const messages: Pick<Message, 'role' | 'content'>[] = []
  const system = [request.systemStatic, request.systemDynamic, request.system].filter(
    (value): value is string => typeof value === 'string' && value.trim() !== '',
  )
  if (system.length > 0) {
    messages.push({ role: 'system', content: system.join('\n\n') })
  }
  for (const message of request.messages) {
    messages.push(
      message.content === undefined
        ? { role: message.role }
        : { role: message.role, content: message.content },
    )
  }
  return messages
}

export class LocalProvider implements Provider {
  private readonly logger: Logger

  constructor(
    private readonly bridge: InferenceBridge,
    private readonly model: string,
    logger?: Logger,
  ) {
    this.logger = logger ?? noopLogger
  }

  name(): string {
    return 'local'
  }

  modelName(): string {
    return this.model
  }

  supportsStructuredDecoding(): boolean {
    return false
  }

  async complete(request: CompletionRequest): Promise<CompletionResponse> {
    const response = await this.bridge.generate({
      messages: buildPromptMessages(request),
      ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
      ...(request.maxTokens === undefined ? {} : { maxTokens: request.maxTokens }),
      ...(request.responseFormat?.type === 'json_object' || request.jsonMode === true
        ? { responseFormat: 'json' as const }
        : {}),
    })

    return {
      content: response.content,
      toolCalls: [],
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
    return await runStructured(
      {
        call: async (messages) => {
          const response = await this.complete({
            ...request,
            messages: messages.map((message) => ({
              role: message.role as Message['role'],
              content: message.content,
            })),
            jsonMode: true,
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
      for await (const chunk of this.bridge.generateStream({
        messages: buildPromptMessages(request),
        ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
        ...(request.maxTokens === undefined ? {} : { maxTokens: request.maxTokens }),
        ...(request.responseFormat?.type === 'json_object' || request.jsonMode === true
          ? { responseFormat: 'json' as const }
          : {}),
      })) {
        if (chunk.type === 'text_delta') {
          yield { type: 'text_delta', text: chunk.text }
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
