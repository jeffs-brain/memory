import {
  LLMError,
  PromptTooLongError,
  ProviderError,
  SchemaValidationError,
  StreamIdleTimeoutError,
  TransportError,
} from './errors.js'
import { type HttpClient, defaultHttpClient, postForStream, postForText } from './http.js'
import { iterateSSE } from './sse.js'
import { extractJson, runStructured, validateAgainstSchema } from './structured.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Logger,
  Message,
  Provider,
  StreamEvent,
  StructuredRequest,
  ToolCall,
  Usage,
} from './types.js'
import { noopLogger } from './types.js'

const DEFAULT_BASE_URL = 'https://api.openai.com'
const DEFAULT_EMBED_MODEL = 'text-embedding-3-small'
const DEFAULT_EMBED_DIM = 1536

export type OpenAIConfig = {
  readonly apiKey: string
  readonly model: string
  readonly baseURL?: string
  readonly defaultMaxTokens?: number
  readonly streamIdleTimeoutMs?: number
  readonly logger?: Logger
  readonly http?: HttpClient
}

export type OpenAIEmbedderConfig = {
  readonly apiKey: string
  readonly model?: string
  readonly baseURL?: string
  readonly dimensions?: number
  readonly logger?: Logger
  readonly http?: HttpClient
}

type OpenAIChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | null
  name?: string
  tool_call_id?: string
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: {
      name: string
      arguments: string
    }
  }>
}

type OpenAIRequestBody = {
  model: string
  messages: OpenAIChatMessage[]
  stream: boolean
  max_tokens?: number
  max_completion_tokens?: number
  temperature?: number
  response_format?: Record<string, unknown>
  tools?: Array<{
    type: 'function'
    function: {
      name: string
      description: string
      parameters: unknown
    }
  }>
  [extra: string]: unknown
}

const usesMaxCompletionTokens = (model: string): boolean => {
  const normalised = model.toLowerCase()
  return (
    normalised.startsWith('gpt-5') ||
    normalised.startsWith('o1') ||
    normalised.startsWith('o3') ||
    normalised.startsWith('o4')
  )
}

export class OpenAIProvider implements Provider {
  private readonly apiKey: string
  private readonly model: string
  private readonly baseURL: string
  private readonly defaultMaxTokens: number
  private readonly streamIdleTimeoutMs: number
  private readonly logger: Logger
  private readonly http: HttpClient

  constructor(config: OpenAIConfig) {
    if (config.apiKey === '') throw new LLMError('openai: apiKey required')
    if (config.model === '') throw new LLMError('openai: model required')
    this.apiKey = config.apiKey
    this.model = config.model
    this.baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.defaultMaxTokens = config.defaultMaxTokens ?? 4096
    this.streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? 90_000
    this.logger = config.logger ?? noopLogger
    this.http = config.http ?? defaultHttpClient
  }

  name(): string {
    return 'openai'
  }

  modelName(): string {
    return this.model
  }

  supportsStructuredDecoding(): boolean {
    return true
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const body = this.buildBody(request, false)
    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/v1/chat/completions`,
      body,
      {
        headers: this.headers(),
        ...(signal === undefined ? {} : { signal }),
      },
    )

    if (!response.ok) {
      if (isContextTooLong(text)) {
        throw new PromptTooLongError(`openai: ${text}`)
      }
      throw new ProviderError(
        `openai: complete failed with status ${response.status}`,
        response.status,
        text,
      )
    }

    return parseCompleteResponse(text)
  }

  async structured(request: StructuredRequest, signal?: AbortSignal): Promise<string> {
    let schemaObject: unknown
    try {
      schemaObject = JSON.parse(request.schema)
    } catch (error) {
      throw new LLMError(
        `openai: invalid JSON schema: ${error instanceof Error ? error.message : String(error)}`,
        error,
      )
    }

    const baseRequest: CompletionRequest = {
      messages: request.messages,
      ...(request.model === undefined ? {} : { model: request.model }),
      ...(request.system === undefined ? {} : { system: request.system }),
      ...(request.systemStatic === undefined ? {} : { systemStatic: request.systemStatic }),
      ...(request.systemDynamic === undefined ? {} : { systemDynamic: request.systemDynamic }),
      ...(request.maxTokens === undefined ? {} : { maxTokens: request.maxTokens }),
      ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
      ...(request.tools === undefined ? {} : { tools: request.tools }),
      ...(request.reasoningEffort === undefined
        ? {}
        : { reasoningEffort: request.reasoningEffort }),
      ...(request.extraBody === undefined ? {} : { extraBody: request.extraBody }),
      responseFormat: {
        type: 'json_schema',
        jsonSchema: {
          name: request.schemaName ?? 'structured_response',
          strict: true,
          schema: request.schema,
        },
      },
    }
    const maxRetries =
      request.maxRetries !== undefined && request.maxRetries > 0 ? request.maxRetries : 5

    const runner = {
      call: async (
        messages: ReadonlyArray<{ readonly role: string; readonly content: string }>,
      ) => {
        const response = await this.complete(
          {
            ...baseRequest,
            messages: messages.map((message) => ({
              role: message.role as Message['role'],
              content: message.content,
            })),
          },
          signal,
        )
        return response.content
      },
    }

    try {
      return await runStructured(
        runner,
        request.messages.map((message) => ({
          role: String(message.role),
          content: message.content ?? '',
        })),
        request.schema,
        maxRetries,
      )
    } catch (error) {
      if (error instanceof SchemaValidationError && error.rawPayload !== '') {
        try {
          const parsed = JSON.parse(extractJson(error.rawPayload))
          validateAgainstSchema(parsed, schemaObject)
          return error.rawPayload
        } catch {
          // Keep the original error.
        }
      }
      throw error
    }
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = this.buildBody(request, true)
    const response = await postForStream(this.http, `${this.baseURL}/v1/chat/completions`, body, {
      headers: this.headers(),
      ...(signal === undefined ? {} : { signal }),
    })

    if (response.body === null) {
      throw new TransportError('openai: stream response missing body')
    }

    yield* this.readStream(response.body, signal, request.onToolUseReady)
  }

  private headers(): Record<string, string> {
    return {
      authorization: `Bearer ${this.apiKey}`,
    }
  }

  private buildBody(request: CompletionRequest, stream: boolean): OpenAIRequestBody {
    const model = request.model !== undefined && request.model !== '' ? request.model : this.model
    const body: OpenAIRequestBody = {
      model,
      messages: convertMessages(request),
      stream,
    }
    const maxTokens = request.maxTokens ?? this.defaultMaxTokens

    if (usesMaxCompletionTokens(model)) {
      body.max_completion_tokens = maxTokens
    } else {
      body.max_tokens = maxTokens
      if (request.temperature !== undefined) {
        body.temperature = request.temperature
      }
    }

    if (request.jsonMode === true && request.responseFormat === undefined) {
      body.response_format = { type: 'json_object' }
    } else if (request.responseFormat !== undefined) {
      if (request.responseFormat.type === 'json_object') {
        body.response_format = { type: 'json_object' }
      } else {
        body.response_format = {
          type: 'json_schema',
          json_schema: {
            name: request.responseFormat.jsonSchema.name,
            strict: request.responseFormat.jsonSchema.strict ?? true,
            schema: safeParseJSON(request.responseFormat.jsonSchema.schema),
          },
        }
      }
    }

    if (request.tools !== undefined && request.tools.length > 0) {
      body.tools = request.tools.map((tool) => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: safeParseJSON(tool.inputSchema),
        },
      }))
    }

    if (request.reasoningEffort !== undefined) {
      body.reasoning = { effort: request.reasoningEffort }
    }

    if (request.extraBody !== undefined) {
      for (const [key, value] of Object.entries(request.extraBody)) {
        body[key] = value
      }
    }

    return body
  }

  private async *readStream(
    body: ReadableStream<Uint8Array>,
    signal: AbortSignal | undefined,
    onToolUseReady: CompletionRequest['onToolUseReady'],
  ): AsyncGenerator<StreamEvent, void, void> {
    let lastSeen = Date.now()
    let idleTripped = false
    const idleTimer = setInterval(
      () => {
        if (Date.now() - lastSeen > this.streamIdleTimeoutMs) {
          idleTripped = true
          try {
            body.cancel().catch(() => {})
          } catch {
            // Ignore cancellation failures.
          }
        }
      },
      Math.min(1000, this.streamIdleTimeoutMs),
    )

    const activeTools = new Map<number, { id: string; name: string; arguments: string }>()
    const usage: { inputTokens: number; outputTokens: number } = {
      inputTokens: 0,
      outputTokens: 0,
    }
    let finishReason = ''

    try {
      for await (const event of iterateSSE(body)) {
        if (signal?.aborted === true) {
          throw new TransportError('openai: stream aborted', signal.reason)
        }
        lastSeen = Date.now()
        if (event.data === '' || event.data === '[DONE]') continue

        let parsed: {
          readonly choices?: Array<{
            readonly index?: number
            readonly delta?: {
              readonly content?: string | null
              readonly tool_calls?: Array<{
                readonly index: number
                readonly id?: string
                readonly function?: {
                  readonly name?: string
                  readonly arguments?: string
                }
              }>
            }
            readonly finish_reason?: string | null
          }>
          readonly usage?: {
            readonly prompt_tokens?: number
            readonly completion_tokens?: number
          }
        }

        try {
          parsed = JSON.parse(event.data) as typeof parsed
        } catch (error) {
          yield {
            type: 'error',
            error: new ProviderError('openai: malformed SSE payload', 0, event.data, error),
          }
          continue
        }

        if (parsed.usage?.prompt_tokens !== undefined) {
          usage.inputTokens = parsed.usage.prompt_tokens
        }
        if (parsed.usage?.completion_tokens !== undefined) {
          usage.outputTokens = parsed.usage.completion_tokens
        }

        for (const choice of parsed.choices ?? []) {
          if (
            choice.delta?.content !== undefined &&
            choice.delta.content !== null &&
            choice.delta.content !== ''
          ) {
            yield { type: 'text_delta', text: choice.delta.content }
          }

          for (const toolCall of choice.delta?.tool_calls ?? []) {
            const existing = activeTools.get(toolCall.index)
            if (existing === undefined) {
              const fresh: ToolCall = {
                id: toolCall.id ?? '',
                name: toolCall.function?.name ?? '',
                arguments: toolCall.function?.arguments ?? '',
              }
              activeTools.set(toolCall.index, fresh)
              yield { type: 'tool_call_start', toolCall: { ...fresh } }
              if (
                toolCall.function?.arguments !== undefined &&
                toolCall.function.arguments !== ''
              ) {
                yield {
                  type: 'tool_call_delta',
                  toolCall: { ...fresh },
                  text: toolCall.function.arguments,
                }
              }
              continue
            }

            if (toolCall.id !== undefined && existing.id === '') {
              existing.id = toolCall.id
            }
            if (toolCall.function?.name !== undefined && existing.name === '') {
              existing.name = toolCall.function.name
            }
            if (toolCall.function?.arguments !== undefined) {
              existing.arguments += toolCall.function.arguments
              yield {
                type: 'tool_call_delta',
                toolCall: { ...existing },
                text: toolCall.function.arguments,
              }
            }
          }

          if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
            finishReason = choice.finish_reason
          }
        }
      }
    } catch (error) {
      clearInterval(idleTimer)
      if (idleTripped) {
        yield {
          type: 'error',
          error: new StreamIdleTimeoutError(
            `openai: stream idle for more than ${this.streamIdleTimeoutMs}ms`,
          ),
        }
        return
      }
      if (error instanceof LLMError) {
        yield { type: 'error', error }
        return
      }
      yield { type: 'error', error: new TransportError('openai: stream read failed', error) }
      return
    } finally {
      clearInterval(idleTimer)
    }

    for (const toolCall of activeTools.values()) {
      if (onToolUseReady !== undefined) {
        try {
          onToolUseReady(
            toolCall.id,
            toolCall.name,
            toolCall.arguments === '' ? '{}' : toolCall.arguments,
          )
        } catch (error) {
          this.logger.warn('openai: onToolUseReady threw', {
            id: toolCall.id,
            name: toolCall.name,
            error: error instanceof Error ? error.message : String(error),
          })
        }
      }
      yield { type: 'tool_call_end', toolCall: { ...toolCall } }
    }

    yield { type: 'done', usage: { ...usage }, stopReason: normaliseFinish(finishReason) }
  }
}

export class OpenAIEmbedder implements Embedder {
  private readonly apiKey: string
  private readonly modelNameValue: string
  private readonly baseURL: string
  private readonly logger: Logger
  private readonly http: HttpClient
  private dimensionValue: number

  constructor(config: OpenAIEmbedderConfig) {
    if (config.apiKey === '') throw new LLMError('openai: apiKey required')
    this.apiKey = config.apiKey
    this.modelNameValue = config.model ?? DEFAULT_EMBED_MODEL
    this.baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.logger = config.logger ?? noopLogger
    this.http = config.http ?? defaultHttpClient
    this.dimensionValue = config.dimensions ?? DEFAULT_EMBED_DIM
  }

  name(): string {
    return 'openai'
  }

  model(): string {
    return this.modelNameValue
  }

  dimension(): number {
    return this.dimensionValue
  }

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return []

    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/v1/embeddings`,
      {
        input: texts,
        model: this.modelNameValue,
      },
      {
        headers: {
          authorization: `Bearer ${this.apiKey}`,
        },
        ...(signal === undefined ? {} : { signal }),
      },
    )

    if (!response.ok) {
      throw new ProviderError(
        `openai: embed failed with status ${response.status}`,
        response.status,
        text,
      )
    }

    let parsed: {
      readonly data?: Array<{
        readonly index?: number
        readonly embedding?: number[]
      }>
      readonly error?: {
        readonly message?: string
      }
    }

    try {
      parsed = JSON.parse(text) as typeof parsed
    } catch (error) {
      throw new ProviderError('openai: embed returned invalid JSON', response.status, text, error)
    }

    if (parsed.error?.message !== undefined && parsed.error.message !== '') {
      throw new LLMError(`openai: embed error: ${parsed.error.message}`)
    }

    const vectors = Array.from({ length: texts.length }, () => [] as number[])
    for (const entry of parsed.data ?? []) {
      const index = typeof entry.index === 'number' ? entry.index : 0
      if (index < 0 || index >= vectors.length) continue
      vectors[index] = Array.isArray(entry.embedding)
        ? entry.embedding.map((value) => Number(value))
        : []
    }

    const first = vectors.find((vector) => vector.length > 0)
    if (first !== undefined) {
      this.dimensionValue = first.length
    } else {
      this.logger.warn('openai: embed returned no vectors')
    }

    return vectors
  }
}

const isContextTooLong = (body: string): boolean => {
  return (
    body.includes('context_length_exceeded') ||
    body.includes('maximum context length') ||
    body.includes('tokens in your prompt')
  )
}

const normaliseFinish = (reason: string): CompletionResponse['stopReason'] => {
  switch (reason) {
    case 'stop':
      return 'end_turn'
    case 'length':
      return 'max_tokens'
    case 'tool_calls':
    case 'function_call':
      return 'tool_use'
    case 'content_filter':
    case 'stop_sequence':
      return 'stop_sequence'
    default:
      return ''
  }
}

type OpenAIFullResponse = {
  readonly choices?: Array<{
    readonly message?: {
      readonly content?: string | null
      readonly tool_calls?: Array<{
        readonly id: string
        readonly function: {
          readonly name: string
          readonly arguments: string
        }
      }>
    }
    readonly finish_reason?: string
  }>
  readonly usage?: {
    readonly prompt_tokens?: number
    readonly completion_tokens?: number
  }
}

const parseCompleteResponse = (text: string): CompletionResponse => {
  let parsed: OpenAIFullResponse
  try {
    parsed = JSON.parse(text) as OpenAIFullResponse
  } catch (error) {
    throw new ProviderError('openai: failed to parse response', 0, text, error)
  }

  const choice = parsed.choices?.[0]
  const toolCalls: ToolCall[] = []
  for (const toolCall of choice?.message?.tool_calls ?? []) {
    toolCalls.push({
      id: toolCall.id,
      name: toolCall.function.name,
      arguments: toolCall.function.arguments,
    })
  }

  return {
    content: choice?.message?.content ?? '',
    toolCalls,
    usage: {
      inputTokens: parsed.usage?.prompt_tokens ?? 0,
      outputTokens: parsed.usage?.completion_tokens ?? 0,
    },
    stopReason: normaliseFinish(choice?.finish_reason ?? ''),
  }
}

const convertMessages = (request: CompletionRequest): OpenAIChatMessage[] => {
  const out: OpenAIChatMessage[] = []
  const systemText = combinedSystem(request)
  if (systemText !== '') {
    out.push({ role: 'system', content: systemText })
  }

  for (const message of request.messages) {
    if (message.role === 'system') {
      if ((message.content ?? '') !== '') {
        out.push({ role: 'system', content: message.content ?? '' })
      }
      continue
    }

    if (message.role === 'assistant') {
      const next: OpenAIChatMessage = {
        role: 'assistant',
        content: message.content ?? '',
      }
      if ((message.toolCalls ?? []).length > 0) {
        next.tool_calls = (message.toolCalls ?? []).map((toolCall) => ({
          id: toolCall.id,
          type: 'function',
          function: {
            name: toolCall.name,
            arguments: toolCall.arguments,
          },
        }))
      }
      out.push(next)
      continue
    }

    if (message.role === 'tool') {
      out.push({
        role: 'tool',
        content: message.content ?? '',
        ...(message.toolCallId === undefined ? {} : { tool_call_id: message.toolCallId }),
        ...(message.name === undefined ? {} : { name: message.name }),
      })
      continue
    }

    out.push({ role: 'user', content: message.content ?? '' })
  }

  return out
}

const combinedSystem = (request: CompletionRequest): string => {
  const parts: string[] = []
  if (request.systemStatic !== undefined && request.systemStatic !== '')
    parts.push(request.systemStatic)
  if (request.systemDynamic !== undefined && request.systemDynamic !== '') {
    parts.push(request.systemDynamic)
  }
  if (parts.length === 0 && request.system !== undefined && request.system !== '') {
    return request.system
  }
  return parts.join('\n\n')
}

const safeParseJSON = (raw: string): unknown => {
  if (raw === '') return {}
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}
