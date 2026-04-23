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
import { validateAgainstSchema } from './structured.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Logger,
  Message,
  Provider,
  StreamEvent,
  StructuredRequest,
  ToolCall,
  ToolDefinition,
  Usage,
} from './types.js'
import { noopLogger } from './types.js'

const DEFAULT_BASE_URL = 'https://api.anthropic.com'
const DEFAULT_VERSION = '2023-06-01'
const STRUCTURED_TOOL_NAME = 'emit_structured'

type MutableToolCall = {
  id: string
  name: string
  arguments: string
}

type MutableUsage = {
  inputTokens: number
  outputTokens: number
  cacheReadTokens?: number
  cacheCreateTokens?: number
}

type SystemBlock = {
  type: 'text'
  text: string
  cache_control?: { type: 'ephemeral' }
}

type AnthropicMessage = {
  role: 'user' | 'assistant'
  content: string | Array<Record<string, unknown>>
}

type AnthropicRequestBody = {
  model: string
  max_tokens: number
  stream: boolean
  system?: string | SystemBlock[]
  messages: AnthropicMessage[]
  tools?: Array<{
    name: string
    description: string
    input_schema: unknown
  }>
  tool_choice?: Record<string, unknown>
  temperature?: number
  [extra: string]: unknown
}

type AnthropicRawContentBlock = {
  type?: string
  text?: string
  id?: string
  name?: string
  input?: unknown
}

type AnthropicRawResponse = {
  content?: AnthropicRawContentBlock[]
  stop_reason?: string
  usage?: {
    input_tokens?: number
    output_tokens?: number
    cache_read_input_tokens?: number
    cache_creation_input_tokens?: number
  }
}

export type AnthropicConfig = {
  readonly apiKey: string
  readonly model: string
  readonly baseURL?: string
  readonly apiVersion?: string
  readonly defaultMaxTokens?: number
  readonly streamIdleTimeoutMs?: number
  readonly logger?: Logger
  readonly http?: HttpClient
}

export class AnthropicProvider implements Provider {
  private readonly apiKey: string
  private readonly model: string
  private readonly baseURL: string
  private readonly apiVersion: string
  private readonly defaultMaxTokens: number
  private readonly streamIdleTimeoutMs: number
  private readonly logger: Logger
  private readonly http: HttpClient

  constructor(config: AnthropicConfig) {
    if (config.apiKey === '') throw new LLMError('anthropic: apiKey required')
    if (config.model === '') throw new LLMError('anthropic: model required')
    this.apiKey = config.apiKey
    this.model = config.model
    this.baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.apiVersion = config.apiVersion ?? DEFAULT_VERSION
    this.defaultMaxTokens = config.defaultMaxTokens ?? 16384
    this.streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? 90_000
    this.logger = config.logger ?? noopLogger
    this.http = config.http ?? defaultHttpClient
  }

  name(): string {
    return 'anthropic'
  }

  modelName(): string {
    return this.model
  }

  supportsStructuredDecoding(): boolean {
    return false
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = this.buildBody(request, true)
    const response = await postForStream(this.http, `${this.baseURL}/v1/messages`, body, {
      headers: this.headers(),
      ...(signal === undefined ? {} : { signal }),
    })
    if (response.body === null) {
      throw new TransportError('anthropic: stream response missing body')
    }
    yield* this.readStream(response.body, signal, request.onToolUseReady)
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const body = this.buildBody(request, false)
    const { response, text } = await postForText(this.http, `${this.baseURL}/v1/messages`, body, {
      headers: this.headers(),
      ...(signal === undefined ? {} : { signal }),
    })

    if (!response.ok) {
      if (isPromptTooLong(text)) {
        throw new PromptTooLongError(`anthropic: ${text}`)
      }
      throw new ProviderError(
        `anthropic: complete failed with status ${response.status}`,
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
        `anthropic: invalid JSON schema: ${error instanceof Error ? error.message : String(error)}`,
        error,
      )
    }

    const baseTool: ToolDefinition = {
      name: STRUCTURED_TOOL_NAME,
      description:
        request.schemaName === undefined
          ? 'Emit the structured payload. Call this tool exactly once.'
          : `Emit the ${request.schemaName} payload. Call this tool exactly once.`,
      inputSchema: request.schema,
    }
    const extraInstruction =
      'Respond by calling the emit_structured tool exactly once with a JSON payload matching its input schema.'
    const maxRetries =
      request.maxRetries !== undefined && request.maxRetries > 0 ? request.maxRetries : 5

    let messages = request.messages.slice()
    let lastPayload = ''
    let lastReason = ''

    for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
      const response = await this.complete(
        {
          messages,
          tools: [baseTool, ...(request.tools ?? [])],
          ...(request.model === undefined ? {} : { model: request.model }),
          ...(request.system === undefined ? {} : { system: request.system }),
          ...(request.systemStatic === undefined ? {} : { systemStatic: request.systemStatic }),
          ...(request.systemDynamic === undefined ? {} : { systemDynamic: request.systemDynamic }),
          ...(request.maxTokens === undefined ? {} : { maxTokens: request.maxTokens }),
          ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
          ...(request.reasoningEffort === undefined
            ? {}
            : { reasoningEffort: request.reasoningEffort }),
          extraBody: {
            ...(request.extraBody ?? {}),
            tool_choice: { type: 'tool', name: STRUCTURED_TOOL_NAME },
            system_suffix: extraInstruction,
          },
        },
        signal,
      )

      const toolCall = response.toolCalls.find((entry) => entry.name === STRUCTURED_TOOL_NAME)
      if (toolCall === undefined) {
        lastPayload = response.content
        lastReason = 'provider did not emit the emit_structured tool call'
      } else {
        lastPayload = toolCall.arguments
        try {
          const payload = JSON.parse(toolCall.arguments)
          validateAgainstSchema(payload, schemaObject)
          return toolCall.arguments
        } catch (error) {
          lastReason = error instanceof Error ? error.message : String(error)
        }
      }

      this.logger.warn('anthropic: structured retry', {
        attempt,
        max: maxRetries,
        reason: lastReason,
      })

      messages = [
        ...messages,
        { role: 'assistant', content: response.content },
        {
          role: 'user',
          content: `Your previous response failed schema validation: ${lastReason}. Call the emit_structured tool with a valid payload.`,
        },
      ]
    }

    throw new SchemaValidationError(lastPayload, lastReason)
  }

  private headers(): Record<string, string> {
    return {
      'x-api-key': this.apiKey,
      'anthropic-version': this.apiVersion,
    }
  }

  private buildBody(request: CompletionRequest, stream: boolean): AnthropicRequestBody {
    const body: AnthropicRequestBody = {
      model: request.model !== undefined && request.model !== '' ? request.model : this.model,
      max_tokens: request.maxTokens ?? this.defaultMaxTokens,
      stream,
      messages: convertMessages(request.messages),
    }

    const system = buildSystemBlocks(request)
    if (system !== null) body.system = system

    if (request.tools !== undefined && request.tools.length > 0) {
      body.tools = request.tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        input_schema: safeParseJSON(tool.inputSchema),
      }))
    }

    if (request.temperature !== undefined) {
      body.temperature = request.temperature
    }

    if (request.extraBody !== undefined) {
      for (const [key, value] of Object.entries(request.extraBody)) {
        if (key === 'system_suffix' && typeof value === 'string') {
          if (Array.isArray(body.system)) {
            body.system = [...body.system, { type: 'text', text: value }]
          } else if (typeof body.system === 'string') {
            body.system = `${body.system}\n\n${value}`
          } else {
            body.system = value
          }
          continue
        }
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
            body.cancel().catch(() => undefined)
          } catch {
            // Ignore cancellation failures.
          }
        }
      },
      Math.min(1000, this.streamIdleTimeoutMs),
    )

    const activeTools = new Map<number, MutableToolCall>()
    const activeInput = new Map<number, string>()
    const usage: MutableUsage = { inputTokens: 0, outputTokens: 0 }
    let stopReason: CompletionResponse['stopReason'] = ''

    try {
      for await (const event of iterateSSE(body)) {
        if (signal?.aborted === true) {
          throw new TransportError('anthropic: stream aborted', signal.reason)
        }
        lastSeen = Date.now()
        if (event.data === '' || event.data === '[DONE]') continue

        let parsed: Record<string, unknown>
        try {
          parsed = JSON.parse(event.data) as Record<string, unknown>
        } catch (error) {
          yield {
            type: 'error',
            error: new ProviderError(
              `anthropic: malformed SSE payload for event ${event.event}`,
              0,
              event.data,
              error,
            ),
          }
          continue
        }

        switch (event.event) {
          case 'message_start': {
            const message = (parsed as { message?: { usage?: Partial<Usage> } }).message
            if (message?.usage !== undefined) {
              mergeUsage(usage, message.usage)
            }
            break
          }
          case 'content_block_start': {
            const block = parsed as {
              index?: number
              content_block?: {
                type?: string
                id?: string
                name?: string
              }
            }
            const index = block.index ?? 0
            if (
              block.content_block?.type === 'tool_use' &&
              block.content_block.id !== undefined &&
              block.content_block.name !== undefined
            ) {
              const toolCall: MutableToolCall = {
                id: block.content_block.id,
                name: block.content_block.name,
                arguments: '',
              }
              activeTools.set(index, toolCall)
              activeInput.set(index, '')
              yield { type: 'tool_call_start', toolCall: { ...toolCall } }
            }
            break
          }
          case 'content_block_delta': {
            const block = parsed as {
              index?: number
              delta?: {
                type?: string
                text?: string
                partial_json?: string
                thinking?: string
              }
            }
            const index = block.index ?? 0
            const delta = block.delta
            if (delta === undefined) break

            if (delta.type === 'text_delta' && typeof delta.text === 'string') {
              yield { type: 'text_delta', text: delta.text }
              break
            }

            if (delta.type === 'thinking_delta' && typeof delta.thinking === 'string') {
              yield { type: 'thinking_delta', text: delta.thinking }
              break
            }

            if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
              const toolCall = activeTools.get(index)
              if (toolCall === undefined) break
              const next = `${activeInput.get(index) ?? ''}${delta.partial_json}`
              activeInput.set(index, next)
              toolCall.arguments = next
              yield {
                type: 'tool_call_delta',
                toolCall: { ...toolCall },
                text: delta.partial_json,
              }
            }
            break
          }
          case 'content_block_stop': {
            const index = (parsed as { index?: number }).index ?? 0
            const toolCall = activeTools.get(index)
            if (toolCall === undefined) break
            toolCall.arguments = activeInput.get(index) ?? '{}'

            if (onToolUseReady !== undefined) {
              try {
                onToolUseReady(toolCall.id, toolCall.name, toolCall.arguments)
              } catch (error) {
                this.logger.warn('anthropic: onToolUseReady threw', {
                  id: toolCall.id,
                  name: toolCall.name,
                  err: error instanceof Error ? error.message : String(error),
                })
              }
            }

            yield { type: 'tool_call_end', toolCall: { ...toolCall } }
            activeTools.delete(index)
            activeInput.delete(index)
            break
          }
          case 'message_delta': {
            const message = parsed as {
              delta?: { stop_reason?: string }
              usage?: Partial<Usage>
            }
            if (typeof message.delta?.stop_reason === 'string') {
              stopReason = normaliseStopReason(message.delta.stop_reason)
            }
            if (message.usage !== undefined) {
              mergeUsage(usage, message.usage)
            }
            break
          }
          case 'message_stop':
            yield { type: 'done', usage: { ...usage }, stopReason }
            return
          case 'error': {
            const message =
              typeof (parsed as { error?: { message?: string } }).error?.message === 'string'
                ? (parsed as { error: { message: string } }).error.message
                : event.data
            yield {
              type: 'error',
              error: isPromptTooLong(message)
                ? new PromptTooLongError(message)
                : new ProviderError(message, 0, event.data),
            }
            return
          }
          default:
            break
        }
      }

      yield { type: 'done', usage: { ...usage }, stopReason }
    } catch (error) {
      if (idleTripped) {
        yield {
          type: 'error',
          error: new StreamIdleTimeoutError(
            `anthropic: stream idle for more than ${this.streamIdleTimeoutMs}ms`,
          ),
        }
        return
      }
      if (error instanceof LLMError) {
        yield { type: 'error', error }
        return
      }
      yield { type: 'error', error: new TransportError('anthropic: stream read failed', error) }
    } finally {
      clearInterval(idleTimer)
    }
  }
}

const isPromptTooLong = (body: string): boolean => {
  return body.includes('prompt is too long') || body.includes('prompt_too_long')
}

const mergeUsage = (usage: MutableUsage, extra: Partial<Usage>): void => {
  if (extra.inputTokens !== undefined && extra.inputTokens > 0) {
    usage.inputTokens = extra.inputTokens
  }
  if (extra.outputTokens !== undefined && extra.outputTokens > 0) {
    usage.outputTokens = extra.outputTokens
  }
  if (extra.cacheReadTokens !== undefined && extra.cacheReadTokens > 0) {
    usage.cacheReadTokens = extra.cacheReadTokens
  }
  if (extra.cacheCreateTokens !== undefined && extra.cacheCreateTokens > 0) {
    usage.cacheCreateTokens = extra.cacheCreateTokens
  }
}

const normaliseStopReason = (reason: string): CompletionResponse['stopReason'] => {
  switch (reason) {
    case 'end_turn':
    case 'tool_use':
    case 'max_tokens':
    case 'stop_sequence':
      return reason
    default:
      return ''
  }
}

const parseCompleteResponse = (text: string): CompletionResponse => {
  let raw: AnthropicRawResponse
  try {
    raw = JSON.parse(text) as AnthropicRawResponse
  } catch (error) {
    throw new ProviderError('anthropic: failed to parse completion response', 0, text, error)
  }

  const content: string[] = []
  const toolCalls: ToolCall[] = []
  for (const block of raw.content ?? []) {
    if (block.type === 'text' && typeof block.text === 'string') {
      content.push(block.text)
      continue
    }
    if (block.type === 'tool_use' && block.id !== undefined && block.name !== undefined) {
      toolCalls.push({
        id: block.id,
        name: block.name,
        arguments: block.input === undefined ? '{}' : JSON.stringify(block.input),
      })
    }
  }

  return {
    content: content.join(''),
    toolCalls,
    usage: {
      inputTokens: raw.usage?.input_tokens ?? 0,
      outputTokens: raw.usage?.output_tokens ?? 0,
      ...(raw.usage?.cache_read_input_tokens === undefined
        ? {}
        : { cacheReadTokens: raw.usage.cache_read_input_tokens }),
      ...(raw.usage?.cache_creation_input_tokens === undefined
        ? {}
        : { cacheCreateTokens: raw.usage.cache_creation_input_tokens }),
    },
    stopReason: normaliseStopReason(raw.stop_reason ?? ''),
  }
}

const buildSystemBlocks = (request: CompletionRequest): string | SystemBlock[] | null => {
  const staticSystem = request.systemStatic ?? ''
  const dynamicSystem = request.systemDynamic ?? ''
  if (staticSystem !== '' || dynamicSystem !== '') {
    const blocks: SystemBlock[] = []
    if (staticSystem !== '') {
      blocks.push({
        type: 'text',
        text: staticSystem,
        cache_control: { type: 'ephemeral' },
      })
    }
    if (dynamicSystem !== '') {
      blocks.push({ type: 'text', text: dynamicSystem })
    }
    return blocks
  }
  if (request.system !== undefined && request.system !== '') {
    return request.system
  }
  return null
}

const convertMessages = (messages: readonly Message[]): AnthropicMessage[] => {
  const out: AnthropicMessage[] = []
  for (const message of messages) {
    if (message.role === 'system') continue

    if (message.role === 'assistant') {
      const blocks: Array<Record<string, unknown>> = []
      if (message.content !== undefined && message.content !== '') {
        blocks.push({ type: 'text', text: message.content })
      }
      for (const toolCall of message.toolCalls ?? []) {
        blocks.push({
          type: 'tool_use',
          id: toolCall.id,
          name: toolCall.name,
          input: safeParseJSON(toolCall.arguments),
        })
      }
      out.push({ role: 'assistant', content: blocks })
      continue
    }

    if (message.role === 'tool') {
      out.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: message.toolCallId ?? '',
            content: message.content ?? '',
          },
        ],
      })
      continue
    }

    if (
      (message.blocks === undefined || message.blocks.length === 0) &&
      message.content !== undefined
    ) {
      out.push({ role: 'user', content: message.content })
      continue
    }

    const blocks: Array<Record<string, unknown>> = []
    for (const block of message.blocks ?? []) {
      if (block.type === 'text' && block.text !== undefined) {
        blocks.push({ type: 'text', text: block.text })
        continue
      }
      if (block.type === 'tool_result' && block.toolResult !== undefined) {
        blocks.push({
          type: 'tool_result',
          tool_use_id: block.toolResult.toolCallId,
          content: block.toolResult.content,
          ...(block.toolResult.isError === true ? { is_error: true } : {}),
        })
        continue
      }
      if (block.type === 'image' && block.image !== undefined) {
        blocks.push({
          type: 'image',
          source: {
            type: 'base64',
            media_type: block.image.source.mediaType,
            data: block.image.source.data,
          },
        })
      }
    }

    if (blocks.length === 0 && message.content !== undefined) {
      blocks.push({ type: 'text', text: message.content })
    }
    out.push({ role: 'user', content: blocks })
  }
  return out
}

const safeParseJSON = (raw: string): unknown => {
  if (raw === '') return {}
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}
