// SPDX-License-Identifier: Apache-2.0

/**
 * Hand-rolled OpenAI Chat Completions client. Streams SSE, maps finish
 * reasons onto the shared stop-reason taxonomy, and supports structured
 * output via the native `response_format: json_schema` field.
 */

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
import { extractJSON, runStructured, validateAgainstSchema } from './structured.js'
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
  apiKey: string
  model: string
  baseURL?: string
  defaultMaxTokens?: number
  streamIdleTimeoutMs?: number
  logger?: Logger
  http?: HttpClient
}

export type OpenAIEmbedderConfig = {
  apiKey: string
  model?: string
  baseURL?: string
  dimensions?: number
  logger?: Logger
  http?: HttpClient
}

type OpenAIChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | null
  name?: string
  tool_call_id?: string
  tool_calls?: Array<{
    id: string
    type: 'function'
    function: { name: string; arguments: string }
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

/**
 * Reasoning and gpt-5 models reject `max_tokens` + non-default `temperature`.
 * They expect `max_completion_tokens` and the default temperature (1). This
 * covers o1/o3/o4 and gpt-5* as of 2026-04.
 */
const usesMaxCompletionTokens = (model: string): boolean => {
  const m = model.toLowerCase()
  return (
    m.startsWith('gpt-5') ||
    m.startsWith('o1') ||
    m.startsWith('o3') ||
    m.startsWith('o4')
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

  constructor(cfg: OpenAIConfig) {
    if (cfg.apiKey === '') throw new LLMError('openai: apiKey required')
    if (cfg.model === '') throw new LLMError('openai: model required')
    this.apiKey = cfg.apiKey
    this.model = cfg.model
    this.baseURL = (cfg.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.defaultMaxTokens = cfg.defaultMaxTokens ?? 4096
    this.streamIdleTimeoutMs = cfg.streamIdleTimeoutMs ?? 90_000
    this.logger = cfg.logger ?? noopLogger
    this.http = cfg.http ?? defaultHttpClient
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

  async *stream(req: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = this.buildBody(req, true)
    const response = await postForStream(
      this.http,
      `${this.baseURL}/v1/chat/completions`,
      body,
      {
        headers: this.headers(),
        ...(signal ? { signal } : {}),
      },
    )
    if (response.body === null) {
      throw new TransportError('openai: stream response missing body')
    }
    yield* this.readStream(response.body, signal, req.onToolUseReady)
  }

  async complete(req: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const body = this.buildBody(req, false)
    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/v1/chat/completions`,
      body,
      {
        headers: this.headers(),
        ...(signal ? { signal } : {}),
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

  async structured(req: StructuredRequest, signal?: AbortSignal): Promise<string> {
    let schemaObj: unknown
    try {
      schemaObj = JSON.parse(req.schema)
    } catch (err) {
      throw new LLMError(
        `openai: invalid JSON schema: ${err instanceof Error ? err.message : String(err)}`,
        err,
      )
    }

    const baseRequest: CompletionRequest = {
      messages: req.messages,
      ...(req.model !== undefined ? { model: req.model } : {}),
      ...(req.system !== undefined ? { system: req.system } : {}),
      ...(req.systemStatic !== undefined ? { systemStatic: req.systemStatic } : {}),
      ...(req.systemDynamic !== undefined ? { systemDynamic: req.systemDynamic } : {}),
      ...(req.maxTokens !== undefined ? { maxTokens: req.maxTokens } : {}),
      ...(req.temperature !== undefined ? { temperature: req.temperature } : {}),
      ...(req.tools !== undefined ? { tools: req.tools } : {}),
      ...(req.extraBody !== undefined ? { extraBody: req.extraBody } : {}),
      responseFormat: {
        type: 'json_schema',
        jsonSchema: {
          name: req.schemaName ?? 'structured_response',
          strict: true,
          schema: req.schema,
        },
      },
    }
    const maxRetries =
      req.maxRetries !== undefined && req.maxRetries > 0 ? req.maxRetries : 5

    const runner = {
      call: async (
        messages: ReadonlyArray<{ role: string; content: string }>,
      ): Promise<string> => {
        const runReq: CompletionRequest = {
          ...baseRequest,
          messages: messages.map((m) => ({
            role: m.role as Message['role'],
            content: m.content,
          })),
        }
        const resp = await this.complete(runReq, signal)
        return resp.content
      },
    }

    try {
      return await runStructured(
        runner,
        req.messages.map((m) => ({
          role: String(m.role),
          content: m.content ?? '',
        })),
        req.schema,
        maxRetries,
      )
    } catch (err) {
      if (err instanceof SchemaValidationError) {
        // Try revalidating the raw payload so the caller gets a cleaner
        // reason than the wrapper's generic message.
        if (err.rawPayload !== '') {
          try {
            const parsed = JSON.parse(extractJSON(err.rawPayload))
            validateAgainstSchema(parsed, schemaObj)
            return err.rawPayload
          } catch {
            // Fall through with the original SchemaValidationError.
          }
        }
      }
      throw err
    }
  }

  private headers(): Record<string, string> {
    return {
      authorization: `Bearer ${this.apiKey}`,
    }
  }

  private buildBody(req: CompletionRequest, stream: boolean): OpenAIRequestBody {
    const model = req.model !== undefined && req.model !== '' ? req.model : this.model
    const body: OpenAIRequestBody = {
      model,
      messages: convertMessages(req),
      stream,
    }
    const maxTokens = req.maxTokens ?? this.defaultMaxTokens
    if (usesMaxCompletionTokens(model)) {
      body.max_completion_tokens = maxTokens
      // Reasoning models enforce the default temperature, so we omit it.
    } else {
      body.max_tokens = maxTokens
      if (req.temperature !== undefined) body.temperature = req.temperature
    }

    if (req.jsonMode === true && req.responseFormat === undefined) {
      body.response_format = { type: 'json_object' }
    } else if (req.responseFormat !== undefined) {
      if (req.responseFormat.type === 'json_object') {
        body.response_format = { type: 'json_object' }
      } else {
        body.response_format = {
          type: 'json_schema',
          json_schema: {
            name: req.responseFormat.jsonSchema.name,
            strict: req.responseFormat.jsonSchema.strict ?? true,
            schema: safeParseJSON(req.responseFormat.jsonSchema.schema),
          },
        }
      }
    }

    if (req.tools !== undefined && req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: safeParseJSON(t.inputSchema),
        },
      }))
    }

    if (req.extraBody !== undefined) {
      for (const [key, value] of Object.entries(req.extraBody)) {
        body[key] = value
      }
    }

    return body
  }

  private async *readStream(
    body: ReadableStream<Uint8Array>,
    signal: AbortSignal | undefined,
    onToolUseReady: ((id: string, name: string, input: string) => void) | undefined,
  ): AsyncGenerator<StreamEvent, void, void> {
    let lastSeen = Date.now()
    let idleTripped = false
    const idleTimer = setInterval(() => {
      if (Date.now() - lastSeen > this.streamIdleTimeoutMs) {
        idleTripped = true
        try {
          body.cancel().catch(() => {})
        } catch {
          // Ignored.
        }
      }
    }, Math.min(1000, this.streamIdleTimeoutMs))

    const activeTools = new Map<number, ToolCall>()
    const accUsage: Usage = { inputTokens: 0, outputTokens: 0 }
    let finishReason = ''

    try {
      for await (const evt of iterateSSE(body)) {
        if (signal?.aborted === true) {
          throw new TransportError('openai: stream aborted', signal.reason)
        }
        lastSeen = Date.now()
        if (evt.data === '' || evt.data === '[DONE]') continue
        let parsed: {
          choices?: Array<{
            index?: number
            delta?: {
              content?: string | null
              tool_calls?: Array<{
                index: number
                id?: string
                function?: { name?: string; arguments?: string }
              }>
            }
            finish_reason?: string | null
          }>
          usage?: {
            prompt_tokens?: number
            completion_tokens?: number
          }
        }
        try {
          parsed = JSON.parse(evt.data)
        } catch (err) {
          yield {
            type: 'error',
            error: new ProviderError('openai: malformed SSE payload', 0, evt.data, err),
          }
          continue
        }
        if (parsed.usage !== undefined) {
          if (parsed.usage.prompt_tokens !== undefined) {
            accUsage.inputTokens = parsed.usage.prompt_tokens
          }
          if (parsed.usage.completion_tokens !== undefined) {
            accUsage.outputTokens = parsed.usage.completion_tokens
          }
        }
        for (const choice of parsed.choices ?? []) {
          const delta = choice.delta
          if (delta?.content !== undefined && delta.content !== null && delta.content !== '') {
            yield { type: 'text_delta', text: delta.content }
          }
          if (delta?.tool_calls !== undefined) {
            for (const tc of delta.tool_calls) {
              const existing = activeTools.get(tc.index)
              if (existing === undefined) {
                const id = tc.id ?? ''
                const name = tc.function?.name ?? ''
                const fresh: ToolCall = { id, name, arguments: tc.function?.arguments ?? '' }
                activeTools.set(tc.index, fresh)
                yield { type: 'tool_call_start', toolCall: { ...fresh } }
                if (tc.function?.arguments !== undefined && tc.function.arguments !== '') {
                  yield {
                    type: 'tool_call_delta',
                    toolCall: { ...fresh },
                    text: tc.function.arguments,
                  }
                }
              } else {
                if (tc.id !== undefined && existing.id === '') existing.id = tc.id
                if (tc.function?.name !== undefined && existing.name === '') {
                  existing.name = tc.function.name
                }
                if (tc.function?.arguments !== undefined) {
                  existing.arguments += tc.function.arguments
                  yield {
                    type: 'tool_call_delta',
                    toolCall: { ...existing },
                    text: tc.function.arguments,
                  }
                }
              }
            }
          }
          if (typeof choice.finish_reason === 'string' && choice.finish_reason !== '') {
            finishReason = choice.finish_reason
          }
        }
      }
    } catch (err) {
      if (idleTripped) {
        yield { type: 'error', error: new StreamIdleTimeoutError(this.streamIdleTimeoutMs) }
        return
      }
      if (err instanceof LLMError) {
        yield { type: 'error', error: err }
        return
      }
      yield {
        type: 'error',
        error: new TransportError('openai: stream read failed', err),
      }
      return
    } finally {
      clearInterval(idleTimer)
    }

    // Close any dangling tool calls.
    for (const tc of activeTools.values()) {
      if (onToolUseReady !== undefined) {
        try {
          onToolUseReady(tc.id, tc.name, tc.arguments === '' ? '{}' : tc.arguments)
        } catch (err) {
          this.logger.warn('openai: onToolUseReady threw', {
            id: tc.id,
            name: tc.name,
            err: err instanceof Error ? err.message : String(err),
          })
        }
      }
      yield { type: 'tool_call_end', toolCall: { ...tc } }
    }
    yield { type: 'done', usage: { ...accUsage }, stopReason: normaliseFinish(finishReason) }
  }
}

export class OpenAIEmbedder implements Embedder {
  private readonly apiKey: string
  private readonly modelNameValue: string
  private readonly baseURL: string
  private readonly logger: Logger
  private readonly http: HttpClient
  private dimensionValue: number

  constructor(cfg: OpenAIEmbedderConfig) {
    if (cfg.apiKey === '') throw new LLMError('openai: apiKey required')
    this.apiKey = cfg.apiKey
    this.modelNameValue = cfg.model ?? DEFAULT_EMBED_MODEL
    this.baseURL = (cfg.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.logger = cfg.logger ?? noopLogger
    this.http = cfg.http ?? defaultHttpClient
    this.dimensionValue = cfg.dimensions ?? DEFAULT_EMBED_DIM
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
        ...(signal !== undefined ? { signal } : {}),
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
      data?: Array<{ index?: number; embedding?: number[] }>
      error?: { message?: string }
    }
    try {
      parsed = JSON.parse(text) as typeof parsed
    } catch (err) {
      throw new ProviderError(
        'openai: embed returned invalid JSON',
        response.status,
        text,
        err,
      )
    }
    if (parsed.error?.message !== undefined && parsed.error.message !== '') {
      throw new LLMError(`openai: embed error: ${parsed.error.message}`)
    }
    const entries = Array.isArray(parsed.data) ? parsed.data : []
    const out = Array.from({ length: texts.length }, () => [] as number[])
    for (const entry of entries) {
      const index = typeof entry.index === 'number' ? entry.index : 0
      if (index < 0 || index >= out.length) continue
      out[index] = Array.isArray(entry.embedding)
        ? entry.embedding.map((value) => Number(value))
        : []
    }
    const first = out.find((row) => row.length > 0)
    if (first !== undefined) {
      this.dimensionValue = first.length
    } else {
      this.logger.warn('openai: embed returned no vectors')
    }
    return out
  }
}

function isContextTooLong(body: string): boolean {
  return (
    body.includes('context_length_exceeded') ||
    body.includes('maximum context length') ||
    body.includes('tokens in your prompt')
  )
}

function normaliseFinish(reason: string): CompletionResponse['stopReason'] {
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
  choices?: Array<{
    message?: {
      content?: string | null
      tool_calls?: Array<{
        id: string
        function: { name: string; arguments: string }
      }>
    }
    finish_reason?: string
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
  }
}

function parseCompleteResponse(text: string): CompletionResponse {
  let raw: OpenAIFullResponse
  try {
    raw = JSON.parse(text) as OpenAIFullResponse
  } catch (err) {
    throw new ProviderError('openai: failed to parse response', 0, text, err)
  }
  const choice = raw.choices?.[0]
  const toolCalls: ToolCall[] = []
  for (const tc of choice?.message?.tool_calls ?? []) {
    toolCalls.push({ id: tc.id, name: tc.function.name, arguments: tc.function.arguments })
  }
  return {
    content: choice?.message?.content ?? '',
    toolCalls,
    usage: {
      inputTokens: raw.usage?.prompt_tokens ?? 0,
      outputTokens: raw.usage?.completion_tokens ?? 0,
    },
    stopReason: normaliseFinish(choice?.finish_reason ?? ''),
  }
}

function convertMessages(req: CompletionRequest): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = []
  const systemText = combinedSystem(req)
  if (systemText !== '') out.push({ role: 'system', content: systemText })
  for (const m of req.messages) {
    if (m.role === 'system') {
      if ((m.content ?? '') !== '') {
        out.push({ role: 'system', content: m.content ?? '' })
      }
      continue
    }
    if (m.role === 'assistant') {
      const msg: OpenAIChatMessage = { role: 'assistant', content: m.content ?? '' }
      if ((m.toolCalls ?? []).length > 0) {
        msg.tool_calls = (m.toolCalls ?? []).map((tc) => ({
          id: tc.id,
          type: 'function',
          function: { name: tc.name, arguments: tc.arguments },
        }))
      }
      out.push(msg)
      continue
    }
    if (m.role === 'tool') {
      out.push({
        role: 'tool',
        content: m.content ?? '',
        ...(m.toolCallId !== undefined ? { tool_call_id: m.toolCallId } : {}),
        ...(m.name !== undefined ? { name: m.name } : {}),
      })
      continue
    }
    // user
    out.push({ role: 'user', content: m.content ?? '' })
  }
  return out
}

function combinedSystem(req: CompletionRequest): string {
  const parts: string[] = []
  if (req.systemStatic !== undefined && req.systemStatic !== '') parts.push(req.systemStatic)
  if (req.systemDynamic !== undefined && req.systemDynamic !== '') parts.push(req.systemDynamic)
  if (parts.length === 0 && req.system !== undefined && req.system !== '') {
    return req.system
  }
  return parts.join('\n\n')
}

function safeParseJSON(raw: string): unknown {
  if (raw === '') return {}
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}
