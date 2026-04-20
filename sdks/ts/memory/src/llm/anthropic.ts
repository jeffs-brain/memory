// SPDX-License-Identifier: Apache-2.0

/**
 * Hand-rolled Anthropic Messages API client. Supports streaming via SSE
 * and structured output via tool-use: we declare a single
 * `emit_structured` tool, run a non-streaming completion, and return
 * the tool's input payload as the structured result.
 *
 * No SDK dependency — just fetch + the SSE parser in ./sse.ts.
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

export type AnthropicConfig = {
  apiKey: string
  model: string
  baseURL?: string
  /** Override the anthropic-version header. */
  apiVersion?: string
  /** Default max_tokens when the request does not specify. */
  defaultMaxTokens?: number
  /** Idle timeout for the SSE stream in ms. Default 90_000. */
  streamIdleTimeoutMs?: number
  /** Optional logger for diagnostic events. Defaults to a no-op. */
  logger?: Logger
  /** Injected HTTP client (for tests). */
  http?: HttpClient
}

type AnthropicRequestBody = {
  model: string
  max_tokens: number
  stream: boolean
  system?: string | Array<{ type: 'text'; text: string; cache_control?: { type: 'ephemeral' } }>
  messages: Array<{
    role: 'user' | 'assistant'
    content: string | Array<Record<string, unknown>>
  }>
  tools?: Array<{ name: string; description: string; input_schema: unknown }>
  tool_choice?: Record<string, unknown>
  temperature?: number
  [extra: string]: unknown
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

  constructor(cfg: AnthropicConfig) {
    if (cfg.apiKey === '') throw new LLMError('anthropic: apiKey required')
    if (cfg.model === '') throw new LLMError('anthropic: model required')
    this.apiKey = cfg.apiKey
    this.model = cfg.model
    this.baseURL = (cfg.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.apiVersion = cfg.apiVersion ?? DEFAULT_VERSION
    this.defaultMaxTokens = cfg.defaultMaxTokens ?? 16384
    this.streamIdleTimeoutMs = cfg.streamIdleTimeoutMs ?? 90_000
    this.logger = cfg.logger ?? noopLogger
    this.http = cfg.http ?? defaultHttpClient
  }

  name(): string {
    return 'anthropic'
  }

  modelName(): string {
    return this.model
  }

  supportsStructuredDecoding(): boolean {
    // Anthropic has no native response_format; we fake it via tool use.
    return false
  }

  async *stream(req: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = this.buildBody(req, true)
    const response = await postForStream(this.http, `${this.baseURL}/v1/messages`, body, {
      headers: this.headers(),
      ...(signal ? { signal } : {}),
    })
    if (response.body === null) {
      throw new TransportError('anthropic: stream response missing body')
    }
    yield* this.readStream(response.body, signal, req.onToolUseReady)
  }

  async complete(req: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const body = this.buildBody(req, false)
    const { response, text } = await postForText(this.http, `${this.baseURL}/v1/messages`, body, {
      headers: this.headers(),
      ...(signal ? { signal } : {}),
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

  async structured(req: StructuredRequest, signal?: AbortSignal): Promise<string> {
    let schemaObj: unknown
    try {
      schemaObj = JSON.parse(req.schema)
    } catch (err) {
      throw new LLMError(
        `anthropic: invalid JSON schema: ${err instanceof Error ? err.message : String(err)}`,
        err,
      )
    }

    const baseTool: ToolDefinition = {
      name: STRUCTURED_TOOL_NAME,
      description:
        req.schemaName !== undefined
          ? `Emit the ${req.schemaName} payload. Call this tool exactly once.`
          : 'Emit the structured payload. Call this tool exactly once.',
      inputSchema: req.schema,
    }

    const extraInstruction =
      'Respond by calling the emit_structured tool exactly once with a JSON payload matching its input schema.'

    const maxRetries = req.maxRetries !== undefined && req.maxRetries > 0 ? req.maxRetries : 5
    let messages = req.messages.slice()
    let lastPayload = ''
    let lastReason = ''

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const completionReq: CompletionRequest = {
        messages,
        tools: [baseTool, ...(req.tools ?? [])],
        ...(req.model !== undefined ? { model: req.model } : {}),
        ...(req.system !== undefined ? { system: req.system } : {}),
        ...(req.systemStatic !== undefined ? { systemStatic: req.systemStatic } : {}),
        ...(req.systemDynamic !== undefined ? { systemDynamic: req.systemDynamic } : {}),
        ...(req.maxTokens !== undefined ? { maxTokens: req.maxTokens } : {}),
        ...(req.temperature !== undefined ? { temperature: req.temperature } : {}),
        ...(req.reasoningEffort !== undefined ? { reasoningEffort: req.reasoningEffort } : {}),
        extraBody: {
          ...(req.extraBody ?? {}),
          tool_choice: { type: 'tool', name: STRUCTURED_TOOL_NAME },
          system_suffix: extraInstruction,
        },
      }
      const resp = await this.complete(completionReq, signal)
      const call = resp.toolCalls.find((tc) => tc.name === STRUCTURED_TOOL_NAME)
      if (call === undefined) {
        lastPayload = resp.content
        lastReason = 'provider did not emit the emit_structured tool call'
      } else {
        lastPayload = call.arguments
        try {
          const parsed = JSON.parse(call.arguments)
          validateAgainstSchema(parsed, schemaObj)
          return call.arguments
        } catch (err) {
          lastReason = err instanceof Error ? err.message : String(err)
        }
      }

      this.logger.warn('anthropic: structured retry', {
        attempt,
        max: maxRetries,
        reason: lastReason,
      })

      messages = [
        ...messages,
        { role: 'assistant', content: resp.content },
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

  private buildBody(req: CompletionRequest, stream: boolean): AnthropicRequestBody {
    const body: AnthropicRequestBody = {
      model: req.model !== undefined && req.model !== '' ? req.model : this.model,
      max_tokens: req.maxTokens ?? this.defaultMaxTokens,
      stream,
      messages: convertMessages(req.messages),
    }

    const systemBlocks = buildSystemBlocks(req)
    if (systemBlocks !== null) body.system = systemBlocks

    if (req.tools !== undefined && req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({
        name: t.name,
        description: t.description,
        input_schema: safeParseJSON(t.inputSchema),
      }))
    }

    if (req.temperature !== undefined) body.temperature = req.temperature

    if (req.extraBody !== undefined) {
      for (const [key, value] of Object.entries(req.extraBody)) {
        if (key === 'system_suffix' && typeof value === 'string') {
          // Append to the system as an uncached dynamic block.
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
    onToolUseReady: ((id: string, name: string, input: string) => void) | undefined,
  ): AsyncGenerator<StreamEvent, void, void> {
    // Idle watchdog: if no data has been seen within the configured
    // window, abort with a StreamIdleTimeoutError.
    let lastSeen = Date.now()
    let idleTripped = false
    const idleTimer = setInterval(
      () => {
        if (Date.now() - lastSeen > this.streamIdleTimeoutMs) {
          idleTripped = true
          try {
            body.cancel().catch(() => {})
          } catch {
            // Ignore; the iterator will throw on the next read.
          }
        }
      },
      Math.min(1000, this.streamIdleTimeoutMs),
    )

    const active = new Map<number, ToolCall>()
    const activeInput = new Map<number, string>()
    const accUsage: Usage = { inputTokens: 0, outputTokens: 0 }
    let stopReason: CompletionResponse['stopReason'] = ''

    try {
      for await (const evt of iterateSSE(body)) {
        if (signal?.aborted === true) {
          throw new TransportError('anthropic: stream aborted', signal.reason)
        }
        lastSeen = Date.now()
        if (evt.data === '' || evt.data === '[DONE]') continue
        let parsed: Record<string, unknown>
        try {
          parsed = JSON.parse(evt.data) as Record<string, unknown>
        } catch (err) {
          yield {
            type: 'error',
            error: new ProviderError(
              `anthropic: malformed SSE payload for event ${evt.event}`,
              0,
              evt.data,
              err,
            ),
          }
          continue
        }

        switch (evt.event) {
          case 'message_start': {
            const msg = (parsed as { message?: { usage?: Partial<Usage> } }).message
            if (msg?.usage !== undefined) {
              mergeUsage(accUsage, msg.usage)
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
            const cb = block.content_block
            if (cb?.type === 'tool_use' && cb.id !== undefined && cb.name !== undefined) {
              const tc: ToolCall = { id: cb.id, name: cb.name, arguments: '' }
              active.set(index, tc)
              activeInput.set(index, '')
              yield { type: 'tool_call_start', toolCall: { ...tc } }
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
            } else if (delta.type === 'thinking_delta' && typeof delta.thinking === 'string') {
              yield { type: 'thinking_delta', text: delta.thinking }
            } else if (
              delta.type === 'input_json_delta' &&
              typeof delta.partial_json === 'string'
            ) {
              const tc = active.get(index)
              if (tc !== undefined) {
                const prev = activeInput.get(index) ?? ''
                const next = prev + delta.partial_json
                activeInput.set(index, next)
                tc.arguments = next
                yield {
                  type: 'tool_call_delta',
                  toolCall: { ...tc },
                  text: delta.partial_json,
                }
              }
            }
            break
          }
          case 'content_block_stop': {
            const block = parsed as { index?: number }
            const index = block.index ?? 0
            const tc = active.get(index)
            if (tc !== undefined) {
              const assembled = activeInput.get(index) ?? ''
              tc.arguments = assembled.length > 0 ? assembled : '{}'
              if (onToolUseReady !== undefined) {
                try {
                  onToolUseReady(tc.id, tc.name, tc.arguments)
                } catch (err) {
                  this.logger.warn('anthropic: onToolUseReady threw', {
                    id: tc.id,
                    name: tc.name,
                    err: err instanceof Error ? err.message : String(err),
                  })
                }
              }
              yield { type: 'tool_call_end', toolCall: { ...tc } }
              active.delete(index)
              activeInput.delete(index)
            }
            break
          }
          case 'message_delta': {
            const md = parsed as {
              delta?: { stop_reason?: string }
              usage?: Partial<Usage>
            }
            if (typeof md.delta?.stop_reason === 'string') {
              stopReason = normaliseStopReason(md.delta.stop_reason)
            }
            if (md.usage !== undefined) {
              mergeUsage(accUsage, md.usage)
            }
            break
          }
          case 'message_stop': {
            yield { type: 'done', usage: { ...accUsage }, stopReason }
            return
          }
          case 'error': {
            const msg =
              typeof (parsed as { error?: { message?: string } }).error?.message === 'string'
                ? (parsed as { error: { message: string } }).error.message
                : evt.data
            if (isPromptTooLong(msg)) {
              yield { type: 'error', error: new PromptTooLongError(msg) }
            } else {
              yield { type: 'error', error: new ProviderError(msg, 0, evt.data) }
            }
            return
          }
          default:
            // ping + unknown: ignore.
            break
        }
      }
      yield { type: 'done', usage: { ...accUsage }, stopReason }
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
        error: new TransportError('anthropic: stream read failed', err),
      }
    } finally {
      clearInterval(idleTimer)
    }
  }
}

function isPromptTooLong(body: string): boolean {
  return body.includes('prompt is too long') || body.includes('prompt_too_long')
}

function mergeUsage(acc: Usage, extra: Partial<Usage>): void {
  if (extra.inputTokens !== undefined && extra.inputTokens > 0) {
    acc.inputTokens = extra.inputTokens
  }
  if (extra.outputTokens !== undefined && extra.outputTokens > 0) {
    acc.outputTokens = extra.outputTokens
  }
  if (extra.cacheReadTokens !== undefined && extra.cacheReadTokens > 0) {
    acc.cacheReadTokens = extra.cacheReadTokens
  }
  if (extra.cacheCreateTokens !== undefined && extra.cacheCreateTokens > 0) {
    acc.cacheCreateTokens = extra.cacheCreateTokens
  }
}

function normaliseStopReason(reason: string): CompletionResponse['stopReason'] {
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

function parseCompleteResponse(text: string): CompletionResponse {
  let raw: AnthropicRawResponse
  try {
    raw = JSON.parse(text) as AnthropicRawResponse
  } catch (err) {
    throw new ProviderError('anthropic: failed to parse completion response', 0, text, err)
  }
  const textParts: string[] = []
  const toolCalls: ToolCall[] = []
  for (const block of raw.content ?? []) {
    if (block.type === 'text' && typeof block.text === 'string') {
      textParts.push(block.text)
    } else if (block.type === 'tool_use' && block.id !== undefined && block.name !== undefined) {
      toolCalls.push({
        id: block.id,
        name: block.name,
        arguments: block.input !== undefined ? JSON.stringify(block.input) : '{}',
      })
    }
  }
  const usage: Usage = {
    inputTokens: raw.usage?.input_tokens ?? 0,
    outputTokens: raw.usage?.output_tokens ?? 0,
    ...(raw.usage?.cache_read_input_tokens !== undefined
      ? { cacheReadTokens: raw.usage.cache_read_input_tokens }
      : {}),
    ...(raw.usage?.cache_creation_input_tokens !== undefined
      ? { cacheCreateTokens: raw.usage.cache_creation_input_tokens }
      : {}),
  }
  return {
    content: textParts.join(''),
    toolCalls,
    usage,
    stopReason: normaliseStopReason(raw.stop_reason ?? ''),
  }
}

function buildSystemBlocks(
  req: CompletionRequest,
): string | Array<{ type: 'text'; text: string; cache_control?: { type: 'ephemeral' } }> | null {
  const staticSys = req.systemStatic ?? ''
  const dynamicSys = req.systemDynamic ?? ''
  if (staticSys !== '' || dynamicSys !== '') {
    const blocks: Array<{
      type: 'text'
      text: string
      cache_control?: { type: 'ephemeral' }
    }> = []
    if (staticSys !== '') {
      blocks.push({ type: 'text', text: staticSys, cache_control: { type: 'ephemeral' } })
    }
    if (dynamicSys !== '') {
      blocks.push({ type: 'text', text: dynamicSys })
    }
    return blocks
  }
  if (req.system !== undefined && req.system !== '') return req.system
  return null
}

function convertMessages(
  messages: readonly Message[],
): Array<{ role: 'user' | 'assistant'; content: string | Array<Record<string, unknown>> }> {
  const out: Array<{
    role: 'user' | 'assistant'
    content: string | Array<Record<string, unknown>>
  }> = []
  for (const m of messages) {
    if (m.role === 'system') continue
    if (m.role === 'assistant') {
      const blocks: Array<Record<string, unknown>> = []
      if (m.content !== undefined && m.content !== '') {
        blocks.push({ type: 'text', text: m.content })
      }
      for (const tc of m.toolCalls ?? []) {
        blocks.push({
          type: 'tool_use',
          id: tc.id,
          name: tc.name,
          input: safeParseJSON(tc.arguments),
        })
      }
      out.push({ role: 'assistant', content: blocks })
      continue
    }
    if (m.role === 'tool') {
      out.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: m.toolCallId ?? '',
            content: m.content ?? '',
          },
        ],
      })
      continue
    }
    // RoleUser
    if ((m.blocks === undefined || m.blocks.length === 0) && typeof m.content === 'string') {
      out.push({ role: 'user', content: m.content })
      continue
    }
    const blocks: Array<Record<string, unknown>> = []
    for (const b of m.blocks ?? []) {
      if (b.type === 'text' && b.text !== undefined) {
        blocks.push({ type: 'text', text: b.text })
      } else if (b.type === 'tool_result' && b.toolResult !== undefined) {
        blocks.push({
          type: 'tool_result',
          tool_use_id: b.toolResult.toolCallId,
          content: b.toolResult.content,
          ...(b.toolResult.isError === true ? { is_error: true } : {}),
        })
      } else if (b.type === 'image' && b.image !== undefined) {
        blocks.push({
          type: 'image',
          source: {
            type: 'base64',
            media_type: b.image.source.mediaType,
            data: b.image.source.data,
          },
        })
      }
    }
    if (blocks.length === 0 && m.content !== undefined) {
      blocks.push({ type: 'text', text: m.content })
    }
    out.push({ role: 'user', content: blocks })
  }
  return out
}

function safeParseJSON(raw: string): unknown {
  if (raw === '') return {}
  try {
    return JSON.parse(raw)
  } catch {
    return raw
  }
}
