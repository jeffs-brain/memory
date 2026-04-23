import { createCacheKey } from './cache-key.js'
import {
  LLMError,
  PromptTooLongError,
  ProviderError,
  StreamIdleTimeoutError,
  TransportError,
} from './errors.js'
import { type HttpClient, defaultHttpClient, postForStream, postForText } from './http.js'
import { LRUCache, SingleFlight } from './lru.js'
import { runStructured } from './structured.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Logger,
  Message,
  Provider,
  StreamEvent,
  StructuredRequest,
  Usage,
} from './types.js'
import { noopLogger } from './types.js'

const DEFAULT_BASE_URL = 'http://localhost:11434'
const DEFAULT_EMBED_MODEL = 'bge-m3'
const DEFAULT_CACHE_SIZE = 10_000
const MAX_EMBED_INPUT_CHARS = 8000

type OllamaChatMessage = {
  role: string
  content: string
  tool_calls?: Array<{
    id?: string
    function: {
      name: string
      arguments: string | Record<string, unknown>
    }
  }>
}

type OllamaChatRequest = {
  model: string
  messages: OllamaChatMessage[]
  stream: boolean
  options?: Record<string, unknown>
  format?: 'json' | Record<string, unknown>
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

type MutableUsage = {
  inputTokens: number
  outputTokens: number
}

export type OllamaConfig = {
  readonly model: string
  readonly baseURL?: string
  readonly defaultMaxTokens?: number
  readonly streamIdleTimeoutMs?: number
  readonly logger?: Logger
  readonly http?: HttpClient
}

export class OllamaProvider implements Provider {
  private readonly model: string
  private readonly baseURL: string
  private readonly defaultMaxTokens: number
  private readonly streamIdleTimeoutMs: number
  private readonly logger: Logger
  private readonly http: HttpClient

  constructor(config: OllamaConfig) {
    if (config.model === '') throw new LLMError('ollama: model required')
    this.model = config.model
    this.baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.defaultMaxTokens = config.defaultMaxTokens ?? 4096
    this.streamIdleTimeoutMs = config.streamIdleTimeoutMs ?? 90_000
    this.logger = config.logger ?? noopLogger
    this.http = config.http ?? defaultHttpClient
  }

  name(): string {
    return 'ollama'
  }

  modelName(): string {
    return this.model
  }

  supportsStructuredDecoding(): boolean {
    return true
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = this.buildBody(request, true)
    const response = await postForStream(this.http, `${this.baseURL}/api/chat`, body, {
      ...(signal === undefined ? {} : { signal }),
    })
    if (response.body === null) {
      throw new TransportError('ollama: stream response missing body')
    }
    yield* this.readStream(response.body, signal)
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const body = this.buildBody(request, false)
    const { response, text } = await postForText(this.http, `${this.baseURL}/api/chat`, body, {
      ...(signal === undefined ? {} : { signal }),
    })

    if (!response.ok) {
      if (isContextTooLong(text)) {
        throw new PromptTooLongError(`ollama: ${text}`)
      }
      throw new ProviderError(
        `ollama: complete failed with status ${response.status}`,
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
        `ollama: invalid JSON schema: ${error instanceof Error ? error.message : String(error)}`,
        error,
      )
    }

    const maxRetries =
      request.maxRetries !== undefined && request.maxRetries > 0 ? request.maxRetries : 5

    return await runStructured(
      {
        call: async (messages) => {
          const response = await this.complete(
            {
              messages: messages.map((message) => ({
                role: message.role as Message['role'],
                content: message.content,
              })),
              ...(request.model === undefined ? {} : { model: request.model }),
              ...(request.system === undefined ? {} : { system: request.system }),
              ...(request.systemStatic === undefined ? {} : { systemStatic: request.systemStatic }),
              ...(request.systemDynamic === undefined
                ? {}
                : { systemDynamic: request.systemDynamic }),
              ...(request.maxTokens === undefined ? {} : { maxTokens: request.maxTokens }),
              ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
              ...(request.tools === undefined ? {} : { tools: request.tools }),
              extraBody: {
                ...(request.extraBody ?? {}),
                format: schemaObject,
              },
            },
            signal,
          )
          return response.content
        },
      },
      request.messages.map((message) => ({
        role: String(message.role),
        content: message.content ?? '',
      })),
      request.schema,
      maxRetries,
    )
  }

  private buildBody(request: CompletionRequest, stream: boolean): OllamaChatRequest {
    const body: OllamaChatRequest = {
      model: request.model !== undefined && request.model !== '' ? request.model : this.model,
      messages: convertMessages(request),
      stream,
    }

    const options: Record<string, unknown> = {
      num_predict: request.maxTokens ?? this.defaultMaxTokens,
    }
    if (request.temperature !== undefined) {
      options.temperature = request.temperature
    }
    body.options = options

    if (request.jsonMode === true && request.responseFormat === undefined) {
      body.format = 'json'
    } else if (request.responseFormat?.type === 'json_schema') {
      body.format = safeParseObject(request.responseFormat.jsonSchema.schema)
    }

    if (request.tools !== undefined && request.tools.length > 0) {
      body.tools = request.tools.map((tool) => ({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: safeParseObject(tool.inputSchema),
        },
      }))
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

    const reader = body.getReader()
    const decoder = new TextDecoder('utf-8')
    const usage: MutableUsage = { inputTokens: 0, outputTokens: 0 }
    let buffer = ''
    let stopReason: CompletionResponse['stopReason'] = ''

    try {
      for (;;) {
        if (signal?.aborted === true) {
          throw new TransportError('ollama: stream aborted', signal.reason)
        }

        const { value, done } = await reader.read()
        if (done) break
        lastSeen = Date.now()
        buffer += decoder.decode(value, { stream: true })

        let newline = buffer.indexOf('\n')
        while (newline !== -1) {
          const line = buffer.slice(0, newline).trim()
          buffer = buffer.slice(newline + 1)
          newline = buffer.indexOf('\n')
          if (line === '') continue

          let parsed: {
            message?: { role?: string; content?: string }
            done?: boolean
            done_reason?: string
            prompt_eval_count?: number
            eval_count?: number
            error?: string
          }
          try {
            parsed = JSON.parse(line) as typeof parsed
          } catch (error) {
            yield {
              type: 'error',
              error: new ProviderError('ollama: malformed chunk', 0, line, error),
            }
            continue
          }

          if (typeof parsed.error === 'string' && parsed.error !== '') {
            yield {
              type: 'error',
              error: isContextTooLong(parsed.error)
                ? new PromptTooLongError(parsed.error)
                : new ProviderError(parsed.error, 0, line),
            }
            return
          }

          if (typeof parsed.message?.content === 'string' && parsed.message.content !== '') {
            yield { type: 'text_delta', text: parsed.message.content }
          }
          if (parsed.prompt_eval_count !== undefined) {
            usage.inputTokens = parsed.prompt_eval_count
          }
          if (parsed.eval_count !== undefined) {
            usage.outputTokens = parsed.eval_count
          }
          if (parsed.done === true) {
            stopReason = normaliseReason(parsed.done_reason ?? '')
            yield { type: 'done', usage: { ...usage }, stopReason }
            return
          }
        }
      }

      if (buffer.trim() !== '') {
        this.logger.debug('ollama: trailing buffer discarded', { len: buffer.length })
      }
      yield { type: 'done', usage: { ...usage }, stopReason }
    } catch (error) {
      if (idleTripped) {
        yield {
          type: 'error',
          error: new StreamIdleTimeoutError(
            `ollama: stream idle for more than ${this.streamIdleTimeoutMs}ms`,
          ),
        }
        return
      }
      if (error instanceof LLMError) {
        yield { type: 'error', error }
        return
      }
      yield { type: 'error', error: new TransportError('ollama: stream read failed', error) }
    } finally {
      try {
        reader.releaseLock()
      } catch {
        // Ignore release failures.
      }
      clearInterval(idleTimer)
    }
  }
}

const normaliseReason = (reason: string): CompletionResponse['stopReason'] => {
  switch (reason) {
    case 'stop':
      return 'end_turn'
    case 'length':
      return 'max_tokens'
    default:
      return ''
  }
}

const isContextTooLong = (body: string): boolean => {
  return body.includes('context') && (body.includes('too long') || body.includes('exceed'))
}

const parseCompleteResponse = (text: string): CompletionResponse => {
  let raw: {
    message?: { content?: string }
    done_reason?: string
    prompt_eval_count?: number
    eval_count?: number
  }
  try {
    raw = JSON.parse(text) as typeof raw
  } catch (error) {
    throw new ProviderError('ollama: failed to parse response', 0, text, error)
  }

  return {
    content: raw.message?.content ?? '',
    toolCalls: [],
    usage: {
      inputTokens: raw.prompt_eval_count ?? 0,
      outputTokens: raw.eval_count ?? 0,
    },
    stopReason: normaliseReason(raw.done_reason ?? ''),
  }
}

const convertMessages = (request: CompletionRequest): OllamaChatMessage[] => {
  const out: OllamaChatMessage[] = []
  const system = combinedSystem(request)
  if (system !== '') {
    out.push({ role: 'system', content: system })
  }
  for (const message of request.messages) {
    out.push({ role: message.role, content: message.content ?? '' })
  }
  return out
}

const combinedSystem = (request: CompletionRequest): string => {
  const parts: string[] = []
  if (request.systemStatic !== undefined && request.systemStatic !== '') {
    parts.push(request.systemStatic)
  }
  if (request.systemDynamic !== undefined && request.systemDynamic !== '') {
    parts.push(request.systemDynamic)
  }
  if (parts.length === 0 && request.system !== undefined && request.system !== '') {
    return request.system
  }
  return parts.join('\n\n')
}

const safeParseObject = (raw: string): Record<string, unknown> => {
  if (raw === '') return {}
  try {
    const parsed = JSON.parse(raw) as unknown
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    // Ignore parse failures.
  }
  return {}
}

export type OllamaEmbedderConfig = {
  readonly model?: string
  readonly baseURL?: string
  readonly cacheSize?: number
  readonly logger?: Logger
  readonly http?: HttpClient
}

export class OllamaEmbedder implements Embedder {
  private readonly modelNameValue: string
  private readonly baseURL: string
  private readonly http: HttpClient
  private readonly logger: Logger
  private readonly cache: LRUCache<string, number[]> | null
  private readonly singleFlight = new SingleFlight<string, number[]>()
  private dimensionValue = 0
  private batchSupported: boolean | null = null

  constructor(config: OllamaEmbedderConfig = {}) {
    this.modelNameValue = config.model ?? DEFAULT_EMBED_MODEL
    this.baseURL = (config.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.http = config.http ?? defaultHttpClient
    this.logger = config.logger ?? noopLogger
    const size = config.cacheSize ?? DEFAULT_CACHE_SIZE
    this.cache = size > 0 ? new LRUCache<string, number[]>(size) : null
  }

  name(): string {
    return 'ollama-embed'
  }

  model(): string {
    return this.modelNameValue
  }

  dimension(): number {
    return this.dimensionValue
  }

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return []

    const out: (number[] | undefined)[] = new Array(texts.length)
    const misses: Array<{ idx: number; text: string; key: string }> = []

    for (let index = 0; index < texts.length; index += 1) {
      const text = sanitise(texts[index] ?? '')
      const key = createCacheKey(this.modelNameValue, text)
      if (this.cache !== null) {
        const cached = this.cache.get(key)
        if (cached !== undefined) {
          out[index] = [...cached]
          continue
        }
      }
      misses.push({ idx: index, text, key })
    }

    if (misses.length > 0) {
      await Promise.all(
        misses.map(async ({ idx, text, key }) => {
          const vector = await this.singleFlight.do(key, () => this.fetchOne(text, signal))
          if (this.cache !== null) {
            this.cache.set(key, vector)
          }
          out[idx] = [...vector]
        }),
      )
    }

    for (let index = 0; index < out.length; index += 1) {
      if (out[index] === undefined) {
        out[index] = this.dimensionValue > 0 ? new Array(this.dimensionValue).fill(0) : []
      }
    }

    return out as number[][]
  }

  private async fetchOne(text: string, signal: AbortSignal | undefined): Promise<number[]> {
    const vectors = await this.fetchBatch([text], signal)
    if (vectors.length !== 1) {
      throw new ProviderError(`ollama: embed returned ${vectors.length} vectors for 1 input`, 0)
    }
    return vectors[0] as number[]
  }

  private async fetchBatch(texts: string[], signal: AbortSignal | undefined): Promise<number[][]> {
    if (this.batchSupported !== false) {
      const result = await this.tryBatch(texts, signal)
      if (result !== null) {
        this.batchSupported = true
        this.recordDimension(result[0])
        return result
      }
      this.batchSupported = false
    }

    const out: number[][] = []
    for (const text of texts) {
      out.push(await this.fetchPerItem(text, signal))
    }
    if (out.length > 0) {
      this.recordDimension(out[0])
    }
    return out
  }

  private async tryBatch(
    texts: string[],
    signal: AbortSignal | undefined,
  ): Promise<number[][] | null> {
    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/api/embed`,
      { model: this.modelNameValue, input: texts },
      { ...(signal === undefined ? {} : { signal }) },
    )

    if (response.status === 404 || response.status === 405) {
      this.logger.debug('ollama: /api/embed unsupported, falling back', {
        status: response.status,
      })
      return null
    }

    if (!response.ok) {
      throw new ProviderError(
        `ollama: /api/embed failed with status ${response.status}`,
        response.status,
        text,
      )
    }

    let parsed: { embeddings?: number[][]; error?: string }
    try {
      parsed = JSON.parse(text) as typeof parsed
    } catch (error) {
      throw new ProviderError(
        'ollama: /api/embed returned invalid JSON',
        response.status,
        text,
        error,
      )
    }

    if (typeof parsed.error === 'string' && parsed.error !== '') {
      throw new ProviderError(`ollama: /api/embed error: ${parsed.error}`, response.status, text)
    }
    if (!Array.isArray(parsed.embeddings)) {
      throw new ProviderError('ollama: /api/embed missing embeddings', response.status, text)
    }
    if (parsed.embeddings.length !== texts.length) {
      throw new ProviderError(
        `ollama: /api/embed returned ${parsed.embeddings.length} vectors for ${texts.length} inputs`,
        response.status,
        text,
      )
    }
    return parsed.embeddings
  }

  private async fetchPerItem(text: string, signal: AbortSignal | undefined): Promise<number[]> {
    const { response, text: body } = await postForText(
      this.http,
      `${this.baseURL}/api/embeddings`,
      { model: this.modelNameValue, prompt: text },
      { ...(signal === undefined ? {} : { signal }) },
    )

    if (!response.ok) {
      throw new ProviderError(
        `ollama: /api/embeddings failed with status ${response.status}`,
        response.status,
        body,
      )
    }

    let parsed: { embedding?: number[]; error?: string }
    try {
      parsed = JSON.parse(body) as typeof parsed
    } catch (error) {
      throw new ProviderError(
        'ollama: /api/embeddings returned invalid JSON',
        response.status,
        body,
        error,
      )
    }

    if (typeof parsed.error === 'string' && parsed.error !== '') {
      throw new ProviderError(
        `ollama: /api/embeddings error: ${parsed.error}`,
        response.status,
        body,
      )
    }
    if (!Array.isArray(parsed.embedding) || parsed.embedding.length === 0) {
      throw new ProviderError(
        'ollama: /api/embeddings returned empty vector',
        response.status,
        body,
      )
    }
    return parsed.embedding
  }

  private recordDimension(sample: number[] | undefined): void {
    if (sample !== undefined && sample.length > 0 && this.dimensionValue === 0) {
      this.dimensionValue = sample.length
    }
  }
}

const sanitise = (value: string): string => {
  if (value === '') return ''
  let out = value
  if (out.includes('\x00')) {
    out = out.replaceAll('\x00', '')
  }
  if (out.length > MAX_EMBED_INPUT_CHARS) {
    out = out.slice(0, MAX_EMBED_INPUT_CHARS)
  }
  return out
}
