/**
 * Ollama chat + embeddings client. Shares the Provider interface via
 * the OpenAI-compatible chat path, and bundles an Embedder with:
 *
 *   - Single-flight deduplication (same-text concurrent calls share a
 *     single underlying fetch).
 *   - LRU cache keyed by model + sha256(text).
 *   - Two-endpoint fallback: try /api/embed (batch), fall back to
 *     /api/embeddings (per-item) on 404/405.
 */

import { createHash } from 'node:crypto'
import {
  LLMError,
  PromptTooLongError,
  ProviderError,
  StreamIdleTimeoutError,
  TransportError,
} from './errors.js'
import { type HttpClient, defaultHttpClient, postForStream, postForText } from './http.js'
import { LRUCache, SingleFlight } from './lru.js'
import { iterateSSE } from './sse.js'
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

export type OllamaConfig = {
  model: string
  baseURL?: string
  defaultMaxTokens?: number
  streamIdleTimeoutMs?: number
  logger?: Logger
  http?: HttpClient
}

type OllamaChatMessage = {
  role: string
  content: string
  tool_calls?: Array<{
    id?: string
    function: { name: string; arguments: string | Record<string, unknown> }
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
    function: { name: string; description: string; parameters: unknown }
  }>
  [extra: string]: unknown
}

export class OllamaProvider implements Provider {
  private readonly model: string
  private readonly baseURL: string
  private readonly defaultMaxTokens: number
  private readonly streamIdleTimeoutMs: number
  private readonly logger: Logger
  private readonly http: HttpClient

  constructor(cfg: OllamaConfig) {
    if (cfg.model === '') throw new LLMError('ollama: model required')
    this.model = cfg.model
    this.baseURL = (cfg.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.defaultMaxTokens = cfg.defaultMaxTokens ?? 4096
    this.streamIdleTimeoutMs = cfg.streamIdleTimeoutMs ?? 90_000
    this.logger = cfg.logger ?? noopLogger
    this.http = cfg.http ?? defaultHttpClient
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

  async *stream(req: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const body = this.buildBody(req, true)
    const response = await postForStream(this.http, `${this.baseURL}/api/chat`, body, {
      ...(signal ? { signal } : {}),
    })
    if (response.body === null) {
      throw new TransportError('ollama: stream response missing body')
    }
    yield* this.readStream(response.body, signal)
  }

  async complete(req: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const body = this.buildBody(req, false)
    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/api/chat`,
      body,
      {
        ...(signal ? { signal } : {}),
      },
    )
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

  async structured(req: StructuredRequest, signal?: AbortSignal): Promise<string> {
    let schemaObj: unknown
    try {
      schemaObj = JSON.parse(req.schema)
    } catch (err) {
      throw new LLMError(
        `ollama: invalid JSON schema: ${err instanceof Error ? err.message : String(err)}`,
        err,
      )
    }
    const maxRetries =
      req.maxRetries !== undefined && req.maxRetries > 0 ? req.maxRetries : 5
    const runner = {
      call: async (
        messages: ReadonlyArray<{ role: string; content: string }>,
      ): Promise<string> => {
        const runReq: CompletionRequest = {
          messages: messages.map((m) => ({
            role: m.role as Message['role'],
            content: m.content,
          })),
          ...(req.model !== undefined ? { model: req.model } : {}),
          ...(req.system !== undefined ? { system: req.system } : {}),
          ...(req.systemStatic !== undefined ? { systemStatic: req.systemStatic } : {}),
          ...(req.systemDynamic !== undefined ? { systemDynamic: req.systemDynamic } : {}),
          ...(req.maxTokens !== undefined ? { maxTokens: req.maxTokens } : {}),
          ...(req.temperature !== undefined ? { temperature: req.temperature } : {}),
          ...(req.tools !== undefined ? { tools: req.tools } : {}),
          extraBody: {
            ...(req.extraBody ?? {}),
            format: schemaObj,
          },
        }
        const resp = await this.complete(runReq, signal)
        return resp.content
      },
    }
    return runStructured(
      runner,
      req.messages.map((m) => ({ role: String(m.role), content: m.content ?? '' })),
      req.schema,
      maxRetries,
    )
  }

  private buildBody(req: CompletionRequest, stream: boolean): OllamaChatRequest {
    const body: OllamaChatRequest = {
      model: req.model !== undefined && req.model !== '' ? req.model : this.model,
      messages: convertMessages(req),
      stream,
    }
    const options: Record<string, unknown> = {}
    if (req.maxTokens !== undefined) options.num_predict = req.maxTokens
    else options.num_predict = this.defaultMaxTokens
    if (req.temperature !== undefined) options.temperature = req.temperature
    if (Object.keys(options).length > 0) body.options = options

    if (req.jsonMode === true && req.responseFormat === undefined) {
      body.format = 'json'
    } else if (req.responseFormat !== undefined && req.responseFormat.type === 'json_schema') {
      body.format = safeParseObject(req.responseFormat.jsonSchema.schema)
    }
    if (req.tools !== undefined && req.tools.length > 0) {
      body.tools = req.tools.map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: safeParseObject(t.inputSchema),
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
  ): AsyncGenerator<StreamEvent, void, void> {
    // Ollama streams NDJSON, not SSE. We still iterate it via the same
    // TextDecoder machinery because each line is a self-contained JSON
    // object.
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

    const reader = body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    const accUsage: Usage = { inputTokens: 0, outputTokens: 0 }
    let stopReason: CompletionResponse['stopReason'] = ''

    try {
      while (true) {
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
            parsed = JSON.parse(line)
          } catch (err) {
            yield {
              type: 'error',
              error: new ProviderError('ollama: malformed chunk', 0, line, err),
            }
            continue
          }
          if (typeof parsed.error === 'string' && parsed.error !== '') {
            if (isContextTooLong(parsed.error)) {
              yield { type: 'error', error: new PromptTooLongError(parsed.error) }
            } else {
              yield { type: 'error', error: new ProviderError(parsed.error, 0, line) }
            }
            return
          }
          if (typeof parsed.message?.content === 'string' && parsed.message.content !== '') {
            yield { type: 'text_delta', text: parsed.message.content }
          }
          if (parsed.prompt_eval_count !== undefined) {
            accUsage.inputTokens = parsed.prompt_eval_count
          }
          if (parsed.eval_count !== undefined) {
            accUsage.outputTokens = parsed.eval_count
          }
          if (parsed.done === true) {
            stopReason = normaliseReason(parsed.done_reason ?? '')
            yield { type: 'done', usage: { ...accUsage }, stopReason }
            return
          }
        }
      }
      if (buffer.trim() !== '') {
        // Trailing partial; ignore.
        this.logger.debug('ollama: trailing buffer discarded', { len: buffer.length })
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
        error: new TransportError('ollama: stream read failed', err),
      }
    } finally {
      try {
        reader.releaseLock()
      } catch {
        // Ignored — reader already released on error.
      }
      clearInterval(idleTimer)
    }
  }
}

function normaliseReason(reason: string): CompletionResponse['stopReason'] {
  switch (reason) {
    case 'stop':
      return 'end_turn'
    case 'length':
      return 'max_tokens'
    default:
      return ''
  }
}

function isContextTooLong(body: string): boolean {
  return body.includes('context') && (body.includes('too long') || body.includes('exceed'))
}

type OllamaCompleteResponse = {
  message?: { content?: string }
  done_reason?: string
  prompt_eval_count?: number
  eval_count?: number
}

function parseCompleteResponse(text: string): CompletionResponse {
  let raw: OllamaCompleteResponse
  try {
    raw = JSON.parse(text)
  } catch (err) {
    throw new ProviderError('ollama: failed to parse response', 0, text, err)
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

function convertMessages(req: CompletionRequest): OllamaChatMessage[] {
  const out: OllamaChatMessage[] = []
  const systemText = combinedSystem(req)
  if (systemText !== '') out.push({ role: 'system', content: systemText })
  for (const m of req.messages) {
    out.push({ role: m.role, content: m.content ?? '' })
  }
  return out
}

function combinedSystem(req: CompletionRequest): string {
  const parts: string[] = []
  if (req.systemStatic !== undefined && req.systemStatic !== '') parts.push(req.systemStatic)
  if (req.systemDynamic !== undefined && req.systemDynamic !== '') parts.push(req.systemDynamic)
  if (parts.length === 0 && req.system !== undefined && req.system !== '') return req.system
  return parts.join('\n\n')
}

function safeParseObject(raw: string): Record<string, unknown> {
  if (raw === '') return {}
  try {
    const parsed: unknown = JSON.parse(raw)
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>
    }
  } catch {
    // fall through
  }
  return {}
}

// ---------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------

export type OllamaEmbedderConfig = {
  model?: string
  baseURL?: string
  /** LRU cache capacity. Default 10_000. Set to 0 to disable caching. */
  cacheSize?: number
  logger?: Logger
  http?: HttpClient
}

export class OllamaEmbedder implements Embedder {
  private readonly modelName: string
  private readonly baseURL: string
  private readonly http: HttpClient
  private readonly logger: Logger
  private readonly cache: LRUCache<string, number[]> | null
  private readonly singleFlight = new SingleFlight<string, number[]>()
  private dim = 0
  /** null means unknown; true means /api/embed works; false means use fallback. */
  private batchSupported: boolean | null = null

  constructor(cfg: OllamaEmbedderConfig = {}) {
    this.modelName = cfg.model ?? DEFAULT_EMBED_MODEL
    this.baseURL = (cfg.baseURL ?? DEFAULT_BASE_URL).replace(/\/+$/, '')
    this.http = cfg.http ?? defaultHttpClient
    this.logger = cfg.logger ?? noopLogger
    const size = cfg.cacheSize ?? DEFAULT_CACHE_SIZE
    this.cache = size > 0 ? new LRUCache<string, number[]>(size) : null
  }

  name(): string {
    return 'ollama-embed'
  }

  model(): string {
    return this.modelName
  }

  dimension(): number {
    return this.dim
  }

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    if (texts.length === 0) return []
    const out: (number[] | undefined)[] = new Array(texts.length)
    const misses: { idx: number; text: string; key: string }[] = []

    for (let i = 0; i < texts.length; i++) {
      const sanitised = sanitise(texts[i] ?? '')
      const key = cacheKey(this.modelName, sanitised)
      if (this.cache !== null) {
        const hit = this.cache.get(key)
        if (hit !== undefined) {
          out[i] = [...hit]
          continue
        }
      }
      misses.push({ idx: i, text: sanitised, key })
    }

    if (misses.length > 0) {
      await Promise.all(
        misses.map(async ({ idx, text, key }) => {
          const vec = await this.singleFlight.do(key, () => this.fetchOne(text, signal))
          if (this.cache !== null) this.cache.set(key, vec)
          out[idx] = [...vec]
        }),
      )
    }

    // Fill zero vectors for empty inputs that somehow slipped through.
    for (let i = 0; i < out.length; i++) {
      if (out[i] === undefined) {
        out[i] = this.dim > 0 ? new Array(this.dim).fill(0) : []
      }
    }
    return out as number[][]
  }

  private async fetchOne(text: string, signal: AbortSignal | undefined): Promise<number[]> {
    const vectors = await this.fetchBatch([text], signal)
    if (vectors.length !== 1) {
      throw new ProviderError(
        `ollama: embed returned ${vectors.length} vectors for 1 input`,
        0,
      )
    }
    return vectors[0] as number[]
  }

  private async fetchBatch(
    texts: string[],
    signal: AbortSignal | undefined,
  ): Promise<number[][]> {
    if (this.batchSupported !== false) {
      const result = await this.tryBatch(texts, signal)
      if (result !== null) {
        this.batchSupported = true
        this.recordDim(result[0])
        return result
      }
      this.batchSupported = false
    }
    // Fallback: per-item.
    const out: number[][] = []
    for (const t of texts) {
      const vec = await this.fetchPerItem(t, signal)
      out.push(vec)
    }
    if (out.length > 0) this.recordDim(out[0])
    return out
  }

  private async tryBatch(
    texts: string[],
    signal: AbortSignal | undefined,
  ): Promise<number[][] | null> {
    const { response, text } = await postForText(
      this.http,
      `${this.baseURL}/api/embed`,
      { model: this.modelName, input: texts },
      { ...(signal ? { signal } : {}) },
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
      parsed = JSON.parse(text)
    } catch (err) {
      throw new ProviderError('ollama: /api/embed returned invalid JSON', response.status, text, err)
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
      { model: this.modelName, prompt: text },
      { ...(signal ? { signal } : {}) },
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
      parsed = JSON.parse(body)
    } catch (err) {
      throw new ProviderError(
        'ollama: /api/embeddings returned invalid JSON',
        response.status,
        body,
        err,
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

  private recordDim(sample: number[] | undefined): void {
    if (sample !== undefined && sample.length > 0 && this.dim === 0) {
      this.dim = sample.length
    }
  }
}

function sanitise(s: string): string {
  if (s === '') return ''
  let out = s
  if (out.includes('\x00')) out = out.replaceAll('\x00', '')
  if (out.length > MAX_EMBED_INPUT_CHARS) out = out.slice(0, MAX_EMBED_INPUT_CHARS)
  return out
}

function cacheKey(model: string, text: string): string {
  const hash = createHash('sha256').update(text).digest('hex').slice(0, 16)
  return `${model}\x1f${hash}`
}
