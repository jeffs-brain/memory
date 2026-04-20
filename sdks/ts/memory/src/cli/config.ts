// SPDX-License-Identifier: Apache-2.0

/**
 * Shared configuration for the memory CLI.
 *
 * Brain directory resolution: `--brain` flag, then `JB_BRAIN`, then
 * process cwd. LLM provider and embedder configuration comes entirely
 * from env so commands stay side-effect free below main().
 */

import { resolve } from 'node:path'
import type { HttpClient } from '../llm/http.js'
import type { Embedder, Provider } from '../llm/index.js'
import {
  OllamaEmbedder,
  OpenAIEmbedder,
  TEIEmbedder,
  TEIReranker,
  createHashEmbedder,
  createProvider,
} from '../llm/index.js'
import {
  AutoReranker,
  CrossEncoderReranker,
  DEFAULT_RERANK_BATCH_SIZE,
  DEFAULT_RERANK_PARALLELISM,
  DEFAULT_SHARED_RERANK_CONCURRENCY,
  LLMReranker,
  type Reranker,
} from '../rerank/index.js'

export type BrainConfig = {
  readonly dir: string
}

export const resolveBrainDir = (flag: string | undefined): string => {
  const trimmed = flag !== undefined && flag !== '' ? flag : undefined
  const fromEnv = process.env.JB_BRAIN
  const picked =
    trimmed !== undefined
      ? trimmed
      : fromEnv !== undefined && fromEnv !== ''
        ? fromEnv
        : process.cwd()
  return resolve(picked)
}

export type ProviderKind = 'anthropic' | 'openai' | 'ollama'

export const isProviderKind = (v: string): v is ProviderKind =>
  v === 'anthropic' || v === 'openai' || v === 'ollama'

export type ProviderSettings = {
  readonly kind: ProviderKind
  readonly model: string
  readonly apiKey: string
  readonly baseURL?: string
}

export class CliUsageError extends Error {
  override readonly name = 'CliUsageError'
}

export class CliError extends Error {
  override readonly name = 'CliError'
}

const providerApiKeyFromEnv = (kind: ProviderKind): string => {
  const explicit = process.env.JB_LLM_API_KEY
  if (explicit !== undefined && explicit !== '') return explicit
  switch (kind) {
    case 'anthropic':
      return process.env.ANTHROPIC_API_KEY ?? ''
    case 'openai':
      return process.env.OPENAI_API_KEY ?? ''
    case 'ollama':
      return ''
  }
}

const providerBaseURLFromEnv = (kind: ProviderKind): string | undefined => {
  const explicit = process.env.JB_LLM_BASE_URL
  if (explicit !== undefined && explicit !== '') return explicit
  switch (kind) {
    case 'anthropic':
      return process.env.ANTHROPIC_BASE_URL
    case 'openai':
      return process.env.OPENAI_BASE_URL
    case 'ollama':
      return process.env.OLLAMA_HOST
  }
}

const inferProviderKindFromFallbackEnv = (): ProviderKind | undefined => {
  if (
    (process.env.ANTHROPIC_API_KEY ?? '') !== '' ||
    (process.env.ANTHROPIC_BASE_URL ?? '') !== ''
  ) {
    return 'anthropic'
  }
  if ((process.env.OPENAI_API_KEY ?? '') !== '' || (process.env.OPENAI_BASE_URL ?? '') !== '') {
    return 'openai'
  }
  return undefined
}

export const providerFromEnv = (): ProviderSettings => {
  const raw = process.env.JB_LLM_PROVIDER
  const kindRaw = raw !== undefined && raw !== '' ? raw : inferProviderKindFromFallbackEnv()
  if (kindRaw === undefined) {
    throw new CliError('JB_LLM_PROVIDER not set; expected one of anthropic|openai|ollama')
  }
  if (!isProviderKind(kindRaw)) {
    throw new CliUsageError(
      `invalid JB_LLM_PROVIDER='${kindRaw}'; expected anthropic|openai|ollama`,
    )
  }
  const model = process.env.JB_LLM_MODEL ?? defaultModelFor(kindRaw)
  const apiKey = providerApiKeyFromEnv(kindRaw)
  const baseURL = providerBaseURLFromEnv(kindRaw)
  if (kindRaw !== 'ollama' && apiKey === '' && baseURL === undefined) {
    throw new CliError(`JB_LLM_API_KEY required for provider '${kindRaw}'`)
  }
  return baseURL ? { kind: kindRaw, model, apiKey, baseURL } : { kind: kindRaw, model, apiKey }
}

export const providerFromEnvOptional = (): ProviderSettings | undefined => {
  const kindRaw = process.env.JB_LLM_PROVIDER
  if (kindRaw === undefined || kindRaw === '') {
    return inferProviderKindFromFallbackEnv() !== undefined ? providerFromEnv() : undefined
  }
  return providerFromEnv()
}

const defaultModelFor = (kind: ProviderKind): string => {
  switch (kind) {
    case 'anthropic':
      return 'claude-opus-4-5'
    case 'openai':
      return 'gpt-4o-mini'
    case 'ollama':
      return 'llama3.1'
  }
}

export const buildProvider = (settings: ProviderSettings): Provider => {
  switch (settings.kind) {
    case 'anthropic':
      return createProvider({
        type: 'anthropic',
        apiKey: settings.apiKey,
        model: settings.model,
        ...(settings.baseURL ? { baseURL: settings.baseURL } : {}),
      })
    case 'openai':
      return createProvider({
        type: 'openai',
        apiKey: settings.apiKey,
        model: settings.model,
        ...(settings.baseURL ? { baseURL: settings.baseURL } : {}),
      })
    case 'ollama':
      return createProvider({
        type: 'ollama',
        model: settings.model,
        ...(settings.baseURL ? { baseURL: settings.baseURL } : {}),
      })
  }
}

export type EmbedderKind = 'hash' | 'openai' | 'ollama' | 'tei'

export const isEmbedderKind = (v: string): v is EmbedderKind =>
  v === 'hash' || v === 'openai' || v === 'ollama' || v === 'tei'

export type EmbedderSettings = {
  readonly kind: EmbedderKind
  readonly baseURL: string
  readonly model: string
  readonly apiKey?: string
}

const DEFAULT_OLLAMA_URL = 'http://localhost:11434'
const DEFAULT_TEI_URL = 'http://localhost:8080'
const DEFAULT_OPENAI_URL = 'https://api.openai.com'

export const embedderFromEnv = (): EmbedderSettings | undefined => {
  const raw = process.env.JB_EMBED_PROVIDER
  const inferred =
    raw !== undefined && raw !== ''
      ? raw
      : (process.env.OPENAI_API_KEY ?? '') !== '' || (process.env.OPENAI_BASE_URL ?? '') !== ''
        ? 'openai'
        : undefined
  if (inferred === undefined) return undefined
  if (!isEmbedderKind(inferred)) {
    throw new CliUsageError(
      `invalid JB_EMBED_PROVIDER='${inferred}'; expected hash|openai|ollama|tei`,
    )
  }
  const baseURL =
    process.env.JB_EMBED_URL ??
    (inferred === 'openai'
      ? (providerBaseURLFromEnv('openai') ?? DEFAULT_OPENAI_URL)
      : inferred === 'ollama'
        ? DEFAULT_OLLAMA_URL
        : DEFAULT_TEI_URL)
  const model =
    process.env.JB_EMBED_MODEL ??
    (inferred === 'hash'
      ? 'hash'
      : inferred === 'openai'
        ? 'text-embedding-3-small'
        : inferred === 'ollama'
          ? 'bge-m3'
          : 'tei')
  return inferred === 'openai'
    ? {
        kind: inferred,
        baseURL,
        model,
        apiKey: providerApiKeyFromEnv('openai'),
      }
    : { kind: inferred, baseURL, model }
}

export const buildEmbedder = (settings: EmbedderSettings): Embedder => {
  switch (settings.kind) {
    case 'hash':
      return createHashEmbedder()
    case 'openai':
      return new OpenAIEmbedder({
        apiKey: settings.apiKey ?? '',
        baseURL: settings.baseURL,
        model: settings.model,
      })
    case 'ollama':
      return new OllamaEmbedder({
        baseURL: settings.baseURL,
        model: settings.model,
      })
    case 'tei':
      return new TEIEmbedder({
        baseURL: settings.baseURL,
        model: settings.model,
      })
  }
}

export type RerankerKind = 'tei' | 'http' | 'llm' | 'auto'

export const isRerankerKind = (v: string): v is RerankerKind =>
  v === 'tei' || v === 'http' || v === 'llm' || v === 'auto'

export type RerankerSettings = {
  readonly kind: RerankerKind
  readonly baseURL: string
  readonly label: string
  readonly batchSize: number
  readonly parallelism: number
  readonly concurrencyCap: number
  readonly preferHttp: boolean
}

type BuildRerankerDeps = {
  readonly provider?: Provider
  readonly http?: HttpClient
}

const parsePositiveIntEnv = (name: string, fallback: number): number => {
  const raw = process.env[name]
  if (raw === undefined || raw === '') return fallback
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new CliUsageError(`invalid ${name}='${raw}'; expected a positive integer`)
  }
  return parsed
}

export const rerankerFromEnv = (): RerankerSettings | undefined => {
  const raw = process.env.JB_RERANK_PROVIDER
  if (raw === undefined || raw === '') return undefined
  if (!isRerankerKind(raw)) {
    throw new CliUsageError(`invalid JB_RERANK_PROVIDER='${raw}'; expected tei|http|llm|auto`)
  }
  const explicitURL = (process.env.JB_RERANK_URL ?? '').trim() !== ''
  const baseURL = process.env.JB_RERANK_URL ?? DEFAULT_TEI_URL
  const label =
    process.env.JB_RERANK_LABEL ??
    (raw === 'auto' ? 'auto-rerank' : raw === 'llm' ? 'llm-rerank' : 'cross-encoder')
  return {
    kind: raw,
    baseURL,
    label,
    batchSize: parsePositiveIntEnv('JB_RERANK_BATCH_SIZE', DEFAULT_RERANK_BATCH_SIZE),
    parallelism: parsePositiveIntEnv('JB_RERANK_PARALLELISM', DEFAULT_RERANK_PARALLELISM),
    concurrencyCap: parsePositiveIntEnv('JB_RERANK_CONCURRENCY', DEFAULT_SHARED_RERANK_CONCURRENCY),
    preferHttp: raw === 'auto' ? explicitURL : raw === 'tei' || raw === 'http',
  }
}

export const buildReranker = (
  settings: RerankerSettings,
  deps: BuildRerankerDeps = {},
): Reranker => {
  const llm = (): Reranker => {
    if (deps.provider === undefined) {
      throw new CliError('JB_RERANK_PROVIDER=llm requires JB_LLM_PROVIDER to be configured')
    }
    return new LLMReranker({
      provider: deps.provider,
      batchSize: settings.batchSize,
      parallelism: settings.parallelism,
      concurrencyCap: settings.concurrencyCap,
      label: settings.label,
    })
  }
  const httpReranker = settings.preferHttp
    ? new CrossEncoderReranker({
        client: new TEIReranker({
          baseURL: settings.baseURL,
          ...(deps.http !== undefined ? { http: deps.http } : {}),
        }),
        label:
          settings.kind === 'auto'
            ? 'tei-rerank'
            : settings.kind === 'http'
              ? 'http-rerank'
              : settings.label,
        concurrencyCap: settings.concurrencyCap,
      })
    : undefined
  switch (settings.kind) {
    case 'tei':
    case 'http':
      if (httpReranker === undefined) {
        throw new CliError('JB_RERANK_PROVIDER requires JB_RERANK_URL')
      }
      return httpReranker
    case 'llm':
      return llm()
    case 'auto': {
      if (httpReranker === undefined) return llm()
      const fallback = deps.provider !== undefined ? llm() : undefined
      return new AutoReranker({
        primary: httpReranker,
        ...(fallback !== undefined ? { fallback } : {}),
        label: settings.label,
      })
    }
  }
}
