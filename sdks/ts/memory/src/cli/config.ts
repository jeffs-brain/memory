// SPDX-License-Identifier: Apache-2.0

/**
 * Shared configuration for the memory CLI.
 *
 * Resolution order for the brain directory:
 *   1. explicit `--brain` flag
 *   2. `JB_BRAIN` environment variable
 *   3. process cwd
 *
 * LLM provider and embedder configuration is driven entirely by env
 * variables so commands stay side-effect free below the main().
 */

import { resolve } from 'node:path'
import type { Embedder, Provider } from '../llm/index.js'
import {
  TEIReranker,
  OllamaEmbedder,
  TEIEmbedder,
  createHashEmbedder,
  createProvider,
} from '../llm/index.js'
import { CrossEncoderReranker, type Reranker } from '../rerank/index.js'

export type BrainConfig = {
  readonly dir: string
}

export const resolveBrainDir = (flag: string | undefined): string => {
  const trimmed = flag !== undefined && flag !== '' ? flag : undefined
  const fromEnv = process.env['JB_BRAIN']
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
}

export class CliUsageError extends Error {
  override readonly name = 'CliUsageError'
}

export class CliError extends Error {
  override readonly name = 'CliError'
}

export const providerFromEnv = (): ProviderSettings => {
  const kindRaw = process.env['JB_LLM_PROVIDER']
  if (kindRaw === undefined || kindRaw === '') {
    throw new CliError(
      'JB_LLM_PROVIDER not set; expected one of anthropic|openai|ollama',
    )
  }
  if (!isProviderKind(kindRaw)) {
    throw new CliUsageError(
      `invalid JB_LLM_PROVIDER='${kindRaw}'; expected anthropic|openai|ollama`,
    )
  }
  const model = process.env['JB_LLM_MODEL'] ?? defaultModelFor(kindRaw)
  const apiKey = process.env['JB_LLM_API_KEY'] ?? ''
  if (kindRaw !== 'ollama' && apiKey === '') {
    throw new CliError(`JB_LLM_API_KEY required for provider '${kindRaw}'`)
  }
  return { kind: kindRaw, model, apiKey }
}

export const providerFromEnvOptional = (): ProviderSettings | undefined => {
  const kindRaw = process.env['JB_LLM_PROVIDER']
  if (kindRaw === undefined || kindRaw === '') return undefined
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
      })
    case 'openai':
      return createProvider({
        type: 'openai',
        apiKey: settings.apiKey,
        model: settings.model,
      })
    case 'ollama':
      return createProvider({
        type: 'ollama',
        model: settings.model,
      })
  }
}

export type EmbedderKind = 'hash' | 'ollama' | 'tei'

export const isEmbedderKind = (v: string): v is EmbedderKind =>
  v === 'hash' || v === 'ollama' || v === 'tei'

export type EmbedderSettings = {
  readonly kind: EmbedderKind
  readonly baseURL: string
  readonly model: string
}

const DEFAULT_OLLAMA_URL = 'http://localhost:11434'
const DEFAULT_TEI_URL = 'http://localhost:8080'

export const embedderFromEnv = (): EmbedderSettings | undefined => {
  const raw = process.env['JB_EMBED_PROVIDER']
  if (raw === undefined || raw === '') return undefined
  if (!isEmbedderKind(raw)) {
    throw new CliUsageError(
      `invalid JB_EMBED_PROVIDER='${raw}'; expected hash|ollama|tei`,
    )
  }
  const baseURL =
    process.env['JB_EMBED_URL'] ??
    (raw === 'ollama' ? DEFAULT_OLLAMA_URL : DEFAULT_TEI_URL)
  const model =
    process.env['JB_EMBED_MODEL'] ??
    (raw === 'hash' ? 'hash' : raw === 'ollama' ? 'bge-m3' : 'tei')
  return { kind: raw, baseURL, model }
}

export const buildEmbedder = (settings: EmbedderSettings): Embedder => {
  switch (settings.kind) {
    case 'hash':
      return createHashEmbedder()
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

export type RerankerKind = 'tei'

export const isRerankerKind = (v: string): v is RerankerKind => v === 'tei'

export type RerankerSettings = {
  readonly kind: RerankerKind
  readonly baseURL: string
  readonly label: string
}

export const rerankerFromEnv = (): RerankerSettings | undefined => {
  const raw = process.env['JB_RERANK_PROVIDER']
  if (raw === undefined || raw === '') return undefined
  if (!isRerankerKind(raw)) {
    throw new CliUsageError(
      `invalid JB_RERANK_PROVIDER='${raw}'; expected tei`,
    )
  }
  const baseURL = process.env['JB_RERANK_URL'] ?? DEFAULT_TEI_URL
  const label = process.env['JB_RERANK_LABEL'] ?? 'cross-encoder'
  return {
    kind: raw,
    baseURL,
    label,
  }
}

export const buildReranker = (settings: RerankerSettings): Reranker => {
  switch (settings.kind) {
    case 'tei':
      return new CrossEncoderReranker({
        client: new TEIReranker({
          baseURL: settings.baseURL,
        }),
        label: settings.label,
      })
  }
}
