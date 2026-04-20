// SPDX-License-Identifier: Apache-2.0

/**
 * Provider + embedder factories. Picks implementations by discriminator so
 * callers can swap backends via config without branching.
 */

import { type AnthropicConfig, AnthropicProvider } from './anthropic.js'
import { LLMError } from './errors.js'
import { HashEmbedder, type HashEmbedderOptions } from './hashembed.js'
import {
  type OllamaConfig,
  OllamaEmbedder,
  type OllamaEmbedderConfig,
  OllamaProvider,
} from './ollama.js'
import {
  type OpenAIConfig,
  OpenAIEmbedder,
  type OpenAIEmbedderConfig,
  OpenAIProvider,
} from './openai.js'
import { TEIEmbedder, type TEIEmbedderConfig } from './tei.js'
import type { Embedder, Provider } from './types.js'

export type ProviderConfig =
  | ({ type: 'anthropic' } & AnthropicConfig)
  | ({ type: 'openai' } & OpenAIConfig)
  | ({ type: 'ollama' } & OllamaConfig)

export function createProvider(config: ProviderConfig): Provider {
  switch (config.type) {
    case 'anthropic':
      return new AnthropicProvider(config)
    case 'openai':
      return new OpenAIProvider(config)
    case 'ollama':
      return new OllamaProvider(config)
    default: {
      const exhaustive: never = config
      throw new LLMError(`unknown provider type: ${JSON.stringify(exhaustive)}`)
    }
  }
}

export type EmbedderConfig =
  | ({ type: 'hash' } & HashEmbedderOptions)
  | ({ type: 'openai' } & OpenAIEmbedderConfig)
  | ({ type: 'ollama' } & OllamaEmbedderConfig)
  | ({ type: 'tei' } & TEIEmbedderConfig)

export function createEmbedder(config: EmbedderConfig): Embedder {
  switch (config.type) {
    case 'hash':
      return new HashEmbedder(config)
    case 'openai':
      return new OpenAIEmbedder(config)
    case 'ollama':
      return new OllamaEmbedder(config)
    case 'tei':
      return new TEIEmbedder(config)
    default: {
      const exhaustive: never = config
      throw new LLMError(`unknown embedder type: ${JSON.stringify(exhaustive)}`)
    }
  }
}
