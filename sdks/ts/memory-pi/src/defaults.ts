// SPDX-License-Identifier: Apache-2.0

/**
 * Auto-detection helpers for the memory-pi extension. Probes the local
 * environment so callers can ship `{ kind: "auto" }` for store, embedder,
 * provider, and reranker and get sensible defaults without ceremony.
 *
 * The detection order mirrors `mcp/ts/src/memory-client.ts`:
 *   - Embedder: Ollama on `localhost:11434` (model `bge-m3`).
 *   - Provider: explicit OPENAI_API_KEY > ANTHROPIC_API_KEY > local Ollama.
 *   - Reranker: off unless explicit configuration says otherwise.
 *   - Store: fs (autodetects to git when the root contains a `.git`).
 */

import { homedir } from 'node:os'
import { join } from 'node:path'

export type ProbeOptions = {
  readonly baseUrl?: string
  readonly timeoutMs?: number
  readonly fetchImpl?: typeof fetch
}

export const DEFAULT_OLLAMA_BASE_URL = 'http://localhost:11434'
export const DEFAULT_EMBED_MODEL = 'bge-m3'
export const DEFAULT_OLLAMA_CHAT_MODEL = 'gemma3'
export const DEFAULT_OPENAI_CHAT_MODEL = 'gpt-4o-mini'
export const DEFAULT_ANTHROPIC_CHAT_MODEL = 'claude-opus-4-5'
export const DEFAULT_BRAIN_ROOT = join(homedir(), '.config', 'memory-pi', 'brains')
export const DEFAULT_BRAIN_ID = 'default'

/**
 * Probe whether an Ollama-compatible endpoint answers at `/api/tags`. Used
 * by `selectEmbedder` and `selectProvider` to decide whether to fall back
 * to local inference.
 */
export const ollamaReachable = async (opts: ProbeOptions = {}): Promise<boolean> => {
  const base = (opts.baseUrl ?? DEFAULT_OLLAMA_BASE_URL).replace(/\/+$/, '')
  const fetchImpl = opts.fetchImpl ?? fetch
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), opts.timeoutMs ?? 1500)
  try {
    const resp = await fetchImpl(`${base}/api/tags`, { signal: controller.signal })
    return resp.ok
  } catch {
    return false
  } finally {
    clearTimeout(timer)
  }
}

export type DetectedProvider =
  | { readonly kind: 'openai'; readonly apiKey: string; readonly model: string }
  | { readonly kind: 'anthropic'; readonly apiKey: string; readonly model: string }
  | { readonly kind: 'ollama'; readonly baseUrl: string; readonly model: string }

export type DetectProviderOptions = ProbeOptions & {
  readonly env?: NodeJS.ProcessEnv
}

/**
 * Choose the best LLM provider available on the host. Returns undefined
 * when none of the three resolve.
 */
export const detectProvider = async (
  opts: DetectProviderOptions = {},
): Promise<DetectedProvider | undefined> => {
  const env = opts.env ?? process.env
  const openaiKey = env.OPENAI_API_KEY
  if (openaiKey !== undefined && openaiKey !== '') {
    return {
      kind: 'openai',
      apiKey: openaiKey,
      model: env.OPENAI_MODEL ?? DEFAULT_OPENAI_CHAT_MODEL,
    }
  }
  const anthropicKey = env.ANTHROPIC_API_KEY
  if (anthropicKey !== undefined && anthropicKey !== '') {
    return {
      kind: 'anthropic',
      apiKey: anthropicKey,
      model: env.ANTHROPIC_MODEL ?? DEFAULT_ANTHROPIC_CHAT_MODEL,
    }
  }
  const baseUrl = opts.baseUrl ?? DEFAULT_OLLAMA_BASE_URL
  if (await ollamaReachable({ ...opts, baseUrl })) {
    return {
      kind: 'ollama',
      baseUrl,
      model: env.OLLAMA_CHAT_MODEL ?? DEFAULT_OLLAMA_CHAT_MODEL,
    }
  }
  return undefined
}

export type DetectedEmbedder =
  | { readonly kind: 'openai'; readonly apiKey: string; readonly model: string }
  | { readonly kind: 'ollama'; readonly baseUrl: string; readonly model: string }

export type DetectEmbedderOptions = ProbeOptions & {
  readonly env?: NodeJS.ProcessEnv
}

/**
 * Prefer local Ollama for embeddings; fall back to OpenAI text-embedding-3
 * only when explicitly configured (cost). Returns undefined when neither
 * is reachable.
 */
export const detectEmbedder = async (
  opts: DetectEmbedderOptions = {},
): Promise<DetectedEmbedder | undefined> => {
  const env = opts.env ?? process.env
  const baseUrl = opts.baseUrl ?? DEFAULT_OLLAMA_BASE_URL
  if (await ollamaReachable({ ...opts, baseUrl })) {
    return {
      kind: 'ollama',
      baseUrl,
      model: env.OLLAMA_EMBED_MODEL ?? DEFAULT_EMBED_MODEL,
    }
  }
  // Fall back to OpenAI embeddings only when the user opts in via a
  // dedicated env var so we never silently bill an OpenAI key set for
  // chat completions for embedding work.
  const openaiKey = env.OPENAI_EMBEDDING_API_KEY ?? env.OPENAI_API_KEY
  if (openaiKey !== undefined && openaiKey !== '' && env.MEMORY_PI_EMBEDDER_OPENAI === 'true') {
    return {
      kind: 'openai',
      apiKey: openaiKey,
      model: env.OPENAI_EMBED_MODEL ?? 'text-embedding-3-small',
    }
  }
  return undefined
}

export type ResolvedBrainPaths = {
  readonly root: string
  readonly id: string
  readonly searchIndexPath: string
}

/**
 * Resolve the on-disk paths used for a single brain. Mirrors the layout
 * used by the MCP local client so the same brain can be opened by either
 * surface.
 */
export const resolveBrainPaths = (root: string, brainId: string): ResolvedBrainPaths => {
  const cleaned = brainId.replace(/[^A-Za-z0-9._-]/g, '_')
  const safe = cleaned === '' || cleaned.startsWith('.') ? DEFAULT_BRAIN_ID : cleaned
  const brainRoot = join(root, safe)
  return {
    root: brainRoot,
    id: safe,
    searchIndexPath: join(brainRoot, 'search.sqlite'),
  }
}
