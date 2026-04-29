// SPDX-License-Identifier: Apache-2.0

/**
 * Distiller rewrites a raw user query into a compact, search-ready form
 * using an LLM. Results are cached in-process by an LRU keyed on
 * `(model, normalised query)` so repeat invocations within the same
 * process skip the LLM round-trip.
 *
 * Parity notes:
 *   - Go exposes a richer `Distiller` that emits structured Query
 *     objects with a Trace (see go/query/query.go). The TS SDK
 *     keeps the public surface deliberately small (string in, string
 *     out) to match the retrieval pipeline's current pre-parse hook.
 *   - The system prompt is sourced verbatim from jeff; see
 *     `./prompt.ts` for the citation.
 */

import type { Provider } from '../llm/index.js'
import { type Cache, cacheKey, createCache } from './cache.js'
import { callDistillLLM } from './prompt.js'

export type Distiller = {
  /** Distil `raw` into a compact query. Returns '' for empty input. */
  distill(raw: string, signal?: AbortSignal): Promise<string>
}

export type DistillerOptions = {
  /** LLM provider used when a cache miss hits the distiller. */
  provider: Provider
  /** Override the model the provider would otherwise pick. */
  model?: string
  /** Inject a pre-built cache (tests, cross-instance sharing). */
  cache?: Cache
  /** LRU capacity when `cache` is omitted. Defaults to 512. */
  cacheCapacity?: number
}

const DEFAULT_CACHE_CAPACITY = 512

/**
 * normaliseForCache lowercases the input and collapses surrounding
 * whitespace so trivially different renderings (casing, stray spaces)
 * hit the same cache slot.
 */
function normaliseForCache(raw: string): string {
  return raw.trim().toLowerCase()
}

/**
 * createDistiller builds a distiller bound to a specific provider plus
 * an LRU cache. The cache is owned by the distiller unless the caller
 * passes one in via `opts.cache`, which is useful for sharing across
 * multiple distillers or for stubbing in tests.
 */
export function createDistiller(opts: DistillerOptions): Distiller {
  const cache =
    opts.cache ?? createCache({ capacity: opts.cacheCapacity ?? DEFAULT_CACHE_CAPACITY })

  return {
    async distill(raw, signal): Promise<string> {
      const normalised = normaliseForCache(raw)
      if (normalised === '') return ''

      const key = cacheKey(opts.model ?? '', normalised)
      const hit = cache.get(key)
      if (hit !== undefined) return hit

      const distilled = await callDistillLLM(opts.provider, raw, opts.model, signal)
      cache.set(key, distilled)
      return distilled
    },
  }
}
