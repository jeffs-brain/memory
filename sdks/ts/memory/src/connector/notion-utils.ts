// SPDX-License-Identifier: Apache-2.0

/**
 * Utility functions for the Notion connector. Contains transport
 * helpers, signal combinators, and configuration parsers.
 */

// ---------- Retry-After parsing ----------

/**
 * Parses the Retry-After header value as seconds. Returns -1 if
 * absent or unparseable (indicating backoff should be used instead).
 */
export const parseRetryAfterHeader = (value: string | null): number => {
  if (value === null || value === '') return -1
  const parsed = parseInt(value, 10)
  return isNaN(parsed) ? -1 : parsed
}

// ---------- Response body size limiting ----------

/**
 * Reads a Response body with a size limit. Throws if the body
 * exceeds the limit.
 */
export const readResponseWithLimit = async (
  response: Response,
  maxBytes: number,
): Promise<string> => {
  // If the Content-Length header is present and exceeds the limit,
  // reject early.
  const contentLength = response.headers.get('Content-Length')
  if (contentLength !== null) {
    const length = parseInt(contentLength, 10)
    if (!isNaN(length) && length > maxBytes) {
      throw new Error(
        `connector/notion: response body exceeds ${String(maxBytes)} bytes limit`,
      )
    }
  }

  // Stream the body, checking size as we go.
  if (response.body !== null) {
    const reader = response.body.getReader()
    const chunks: Uint8Array[] = []
    let totalBytes = 0

    let readResult = await reader.read()
    while (!readResult.done) {
      totalBytes += readResult.value.byteLength
      if (totalBytes > maxBytes) {
        reader.cancel().catch(() => { /* intentionally empty */ })
        throw new Error(
          `connector/notion: response body exceeds ${String(maxBytes)} bytes limit`,
        )
      }
      chunks.push(readResult.value)
      readResult = await reader.read()
    }

    const decoder = new TextDecoder()
    return chunks.map((chunk) => decoder.decode(chunk, { stream: true })).join('') +
      decoder.decode()
  }

  return response.text()
}

// ---------- Signal helpers ----------

/**
 * Sleeps for the given duration, but resolves early if the signal
 * fires.
 */
export const interruptibleSleep = (
  signal: AbortSignal,
  ms: number,
): Promise<void> =>
  new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, ms)
    const onAbort = (): void => {
      clearTimeout(timer)
      resolve()
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })

/**
 * Combines two abort signals into one that fires when either fires.
 */
export const combineSignals = (a: AbortSignal, b: AbortSignal): AbortSignal => {
  if (a.aborted) return a
  if (b.aborted) return b

  const controller = new AbortController()
  const onAbort = (): void => controller.abort()
  a.addEventListener('abort', onAbort, { once: true })
  b.addEventListener('abort', onAbort, { once: true })
  return controller.signal
}

// ---------- Configuration parsers ----------

/**
 * Parses a value that might be a string array or undefined.
 */
export const parseStringArray = (
  value: unknown,
): readonly string[] | undefined => {
  if (value === undefined || value === null) return undefined
  if (Array.isArray(value)) {
    return value.filter((v): v is string => typeof v === 'string')
  }
  return undefined
}

/**
 * Parses a block type filter from raw config. Accepts a string
 * array and converts to a Set.
 */
export const parseBlockTypeFilter = (
  value: unknown,
): ReadonlySet<string> | undefined => {
  if (value === undefined || value === null) return undefined
  if (Array.isArray(value)) {
    const filtered = value.filter((v): v is string => typeof v === 'string')
    return filtered.length > 0 ? new Set(filtered) : undefined
  }
  return undefined
}

// ---------- HTTP ----------

/**
 * HTTP fetcher interface for dependency injection in tests.
 */
export type NotionHTTPFetcher = (
  url: string,
  options: RequestInit,
) => Promise<Response>

/**
 * Global fetch wrapper for the default HTTP fetcher.
 */
export const globalFetch: NotionHTTPFetcher = (url, options) =>
  fetch(url, options)
