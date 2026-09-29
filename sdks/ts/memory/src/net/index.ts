// SPDX-License-Identifier: Apache-2.0

/**
 * Outbound network helpers shared by the daemon, the MCP server and the
 * pi extension. Use {@link safeFetch} for any URL a user or a model
 * supplied.
 */

export {
  DEFAULT_SAFE_FETCH_MAX_BYTES,
  DEFAULT_SAFE_FETCH_MAX_REDIRECTS,
  DEFAULT_SAFE_FETCH_TIMEOUT_MS,
  FetchFailedError,
  type Resolver,
  type SafeFetchOptions,
  type SafeFetchResult,
  UnsafeUrlError,
  isBlockedAddress,
  resolvePublicAddress,
  safeFetch,
} from './safe-fetch.js'
