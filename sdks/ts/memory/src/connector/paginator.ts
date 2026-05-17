// SPDX-License-Identifier: Apache-2.0

/**
 * Pagination abstractor supporting cursor, offset, and token patterns.
 * Provides an async iterable interface for transparent multi-page
 * traversal.
 */

import type { PaginatorConfig, RateLimiter } from './types.js'

/** Thrown when the maximum page count guard is hit. */
export class MaxPagesExceededError extends Error {
  constructor(maxPages: number) {
    super(`Paginator exceeded maximum page count: ${maxPages}`)
    this.name = 'MaxPagesExceededError'
  }
}

const DEFAULT_PAGE_SIZE = 100
const DEFAULT_MAX_PAGES = 10_000

/**
 * Create an async iterable that iterates through all pages,
 * transparently handling pagination, rate limiting, and termination.
 */
export async function* paginate<T>(config: PaginatorConfig<T>): AsyncGenerator<T> {
  const maxPages = config.maxPages ?? DEFAULT_MAX_PAGES

  let cursor: string | undefined
  for (let page = 0; page < maxPages; page++) {
    if (config.rateLimiter) {
      await config.rateLimiter.acquire(1)
    }

    const result = await config.fetchPage(cursor)

    for (const item of result.items) {
      yield item
    }

    if (!result.nextCursor) {
      return
    }
    cursor = result.nextCursor
  }

  throw new MaxPagesExceededError(maxPages)
}

/**
 * Collect all paginated items into a single array. Suitable for
 * moderate result sets.
 */
export const collectPages = async <T>(config: PaginatorConfig<T>): Promise<readonly T[]> => {
  const items: T[] = []
  for await (const item of paginate(config)) {
    items.push(item)
  }
  return items
}
