// SPDX-License-Identifier: Apache-2.0

/**
 * Notion connector for syncing pages and databases into a brain.
 * Uses Notion's native Markdown API (February 2026) for page content
 * retrieval, with fallback to block-by-block reconstruction.
 * Supports incremental sync via last_edited_time filtering.
 */

import type {
  Connector,
  ConnectorConfig,
  ConnectorDocument,
  SyncCursor,
} from './types.js'
import { createRateLimiter } from './rate-limiter.js'
import {
  blockToMarkdown,
  isListBlock,
  type NotionBlock,
} from './notion-blocks.js'
import {
  parseDatabaseEntry,
  parsePageResponse,
} from './notion-properties.js'
import {
  combineSignals,
  globalFetch,
  interruptibleSleep,
  parseBlockTypeFilter,
  parseRetryAfterHeader,
  parseStringArray,
  readResponseWithLimit,
  type NotionHTTPFetcher,
} from './notion-utils.js'

// Re-export types that consumers may need.
export type { NotionBlock } from './notion-blocks.js'
export type { NotionRichText } from './notion-blocks.js'
export type { NotionHTTPFetcher } from './notion-utils.js'

// ---------- Constants ----------

const NOTION_API_VERSION = '2022-06-28'
const NOTION_DEFAULT_BASE_URL = 'https://api.notion.com/v1'
const NOTION_DEFAULT_MAX_DEPTH = 10
const NOTION_DEFAULT_POLL_INTERVAL = 15 * 60 * 1000
const NOTION_DEFAULT_TIMEOUT = 30_000
const NOTION_DEFAULT_PAGE_SIZE = 100
const NOTION_DEFAULT_RATE_LIMIT = 3
const NOTION_MAX_RETRY_ATTEMPTS = 5
const NOTION_MAX_RESPONSE_BYTES = 10 * 1024 * 1024

// ---------- Types ----------

/**
 * Notion-specific connector configuration.
 */
export type NotionConnectorConfig = {
  readonly apiToken: string
  readonly rootPageIds?: readonly string[] | undefined
  readonly databaseIds?: readonly string[] | undefined
  readonly includeChildPages?: boolean | undefined // default: true
  readonly includeDatabases?: boolean | undefined // default: true
  readonly maxDepth?: number | undefined // default: 10
  readonly blockTypeFilter?: ReadonlySet<string> | undefined
}

/**
 * Notion search/database query response shape.
 */
type NotionPaginatedResponse = {
  readonly results: readonly Record<string, unknown>[]
  readonly next_cursor: string | null
  readonly has_more: boolean
}

/**
 * Options for creating a Notion connector.
 */
export type NotionConnectorOptions = {
  readonly baseUrl?: string
  readonly fetcher?: NotionHTTPFetcher
}

// ---------- Implementation ----------

/**
 * Creates a Notion connector that implements the Connector interface.
 * A default rate limiter of 3 req/sec is created if none is provided
 * via deps.rateLimiter.
 */
export const createNotionConnector = (
  deps: ConnectorConfig,
  options?: NotionConnectorOptions,
): Connector & { readonly name: 'notion' } => {
  let config: NotionConnectorConfig | undefined
  let abortController: AbortController | undefined
  const baseUrl = options?.baseUrl ?? NOTION_DEFAULT_BASE_URL
  const fetcher: NotionHTTPFetcher = options?.fetcher ?? globalFetch
  const rateLimiter = deps.rateLimiter ?? createRateLimiter({
    maxTokens: NOTION_DEFAULT_RATE_LIMIT,
    refillRate: NOTION_DEFAULT_RATE_LIMIT,
  })

  const ensureConfigured = (): NotionConnectorConfig => {
    if (config === undefined) {
      throw new Error('connector/notion: not configured, call configure first')
    }
    return config
  }

  const acquireRateLimit = async (signal: AbortSignal): Promise<void> => {
    await rateLimiter.acquire(1, signal)
  }

  /**
   * Performs a single HTTP request to the Notion API. Retries on 429
   * responses using the Retry-After header with exponential backoff.
   */
  const doRequest = async (
    signal: AbortSignal,
    method: string,
    path: string,
    body?: Record<string, unknown>,
  ): Promise<unknown> => {
    for (let attempt = 0; attempt <= NOTION_MAX_RETRY_ATTEMPTS; attempt++) {
      await acquireRateLimit(signal)

      const { data, retryAfterSeconds, error } = await doSingleRequest(
        signal,
        method,
        path,
        body,
      )

      if (error === undefined) {
        return data
      }

      // Only retry on 429.
      if (retryAfterSeconds === undefined) {
        throw error
      }

      if (attempt >= NOTION_MAX_RETRY_ATTEMPTS) {
        throw error
      }

      const waitMs = retryAfterSeconds >= 0
        ? retryAfterSeconds * 1000
        : Math.min(1000 * Math.pow(2, attempt) + Math.random() * 500, 60_000)

      await interruptibleSleep(signal, waitMs)
    }

    throw new Error(
      `connector/notion: exhausted retries for ${method} ${path}`,
    )
  }

  /**
   * Executes a single HTTP request. Returns the parsed body on
   * success, or an error with optional retryAfterSeconds for 429.
   */
  const doSingleRequest = async (
    signal: AbortSignal,
    method: string,
    path: string,
    body?: Record<string, unknown>,
  ): Promise<{
    data?: unknown
    retryAfterSeconds?: number
    error?: Error
  }> => {
    const cfg = ensureConfigured()

    const timeoutController = new AbortController()
    const timeoutId = setTimeout(
      () => timeoutController.abort(new Error('request timeout')),
      NOTION_DEFAULT_TIMEOUT,
    )

    const combinedSignal = combineSignals(signal, timeoutController.signal)

    try {
      const headers: Record<string, string> = {
        Authorization: `Bearer ${cfg.apiToken}`,
        'Notion-Version': NOTION_API_VERSION,
      }

      const init: RequestInit = {
        method,
        headers,
        signal: combinedSignal,
      }

      if (body !== undefined) {
        headers['Content-Type'] = 'application/json'
        init.body = JSON.stringify(body)
      }

      const response = await fetcher(`${baseUrl}${path}`, init)

      if (response.status === 429) {
        const retryHeader = response.headers.get('Retry-After')
        const retryAfterSeconds = parseRetryAfterHeader(retryHeader)
        return {
          retryAfterSeconds,
          error: new Error('connector/notion: rate limited (429)'),
        }
      }

      if (response.status === 404) {
        return {
          error: new Error('connector/notion: not found (404)'),
        }
      }

      if (response.status < 200 || response.status >= 300) {
        const text = await response.text()
        return {
          error: new Error(
            `connector/notion: HTTP ${String(response.status)}: ${text}`,
          ),
        }
      }

      // Enforce response body size limit (10 MiB).
      const text = await readResponseWithLimit(response, NOTION_MAX_RESPONSE_BYTES)
      const data = JSON.parse(text) as unknown
      return { data }
    } catch (err) {
      if (err instanceof Error && err.message.startsWith('connector/notion:')) {
        return { error: err }
      }
      return { error: err instanceof Error ? err : new Error(String(err)) }
    } finally {
      clearTimeout(timeoutId)
    }
  }

  // ---------- Block children pagination ----------

  /**
   * Shared pagination helper for fetching all block children.
   * Used by both fetchBlocksAsMarkdown and fetchChildPageIds.
   */
  const fetchBlockChildren = async (
    signal: AbortSignal,
    blockId: string,
  ): Promise<readonly NotionBlock[]> => {
    const allBlocks: NotionBlock[] = []
    let cursor: string | undefined

    while (!signal.aborted) {
      let endpoint = `/blocks/${blockId}/children?page_size=${String(NOTION_DEFAULT_PAGE_SIZE)}`
      if (cursor !== undefined) {
        endpoint += `&start_cursor=${cursor}`
      }

      const resp = (await doRequest(
        signal,
        'GET',
        endpoint,
      )) as NotionPaginatedResponse
      for (const result of resp.results) {
        allBlocks.push(result as unknown as NotionBlock)
      }

      if (!resp.has_more || resp.next_cursor === null) break
      cursor = resp.next_cursor
    }

    return allBlocks
  }

  // ---------- Page fetching ----------

  const fetchPageContent = async (
    signal: AbortSignal,
    pageId: string,
    maxDepth: number,
  ): Promise<string> => {
    // Try Markdown API first.
    try {
      const mdResp = (await doRequest(
        signal,
        'GET',
        `/pages/${pageId}/markdown`,
      )) as { markdown?: string }
      if (mdResp.markdown !== undefined) {
        return mdResp.markdown
      }
    } catch {
      // Fall back to block retrieval.
    }

    return fetchBlocksAsMarkdown(signal, pageId, 0, maxDepth)
  }

  const fetchBlocksAsMarkdown = async (
    signal: AbortSignal,
    blockId: string,
    depth: number,
    maxDepth: number,
  ): Promise<string> => {
    if (depth > maxDepth) return ''

    const allBlocks = await fetchBlockChildren(signal, blockId)
    const parts: string[] = []

    for (const block of allBlocks) {
      const md = blockToMarkdown(block, config?.blockTypeFilter)
      if (md !== '') {
        parts.push(md)
      }

      if (block.has_children) {
        const childContent = await fetchBlocksAsMarkdown(
          signal,
          block.id,
          depth + 1,
          maxDepth,
        )
        if (childContent !== '') {
          const lines = childContent.split('\n')
          const prefix = isListBlock(block.type) ? '  ' : ''
          for (const line of lines) {
            parts.push(`${prefix}${line}`)
          }
        }
      }
    }

    return parts.join('\n')
  }

  const fetchChildPageIds = async (
    signal: AbortSignal,
    blockId: string,
  ): Promise<readonly string[]> => {
    const allBlocks = await fetchBlockChildren(signal, blockId)
    const childIds: string[] = []
    for (const block of allBlocks) {
      if (block.type === 'child_page' || block.type === 'child_database') {
        childIds.push(block.id)
      }
    }
    return childIds
  }

  // ---------- Page tree traversal ----------

  async function* fetchPageTree(
    signal: AbortSignal,
    pageId: string,
    depth: number,
    sinceISO: string | undefined,
    visited: Set<string>,
    cfg: NotionConnectorConfig,
  ): AsyncGenerator<ConnectorDocument> {
    if (depth > (cfg.maxDepth ?? NOTION_DEFAULT_MAX_DEPTH)) return
    if (visited.has(pageId)) return
    visited.add(pageId)

    let pageData: Record<string, unknown>
    try {
      pageData = (await doRequest(
        signal,
        'GET',
        `/pages/${pageId}`,
      )) as Record<string, unknown>
    } catch {
      return
    }

    const page = parsePageResponse(pageData)

    // For incremental sync, skip pages older than cursor.
    if (sinceISO !== undefined) {
      const cursorTime = new Date(sinceISO)
      if (page.lastEditedTime < cursorTime) return
    }

    const content = await fetchPageContent(
      signal,
      pageId,
      cfg.maxDepth ?? NOTION_DEFAULT_MAX_DEPTH,
    )

    yield {
      externalId: pageId,
      content,
      mime: 'text/markdown',
      title: page.title,
      url: page.url,
      metadata: {
        source: 'notion',
        type: 'page',
        last_edited_time: page.lastEditedTime.toISOString(),
      },
      modifiedAt: page.lastEditedTime,
    }

    // Recursively fetch child pages.
    if (cfg.includeChildPages !== false) {
      const childIds = await fetchChildPageIds(signal, pageId)
      for (const childId of childIds) {
        yield* fetchPageTree(signal, childId, depth + 1, sinceISO, visited, cfg)
      }
    }
  }

  // ---------- Database fetching ----------

  async function* fetchDatabaseEntries(
    signal: AbortSignal,
    dbId: string,
    sinceISO: string | undefined,
    visited: Set<string>,
    cfg: NotionConnectorConfig,
  ): AsyncGenerator<ConnectorDocument> {
    let cursor: string | undefined

    while (!signal.aborted) {
      const body: Record<string, unknown> = {
        page_size: NOTION_DEFAULT_PAGE_SIZE,
      }
      if (cursor !== undefined) {
        body.start_cursor = cursor
      }
      if (sinceISO !== undefined) {
        body.filter = {
          timestamp: 'last_edited_time',
          last_edited_time: { on_or_after: sinceISO },
        }
      }

      const resp = (await doRequest(
        signal,
        'POST',
        `/databases/${dbId}/query`,
        body,
      )) as NotionPaginatedResponse

      for (const result of resp.results) {
        const entry = parseDatabaseEntry(result)
        if (visited.has(entry.pageId)) continue
        visited.add(entry.pageId)

        let pageContent = ''
        try {
          const content = await fetchPageContent(
            signal,
            entry.pageId,
            cfg.maxDepth ?? NOTION_DEFAULT_MAX_DEPTH,
          )
          if (content !== '') {
            pageContent = `\n\n${content}`
          }
        } catch {
          // Page content fetch is optional; proceed without it.
        }

        yield {
          externalId: entry.pageId,
          content: entry.propertiesMarkdown + pageContent,
          mime: 'text/markdown',
          title: entry.title,
          url: entry.url,
          metadata: {
            source: 'notion',
            type: 'database_entry',
            database_id: dbId,
            last_edited_time: entry.lastEditedTime.toISOString(),
          },
          modifiedAt: entry.lastEditedTime,
        }
      }

      if (!resp.has_more || resp.next_cursor === null) break
      cursor = resp.next_cursor
    }
  }

  // ---------- Workspace search ----------

  async function* searchWorkspace(
    signal: AbortSignal,
    sinceISO: string | undefined,
    visited: Set<string>,
    cfg: NotionConnectorConfig,
  ): AsyncGenerator<ConnectorDocument> {
    let cursor: string | undefined

    while (!signal.aborted) {
      const body: Record<string, unknown> = {
        page_size: NOTION_DEFAULT_PAGE_SIZE,
      }
      if (sinceISO !== undefined) {
        body.sort = {
          direction: 'descending',
          timestamp: 'last_edited_time',
        }
      }
      if (cursor !== undefined) {
        body.start_cursor = cursor
      }

      const resp = (await doRequest(
        signal,
        'POST',
        '/search',
        body,
      )) as NotionPaginatedResponse

      for (const result of resp.results) {
        const objectType = result.object as string
        const id = result.id as string
        const lastEditedStr = result.last_edited_time as string

        // For incremental sync with descending sort, stop when we
        // pass the cursor boundary.
        if (sinceISO !== undefined && lastEditedStr !== undefined) {
          const editedTime = new Date(lastEditedStr)
          const cursorTime = new Date(sinceISO)
          if (editedTime < cursorTime) return
        }

        if (objectType === 'page') {
          yield* fetchPageTree(signal, id, 0, sinceISO, visited, cfg)
        } else if (objectType === 'database' && cfg.includeDatabases !== false) {
          yield* fetchDatabaseEntries(signal, id, sinceISO, visited, cfg)
        }
      }

      if (!resp.has_more || resp.next_cursor === null) break
      cursor = resp.next_cursor
    }
  }

  // ---------- Sync orchestrator ----------

  async function* syncPages(
    signal: AbortSignal,
    sinceISO: string | undefined,
  ): AsyncGenerator<ConnectorDocument> {
    const cfg = ensureConfigured()
    const visited = new Set<string>()

    // Fetch specific root pages.
    const rootPageIds = cfg.rootPageIds ?? []
    for (const pageId of rootPageIds) {
      yield* fetchPageTree(signal, pageId, 0, sinceISO, visited, cfg)
    }

    // Fetch specific databases.
    if (cfg.includeDatabases !== false) {
      const databaseIds = cfg.databaseIds ?? []
      for (const dbId of databaseIds) {
        yield* fetchDatabaseEntries(signal, dbId, sinceISO, visited, cfg)
      }
    }

    // If no specific targets, search entire workspace.
    if (rootPageIds.length === 0 && (cfg.databaseIds ?? []).length === 0) {
      yield* searchWorkspace(signal, sinceISO, visited, cfg)
    }
  }

  // ---------- Connector interface ----------

  return {
    name: 'notion' as const,

    async configure(rawConfig: Record<string, unknown>): Promise<void> {
      const token = rawConfig.apiToken as string | undefined
      if (token === undefined || token.trim() === '') {
        throw new Error('connector/notion: apiToken is required')
      }

      config = {
        apiToken: token,
        rootPageIds: parseStringArray(rawConfig.rootPageIds),
        databaseIds: parseStringArray(rawConfig.databaseIds),
        includeChildPages: (rawConfig.includeChildPages as boolean) ?? true,
        includeDatabases: (rawConfig.includeDatabases as boolean) ?? true,
        maxDepth: (rawConfig.maxDepth as number) ?? NOTION_DEFAULT_MAX_DEPTH,
        blockTypeFilter: parseBlockTypeFilter(rawConfig.blockTypeFilter),
      }
    },

    async *fetchAll(signal: AbortSignal): AsyncIterable<ConnectorDocument> {
      yield* syncPages(signal, undefined)
    },

    async *fetchSince(
      signal: AbortSignal,
      cursor: SyncCursor,
    ): AsyncIterable<ConnectorDocument> {
      yield* syncPages(signal, cursor.value)
    },

    async start(signal: AbortSignal): Promise<void> {
      ensureConfigured()
      abortController = new AbortController()

      const interval = deps.pollInterval ?? NOTION_DEFAULT_POLL_INTERVAL

      while (!signal.aborted && !abortController.signal.aborted) {
        const combinedSignal = combineSignals(
          signal,
          abortController.signal,
        )
        for await (const _doc of syncPages(combinedSignal, undefined)) {
          // Documents consumed by the sync loop.
        }

        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, interval)
          const onAbort = (): void => {
            clearTimeout(timer)
            resolve()
          }
          signal.addEventListener('abort', onAbort, { once: true })
          abortController?.signal.addEventListener('abort', onAbort, {
            once: true,
          })
        })
      }
    },

    async stop(): Promise<void> {
      abortController?.abort()
    },
  }
}

