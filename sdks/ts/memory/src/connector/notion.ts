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
  RateLimiter,
  SyncCursor,
} from './types.js'

// ---------- Constants ----------

const NOTION_API_VERSION = '2022-06-28'
const NOTION_DEFAULT_BASE_URL = 'https://api.notion.com/v1'
const NOTION_DEFAULT_MAX_DEPTH = 10
const NOTION_DEFAULT_POLL_INTERVAL = 15 * 60 * 1000
const NOTION_DEFAULT_TIMEOUT = 30_000
const NOTION_DEFAULT_PAGE_SIZE = 100

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
}

/**
 * Notion rich text object.
 */
type NotionRichText = {
  readonly type: string
  readonly plain_text: string
  readonly href?: string
  readonly annotations?: NotionAnnotations
}

/**
 * Notion text formatting annotations.
 */
type NotionAnnotations = {
  readonly bold?: boolean
  readonly italic?: boolean
  readonly strikethrough?: boolean
  readonly underline?: boolean
  readonly code?: boolean
  readonly color?: string
}

/**
 * Notion block object from the blocks API.
 */
type NotionBlock = {
  readonly id: string
  readonly type: string
  readonly has_children: boolean
  readonly [key: string]: unknown
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
 * Parsed page metadata.
 */
type ParsedPage = {
  readonly id: string
  readonly title: string
  readonly url: string
  readonly lastEditedTime: Date
}

/**
 * Parsed database entry.
 */
type ParsedDatabaseEntry = {
  readonly pageId: string
  readonly title: string
  readonly url: string
  readonly lastEditedTime: Date
  readonly propertiesMarkdown: string
}

/**
 * HTTP fetcher interface for dependency injection in tests.
 */
export type NotionHTTPFetcher = (
  url: string,
  options: RequestInit,
) => Promise<Response>

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
 */
export const createNotionConnector = (
  deps: ConnectorConfig,
  options?: NotionConnectorOptions,
): Connector & { readonly name: 'notion' } => {
  let config: NotionConnectorConfig | undefined
  let abortController: AbortController | undefined
  const baseUrl = options?.baseUrl ?? NOTION_DEFAULT_BASE_URL
  const fetcher: NotionHTTPFetcher = options?.fetcher ?? globalFetch
  const rateLimiter: RateLimiter | undefined = deps.rateLimiter

  const ensureConfigured = (): NotionConnectorConfig => {
    if (config === undefined) {
      throw new Error('connector/notion: not configured, call configure first')
    }
    return config
  }

  const acquireRateLimit = async (signal: AbortSignal): Promise<void> => {
    if (rateLimiter !== undefined) {
      await rateLimiter.acquire(signal, 1)
    }
  }

  const doRequest = async (
    signal: AbortSignal,
    method: string,
    path: string,
    body?: Record<string, unknown>,
  ): Promise<unknown> => {
    const cfg = ensureConfigured()

    await acquireRateLimit(signal)

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
        throw new Error('connector/notion: rate limited (429)')
      }

      if (response.status === 404) {
        throw new Error('connector/notion: not found (404)')
      }

      if (response.status < 200 || response.status >= 300) {
        const text = await response.text()
        throw new Error(
          `connector/notion: HTTP ${String(response.status)}: ${text}`,
        )
      }

      return (await response.json()) as unknown
    } finally {
      clearTimeout(timeoutId)
    }
  }

  // ---------- Page fetching ----------

  const parsePageResponse = (data: Record<string, unknown>): ParsedPage => {
    const properties = (data.properties ?? {}) as Record<
      string,
      Record<string, unknown>
    >
    const title = extractTitleFromProperties(properties)
    const lastEditedStr = data.last_edited_time as string
    const lastEditedTime = new Date(lastEditedStr)

    return {
      id: data.id as string,
      title,
      url: data.url as string,
      lastEditedTime,
    }
  }

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

    const parts: string[] = []

    for (const block of allBlocks) {
      const md = blockToMarkdown(block)
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
    const childIds: string[] = []
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
        const block = result as unknown as NotionBlock
        if (block.type === 'child_page' || block.type === 'child_database') {
          childIds.push(block.id)
        }
      }

      if (!resp.has_more || resp.next_cursor === null) break
      cursor = resp.next_cursor
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
    } catch (err) {
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

        await new Promise<void>((resolve, reject) => {
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

// ---------- Helper functions ----------

/**
 * Renders a Notion rich text array to markdown with formatting.
 */
const renderRichText = (texts: readonly NotionRichText[]): string => {
  const parts: string[] = []

  for (const t of texts) {
    let text = t.plain_text

    if (t.annotations?.code === true) {
      text = '`' + text + '`'
    }
    if (t.annotations?.bold === true) {
      text = '**' + text + '**'
    }
    if (t.annotations?.italic === true) {
      text = '*' + text + '*'
    }
    if (t.annotations?.strikethrough === true) {
      text = '~~' + text + '~~'
    }

    if (t.href !== undefined && t.href !== '') {
      text = `[${text}](${t.href})`
    }

    parts.push(text)
  }

  return parts.join('')
}

/**
 * Extracts plain text from rich text without formatting.
 */
const renderPlainRichText = (texts: readonly NotionRichText[]): string =>
  texts.map((t) => t.plain_text).join('')

/**
 * Converts a single Notion block to markdown.
 */
const blockToMarkdown = (block: NotionBlock): string => {
  const blockData = block[block.type] as Record<string, unknown> | undefined
  if (blockData === undefined) return ''

  const richText = (blockData.rich_text ?? []) as readonly NotionRichText[]
  const text = renderRichText(richText)
  const caption = (blockData.caption ?? []) as readonly NotionRichText[]

  const renderers: Readonly<Record<string, () => string>> = {
    paragraph: () => `${text}\n`,
    heading_1: () => `# ${text}\n`,
    heading_2: () => `## ${text}\n`,
    heading_3: () => `### ${text}\n`,
    bulleted_list_item: () => `- ${text}`,
    numbered_list_item: () => `1. ${text}`,
    to_do: () => {
      const checked = blockData.checked === true
      const marker = checked ? '[x]' : '[ ]'
      return `- ${marker} ${text}`
    },
    toggle: () => `- ${text}`,
    quote: () => `> ${text}\n`,
    callout: () => `> ${text}\n`,
    divider: () => '---\n',
    code: () => {
      const lang = (blockData.language as string) ?? ''
      return '```' + lang + '\n' + text + '\n```\n'
    },
    equation: () => {
      const expression = (blockData.expression as string) ?? ''
      return `$$\n${expression}\n$$\n`
    },
    image: () => {
      const capText = renderRichText(caption)
      const url = extractFileUrl(blockData)
      return capText !== ''
        ? `![${capText}](${url})\n`
        : `![](${url})\n`
    },
    file: () => {
      const url = extractFileUrl(blockData)
      const capText = renderRichText(caption)
      const label = capText !== '' ? capText : 'file'
      return `[${label}](${url})\n`
    },
    bookmark: () => {
      const capText = renderRichText(caption)
      const url = blockData.url as string | undefined
      if (url !== undefined) {
        return capText !== '' ? `[${capText}](${url})\n` : `${url}\n`
      }
      return ''
    },
    embed: () => {
      const url = blockData.url as string | undefined
      return url !== undefined ? `${url}\n` : ''
    },
    table_of_contents: () => '',
    breadcrumb: () => '',
    column_list: () => '',
    column: () => '',
    child_page: () => '',
    child_database: () => '',
    synced_block: () => '',
    link_preview: () => {
      const url = blockData.url as string | undefined
      return url !== undefined ? `${url}\n` : ''
    },
    table: () => '',
    table_row: () => {
      const cells = (blockData.cells ?? []) as readonly (readonly NotionRichText[])[]
      const rendered = cells.map((cell) => renderRichText(cell))
      return `| ${rendered.join(' | ')} |`
    },
  }

  const renderer = renderers[block.type]
  return renderer !== undefined ? renderer() : text
}

/**
 * Extracts the title from a page's properties map.
 */
const extractTitleFromProperties = (
  properties: Record<string, Record<string, unknown>>,
): string => {
  for (const prop of Object.values(properties)) {
    if (prop.type === 'title') {
      const titleArr = (prop.title ?? []) as readonly NotionRichText[]
      return renderPlainRichText(titleArr)
    }
  }
  return ''
}

/**
 * Converts a Notion property to its string representation.
 */
const extractPropertyValue = (
  prop: Record<string, unknown>,
): string => {
  const propType = prop.type as string

  const renderers: Readonly<Record<string, () => string>> = {
    title: () =>
      renderPlainRichText((prop.title ?? []) as readonly NotionRichText[]),
    rich_text: () =>
      renderPlainRichText(
        (prop.rich_text ?? []) as readonly NotionRichText[],
      ),
    number: () => {
      const num = prop.number as number | null
      return num !== null && num !== undefined ? String(num) : ''
    },
    select: () => {
      const sel = prop.select as { name: string } | null
      return sel?.name ?? ''
    },
    multi_select: () => {
      const items = (prop.multi_select ?? []) as readonly { name: string }[]
      return items.map((i) => i.name).join(', ')
    },
    date: () => {
      const d = prop.date as { start: string; end?: string } | null
      if (d === null || d === undefined) return ''
      return d.end !== undefined && d.end !== ''
        ? `${d.start} to ${d.end}`
        : d.start
    },
    checkbox: () => (prop.checkbox === true ? 'true' : 'false'),
    url: () => (prop.url as string) ?? '',
    email: () => (prop.email as string) ?? '',
    phone_number: () => (prop.phone_number as string) ?? '',
    status: () => {
      const st = prop.status as { name: string } | null
      return st?.name ?? ''
    },
    people: () => {
      const people = (prop.people ?? []) as readonly { name: string }[]
      return people.map((p) => p.name).join(', ')
    },
    relation: () => {
      const rels = (prop.relation ?? []) as readonly { id: string }[]
      return rels.map((r) => r.id).join(', ')
    },
  }

  const renderer = renderers[propType]
  return renderer !== undefined ? renderer() : ''
}

/**
 * Parses a database entry from the query response.
 */
const parseDatabaseEntry = (
  raw: Record<string, unknown>,
): ParsedDatabaseEntry => {
  const properties = (raw.properties ?? {}) as Record<
    string,
    Record<string, unknown>
  >
  const lastEditedStr = raw.last_edited_time as string
  const lastEditedTime = new Date(lastEditedStr)

  let title = ''
  const propLines: string[] = []

  // Sort keys for deterministic output.
  const sortedKeys = Object.keys(properties).sort()

  for (const key of sortedKeys) {
    const prop = properties[key]
    if (prop === undefined) continue
    const val = extractPropertyValue(prop)

    if (prop.type === 'title') {
      title = val
      continue
    }

    if (val !== '') {
      propLines.push(`- **${key}**: ${val}`)
    }
  }

  let markdown = ''
  if (title !== '') {
    markdown += `## ${title}\n\n`
  }
  if (propLines.length > 0) {
    markdown += propLines.join('\n') + '\n'
  }

  return {
    pageId: raw.id as string,
    title,
    url: raw.url as string,
    lastEditedTime,
    propertiesMarkdown: markdown,
  }
}

/**
 * Extracts a file URL from a file-type block's data.
 */
const extractFileUrl = (data: Record<string, unknown>): string => {
  const file = data.file as { url?: string } | undefined
  const external = data.external as { url?: string } | undefined
  return file?.url ?? external?.url ?? ''
}

/**
 * Checks whether a block type is a list item.
 */
const isListBlock = (blockType: string): boolean => {
  const listTypes = new Set([
    'bulleted_list_item',
    'numbered_list_item',
    'to_do',
    'toggle',
  ])
  return listTypes.has(blockType)
}

/**
 * Parses a value that might be a string array or undefined.
 */
const parseStringArray = (
  value: unknown,
): readonly string[] | undefined => {
  if (value === undefined || value === null) return undefined
  if (Array.isArray(value)) {
    return value.filter((v): v is string => typeof v === 'string')
  }
  return undefined
}

/**
 * Combines two abort signals into one that fires when either fires.
 */
const combineSignals = (a: AbortSignal, b: AbortSignal): AbortSignal => {
  if (a.aborted) return a
  if (b.aborted) return b

  const controller = new AbortController()
  const onAbort = (): void => controller.abort()
  a.addEventListener('abort', onAbort, { once: true })
  b.addEventListener('abort', onAbort, { once: true })
  return controller.signal
}

/**
 * Global fetch wrapper for the default HTTP fetcher.
 */
const globalFetch: NotionHTTPFetcher = (url, options) => fetch(url, options)
