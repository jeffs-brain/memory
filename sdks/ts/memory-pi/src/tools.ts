// SPDX-License-Identifier: Apache-2.0

/**
 * The eleven `memory_*` tools registered with the pi runtime. Each tool
 * wraps an operation on the underlying `@jeffs-brain/memory` pipeline
 * (Memory + Retrieval + Store + SearchIndex). Tools are intentionally
 * thin: validation lives in the TypeBox schema, the heavy lifting lives
 * in the SDK.
 *
 * Reference parity with `mcp/ts/src/tools/*` so MCP clients and pi
 * extensions surface the same canonical eleven operations.
 */

import { createHash, randomUUID } from 'node:crypto'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { extname, isAbsolute, join, resolve as resolvePath } from 'node:path'
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent'
import type { Batch, ExtractedMemory, Message, RecallHit, Scope } from '@jeffs-brain/memory'
import { ingestDocument } from '@jeffs-brain/memory/ingest'
import type { Static, TSchema } from 'typebox'
import { Type } from 'typebox'
import { MEMORY_TOOL_NAMES, type MemoryToolName } from './config.js'
import type { MemoryRuntime } from './runtime.js'

const FILE_LIMIT_BYTES = 25 * 1024 * 1024
const URL_FETCH_LIMIT_BYTES = 5 * 1024 * 1024

const SCOPE_VALUES = ['global', 'project', 'agent'] as const

const SEARCH_SCOPE_VALUES = ['all', ...SCOPE_VALUES] as const

const SORT_VALUES = ['relevance', 'recency', 'relevance_then_recency'] as const

const VISIBILITY_VALUES = ['private', 'tenant', 'public'] as const

const INGEST_AS_VALUES = ['markdown', 'text', 'pdf', 'json'] as const

const ROLE_VALUES = ['system', 'user', 'assistant', 'tool'] as const

const FS_MIME_BY_AS: Record<(typeof INGEST_AS_VALUES)[number], string> = {
  markdown: 'text/markdown',
  text: 'text/plain',
  pdf: 'application/pdf',
  json: 'application/json',
}

const FS_MIME_BY_EXT: Record<string, string> = {
  '.md': 'text/markdown',
  '.markdown': 'text/markdown',
  '.txt': 'text/plain',
  '.pdf': 'application/pdf',
  '.json': 'application/json',
  '.html': 'text/html',
  '.htm': 'text/html',
}

const mimeForFile = (path: string, hint: string | undefined): string => {
  if (hint !== undefined && hint in FS_MIME_BY_AS) {
    return FS_MIME_BY_AS[hint as keyof typeof FS_MIME_BY_AS]
  }
  const ext = extname(path).toLowerCase()
  return FS_MIME_BY_EXT[ext] ?? 'application/octet-stream'
}

const deriveTitle = (content: string, fallback: string | undefined): string => {
  if (fallback !== undefined && fallback !== '') return fallback
  const firstHeading = content.match(/^#\s+(.+)$/m)?.[1]?.trim()
  if (firstHeading !== undefined && firstHeading !== '') return firstHeading
  const firstLine = content
    .split('\n')
    .find((line) => line.trim() !== '')
    ?.trim()
  return firstLine ?? 'Untitled memory'
}

const slugFromTitle = (title: string): string => {
  const slug = title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return slug === '' ? `note-${Date.now()}` : slug.slice(0, 64)
}

const inferMemoryScopeFromPath = (path: string): Scope => {
  if (path.startsWith('memory/project/')) return 'project'
  if (path.startsWith('memory/agent/')) return 'agent'
  return 'global'
}

const buildFrontmatterBlock = (frontmatter: Record<string, unknown>): string => {
  const lines: string[] = ['---']
  for (const [key, value] of Object.entries(frontmatter)) {
    if (value === undefined || value === null) continue
    if (Array.isArray(value)) {
      lines.push(`${key}: [${value.map((v) => JSON.stringify(String(v))).join(', ')}]`)
      continue
    }
    lines.push(`${key}: ${JSON.stringify(value)}`)
  }
  lines.push('---')
  return lines.join('\n')
}

type RankedHit = { readonly path: string; readonly score: number }

const sortLocalHits = async <T extends RankedHit>(
  store: MemoryRuntime['store'],
  hits: readonly T[],
  sort: (typeof SORT_VALUES)[number] | undefined,
): Promise<readonly T[]> => {
  if (sort === undefined || sort === 'relevance') return [...hits]
  const stamped: Array<{ readonly hit: T; readonly index: number; readonly timestamp: number }> =
    await Promise.all(
      hits.map(async (hit, index) => {
        try {
          const info = await store.stat(hit.path as never)
          return { hit, index, timestamp: info.modTime.getTime() }
        } catch {
          return { hit, index, timestamp: Number.NEGATIVE_INFINITY }
        }
      }),
    )
  return stamped
    .sort((left, right) => {
      if (sort === 'relevance_then_recency' && left.hit.score !== right.hit.score) {
        return right.hit.score - left.hit.score
      }
      if (left.timestamp !== right.timestamp) return right.timestamp - left.timestamp
      if (sort !== 'relevance_then_recency' && left.hit.score !== right.hit.score) {
        return right.hit.score - left.hit.score
      }
      return left.index - right.index
    })
    .map(({ hit }) => hit)
}

const renderResult = (
  value: unknown,
): {
  readonly content: ReadonlyArray<{ readonly type: 'text'; readonly text: string }>
  readonly details: { readonly result: unknown }
} => ({
  content: [{ type: 'text', text: JSON.stringify(value, null, 2) }],
  details: { result: value },
})

const noProviderError = (): Error =>
  new Error(
    'memory-pi: no LLM provider configured; set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally',
  )

// ----- Schemas ----------------------------------------------------------

const rememberSchema = Type.Object({
  content: Type.String({ description: 'Markdown body of the new memory.' }),
  title: Type.Optional(
    Type.String({ description: 'Title. Derived from the first heading if omitted.' }),
  ),
  tags: Type.Optional(Type.Array(Type.String())),
  path: Type.Optional(Type.String({ description: 'Destination path within the brain.' })),
})

const recallSchema = Type.Object({
  query: Type.String({ description: 'Free-text query the recall pipeline scores.' }),
  scope: Type.Optional(Type.Union(SCOPE_VALUES.map((v) => Type.Literal(v)))),
  session_id: Type.Optional(Type.String()),
  top_k: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
})

const searchSchema = Type.Object({
  query: Type.String(),
  top_k: Type.Optional(Type.Integer({ minimum: 1, maximum: 100 })),
  scope: Type.Optional(Type.Union(SEARCH_SCOPE_VALUES.map((v) => Type.Literal(v)))),
  sort: Type.Optional(Type.Union(SORT_VALUES.map((v) => Type.Literal(v)))),
})

const askSchema = Type.Object({
  query: Type.String(),
  top_k: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
})

const ingestFileSchema = Type.Object({
  path: Type.String({ description: 'Absolute or relative local path.' }),
  as: Type.Optional(Type.Union(INGEST_AS_VALUES.map((v) => Type.Literal(v)))),
})

const ingestUrlSchema = Type.Object({
  url: Type.String({ description: 'URL to fetch and ingest.' }),
})

const extractMessageSchema = Type.Object({
  role: Type.Union(ROLE_VALUES.map((v) => Type.Literal(v))),
  content: Type.String(),
})

const extractSchema = Type.Object({
  messages: Type.Array(extractMessageSchema),
  actor_id: Type.Optional(Type.String()),
  session_id: Type.Optional(Type.String()),
})

const reflectSchema = Type.Object({
  session_id: Type.String(),
})

const consolidateSchema = Type.Object({})

const createBrainSchema = Type.Object({
  name: Type.String(),
  slug: Type.Optional(Type.String()),
  visibility: Type.Optional(Type.Union(VISIBILITY_VALUES.map((v) => Type.Literal(v)))),
})

const listBrainsSchema = Type.Object({})

// ----- Tool factory -----------------------------------------------------

type AnyToolParams<T extends TSchema> = T

const safeScope = (raw: 'all' | Scope | undefined): { readonly scope: Scope } | undefined => {
  if (raw === undefined || raw === 'all') return undefined
  return { scope: raw }
}

/**
 * Build the set of pi tool definitions for a runtime. The factory does
 * not register them; the caller drives `pi.registerTool(...)` so we can
 * filter by `tools.expose` if the user supplied an allowlist.
 */
export const buildTools = (
  runtime: MemoryRuntime,
): Record<MemoryToolName, ReturnType<typeof defineMemoryTool<TSchema>>> => {
  const cfg = runtime.config
  return {
    remember: defineMemoryTool({
      name: 'memory_remember',
      label: 'Remember',
      description:
        'Store a new memory note (markdown) in the brain. Returns the created note path and digest.',
      parameters: rememberSchema,
      async execute(_id, params) {
        const title = deriveTitle(params.content, params.title)
        const slug = slugFromTitle(title)
        const relativePath =
          params.path !== undefined && params.path !== '' ? params.path : `memory/global/${slug}.md`
        const scope = inferMemoryScopeFromPath(relativePath)
        const nowIso = new Date().toISOString()
        const frontmatter = buildFrontmatterBlock({
          title,
          scope,
          type: scope === 'global' ? 'user' : 'project',
          created: nowIso,
          modified: nowIso,
          ...(params.tags !== undefined && params.tags.length > 0 ? { tags: params.tags } : {}),
        })
        const body = `${frontmatter}\n\n${params.content.trimEnd()}\n`
        const buf = Buffer.from(body, 'utf8')
        const embedder = runtime.embedder
        if (embedder !== undefined) {
          const result = await ingestDocument({
            store: runtime.store,
            searchIndex: runtime.searchIndex,
            embedder,
            doc: {
              brainId: cfg.brainId,
              path: relativePath,
              content: buf,
              title,
              mime: 'text/markdown',
              metadata: {
                scope,
                ...(params.tags !== undefined && params.tags.length > 0
                  ? { tags: params.tags.join(',') }
                  : {}),
              },
            },
          })
          return renderResult({
            id: result.documentId,
            path: result.path,
            byte_size: buf.length,
            hash: result.hash,
            chunk_count: result.chunkCount,
            embedded_count: result.embeddedCount,
            reused: result.reused,
          })
        }
        // No embedder: write the note directly and index a single BM25
        // chunk so search still finds it.
        await runtime.store.batch({ reason: 'memory_remember' }, async (batch: Batch) => {
          await batch.write(relativePath as never, buf)
        })
        const hash = createHash('sha256').update(buf).digest('hex')
        runtime.searchIndex.upsertChunk({
          id: `${cfg.brainId}:${hash.slice(0, 16)}:0`,
          path: relativePath,
          ordinal: 0,
          title,
          content: body,
          metadata: {
            path: relativePath,
            brainId: cfg.brainId,
            scope,
            ...(params.tags !== undefined && params.tags.length > 0
              ? { tags: params.tags.join(',') }
              : {}),
          },
        })
        return renderResult({
          id: `${cfg.brainId}:${hash.slice(0, 16)}`,
          path: relativePath,
          byte_size: buf.length,
          hash,
          chunk_count: 1,
          embedded_count: 0,
          reused: false,
        })
      },
    }),

    recall: defineMemoryTool({
      name: 'memory_recall',
      label: 'Recall',
      description:
        'Recall memory notes scored for a query. `scope` selects the memory namespace (global, project, agent).',
      parameters: recallSchema,
      async execute(_id, params) {
        const topK = params.top_k ?? cfg.recallTopK
        const hits = await runtime.memory.recall({
          query: params.query,
          k: topK,
          scope: (params.scope as Scope | undefined) ?? cfg.scope,
          actorId: cfg.actorId,
          ...(cfg.fallbackScopes.length > 0 ? { fallbackScopes: cfg.fallbackScopes } : {}),
        })
        return renderResult({
          query: params.query,
          brain_id: cfg.brainId,
          session_id: params.session_id ?? null,
          chunks: hits.map((hit: RecallHit) => ({
            score: hit.score,
            path: hit.path,
            content: hit.content,
            title: hit.note.name,
            scope: hit.note.scope,
            tags: hit.note.tags,
          })),
        })
      },
    }),

    search: defineMemoryTool({
      name: 'memory_search',
      label: 'Search',
      description:
        'Hybrid BM25 + vector retrieval across the brain. Returns ranked chunks with citations.',
      parameters: searchSchema,
      async execute(_id, params) {
        const topK = params.top_k ?? 10
        const filters = safeScope(params.scope)
        const results = await runtime.retrieval.search({
          query: params.query,
          topK,
          rerank: false,
          ...(filters !== undefined ? { filters } : {}),
        })
        const ordered = await sortLocalHits(runtime.store, results, params.sort)
        return renderResult({
          query: params.query,
          brain_id: cfg.brainId,
          hits: ordered.map((result) => ({
            score: result.score,
            path: result.path,
            content: result.content,
            title: result.title,
            summary: result.summary,
            chunk_id: result.id,
          })),
        })
      },
    }),

    ask: defineMemoryTool({
      name: 'memory_ask',
      label: 'Ask',
      description:
        'Answer a question grounded in the brain. Retrieves relevant chunks, calls the configured LLM provider, and returns the answer with citations.',
      parameters: askSchema,
      async execute(_id, params, signal) {
        const provider = runtime.provider
        if (provider === undefined) throw noProviderError()
        const topK = params.top_k ?? 8
        const retrieved = await runtime.retrieval.search({
          query: params.query,
          topK,
          rerank: false,
        })
        const contextLines = retrieved.map((hit, idx) => {
          const body = (hit.content ?? '').slice(0, 2000)
          return `### [${idx + 1}] ${hit.path}\n${body}`
        })
        const prompt = [
          'Answer the user question using the retrieved memory chunks below.',
          'Cite supporting chunks inline using the numeric markers [1], [2], etc.',
          'If the chunks do not answer the question, say so plainly.',
          '',
          '## Question',
          params.query,
          '',
          '## Retrieved memory',
          contextLines.join('\n\n') || '_no relevant chunks_',
        ].join('\n')
        const completion = await provider.complete(
          {
            system: 'You are an assistant grounded in this brain. Use British English.',
            messages: [{ role: 'user', content: prompt }],
            maxTokens: 1024,
            temperature: 0.2,
          },
          signal,
        )
        return renderResult({
          answer: completion.content,
          citations: retrieved.slice(0, 5).map((hit, idx) => ({
            chunk_id: hit.id,
            document_id: hit.path,
            quote: (hit.content ?? '').slice(0, 200),
            rank: idx + 1,
          })),
          retrieved: retrieved.map((hit) => ({
            chunk_id: hit.id,
            document_id: hit.path,
            score: hit.score,
            preview: (hit.content ?? '').slice(0, 512),
          })),
        })
      },
    }),

    ingest_file: defineMemoryTool({
      name: 'memory_ingest_file',
      label: 'Ingest File',
      description: 'Ingest a local file (<= 25 MiB) into the brain. Returns the ingest result.',
      parameters: ingestFileSchema,
      async execute(_id, params) {
        const absPath = isAbsolute(params.path) ? params.path : resolvePath(params.path)
        const info = await stat(absPath)
        if (!info.isFile()) {
          throw new Error(`memory_ingest_file: not a regular file: ${absPath}`)
        }
        if (info.size > FILE_LIMIT_BYTES) {
          throw new Error('memory_ingest_file: 25 MiB limit exceeded')
        }
        const content = await readFile(absPath)
        const mime = mimeForFile(absPath, params.as)
        const title = absPath.split(/[\\/]/).pop() ?? absPath
        const embedder = runtime.embedder
        if (embedder === undefined) {
          const hash = createHash('sha256').update(content).digest('hex')
          const storedPath = `raw/documents/${hash}${extname(absPath).toLowerCase() || '.bin'}`
          await runtime.store.batch({ reason: 'memory_ingest_file' }, async (batch: Batch) => {
            await batch.write(storedPath as never, content)
          })
          runtime.searchIndex.upsertChunk({
            id: `${cfg.brainId}:${hash.slice(0, 16)}:0`,
            path: storedPath,
            ordinal: 0,
            title,
            content: content.toString('utf8').slice(0, 32_000),
            metadata: { path: storedPath, brainId: cfg.brainId, mime },
          })
          return renderResult({
            status: 'completed',
            document_id: `${cfg.brainId}:${hash.slice(0, 16)}`,
            path: storedPath,
            hash,
            chunk_count: 1,
            embedded_count: 0,
            reused: false,
          })
        }
        const result = await ingestDocument({
          store: runtime.store,
          searchIndex: runtime.searchIndex,
          embedder,
          doc: {
            brainId: cfg.brainId,
            content,
            mime,
            title,
            metadata: { source: absPath },
          },
        })
        return renderResult({
          status: 'completed',
          document_id: result.documentId,
          path: result.path,
          hash: result.hash,
          chunk_count: result.chunkCount,
          embedded_count: result.embeddedCount,
          duration_ms: result.durationMs,
          reused: result.reused,
        })
      },
    }),

    ingest_url: defineMemoryTool({
      name: 'memory_ingest_url',
      label: 'Ingest URL',
      description: 'Fetch a URL and ingest its contents into the brain.',
      parameters: ingestUrlSchema,
      async execute(_id, params, signal) {
        const init: RequestInit = signal !== undefined ? { signal } : {}
        const resp = await fetch(params.url, init)
        if (!resp.ok) {
          throw new Error(`memory_ingest_url: fetch failed ${resp.status} ${resp.statusText}`)
        }
        const contentTypeHeader = resp.headers.get('content-type') ?? 'text/plain'
        const mime = contentTypeHeader.split(';')[0]?.trim() ?? 'text/plain'
        const buffer = Buffer.from(await resp.arrayBuffer())
        if (buffer.length > URL_FETCH_LIMIT_BYTES) {
          throw new Error('memory_ingest_url: body exceeds 5 MiB fallback limit')
        }
        const embedder = runtime.embedder
        if (embedder === undefined) {
          const hash = createHash('sha256').update(buffer).digest('hex')
          const storedPath = `raw/documents/${hash}.txt`
          await runtime.store.batch({ reason: 'memory_ingest_url' }, async (batch: Batch) => {
            await batch.write(storedPath as never, buffer)
          })
          runtime.searchIndex.upsertChunk({
            id: `${cfg.brainId}:${hash.slice(0, 16)}:0`,
            path: storedPath,
            ordinal: 0,
            title: params.url,
            content: buffer.toString('utf8').slice(0, 32_000),
            metadata: { path: storedPath, brainId: cfg.brainId, source_url: params.url, mime },
          })
          return renderResult({
            status: 'completed',
            document_id: `${cfg.brainId}:${hash.slice(0, 16)}`,
            path: storedPath,
            hash,
            chunk_count: 1,
            embedded_count: 0,
            reused: false,
          })
        }
        const result = await ingestDocument({
          store: runtime.store,
          searchIndex: runtime.searchIndex,
          embedder,
          doc: {
            brainId: cfg.brainId,
            content: buffer,
            mime,
            title: params.url,
            metadata: { source_url: params.url },
          },
        })
        return renderResult({
          status: 'completed',
          document_id: result.documentId,
          path: result.path,
          hash: result.hash,
          chunk_count: result.chunkCount,
          embedded_count: result.embeddedCount,
          duration_ms: result.durationMs,
          reused: result.reused,
        })
      },
    }),

    extract: defineMemoryTool({
      name: 'memory_extract',
      label: 'Extract',
      description:
        'Submit a conversation transcript so the brain extracts memorable facts. Returns the extracted notes.',
      parameters: extractSchema,
      async execute(_id, params) {
        if (runtime.provider === undefined) throw noProviderError()
        const messages: readonly Message[] = params.messages.map((m) => ({
          role: m.role,
          content: m.content,
        }))
        const extracted: readonly ExtractedMemory[] = await runtime.memory.extract({
          messages,
          ...(params.session_id !== undefined ? { sessionId: params.session_id } : {}),
          ...(params.actor_id !== undefined ? { actorId: params.actor_id } : {}),
        })
        runtime.lastExtracted = extracted
        return renderResult({
          mode: params.session_id !== undefined ? 'session' : 'transcript',
          extracted: extracted.map((m) => ({
            filename: m.filename,
            name: m.name,
            description: m.description,
            type: m.type,
            scope: m.scope,
            tags: m.tags ?? [],
            action: m.action,
          })),
        })
      },
    }),

    reflect: defineMemoryTool({
      name: 'memory_reflect',
      label: 'Reflect',
      description: 'Close a session and trigger reflection over its messages.',
      parameters: reflectSchema,
      async execute(_id, params) {
        if (runtime.provider === undefined) throw noProviderError()
        const result = await runtime.memory.reflect({
          sessionId: params.session_id,
          messages: [],
        })
        if (result === undefined) {
          return renderResult({
            reflection_status: 'no_result',
            ended_at: new Date().toISOString(),
          })
        }
        return renderResult({
          reflection_status: 'completed',
          reflection: {
            outcome: result.outcome,
            should_record_episode: result.shouldRecordEpisode ?? false,
            path: result.path,
            retry_feedback: result.retryFeedback ?? '',
          },
          ended_at: new Date().toISOString(),
        })
      },
    }),

    consolidate: defineMemoryTool({
      name: 'memory_consolidate',
      label: 'Consolidate',
      description:
        'Run a consolidation pass on the brain (compile summaries, promote stable notes, prune stale episodic memory).',
      parameters: consolidateSchema,
      async execute(_id) {
        if (runtime.provider === undefined) throw noProviderError()
        const report = await runtime.memory.consolidate({})
        return renderResult({ result: report })
      },
    }),

    create_brain: defineMemoryTool({
      name: 'memory_create_brain',
      label: 'Create Brain',
      description: 'Create a new brain alongside the current one under the configured brain root.',
      parameters: createBrainSchema,
      async execute(_id, params) {
        const slug = params.slug ?? slugFromTitle(params.name)
        const brainDir = join(cfg.brainRoot, slug.replace(/[^A-Za-z0-9._-]/g, '_'))
        await mkdir(brainDir, { recursive: true })
        const configPath = join(brainDir, 'config.json')
        const configBody = {
          version: 1,
          name: params.name,
          slug,
          visibility: params.visibility ?? 'private',
          createdAt: new Date().toISOString(),
        }
        await writeFile(configPath, `${JSON.stringify(configBody, null, 2)}\n`, 'utf8')
        return renderResult({
          id: slug,
          slug,
          name: params.name,
          visibility: configBody.visibility,
          created_at: configBody.createdAt,
        })
      },
    }),

    list_brains: defineMemoryTool({
      name: 'memory_list_brains',
      label: 'List Brains',
      description: 'List the brains discoverable under the configured brain root.',
      parameters: listBrainsSchema,
      async execute() {
        const exists = await stat(cfg.brainRoot).then(
          (s) => s.isDirectory(),
          () => false,
        )
        if (!exists) return renderResult({ items: [] })
        const entries = await readdir(cfg.brainRoot, { withFileTypes: true })
        const items: Array<Record<string, unknown>> = []
        for (const entry of entries) {
          if (!entry.isDirectory()) continue
          const configPath = join(cfg.brainRoot, entry.name, 'config.json')
          let parsed: Record<string, unknown> = {}
          try {
            const raw = await readFile(configPath, 'utf8')
            parsed = JSON.parse(raw) as Record<string, unknown>
          } catch {
            // No config file; treat the directory as a slug.
          }
          items.push({
            id: (parsed.slug as string | undefined) ?? entry.name,
            slug: (parsed.slug as string | undefined) ?? entry.name,
            name: (parsed.name as string | undefined) ?? entry.name,
            visibility: (parsed.visibility as string | undefined) ?? 'private',
            created_at: (parsed.createdAt as string | undefined) ?? null,
          })
        }
        return renderResult({ items })
      },
    }),
  }
}

/**
 * Minimal helper to type the tool definition shape we hand to
 * `pi.registerTool`. Mirrors the pi `defineTool` helper but does not
 * depend on `@earendil-works/pi-coding-agent`'s `defineTool` so the test
 * harness can build tools without booting the loader.
 */
type ToolHandlerResult = {
  readonly content: ReadonlyArray<{ readonly type: 'text'; readonly text: string }>
  readonly details: { readonly result: unknown }
}

type DefinedTool<TParams extends TSchema> = {
  readonly name: string
  readonly label: string
  readonly description: string
  readonly parameters: TParams
  execute(
    toolCallId: string,
    params: Static<TParams>,
    signal?: AbortSignal,
    onUpdate?: () => void,
    ctx?: unknown,
  ): Promise<ToolHandlerResult>
}

const defineMemoryTool = <TParams extends TSchema>(
  tool: DefinedTool<TParams>,
): DefinedTool<TParams> => tool

/**
 * Register the eleven memory tools on the supplied pi runtime. Respects
 * `tools.expose` if set, otherwise registers everything.
 */
export const registerMemoryTools = (pi: ExtensionAPI, runtime: MemoryRuntime): void => {
  const all = buildTools(runtime)
  const allow = runtime.config.exposedTools
  for (const name of MEMORY_TOOL_NAMES) {
    if (allow !== undefined && allow.length > 0 && !allow.includes(name)) continue
    const tool = all[name]
    // The pi ToolDefinition is structurally compatible with our
    // DefinedTool shape; we cast through `unknown` so we do not bring
    // pi's private types into our public surface.
    pi.registerTool(tool as unknown as Parameters<typeof pi.registerTool>[0])
  }
}

export type { MemoryToolName }
