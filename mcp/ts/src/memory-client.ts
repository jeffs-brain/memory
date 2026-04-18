// SPDX-License-Identifier: Apache-2.0

/**
 * Unified client used by every tool handler. Hides whether we are wired
 * to a local FsStore (plus sqlite search and optional Ollama embeddings)
 * or a remote HttpStore against the hosted platform.
 *
 * Both backends expose the same 11-tool surface. Local mode builds the
 * full pipeline out of `@jeffs-brain/memory` primitives; hosted mode
 * proxies through to the platform REST API described in
 * `spec/MCP-TOOLS.md`.
 */

import { createHash, randomUUID } from 'node:crypto'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { extname, isAbsolute, join, resolve } from 'node:path'
import {
  createFsStore,
  createMemory,
  createSearchIndex,
  createStoreBackedCursorStore,
  OllamaEmbedder,
  OllamaProvider,
  AnthropicProvider,
  OpenAIProvider,
  type Embedder,
  type Message,
  type Provider,
  type SqliteSearchIndex,
  type Store,
} from '@jeffs-brain/memory'
import { ingestDocument } from '@jeffs-brain/memory/ingest'
import { createRetrieval } from '@jeffs-brain/memory/retrieval'
import type { ConfigMode, HostedConfig, LocalConfig } from './config.js'

export type SearchArgs = {
  readonly query: string
  readonly brain?: string | undefined
  readonly topK?: number | undefined
  readonly scope?: 'all' | 'global' | 'project' | 'agent' | undefined
  readonly sort?: 'relevance' | 'recency' | 'relevance_then_recency' | undefined
}

export type RecallArgs = {
  readonly query: string
  readonly brain?: string | undefined
  readonly scope?: 'global' | 'project' | 'agent' | undefined
  readonly sessionId?: string | undefined
  readonly topK?: number | undefined
}

export type RememberArgs = {
  readonly content: string
  readonly title?: string | undefined
  readonly brain?: string | undefined
  readonly tags?: readonly string[] | undefined
  readonly path?: string | undefined
}

export type IngestFileArgs = {
  readonly path: string
  readonly brain?: string | undefined
  readonly as?: 'markdown' | 'text' | 'pdf' | 'json' | undefined
}

export type IngestUrlArgs = {
  readonly url: string
  readonly brain?: string | undefined
}

export type ExtractMessage = {
  readonly role: 'system' | 'user' | 'assistant' | 'tool'
  readonly content: string
}

export type ExtractArgs = {
  readonly messages: readonly ExtractMessage[]
  readonly brain?: string | undefined
  readonly actorId?: string | undefined
  readonly sessionId?: string | undefined
}

export type ReflectArgs = {
  readonly sessionId: string
  readonly brain?: string | undefined
}

export type ConsolidateArgs = {
  readonly brain?: string | undefined
}

export type AskArgs = {
  readonly query: string
  readonly brain?: string | undefined
  readonly topK?: number | undefined
}

export type CreateBrainArgs = {
  readonly name: string
  readonly slug?: string | undefined
  readonly visibility?: 'private' | 'tenant' | 'public' | undefined
}

/**
 * Optional hook every tool handler may call to emit incremental
 * progress. Implementations map this onto `extra.sendNotification` when
 * the MCP client supplied a progress token; otherwise it is a no-op.
 */
export type ProgressEmitter = (progress: number, message?: string) => void

export type MemoryClient = {
  readonly mode: ConfigMode['kind']
  remember(args: RememberArgs): Promise<unknown>
  recall(args: RecallArgs): Promise<unknown>
  search(args: SearchArgs): Promise<unknown>
  ask(args: AskArgs, progress?: ProgressEmitter): Promise<unknown>
  ingestFile(args: IngestFileArgs, progress?: ProgressEmitter): Promise<unknown>
  ingestUrl(args: IngestUrlArgs, progress?: ProgressEmitter): Promise<unknown>
  extract(args: ExtractArgs, progress?: ProgressEmitter): Promise<unknown>
  reflect(args: ReflectArgs, progress?: ProgressEmitter): Promise<unknown>
  consolidate(args: ConsolidateArgs, progress?: ProgressEmitter): Promise<unknown>
  createBrain(args: CreateBrainArgs): Promise<unknown>
  listBrains(): Promise<unknown>
  close(): Promise<void>
}

// ---------------------------------------------------------------------
// Local mode
// ---------------------------------------------------------------------

const OLLAMA_EMBED_MODEL = 'bge-m3'
const OLLAMA_CHAT_MODEL = 'gemma3'
const DEFAULT_BRAIN_ID = 'default'
const FS_MIME_BY_AS: Record<NonNullable<IngestFileArgs['as']>, string> = {
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
const URL_FETCH_LIMIT_BYTES = 5 * 1024 * 1024
const FILE_LIMIT_BYTES = 25 * 1024 * 1024

type BrainResources = {
  readonly id: string
  readonly root: string
  readonly store: Store
  readonly searchIndex: SqliteSearchIndex
  close(): Promise<void>
}

type LocalDeps = {
  readonly cfg: LocalConfig
  embedder: Embedder | undefined
  provider: Provider | undefined
  readonly brains: Map<string, BrainResources>
}

const ensureDir = async (path: string): Promise<void> => {
  await mkdir(path, { recursive: true })
}

const ollamaReachable = async (baseUrl: string): Promise<boolean> => {
  try {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 1500)
    const resp = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/tags`, {
      signal: controller.signal,
    }).catch(() => undefined)
    clearTimeout(timer)
    return resp !== undefined && resp.ok
  } catch {
    return false
  }
}

const selectLocalProvider = async (cfg: LocalConfig): Promise<Provider | undefined> => {
  const openaiKey = process.env.OPENAI_API_KEY
  if (openaiKey !== undefined && openaiKey !== '') {
    return new OpenAIProvider({
      apiKey: openaiKey,
      model: process.env.OPENAI_MODEL ?? 'gpt-4o-mini',
    })
  }
  const anthropicKey = process.env.ANTHROPIC_API_KEY
  if (anthropicKey !== undefined && anthropicKey !== '') {
    return new AnthropicProvider({
      apiKey: anthropicKey,
      model: process.env.ANTHROPIC_MODEL ?? 'claude-opus-4-5',
    })
  }
  if (await ollamaReachable(cfg.ollamaBaseUrl)) {
    return new OllamaProvider({
      model: process.env.OLLAMA_CHAT_MODEL ?? OLLAMA_CHAT_MODEL,
      baseURL: cfg.ollamaBaseUrl,
    })
  }
  return undefined
}

const selectLocalEmbedder = async (cfg: LocalConfig): Promise<Embedder | undefined> => {
  if (!(await ollamaReachable(cfg.ollamaBaseUrl))) return undefined
  return new OllamaEmbedder({
    baseURL: cfg.ollamaBaseUrl,
    model: process.env.OLLAMA_EMBED_MODEL ?? OLLAMA_EMBED_MODEL,
  })
}

const resolveBrainId = (cfg: LocalConfig, override: string | undefined): string => {
  if (override !== undefined && override !== '') return override
  if (cfg.defaultBrain !== undefined && cfg.defaultBrain !== '') return cfg.defaultBrain
  return DEFAULT_BRAIN_ID
}

const openBrainResources = async (
  deps: LocalDeps,
  brainId: string,
): Promise<BrainResources> => {
  const existing = deps.brains.get(brainId)
  if (existing !== undefined) return existing
  const brainRoot = join(deps.cfg.brainRoot, sanitiseBrainId(brainId))
  await ensureDir(brainRoot)
  const store = await createFsStore({ root: brainRoot })
  const searchIndex = await createSearchIndex({
    dbPath: join(brainRoot, 'search.sqlite'),
  })
  const resource: BrainResources = {
    id: brainId,
    root: brainRoot,
    store,
    searchIndex,
    close: async () => {
      await searchIndex.close().catch(() => undefined)
      await store.close().catch(() => undefined)
    },
  }
  deps.brains.set(brainId, resource)
  return resource
}

const sanitiseBrainId = (id: string): string => {
  const cleaned = id.replace(/[^A-Za-z0-9._-]/g, '_')
  if (cleaned === '' || cleaned.startsWith('.')) return '_invalid'
  return cleaned
}

const slugFromTitle = (title: string): string => {
  const slug = title
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return slug === '' ? `note-${Date.now()}` : slug.slice(0, 64)
}

const deriveTitle = (content: string, fallback: string | undefined): string => {
  if (fallback !== undefined && fallback !== '') return fallback
  const firstHeading = content.match(/^#\s+(.+)$/m)?.[1]?.trim()
  if (firstHeading !== undefined && firstHeading !== '') return firstHeading
  const firstLine = content.split('\n').find((line) => line.trim() !== '')?.trim()
  return firstLine ?? 'Untitled memory'
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

const mimeForFile = (path: string, hint: IngestFileArgs['as']): string => {
  if (hint !== undefined) return FS_MIME_BY_AS[hint]
  const ext = extname(path).toLowerCase()
  return FS_MIME_BY_EXT[ext] ?? 'application/octet-stream'
}

const noProvider = (): never => {
  throw new Error(
    'memory-mcp: no LLM provider configured; set OPENAI_API_KEY, ANTHROPIC_API_KEY, or run Ollama locally',
  )
}

const createLocalClient = (cfg: LocalConfig): MemoryClient => {
  const deps: LocalDeps = {
    cfg,
    embedder: undefined,
    provider: undefined,
    brains: new Map<string, BrainResources>(),
  }
  let bootstrapped = false
  const ensureBootstrap = async (): Promise<void> => {
    if (bootstrapped) return
    bootstrapped = true
    await ensureDir(cfg.brainRoot)
    deps.embedder = await selectLocalEmbedder(cfg)
    deps.provider = await selectLocalProvider(cfg)
  }

  const buildRetrieval = (brain: BrainResources) =>
    createRetrieval({
      index: brain.searchIndex,
      ...(deps.embedder !== undefined ? { embedder: deps.embedder } : {}),
    })

  const buildMemory = (brain: BrainResources, provider: Provider) =>
    createMemory({
      store: brain.store,
      provider,
      ...(deps.embedder !== undefined ? { embedder: deps.embedder } : {}),
      scope: 'global',
      actorId: 'default',
      cursorStore: createStoreBackedCursorStore(brain.store),
    })

  return {
    mode: 'local',
    async remember(args) {
      await ensureBootstrap()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const title = deriveTitle(args.content, args.title)
      const slug = slugFromTitle(title)
      const relativePath = args.path !== undefined && args.path !== ''
        ? args.path
        : `memory/global/${slug}.md`
      const nowIso = new Date().toISOString()
      const frontmatter = buildFrontmatterBlock({
        title,
        scope: 'global',
        type: 'user',
        created: nowIso,
        modified: nowIso,
        ...(args.tags !== undefined && args.tags.length > 0 ? { tags: args.tags } : {}),
      })
      const body = `${frontmatter}\n\n${args.content.trimEnd()}\n`
      const content = Buffer.from(body, 'utf8')
      const embedder = deps.embedder
      if (embedder !== undefined) {
        const result = await ingestDocument({
          store: brain.store,
          searchIndex: brain.searchIndex,
          embedder,
          doc: {
            brainId,
            path: relativePath,
            content,
            title,
            mime: 'text/markdown',
            metadata: {
              ...(args.tags !== undefined && args.tags.length > 0
                ? { tags: args.tags.join(',') }
                : {}),
            },
          },
        })
        return {
          id: result.documentId,
          path: result.path,
          byte_size: content.length,
          hash: result.hash,
          chunk_count: result.chunkCount,
          embedded_count: result.embeddedCount,
          reused: result.reused,
        }
      }
      // Fallback: write the note directly and index a single BM25 chunk.
      await brain.store.batch({ reason: 'remember' }, async (batch) => {
        await batch.write(relativePath as never, content)
      })
      const hash = createHash('sha256').update(content).digest('hex')
      brain.searchIndex.upsertChunk({
        id: `${brainId}:${hash.slice(0, 16)}:0`,
        path: relativePath,
        ordinal: 0,
        title,
        content: body,
        metadata: {
          path: relativePath,
          brainId,
          ...(args.tags !== undefined && args.tags.length > 0
            ? { tags: args.tags.join(',') }
            : {}),
        },
      })
      return {
        id: `${brainId}:${hash.slice(0, 16)}`,
        path: relativePath,
        byte_size: content.length,
        hash,
        chunk_count: 1,
        embedded_count: 0,
        reused: false,
      }
    },

    async search(args) {
      await ensureBootstrap()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const retrieval = buildRetrieval(brain)
      const results = await retrieval.search({
        query: args.query,
        topK: args.topK ?? 10,
        rerank: false,
      })
      const hits = results.map((result) => ({
        score: result.score,
        path: result.path,
        content: result.content,
        title: result.title,
        summary: result.summary,
        chunk_id: result.id,
      }))
      return {
        query: args.query,
        brain_id: brainId,
        hits,
        took_ms: 0,
      }
    },

    async recall(args) {
      await ensureBootstrap()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const retrieval = buildRetrieval(brain)
      const results = await retrieval.search({
        query: args.query,
        topK: args.topK ?? 5,
        rerank: false,
      })
      const chunks = results.map((result) => ({
        score: result.score,
        path: result.path,
        content: result.content,
        title: result.title,
        summary: result.summary,
        chunk_id: result.id,
      }))
      return {
        query: args.query,
        brain_id: brainId,
        session_id: args.sessionId ?? null,
        chunks,
      }
    },

    async ask(args, progress) {
      await ensureBootstrap()
      const provider = deps.provider
      if (provider === undefined) noProvider()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const retrieval = buildRetrieval(brain)
      const retrieved = await retrieval.search({
        query: args.query,
        topK: args.topK ?? 8,
        rerank: false,
      })
      progress?.(0, 'retrieved')
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
        args.query,
        '',
        '## Retrieved memory',
        contextLines.join('\n\n') || '_no relevant chunks_',
      ].join('\n')
      const completion = await (provider as Provider).complete({
        system: 'You are Jeff, a personal memory assistant. Use British English.',
        messages: [{ role: 'user', content: prompt }],
        maxTokens: 1024,
        temperature: 0.2,
      })
      progress?.(1, 'answered')
      const citations = retrieved.slice(0, 5).map((hit, idx) => ({
        type: 'citation' as const,
        chunk_id: hit.id,
        document_id: hit.path,
        answer_start: 0,
        answer_end: 0,
        quote: (hit.content ?? '').slice(0, 200),
        rank: idx + 1,
      }))
      const retrievedChunks = retrieved.map((hit) => ({
        chunk_id: hit.id,
        document_id: hit.path,
        score: hit.score,
        preview: (hit.content ?? '').slice(0, 512),
      }))
      return {
        answer: completion.content,
        citations,
        retrieved: retrievedChunks,
      }
    },

    async ingestFile(args, progress) {
      await ensureBootstrap()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const absPath = isAbsolute(args.path) ? args.path : resolve(args.path)
      const info = await stat(absPath)
      if (!info.isFile()) {
        throw new Error(`memory_ingest_file: not a regular file: ${absPath}`)
      }
      if (info.size > FILE_LIMIT_BYTES) {
        throw new Error('file_too_large: 25 MiB limit exceeded')
      }
      const content = await readFile(absPath)
      progress?.(0, 'read')
      const mime = mimeForFile(absPath, args.as)
      const title = absPath.split(/[\\/]/).pop() ?? absPath
      const embedder = deps.embedder
      if (embedder === undefined) {
        // Fallback path: store the file and index a single BM25 chunk.
        const hash = createHash('sha256').update(content).digest('hex')
        const storedPath = `ingested/${hash}${extname(absPath).toLowerCase() || '.bin'}`
        await brain.store.batch({ reason: 'ingest' }, async (batch) => {
          await batch.write(storedPath as never, content)
        })
        brain.searchIndex.upsertChunk({
          id: `${brainId}:${hash.slice(0, 16)}:0`,
          path: storedPath,
          ordinal: 0,
          title,
          content: content.toString('utf8').slice(0, 32_000),
          metadata: { path: storedPath, brainId, mime },
        })
        progress?.(1, 'indexed')
        return {
          status: 'completed',
          document_id: `${brainId}:${hash.slice(0, 16)}`,
          path: storedPath,
          hash,
          chunk_count: 1,
          embedded_count: 0,
          duration_ms: 0,
          reused: false,
        }
      }
      const result = await ingestDocument({
        store: brain.store,
        searchIndex: brain.searchIndex,
        embedder,
        doc: {
          brainId,
          content,
          mime,
          title,
          metadata: { source: absPath },
        },
        onProgress: (event) => {
          progress?.(event.completed, `${event.stage} ${event.completed}/${event.total}`)
        },
      })
      return {
        status: 'completed',
        document_id: result.documentId,
        path: result.path,
        hash: result.hash,
        chunk_count: result.chunkCount,
        embedded_count: result.embeddedCount,
        duration_ms: result.durationMs,
        reused: result.reused,
      }
    },

    async ingestUrl(args, progress) {
      await ensureBootstrap()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const resp = await fetch(args.url)
      if (!resp.ok) {
        throw new Error(`memory_ingest_url: fetch failed ${resp.status} ${resp.statusText}`)
      }
      const contentTypeHeader = resp.headers.get('content-type') ?? 'text/plain'
      const mime = contentTypeHeader.split(';')[0]?.trim() ?? 'text/plain'
      const buffer = Buffer.from(await resp.arrayBuffer())
      if (buffer.length > URL_FETCH_LIMIT_BYTES) {
        throw new Error('memory_ingest_url: body exceeds 5 MiB fallback limit')
      }
      progress?.(0, 'fetched')
      const embedder = deps.embedder
      if (embedder === undefined) {
        const hash = createHash('sha256').update(buffer).digest('hex')
        const storedPath = `ingested/${hash}.txt`
        await brain.store.batch({ reason: 'ingest-url' }, async (batch) => {
          await batch.write(storedPath as never, buffer)
        })
        brain.searchIndex.upsertChunk({
          id: `${brainId}:${hash.slice(0, 16)}:0`,
          path: storedPath,
          ordinal: 0,
          title: args.url,
          content: buffer.toString('utf8').slice(0, 32_000),
          metadata: { path: storedPath, brainId, source_url: args.url, mime },
        })
        progress?.(1, 'indexed')
        return {
          path: 'fallback',
          document: {
            id: `${brainId}:${hash.slice(0, 16)}`,
            brain_id: brainId,
            title: args.url,
            path: storedPath,
            source: 'ingest',
            content_type: mime,
            byte_size: buffer.length,
            checksum_sha256: hash,
            metadata: { source_url: args.url },
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            deleted_at: null,
          },
        }
      }
      const result = await ingestDocument({
        store: brain.store,
        searchIndex: brain.searchIndex,
        embedder,
        doc: {
          brainId,
          content: buffer,
          mime,
          title: args.url,
          metadata: { source_url: args.url },
        },
        onProgress: (event) => {
          progress?.(event.completed, `${event.stage} ${event.completed}/${event.total}`)
        },
      })
      return {
        path: 'server',
        result: {
          status: 'completed',
          document_id: result.documentId,
          path: result.path,
          hash: result.hash,
          chunk_count: result.chunkCount,
          embedded_count: result.embeddedCount,
          duration_ms: result.durationMs,
          reused: result.reused,
        },
      }
    },

    async extract(args, progress) {
      await ensureBootstrap()
      const provider = deps.provider
      if (provider === undefined) noProvider()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const memory = buildMemory(brain, provider as Provider)
      const messages: readonly Message[] = args.messages.map((m) => ({
        role: m.role,
        content: m.content,
      }))
      progress?.(0, 'extracting')
      const extracted = await memory.extract({
        messages,
        ...(args.sessionId !== undefined ? { sessionId: args.sessionId } : {}),
        ...(args.actorId !== undefined ? { actorId: args.actorId } : {}),
      })
      progress?.(1, `${extracted.length} memories`)
      if (args.sessionId !== undefined) {
        return {
          mode: 'session',
          messages: extracted.map((m) => ({
            id: `${args.sessionId}:${m.filename}`,
            session_id: args.sessionId,
            role: 'assistant',
            content: m.content,
            created_at: new Date().toISOString(),
          })),
        }
      }
      return {
        mode: 'transcript',
        document: {
          id: `${brainId}:transcript:${randomUUID()}`,
          brain_id: brainId,
          title: 'Transcript',
          path: 'transcripts/session.md',
          source: 'extract',
          content_type: 'text/markdown',
          byte_size: 0,
          checksum_sha256: '0'.repeat(64),
          metadata: { extracted_count: extracted.length },
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          deleted_at: null,
        },
        extracted,
      }
    },

    async reflect(args, progress) {
      await ensureBootstrap()
      const provider = deps.provider
      if (provider === undefined) noProvider()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const memory = buildMemory(brain, provider as Provider)
      progress?.(0, 'reflecting')
      const result = await memory.reflect({
        sessionId: args.sessionId,
        messages: [],
      })
      progress?.(1, 'done')
      if (result === undefined) {
        return {
          reflection_status: 'no_result',
          reflection_attempted: true,
          ended_at: new Date().toISOString(),
        }
      }
      return {
        reflection_status: 'completed',
        reflection: {
          outcome: result.outcome,
          should_record_episode: result.shouldRecordEpisode ?? false,
          path: result.path,
          retry_feedback: result.retryFeedback ?? '',
        },
        reflection_attempted: true,
        ended_at: new Date().toISOString(),
      }
    },

    async consolidate(args, progress) {
      await ensureBootstrap()
      const provider = deps.provider
      if (provider === undefined) noProvider()
      const brainId = resolveBrainId(cfg, args.brain)
      const brain = await openBrainResources(deps, brainId)
      const memory = buildMemory(brain, provider as Provider)
      progress?.(0, 'consolidating')
      const report = await memory.consolidate({})
      progress?.(1, 'done')
      return { result: report }
    },

    async createBrain(args) {
      await ensureBootstrap()
      const slug = args.slug ?? slugFromTitle(args.name)
      const brainDir = join(cfg.brainRoot, sanitiseBrainId(slug))
      await ensureDir(brainDir)
      const configPath = join(brainDir, 'config.json')
      const configBody = {
        version: 1,
        name: args.name,
        slug,
        visibility: args.visibility ?? 'private',
        createdAt: new Date().toISOString(),
      }
      await writeFile(configPath, `${JSON.stringify(configBody, null, 2)}\n`, 'utf8')
      return {
        id: slug,
        slug,
        name: args.name,
        visibility: configBody.visibility,
        created_at: configBody.createdAt,
      }
    },

    async listBrains() {
      await ensureBootstrap()
      const exists = await stat(cfg.brainRoot).then(
        (s) => s.isDirectory(),
        () => false,
      )
      if (!exists) return { items: [] }
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
          // Brain without config — surface the directory name as the slug
        }
        items.push({
          id: (parsed.slug as string | undefined) ?? entry.name,
          slug: (parsed.slug as string | undefined) ?? entry.name,
          name: (parsed.name as string | undefined) ?? entry.name,
          visibility: (parsed.visibility as string | undefined) ?? 'private',
          created_at: (parsed.createdAt as string | undefined) ?? null,
        })
      }
      return { items }
    },

    async close() {
      for (const brain of deps.brains.values()) {
        await brain.close()
      }
      deps.brains.clear()
    },
  }
}

// ---------------------------------------------------------------------
// Hosted mode
// ---------------------------------------------------------------------

type HostedDeps = {
  readonly endpoint: string
  readonly token: string
  readonly defaultBrain: string | undefined
}

const joinUrl = (base: string, suffix: string): string => {
  const trimmed = base.endsWith('/') ? base.slice(0, -1) : base
  const rooted = suffix.startsWith('/') ? suffix : `/${suffix}`
  return `${trimmed}${rooted}`
}

const hostedResolveBrain = (deps: HostedDeps, override: string | undefined): string => {
  if (override !== undefined && override !== '') return override
  if (deps.defaultBrain !== undefined && deps.defaultBrain !== '') return deps.defaultBrain
  throw new Error('memory-mcp: brain id required in hosted mode; set JB_BRAIN or pass `brain`')
}

const hostedFetch = async (
  deps: HostedDeps,
  path: string,
  init: RequestInit = {},
): Promise<unknown> => {
  const resp = await fetch(joinUrl(deps.endpoint, path), {
    ...init,
    headers: {
      authorization: `Bearer ${deps.token}`,
      'content-type': 'application/json',
      accept: 'application/json',
      ...(init.headers ?? {}),
    },
  })
  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    throw new Error(`memory-mcp: hosted ${path} failed ${resp.status}: ${text.slice(0, 256)}`)
  }
  if (resp.status === 204) return undefined
  return resp.json()
}

const createHostedClient = (cfg: HostedConfig): MemoryClient => {
  const deps: HostedDeps = {
    endpoint: cfg.endpoint,
    token: cfg.token,
    defaultBrain: cfg.defaultBrain,
  }
  return {
    mode: 'hosted',
    async remember(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      return hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/documents`, {
        method: 'POST',
        body: JSON.stringify({
          title: args.title,
          content: args.content,
          path: args.path,
          metadata: args.tags !== undefined && args.tags.length > 0 ? { tags: args.tags.join(',') } : {},
        }),
      })
    },
    async search(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      const qs = new URLSearchParams({ q: args.query })
      if (args.topK !== undefined) qs.set('top_k', String(args.topK))
      if (args.scope !== undefined) qs.set('scope', args.scope)
      if (args.sort !== undefined) qs.set('sort', args.sort)
      return hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/search?${qs.toString()}`)
    },
    async recall(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      return hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/recall`, {
        method: 'POST',
        body: JSON.stringify({
          query: args.query,
          scope: args.scope,
          session_id: args.sessionId,
          top_k: args.topK,
        }),
      })
    },
    async ask(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      return hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/ask`, {
        method: 'POST',
        body: JSON.stringify({ query: args.query, top_k: args.topK }),
      })
    },
    async ingestFile(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      const absPath = isAbsolute(args.path) ? args.path : resolve(args.path)
      const info = await stat(absPath)
      if (info.size > FILE_LIMIT_BYTES) {
        throw new Error('file_too_large: 25 MiB limit exceeded')
      }
      const content = await readFile(absPath)
      const form = new FormData()
      const blob = new Blob([new Uint8Array(content)], { type: mimeForFile(absPath, args.as) })
      form.append('file', blob, absPath.split(/[\\/]/).pop() ?? 'upload')
      const resp = await fetch(joinUrl(deps.endpoint, `/v1/brains/${encodeURIComponent(brain)}/documents/ingest/file`), {
        method: 'POST',
        headers: { authorization: `Bearer ${deps.token}` },
        body: form,
      })
      if (!resp.ok) {
        const text = await resp.text().catch(() => '')
        throw new Error(`memory_ingest_file hosted ${resp.status}: ${text.slice(0, 256)}`)
      }
      return resp.json()
    },
    async ingestUrl(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      return hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/documents/ingest/url`, {
        method: 'POST',
        body: JSON.stringify({ url: args.url }),
      })
    },
    async extract(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      if (args.sessionId !== undefined && args.sessionId !== '') {
        const results: unknown[] = []
        for (let i = 0; i < args.messages.length; i++) {
          const msg = args.messages[i]
          if (msg === undefined) continue
          const isLast = i === args.messages.length - 1
          const payload = {
            role: msg.role,
            content: msg.content,
            metadata: {
              ...(args.actorId !== undefined ? { actor_id: args.actorId } : {}),
              skip_extract: !isLast,
            },
          }
          const result = await hostedFetch(
            deps,
            `/v1/brains/${encodeURIComponent(brain)}/sessions/${encodeURIComponent(args.sessionId)}/messages`,
            { method: 'POST', body: JSON.stringify(payload) },
          )
          results.push(result)
        }
        return { mode: 'session', messages: results }
      }
      const doc = await hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/documents`, {
        method: 'POST',
        body: JSON.stringify({
          title: `Transcript ${new Date().toISOString()}`,
          content: args.messages
            .map((m) => `[${m.role}] ${m.content}`)
            .join('\n\n'),
          source: 'extract',
        }),
      })
      return { mode: 'transcript', document: doc }
    },
    async reflect(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      return hostedFetch(
        deps,
        `/v1/brains/${encodeURIComponent(brain)}/sessions/${encodeURIComponent(args.sessionId)}/close`,
        { method: 'POST', body: JSON.stringify({}) },
      )
    },
    async consolidate(args) {
      const brain = hostedResolveBrain(deps, args.brain)
      return hostedFetch(deps, `/v1/brains/${encodeURIComponent(brain)}/consolidate`, {
        method: 'POST',
        body: JSON.stringify({}),
      })
    },
    async createBrain(args) {
      return hostedFetch(deps, '/v1/brains', {
        method: 'POST',
        body: JSON.stringify({
          name: args.name,
          slug: args.slug,
          visibility: args.visibility ?? 'private',
        }),
      })
    },
    async listBrains() {
      return hostedFetch(deps, '/v1/brains')
    },
    async close() {
      /* no persistent connections */
    },
  }
}

export const createMemoryClient = (cfg: ConfigMode): MemoryClient => {
  if (cfg.kind === 'hosted') return createHostedClient(cfg)
  return createLocalClient(cfg)
}
