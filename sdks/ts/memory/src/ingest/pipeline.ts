/**
 * Ingest pipeline orchestrator: raw content → store → chunks → embeddings
 * → search index. Re-entrant: ingesting identical bytes a second time
 * returns the original documentId without re-embedding.
 */

import { createHash } from 'node:crypto'
import type { Embedder } from '../llm/index.js'
import type { Store } from '../store/index.js'
import { toPath } from '../store/index.js'
import { appendLogInBatch } from '../knowledge/log.js'
import type { Chunk as IndexChunk, SearchIndex as SqliteSearchIndex } from '../search/index.js'
import { INGESTED_PREFIX } from '../knowledge/ingest.js'
import { chunkAuto, chunkMarkdown, chunkPlainText, type Chunk } from './chunker.js'

export type IngestProgressStage = 'store' | 'chunk' | 'embed' | 'index'

export type IngestProgress = {
  readonly stage: IngestProgressStage
  readonly completed: number
  readonly total: number
}

export type IngestPipelineInput = {
  readonly brainId: string
  readonly path?: string
  readonly content: Buffer | string
  readonly source?: string
  readonly mime?: string
  readonly tenantId?: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly title?: string
}

export type IngestPipelineResult = {
  readonly documentId: string
  readonly path: string
  readonly hash: string
  readonly chunkCount: number
  readonly embeddedCount: number
  readonly skippedChunks: readonly string[]
  readonly durationMs: number
  readonly reused: boolean
}

export type IngestPipelineDeps = {
  readonly store: Store
  readonly searchIndex: SqliteSearchIndex
  readonly embedder: Embedder
  readonly doc: IngestPipelineInput
  readonly onProgress?: (event: IngestProgress) => void
  /** Override the clock. Tests pin this to keep audit entries deterministic. */
  readonly now?: () => Date
}

const hashContent = (buf: Buffer): string =>
  createHash('sha256').update(buf).digest('hex')

const extensionForMime = (mime: string | undefined): string => {
  if (mime === undefined) return '.md'
  if (mime.includes('markdown')) return '.md'
  if (mime.startsWith('text/')) return '.txt'
  if (mime.includes('json')) return '.json'
  if (mime.includes('pdf')) return '.pdf'
  if (mime.includes('html')) return '.html'
  return '.bin'
}

const sniffMime = (buf: Buffer, fallback: string): string => {
  if (buf.length >= 4 && buf.slice(0, 4).toString('binary') === '%PDF') {
    return 'application/pdf'
  }
  // UTF-8 BOM
  if (buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf) {
    return fallback
  }
  // Detect likely HTML.
  const head = buf.slice(0, 512).toString('utf8').trim().toLowerCase()
  if (head.startsWith('<!doctype html') || head.startsWith('<html')) {
    return 'text/html'
  }
  return fallback
}

const pickChunker = (mime: string): ((text: string) => readonly Chunk[]) => {
  if (mime === 'text/markdown') return (text: string) => chunkMarkdown(text)
  if (mime === 'text/plain') return (text: string) => chunkPlainText(text)
  return (text: string) => chunkAuto(text)
}

/**
 * Run the full ingest pipeline for a single document. Steps are emitted
 * via `onProgress` so UIs can render a status line.
 */
export const ingestDocument = async (
  deps: IngestPipelineDeps,
): Promise<IngestPipelineResult> => {
  const start = (deps.now ?? (() => new Date()))().getTime()
  const { store, searchIndex, embedder, doc, onProgress } = deps
  const buffer = typeof doc.content === 'string' ? Buffer.from(doc.content, 'utf8') : doc.content
  const mimeFallback = doc.mime ?? 'text/markdown'
  const mime = doc.mime ?? sniffMime(buffer, mimeFallback)
  const hash = hashContent(buffer)
  const ext = extensionForMime(mime)
  const storedPath = doc.path ?? `${INGESTED_PREFIX}/${hash}${ext}`
  const documentId = `${doc.brainId}:${hash.slice(0, 16)}`

  // Step 1: write raw content inside a single batch. If the exact file
  // already exists with the same hash we treat this as a dedup hit and
  // return the existing id without re-chunking.
  const existing = await store.exists(toPath(storedPath))
  let reused = false
  if (existing) {
    const prior = await store.read(toPath(storedPath))
    if (hashContent(prior) === hash) {
      reused = true
    }
  }

  if (!reused) {
    await store.batch({ reason: 'ingest' }, async (batch) => {
      await batch.write(toPath(storedPath), buffer)
      await appendLogInBatch(batch, {
        kind: 'ingest',
        title: doc.title ?? hash.slice(0, 12),
        detail: `ingested ${buffer.length} bytes → ${storedPath}`,
        when: new Date().toISOString(),
      })
    })
  }
  onProgress?.({ stage: 'store', completed: 1, total: 1 })

  // Step 2: chunk — skipped on replay because the chunk rows already exist.
  if (reused) {
    return {
      documentId,
      path: storedPath,
      hash,
      chunkCount: 0,
      embeddedCount: 0,
      skippedChunks: [],
      durationMs: (deps.now ?? (() => new Date()))().getTime() - start,
      reused: true,
    }
  }

  const text = buffer.toString('utf8')
  const chunker = pickChunker(mime)
  let chunks: readonly Chunk[] = []
  const skipped: string[] = []
  // Binary / PDF / non-UTF-8 bodies fall through to a single "raw" chunk
  // whose content is a placeholder so search still returns the document.
  if (mime === 'application/pdf' || mime === 'application/octet-stream') {
    skipped.push('binary-body-skipped-chunking')
    chunks = [
      {
        content: `[binary content ${mime}, ${buffer.length} bytes]`,
        ordinal: 0,
        headingPath: [],
        startLine: 0,
        endLine: 0,
        tokens: 1,
      },
    ]
  } else {
    chunks = chunker(text)
  }
  onProgress?.({ stage: 'chunk', completed: chunks.length, total: chunks.length })

  if (chunks.length === 0) {
    return {
      documentId,
      path: storedPath,
      hash,
      chunkCount: 0,
      embeddedCount: 0,
      skippedChunks: skipped,
      durationMs: (deps.now ?? (() => new Date()))().getTime() - start,
      reused: false,
    }
  }

  // Step 3: embed.
  const texts = chunks.map((c) => c.content)
  const embeddings = await embedder.embed(texts)
  onProgress?.({ stage: 'embed', completed: embeddings.length, total: chunks.length })

  // Step 4: index.
  const indexChunks: IndexChunk[] = chunks.map((chunk, idx) => {
    const embedding = embeddings[idx]
    const titleParts: string[] = []
    if (doc.title !== undefined && doc.title !== '') titleParts.push(doc.title)
    if (chunk.headingPath.length > 0) titleParts.push(chunk.headingPath.join(' › '))
    const title = titleParts.join(' — ')
    const metadata: Record<string, unknown> = {
      headingPath: chunk.headingPath,
      path: storedPath,
      startLine: chunk.startLine,
      endLine: chunk.endLine,
      tokens: chunk.tokens,
      documentId,
      brainId: doc.brainId,
      ...(doc.tenantId !== undefined ? { tenantId: doc.tenantId } : {}),
      ...(doc.metadata ?? {}),
    }
    return {
      id: `${documentId}:${chunk.ordinal}`,
      path: storedPath,
      ordinal: chunk.ordinal,
      title,
      content: chunk.content,
      metadata,
      ...(embedding !== undefined ? { embedding } : {}),
    }
  })
  searchIndex.upsertChunks(indexChunks)
  onProgress?.({ stage: 'index', completed: indexChunks.length, total: indexChunks.length })

  return {
    documentId,
    path: storedPath,
    hash,
    chunkCount: indexChunks.length,
    embeddedCount: embeddings.length,
    skippedChunks: skipped,
    durationMs: (deps.now ?? (() => new Date()))().getTime() - start,
    reused: false,
  }
}
