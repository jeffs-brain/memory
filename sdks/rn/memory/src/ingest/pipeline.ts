import {
  type HashContentInput,
  contentByteLength,
  hashContent,
  toBytes,
  toText,
} from '../knowledge/hash.js'
import { RAW_DOCUMENTS_PREFIX } from '../knowledge/ingest.js'
import { appendLogInBatch } from '../knowledge/log.js'
import type { Embedder } from '../llm/types.js'
import type { Chunk as IndexChunk, SearchIndex } from '../search/index.js'
import type { Store } from '../store/index.js'
import { toPath } from '../store/index.js'
import { type Chunk, chunkAuto, chunkMarkdown, chunkPlainText } from './chunker.js'

export type IngestProgressStage = 'store' | 'chunk' | 'embed' | 'index'

export type IngestProgress = {
  readonly stage: IngestProgressStage
  readonly completed: number
  readonly total: number
}

export type IngestPipelineInput = {
  readonly brainId: string
  readonly path?: string
  readonly content: HashContentInput
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
  readonly searchIndex: SearchIndex
  readonly embedder: Embedder
  readonly doc: IngestPipelineInput
  readonly onProgress?: (event: IngestProgress) => void
  readonly now?: () => Date
}

const extensionForMime = (mime: string | undefined): string => {
  if (mime === undefined) return '.md'
  if (mime.includes('markdown')) return '.md'
  if (mime.startsWith('text/')) return '.txt'
  if (mime.includes('json')) return '.json'
  if (mime.includes('pdf')) return '.pdf'
  if (mime.includes('html')) return '.html'
  return '.bin'
}

const sniffMime = (bytes: Uint8Array, fallback: string): string => {
  if (
    bytes.length >= 4 &&
    bytes[0] === 0x25 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x44 &&
    bytes[3] === 0x46
  ) {
    return 'application/pdf'
  }
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    return fallback
  }
  const head = toText(bytes.slice(0, 512)).trim().toLowerCase()
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

export const ingestDocument = async (deps: IngestPipelineDeps): Promise<IngestPipelineResult> => {
  const startedAt = (deps.now ?? (() => new Date()))().getTime()
  const { store, searchIndex, embedder, doc, onProgress } = deps
  const bytes = toBytes(doc.content)
  const text = toText(doc.content)
  const mimeFallback = doc.mime ?? 'text/markdown'
  const mime = doc.mime ?? sniffMime(bytes, mimeFallback)
  const hash = hashContent(doc.content)
  const ext = extensionForMime(mime)
  const storedPath = doc.path ?? `${RAW_DOCUMENTS_PREFIX}/${hash}${ext}`
  const documentId = `${doc.brainId}:${hash.slice(0, 16)}`

  let reused = false
  if (await store.exists(toPath(storedPath)).catch(() => false)) {
    const prior = await store.read(toPath(storedPath))
    if (hashContent(prior) === hash) {
      reused = true
    }
  }

  if (!reused) {
    await store.batch({ reason: 'ingest' }, async (batch) => {
      await batch.write(toPath(storedPath), text)
      await appendLogInBatch(batch, {
        kind: 'ingest',
        title: doc.title ?? hash.slice(0, 12),
        detail: `ingested ${contentByteLength(doc.content)} bytes → ${storedPath}`,
        when: new Date().toISOString(),
      })
    })
  }
  onProgress?.({ stage: 'store', completed: 1, total: 1 })

  if (reused) {
    return {
      documentId,
      path: storedPath,
      hash,
      chunkCount: 0,
      embeddedCount: 0,
      skippedChunks: [],
      durationMs: (deps.now ?? (() => new Date()))().getTime() - startedAt,
      reused: true,
    }
  }

  const chunker = pickChunker(mime)
  let chunks: readonly Chunk[] = []
  const skipped: string[] = []
  if (mime === 'application/pdf' || mime === 'application/octet-stream') {
    skipped.push('binary-body-skipped-chunking')
    chunks = [
      {
        content: `[binary content ${mime}, ${bytes.byteLength} bytes]`,
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
      durationMs: (deps.now ?? (() => new Date()))().getTime() - startedAt,
      reused: false,
    }
  }

  const embeddings = await embedder.embed(chunks.map((chunk) => chunk.content))
  onProgress?.({ stage: 'embed', completed: embeddings.length, total: chunks.length })

  const indexChunks: IndexChunk[] = chunks.map((chunk, index) => {
    const embedding = embeddings[index]
    const titleParts: string[] = []
    if (doc.title !== undefined && doc.title !== '') titleParts.push(doc.title)
    if (chunk.headingPath.length > 0) titleParts.push(chunk.headingPath.join(' › '))
    const title = titleParts.join(' - ')
    return {
      id: `${documentId}:${chunk.ordinal}`,
      path: storedPath,
      ordinal: chunk.ordinal,
      title,
      content: chunk.content,
      metadata: {
        headingPath: chunk.headingPath,
        path: storedPath,
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        tokens: chunk.tokens,
        documentId,
        brainId: doc.brainId,
        ...(doc.source !== undefined ? { source: doc.source } : {}),
        ...(doc.tenantId !== undefined ? { tenantId: doc.tenantId } : {}),
        ...(doc.metadata ?? {}),
      },
      ...(embedding !== undefined ? { embedding } : {}),
      embeddingModel: embedder.model(),
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
    durationMs: (deps.now ?? (() => new Date()))().getTime() - startedAt,
    reused: false,
  }
}
