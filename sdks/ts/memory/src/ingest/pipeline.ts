// SPDX-License-Identifier: Apache-2.0

/**
 * Ingest pipeline orchestrator: raw content -> store -> chunks -> embeddings
 * -> search index. Crash-safe: persists pipeline state after each stage so
 * that a restart resumes from the last completed stage rather than silently
 * skipping documents that were stored but never indexed.
 */

import { createHash } from 'node:crypto'
import { RAW_DOCUMENTS_PREFIX } from '../knowledge/ingest.js'
import { appendLogInBatch } from '../knowledge/log.js'
import type { Embedder, Logger } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import type { Chunk as IndexChunk, SearchIndex as SqliteSearchIndex } from '../search/index.js'
import type { Store } from '../store/index.js'
import { toPath } from '../store/index.js'
import { type Chunk, chunkAuto, chunkMarkdown, chunkPlainText } from './chunker.js'
import {
  type PipelineStage,
  type PipelineState,
  deletePipelineState,
  isStageComplete,
  readPipelineState,
  writePipelineState,
} from './pipeline-state.js'

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
  readonly logger?: Logger
  /** Override the clock. Tests pin this to keep audit entries deterministic. */
  readonly now?: () => Date
}

const hashContent = (buf: Buffer): string => createHash('sha256').update(buf).digest('hex')

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
  if (buf.length >= 3 && buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf) {
    return fallback
  }
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
 *
 * Crash recovery: on re-entry, reads pipeline state from the Store. If
 * a prior run completed partially (stored but not indexed), resumes from
 * the last completed stage. Only returns `reused: true` when the full
 * pipeline completed previously with the same content hash.
 */
export const ingestDocument = async (deps: IngestPipelineDeps): Promise<IngestPipelineResult> => {
  const start = (deps.now ?? (() => new Date()))().getTime()
  const { store, searchIndex, embedder, doc, onProgress } = deps
  const logger = deps.logger ?? noopLogger
  const buffer = typeof doc.content === 'string' ? Buffer.from(doc.content, 'utf8') : doc.content
  const mimeFallback = doc.mime ?? 'text/markdown'
  const mime = doc.mime ?? sniffMime(buffer, mimeFallback)
  const hash = hashContent(buffer)
  const ext = extensionForMime(mime)
  const storedPath = doc.path ?? `${RAW_DOCUMENTS_PREFIX}/${hash}${ext}`
  const documentId = `${doc.brainId}:${hash.slice(0, 16)}`
  const nowFn = deps.now ?? (() => new Date())

  const priorState = await readPipelineState(store, hash, logger)
  const resumeDecision = resolveResumeStage(priorState, hash)

  if (resumeDecision === 'complete') {
    onProgress?.({ stage: 'store', completed: 1, total: 1 })
    return {
      documentId,
      path: storedPath,
      hash,
      chunkCount: 0,
      embeddedCount: 0,
      skippedChunks: [],
      durationMs: nowFn().getTime() - start,
      reused: true,
    }
  }

  // After the 'complete' early-return above, resumeDecision is 'none' | PipelineStage.
  const lastCompletedStage: PipelineStage | undefined =
    resumeDecision === 'none' ? undefined : resumeDecision

  const shouldSkip = (target: PipelineStage): boolean =>
    lastCompletedStage !== undefined && isStageComplete(lastCompletedStage, target)

  // Step 1: write raw content if not already stored.
  if (!shouldSkip('stored')) {
    await store.batch({ reason: 'ingest' }, async (batch) => {
      await batch.write(toPath(storedPath), buffer)
      await appendLogInBatch(batch, {
        kind: 'ingest',
        title: doc.title ?? hash.slice(0, 12),
        detail: `ingested ${buffer.length} bytes -> ${storedPath}`,
        when: nowFn().toISOString(),
      })
    })
    await writePipelineState(store, {
      documentId,
      hash,
      stage: 'stored',
      updatedAt: nowFn().toISOString(),
    })
  }
  onProgress?.({ stage: 'store', completed: 1, total: 1 })

  // Step 2: chunk.
  const text = buffer.toString('utf8')
  const chunker = pickChunker(mime)
  let chunks: readonly Chunk[] = []
  const skipped: string[] = []

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

  if (chunks.length === 0) {
    await deletePipelineState(store, hash, logger)
    return {
      documentId,
      path: storedPath,
      hash,
      chunkCount: 0,
      embeddedCount: 0,
      skippedChunks: skipped,
      durationMs: nowFn().getTime() - start,
      reused: false,
    }
  }

  if (!shouldSkip('chunked')) {
    await writePipelineState(store, {
      documentId,
      hash,
      stage: 'chunked',
      updatedAt: nowFn().toISOString(),
      chunkCount: chunks.length,
    })
  }
  onProgress?.({ stage: 'chunk', completed: chunks.length, total: chunks.length })

  // Step 3: embed.
  const texts = chunks.map((c) => c.content)
  const embeddings = await embedder.embed(texts)

  if (!shouldSkip('embedded')) {
    await writePipelineState(store, {
      documentId,
      hash,
      stage: 'embedded',
      updatedAt: nowFn().toISOString(),
      chunkCount: chunks.length,
    })
  }
  onProgress?.({ stage: 'embed', completed: embeddings.length, total: chunks.length })

  // Step 4: index.
  const indexChunks: IndexChunk[] = chunks.map((chunk, idx) => {
    const embedding = embeddings[idx]
    const titleParts: string[] = []
    if (doc.title !== undefined && doc.title !== '') titleParts.push(doc.title)
    if (chunk.headingPath.length > 0) titleParts.push(chunk.headingPath.join(' > '))
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

  // Full pipeline complete: mark state as indexed for future dedup.
  await writePipelineState(store, {
    documentId,
    hash,
    stage: 'indexed',
    updatedAt: nowFn().toISOString(),
    chunkCount: indexChunks.length,
  })

  return {
    documentId,
    path: storedPath,
    hash,
    chunkCount: indexChunks.length,
    embeddedCount: embeddings.length,
    skippedChunks: skipped,
    durationMs: nowFn().getTime() - start,
    reused: false,
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type ResumeDecision = PipelineStage | 'none' | 'complete'

/**
 * Determine where to resume from based on prior state and current hash.
 * Returns 'complete' if the document was fully indexed with a matching hash.
 * Returns 'none' if no prior state exists or hash mismatches (start fresh).
 * Returns the last completed stage if partially complete with matching hash.
 */
const resolveResumeStage = (
  priorState: PipelineState | undefined,
  currentHash: string,
): ResumeDecision => {
  if (priorState === undefined) return 'none'
  if (priorState.hash !== currentHash) return 'none'
  if (priorState.stage === 'indexed') return 'complete'
  return priorState.stage
}
