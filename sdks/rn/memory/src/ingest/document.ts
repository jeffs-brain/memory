import type { Embedder } from '../llm/types.js'
import type { SearchIndex } from '../search/index.js'
import type { Store } from '../store/index.js'
import { toPath } from '../store/index.js'
import {
  type Chunk,
  type ChunkOptions,
  chunkAuto,
  chunkMarkdown,
  chunkPlainText,
} from './chunker.js'

export type IngestDocumentArgs = {
  readonly path: string
  readonly content: string
  readonly title?: string
  readonly mime?: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly searchIndex: SearchIndex
  readonly embedder?: Embedder
  readonly store?: Store
  readonly chunking?: ChunkOptions
}

export type IngestDocumentResult = {
  readonly path: string
  readonly title: string
  readonly mime: string
  readonly chunkCount: number
  readonly embeddedCount: number
}

const pickChunker = (
  mime: string,
): ((text: string, options?: ChunkOptions) => readonly Chunk[]) => {
  if (mime === 'text/markdown') return chunkMarkdown
  if (mime === 'text/plain') return chunkPlainText
  return chunkAuto
}

const inferTitle = (path: string, explicitTitle: string | undefined): string => {
  if (explicitTitle !== undefined && explicitTitle.trim() !== '') {
    return explicitTitle.trim()
  }
  const filename = path.split('/').pop() ?? path
  return filename.replace(/\.[^.]+$/, '')
}

export const ingestDocument = async (args: IngestDocumentArgs): Promise<IngestDocumentResult> => {
  const path = toPath(args.path)
  const mime = args.mime ?? (looksLikeMarkdownMime(args.content) ? 'text/markdown' : 'text/plain')
  const title = inferTitle(args.path, args.title)
  const chunker = pickChunker(mime)
  const chunks = chunker(args.content, args.chunking)
  const embeddings =
    args.embedder === undefined || chunks.length === 0
      ? []
      : await args.embedder.embed(chunks.map((chunk) => chunk.content))

  if (args.store !== undefined) {
    await args.store.write(path, args.content)
  }

  args.searchIndex.deleteByPath(path)
  args.searchIndex.upsertChunks(
    chunks.map((chunk, index) => ({
      id: `${path}:${chunk.ordinal}`,
      path,
      ordinal: chunk.ordinal,
      title: chunk.headingPath.length === 0 ? title : `${title} - ${chunk.headingPath.join(' / ')}`,
      summary: buildChunkSummary(title, chunk),
      content: chunk.content,
      metadata: {
        type: 'document',
        mime,
        headingPath: [...chunk.headingPath],
        startLine: chunk.startLine,
        endLine: chunk.endLine,
        tokens: chunk.tokens,
        ...(args.metadata ?? {}),
      },
      ...(embeddings[index] === undefined ? {} : { embedding: embeddings[index] }),
      ...(args.embedder === undefined ? {} : { embeddingModel: args.embedder.model() }),
    })),
  )

  return {
    path,
    title,
    mime,
    chunkCount: chunks.length,
    embeddedCount: embeddings.filter((embedding) => embedding !== undefined).length,
  }
}

const buildChunkSummary = (title: string, chunk: Chunk): string => {
  if (chunk.headingPath.length === 0) return title
  return `${title}: ${chunk.headingPath.join(' / ')}`
}

const looksLikeMarkdownMime = (content: string): boolean => {
  return (
    /^#{1,6}\s+\S/m.test(content) || /^\s*[-*+]\s+\S/m.test(content) || /^>\s+\S/m.test(content)
  )
}
