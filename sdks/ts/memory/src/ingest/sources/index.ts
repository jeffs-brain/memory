/**
 * Dispatch façade for ingest source adapters. Callers pass a classified
 * input + the raw bytes (or URL) and get a `LoadedSource` back.
 */

import { detectSource, type DetectInput, type SourceKind } from './detect.js'
import { loadJsonTranscript } from './json-transcript.js'
import {
  isMarkitdownCandidate,
  loadMarkitdownFile,
  type MarkitdownServiceConfig,
} from './markitdown.js'
import { loadMarkdown } from './markdown.js'
import { loadPdf } from './pdf.js'
import { loadText } from './text.js'
import type { LoadedSource, SourceLoadOptions, SourceFetchLike } from './types.js'
import { fetchUrlSource, htmlToMarkdown, type LoadUrlOptions } from './url.js'

export type { DetectInput, SourceKind } from './detect.js'
export { detectSource } from './detect.js'
export type { LoadedSource, SourceLoadOptions, SourceFetchLike } from './types.js'
export {
  MarkitdownConfigurationError,
  MarkitdownConvertError,
  isMarkitdownCandidate,
  loadMarkitdownFile,
  type MarkitdownServiceConfig,
} from './markitdown.js'
export { htmlToMarkdown, loadUrl } from './url.js'
export { loadMarkdown } from './markdown.js'
export { loadText } from './text.js'
export { loadPdf } from './pdf.js'
export { loadJsonTranscript } from './json-transcript.js'

export type LoadInput =
  | {
      readonly kind: 'bytes'
      readonly bytes: Buffer
      readonly filename?: string
      readonly mime?: string
    }
  | { readonly kind: 'url'; readonly url: string }

export type LoadSourceOptions = SourceLoadOptions & {
  readonly fetch?: SourceFetchLike
  readonly forceKind?: SourceKind
  readonly markitdown?: MarkitdownServiceConfig
}

const mergeMeta = (
  base: Readonly<Record<string, unknown>> | undefined,
  extra: Readonly<Record<string, unknown>>,
): Readonly<Record<string, unknown>> => ({
  ...(base ?? {}),
  ...extra,
})

export const loadSource = async (
  input: LoadInput,
  opts: LoadSourceOptions = {},
): Promise<LoadedSource> => {
  if (input.kind === 'url') {
    const urlOpts: LoadUrlOptions = {
      ...(opts.title !== undefined ? { title: opts.title } : {}),
      ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
    }
    const fetched = await fetchUrlSource(input.url, urlOpts)
    const resolvedTitle = opts.title ?? fetched.title
    const loaded = await loadSource(
      {
        kind: 'bytes',
        bytes: fetched.content,
        ...(fetched.filename !== undefined ? { filename: fetched.filename } : {}),
        ...(fetched.mime !== undefined ? { mime: fetched.mime } : {}),
      },
      {
        ...(resolvedTitle !== undefined ? { title: resolvedTitle } : {}),
        ...(opts.forceKind !== undefined ? { forceKind: opts.forceKind } : {}),
        ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
        ...(opts.markitdown !== undefined ? { markitdown: opts.markitdown } : {}),
      },
    )
    return {
      ...loaded,
      meta: mergeMeta(loaded.meta, {
        url: input.url,
        sourceMime: fetched.mime,
      }),
    }
  }
  const { bytes, filename, mime } = input
  const kind =
    opts.forceKind ?? detectSource({ kind: 'bytes', bytes, ...(filename !== undefined ? { filename } : {}), ...(mime !== undefined ? { mime } : {}) })
  const sub: SourceLoadOptions = {
    ...(opts.title !== undefined ? { title: opts.title } : {}),
    ...(filename !== undefined ? { filename } : {}),
    ...(mime !== undefined ? { mime } : {}),
  }
  const candidate = isMarkitdownCandidate({
    ...(filename !== undefined ? { filename } : {}),
    ...(mime !== undefined ? { mime } : {}),
  })
  if ((opts.forceKind === undefined || opts.forceKind === 'binary') && candidate) {
    return loadMarkitdownFile(bytes, {
      ...sub,
      ...(opts.fetch !== undefined ? { fetch: opts.fetch } : {}),
      ...(opts.markitdown !== undefined ? { service: opts.markitdown } : {}),
    })
  }
  switch (kind) {
    case 'markdown':
      return loadMarkdown(bytes, sub)
    case 'html': {
      const html = bytes.toString('utf8')
      const body = htmlToMarkdown(html)
      const resolvedTitle = opts.title
      const content =
        resolvedTitle !== undefined && resolvedTitle !== ''
          ? `# ${resolvedTitle}\n\n${body}`
          : body
      return {
        content: Buffer.from(content, 'utf8'),
        mime: 'text/markdown',
        ...(resolvedTitle !== undefined ? { title: resolvedTitle } : {}),
      }
    }
    case 'pdf':
      return loadPdf(bytes, sub)
    case 'json-transcript':
      return loadJsonTranscript(bytes, sub)
    case 'text':
      return loadText(bytes, sub)
    case 'url':
      throw new Error('loadSource: URL kind requires input.kind === "url"')
    case 'binary':
      return {
        content: bytes,
        mime: sub.mime ?? 'application/octet-stream',
        ...(opts.title !== undefined ? { title: opts.title } : {}),
        meta: { binary: true },
      }
  }
}

export const sourceKindToChunkerMime = (kind: SourceKind): string => {
  switch (kind) {
    case 'markdown':
    case 'html':
    case 'json-transcript':
    case 'pdf':
    case 'url':
      return 'text/markdown'
    case 'text':
      return 'text/plain'
    case 'binary':
      return 'application/octet-stream'
  }
}

export const detectInputFromBytes = (
  bytes: Buffer,
  filename?: string,
  mime?: string,
): DetectInput => ({
  kind: 'bytes',
  bytes,
  ...(filename !== undefined ? { filename } : {}),
  ...(mime !== undefined ? { mime } : {}),
})
