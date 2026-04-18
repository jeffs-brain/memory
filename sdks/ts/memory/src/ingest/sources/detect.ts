// SPDX-License-Identifier: Apache-2.0

/**
 * Source detection. Uses a combination of URL scheme, magic bytes, and
 * file extension to classify an ingest input. All heuristics are pure and
 * dependency-free.
 */

export type SourceKind =
  | 'markdown'
  | 'text'
  | 'url'
  | 'pdf'
  | 'json-transcript'
  | 'html'
  | 'binary'

export type DetectInput =
  | { readonly kind: 'url'; readonly url: string }
  | {
      readonly kind: 'bytes'
      readonly bytes: Buffer
      readonly filename?: string
      readonly mime?: string
    }

const EXT_MAP: Record<string, SourceKind> = {
  md: 'markdown',
  markdown: 'markdown',
  txt: 'text',
  text: 'text',
  log: 'text',
  pdf: 'pdf',
  html: 'html',
  htm: 'html',
  json: 'text',
}

const extOf = (filename: string | undefined): string | undefined => {
  if (filename === undefined) return undefined
  const idx = filename.lastIndexOf('.')
  if (idx === -1) return undefined
  return filename.slice(idx + 1).toLowerCase()
}

const isPdfMagic = (buf: Buffer): boolean =>
  buf.length >= 4 && buf.slice(0, 4).toString('binary') === '%PDF'

const looksBinary = (buf: Buffer): boolean => {
  // Any NUL byte in the first 4 KiB flags the buffer as binary. UTF-16 is
  // not supported here; callers that need it should pass a `mime` hint.
  const head = buf.slice(0, 4096)
  for (let i = 0; i < head.length; i++) {
    if (head[i] === 0) return true
  }
  return false
}

const sniffJsonTranscript = (buf: Buffer): boolean => {
  const snippet = buf.slice(0, 4096).toString('utf8').trimStart()
  if (!snippet.startsWith('{')) return false
  // Cheap check for the `messages` key. Avoids parsing the full document.
  return /"messages"\s*:/.test(snippet)
}

const sniffHtml = (buf: Buffer): boolean => {
  const snippet = buf.slice(0, 512).toString('utf8').trimStart().toLowerCase()
  return snippet.startsWith('<!doctype html') || snippet.startsWith('<html')
}

const sniffMarkdown = (buf: Buffer): boolean => {
  const snippet = buf.slice(0, 4096).toString('utf8')
  if (/^#{1,6}\s+\S/m.test(snippet)) return true
  if (/^\s*[-*+]\s+\S/m.test(snippet)) return true
  if (/^>\s+\S/m.test(snippet)) return true
  return false
}

export const detectSource = (input: DetectInput): SourceKind => {
  if (input.kind === 'url') {
    const lower = input.url.toLowerCase()
    if (lower.endsWith('.pdf')) return 'pdf'
    return 'url'
  }
  const { bytes, filename, mime } = input
  if (mime !== undefined && mime !== '') {
    if (mime.includes('pdf')) return 'pdf'
    if (mime.includes('markdown')) return 'markdown'
    if (mime.includes('html')) return 'html'
    if (mime.startsWith('text/')) {
      if (sniffMarkdown(bytes)) return 'markdown'
      return 'text'
    }
    if (mime.includes('json')) {
      if (sniffJsonTranscript(bytes)) return 'json-transcript'
      return 'text'
    }
  }
  if (isPdfMagic(bytes)) return 'pdf'
  if (looksBinary(bytes)) return 'binary'
  if (sniffJsonTranscript(bytes)) return 'json-transcript'
  if (sniffHtml(bytes)) return 'html'
  const ext = extOf(filename)
  if (ext !== undefined) {
    const mapped = EXT_MAP[ext]
    if (mapped !== undefined) {
      if (mapped === 'text' && sniffMarkdown(bytes)) return 'markdown'
      return mapped
    }
  }
  if (sniffMarkdown(bytes)) return 'markdown'
  return 'text'
}
