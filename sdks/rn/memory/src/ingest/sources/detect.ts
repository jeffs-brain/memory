import { toBytes } from '../../knowledge/hash.js'

export type SourceKind = 'markdown' | 'text' | 'url' | 'pdf' | 'json-transcript' | 'html' | 'binary'

export type DetectInput =
  | { readonly kind: 'url'; readonly url: string }
  | {
      readonly kind: 'bytes'
      readonly bytes: Uint8Array | ArrayBuffer
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
  const index = filename.lastIndexOf('.')
  if (index === -1) return undefined
  return filename.slice(index + 1).toLowerCase()
}

const isPdfMagic = (bytes: Uint8Array): boolean =>
  bytes.length >= 4 &&
  bytes[0] === 0x25 &&
  bytes[1] === 0x50 &&
  bytes[2] === 0x44 &&
  bytes[3] === 0x46

const looksBinary = (bytes: Uint8Array): boolean => {
  const head = bytes.slice(0, 4096)
  for (const byte of head) {
    if (byte === 0) return true
  }
  return false
}

const sniffJsonTranscript = (bytes: Uint8Array): boolean => {
  const snippet = new TextDecoder('utf-8').decode(bytes.slice(0, 4096)).trimStart()
  return snippet.startsWith('{') && /"messages"\s*:/.test(snippet)
}

const sniffHtml = (bytes: Uint8Array): boolean => {
  const snippet = new TextDecoder('utf-8').decode(bytes.slice(0, 512)).trimStart().toLowerCase()
  return snippet.startsWith('<!doctype html') || snippet.startsWith('<html')
}

const sniffMarkdown = (bytes: Uint8Array): boolean => {
  const snippet = new TextDecoder('utf-8').decode(bytes.slice(0, 4096))
  return (
    /^#{1,6}\s+\S/m.test(snippet) || /^\s*[-*+]\s+\S/m.test(snippet) || /^>\s+\S/m.test(snippet)
  )
}

export const detectSource = (input: DetectInput): SourceKind => {
  if (input.kind === 'url') {
    const lower = input.url.toLowerCase()
    return lower.endsWith('.pdf') ? 'pdf' : 'url'
  }

  const bytes = toBytes(input.bytes)
  if (input.mime !== undefined && input.mime !== '') {
    if (input.mime.includes('pdf')) return 'pdf'
    if (input.mime.includes('markdown')) return 'markdown'
    if (input.mime.includes('html')) return 'html'
    if (input.mime.startsWith('text/')) {
      return sniffMarkdown(bytes) ? 'markdown' : 'text'
    }
    if (input.mime.includes('json')) {
      return sniffJsonTranscript(bytes) ? 'json-transcript' : 'text'
    }
  }

  if (isPdfMagic(bytes)) return 'pdf'
  if (looksBinary(bytes)) return 'binary'
  if (sniffJsonTranscript(bytes)) return 'json-transcript'
  if (sniffHtml(bytes)) return 'html'

  const ext = extOf(input.filename)
  if (ext !== undefined) {
    const mapped = EXT_MAP[ext]
    if (mapped !== undefined) {
      if (mapped === 'text' && sniffMarkdown(bytes)) return 'markdown'
      return mapped
    }
  }

  return sniffMarkdown(bytes) ? 'markdown' : 'text'
}
