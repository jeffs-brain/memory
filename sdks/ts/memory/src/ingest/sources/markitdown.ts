// SPDX-License-Identifier: Apache-2.0

import { extname } from 'node:path'
import type { LoadedSource, SourceFetchLike, SourceLoadOptions } from './types.js'

export type MarkitdownServiceConfig = {
  readonly url?: string
  readonly bearerToken?: string
}

export class MarkitdownConfigurationError extends Error {
  override readonly name = 'MarkitdownConfigurationError'
}

export type MarkitdownConvertErrorOptions = {
  readonly status?: number
  readonly clientSafe?: boolean
}

export class MarkitdownConvertError extends Error {
  override readonly name = 'MarkitdownConvertError'
  readonly status: number | undefined
  readonly clientSafe: boolean

  constructor(message: string, opts: MarkitdownConvertErrorOptions = {}) {
    super(message)
    this.status = opts.status
    this.clientSafe = opts.clientSafe ?? false
  }
}

const MARKITDOWN_EXTENSIONS = new Set([
  '.doc',
  '.docm',
  '.docx',
  '.epub',
  '.odp',
  '.ods',
  '.odt',
  '.potm',
  '.potx',
  '.pps',
  '.ppsm',
  '.ppsx',
  '.ppt',
  '.pptm',
  '.pptx',
  '.rtf',
  '.xls',
  '.xlsb',
  '.xlsm',
  '.xlsx',
])

const MARKITDOWN_MIME_EXACT = new Set([
  'application/epub+zip',
  'application/msword',
  'application/rtf',
  'application/vnd.ms-excel',
  'application/vnd.ms-powerpoint',
  'text/rtf',
])

const MARKITDOWN_MIME_PREFIXES = [
  'application/vnd.ms-',
  'application/vnd.oasis.opendocument.',
  'application/vnd.openxmlformats-officedocument.',
]

const normalise = (value: string | undefined): string | undefined => {
  if (value === undefined) return undefined
  const trimmed = value.trim()
  return trimmed === '' ? undefined : trimmed
}

const defaultFetch = (): SourceFetchLike => {
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as SourceFetchLike
  }
  throw new Error('loadMarkitdownFile: no global fetch available; pass opts.fetch')
}

const fileExtension = (filename: string | undefined): string | undefined => {
  if (filename === undefined || filename === '') return undefined
  const ext = extname(filename).toLowerCase()
  return ext === '' ? undefined : ext
}

const isMarkitdownMime = (mime: string | undefined): boolean => {
  const cleaned = normalise(mime)?.toLowerCase()
  if (cleaned === undefined) return false
  if (MARKITDOWN_MIME_EXACT.has(cleaned)) return true
  return MARKITDOWN_MIME_PREFIXES.some((prefix) => cleaned.startsWith(prefix))
}

export const isMarkitdownCandidate = (input: {
  readonly filename?: string
  readonly mime?: string
}): boolean => {
  const ext = fileExtension(input.filename)
  if (ext !== undefined && MARKITDOWN_EXTENSIONS.has(ext)) return true
  return isMarkitdownMime(input.mime)
}

const responseDetail = (raw: string): string | undefined => {
  const trimmed = raw.trim()
  if (trimmed === '') return undefined
  try {
    const parsed = JSON.parse(trimmed) as unknown
    if (
      parsed !== null &&
      typeof parsed === 'object' &&
      'detail' in parsed &&
      typeof (parsed as { detail: unknown }).detail === 'string'
    ) {
      return (parsed as { detail: string }).detail
    }
  } catch {
    return trimmed
  }
  return trimmed
}

const isClientSafeStatus = (status: number): boolean =>
  status === 400 || status === 413 || status === 415 || status === 422

const parseMarkdownResponse = (raw: string): {
  readonly markdown: string
  readonly metadata?: Readonly<Record<string, unknown>>
} => {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw) as unknown
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new MarkitdownConvertError(`markitdown returned invalid JSON (${message})`)
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new MarkitdownConvertError('markitdown returned an invalid response body')
  }
  const markdown = typeof (parsed as { markdown?: unknown }).markdown === 'string'
    ? (parsed as { markdown: string }).markdown.trim()
    : ''
  if (markdown === '') {
    throw new MarkitdownConvertError('markitdown returned empty markdown')
  }
  const metadataValue = (parsed as { metadata?: unknown }).metadata
  const metadata =
    metadataValue !== undefined &&
    metadataValue !== null &&
    typeof metadataValue === 'object' &&
    !Array.isArray(metadataValue)
      ? (metadataValue as Readonly<Record<string, unknown>>)
      : undefined
  return {
    markdown,
    ...(metadata !== undefined ? { metadata } : {}),
  }
}

const resolveServiceConfig = (
  config: MarkitdownServiceConfig | undefined,
): {
  readonly url: string
  readonly bearerToken: string
} => {
  const url = normalise(config?.url)
  const bearerToken = normalise(config?.bearerToken)
  if (url === undefined || bearerToken === undefined) {
    throw new MarkitdownConfigurationError(
      'markitdown service requires MARKITDOWN_SERVICE_URL and MARKITDOWN_SERVICE_BEARER_TOKEN',
    )
  }
  return {
    url: url.replace(/\/+$/, ''),
    bearerToken,
  }
}

export type LoadMarkitdownFileOptions = SourceLoadOptions & {
  readonly filename?: string
  readonly mime?: string
  readonly fetch?: SourceFetchLike
  readonly service?: MarkitdownServiceConfig
}

export const loadMarkitdownFile = async (
  bytes: Buffer,
  opts: LoadMarkitdownFileOptions,
): Promise<LoadedSource> => {
  if (!isMarkitdownCandidate(opts)) {
    throw new MarkitdownConvertError(
      'markitdown was requested for an unsupported file type',
      { clientSafe: true },
    )
  }
  const service = resolveServiceConfig(opts.service)
  const fetchImpl = opts.fetch ?? defaultFetch()
  const filename = opts.filename ?? 'upload.bin'
  const mime = normalise(opts.mime) ?? 'application/octet-stream'
  const form = new FormData()
  form.set('file', new Blob([bytes], { type: mime }), filename)
  const response = await fetchImpl(`${service.url}/convert-file`, {
    method: 'POST',
    headers: { authorization: `Bearer ${service.bearerToken}` },
    body: form,
  })
  const raw = await response.text()
  if (!response.ok) {
    const detail = responseDetail(raw)
    throw new MarkitdownConvertError(
      detail === undefined
        ? `markitdown conversion failed with ${response.status} ${response.statusText}`
        : `markitdown conversion failed with ${response.status} ${response.statusText}: ${detail}`,
      {
        status: response.status,
        clientSafe: isClientSafeStatus(response.status),
      },
    )
  }
  const parsed = parseMarkdownResponse(raw)
  return {
    content: Buffer.from(parsed.markdown, 'utf8'),
    mime: 'text/markdown',
    ...(opts.title !== undefined ? { title: opts.title } : {}),
    meta: {
      converted_by: 'markitdown-service',
      original_filename: filename,
      original_mime: mime,
      ...(parsed.metadata !== undefined ? parsed.metadata : {}),
    },
  }
}
