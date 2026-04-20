// SPDX-License-Identifier: Apache-2.0

/**
 * URL source: fetch the resource and strip it down to a markdown-like text
 * blob. The HTML stripper is intentionally simple — it removes script /
 * style / nav / footer / aside blocks and converts a handful of block
 * elements into markdown equivalents.
 */

import type { LoadedSource, SourceFetchLike, SourceLoadOptions } from './types.js'

export type LoadUrlOptions = SourceLoadOptions & {
  readonly fetch?: SourceFetchLike
  readonly signal?: AbortSignal
  readonly maxBytes?: number
}

export type FetchedUrlSource = {
  readonly content: Buffer
  readonly mime: string
  readonly title?: string
  readonly filename?: string
  readonly url: string
}

const DEFAULT_MAX_BYTES = 10 * 1024 * 1024

const defaultFetch = (): SourceFetchLike => {
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as SourceFetchLike
  }
  throw new Error('loadUrl: no global fetch available; pass opts.fetch')
}

const DROP_BLOCKS = ['script', 'style', 'nav', 'footer', 'aside', 'svg', 'noscript']

const stripBlock = (html: string, tag: string): string => {
  const re = new RegExp(`<${tag}[\\s\\S]*?<\\/${tag}>`, 'gi')
  return html.replace(re, '')
}

const decodeEntities = (text: string): string =>
  text
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))

const extractTitle = (html: string): string | undefined => {
  const match = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(html)
  if (match === null) return undefined
  const raw = (match[1] ?? '').trim()
  if (raw === '') return undefined
  return decodeEntities(raw)
}

/**
 * Minimal HTML → markdown-ish stripper. Covers the shapes most ingest
 * consumers care about without pulling in a DOM library.
 */
export const htmlToMarkdown = (html: string): string => {
  let out = html
  for (const tag of DROP_BLOCKS) out = stripBlock(out, tag)
  // Headings
  out = out.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h\1>/gi, (_, level, inner) => {
    const hashes = '#'.repeat(Number(level))
    return `\n\n${hashes} ${stripTags(String(inner)).trim()}\n\n`
  })
  // Lists
  out = out.replace(
    /<li[^>]*>([\s\S]*?)<\/li>/gi,
    (_, inner) => `\n- ${stripTags(String(inner)).trim()}`,
  )
  out = out.replace(/<\/(ul|ol)>/gi, '\n')
  // Blockquotes
  out = out.replace(/<blockquote[^>]*>([\s\S]*?)<\/blockquote>/gi, (_, inner) => {
    const body = stripTags(String(inner)).trim()
    return body
      .split(/\n+/)
      .map((l) => `> ${l}`)
      .join('\n')
  })
  // Code blocks
  out = out.replace(/<pre[^>]*>([\s\S]*?)<\/pre>/gi, (_, inner) => {
    const body = stripTags(String(inner))
    return `\n\n\`\`\`\n${body.trim()}\n\`\`\`\n\n`
  })
  out = out.replace(
    /<code[^>]*>([\s\S]*?)<\/code>/gi,
    (_, inner) => `\`${stripTags(String(inner)).trim()}\``,
  )
  // Paragraphs / breaks
  out = out.replace(/<br\s*\/?>(\s*)/gi, '\n')
  out = out.replace(/<\/p>/gi, '\n\n')
  out = out.replace(/<p[^>]*>/gi, '')
  out = stripTags(out)
  out = decodeEntities(out)
  return out
    .replace(/\n{3,}/g, '\n\n')
    .split('\n')
    .map((l) => l.replace(/[\t ]+/g, ' ').replace(/\s+$/, ''))
    .join('\n')
    .trim()
}

const stripTags = (html: string): string => html.replace(/<[^>]+>/g, '')

const filenameFromUrl = (url: string): string | undefined => {
  try {
    const pathname = new URL(url).pathname
    const filename = pathname.split('/').pop()
    return filename === undefined || filename === '' ? undefined : filename
  } catch {
    return undefined
  }
}

export const fetchUrlSource = async (
  url: string,
  opts: LoadUrlOptions = {},
): Promise<FetchedUrlSource> => {
  const fetchImpl = opts.fetch ?? defaultFetch()
  const max = opts.maxBytes ?? DEFAULT_MAX_BYTES
  const resp = await fetchImpl(url, opts.signal !== undefined ? { signal: opts.signal } : {})
  if (!resp.ok) {
    throw new Error(`loadUrl: ${url} failed with ${resp.status} ${resp.statusText}`)
  }
  const ab = await resp.arrayBuffer()
  if (ab.byteLength > max) {
    throw new Error(`loadUrl: ${url} exceeded maxBytes=${max}`)
  }
  const content = Buffer.from(ab)
  const mime = resp.headers.get('content-type')?.split(';')[0]?.trim() || 'text/plain'
  const title = mime.includes('html') ? extractTitle(content.toString('utf8')) : undefined
  const filename = filenameFromUrl(url)
  return {
    content,
    mime,
    ...(title !== undefined ? { title } : {}),
    ...(filename !== undefined ? { filename } : {}),
    url,
  }
}

export const loadUrl = async (url: string, opts: LoadUrlOptions = {}): Promise<LoadedSource> => {
  const fetched = await fetchUrlSource(url, opts)
  const contentType = fetched.mime
  const buf = fetched.content
  const isHtml =
    contentType.includes('html') || /<html[\s>]/i.test(buf.slice(0, 512).toString('utf8'))
  if (!isHtml) {
    return {
      content: buf,
      mime: contentType,
      ...(opts.title !== undefined ? { title: opts.title } : {}),
      meta: { url },
    }
  }
  const html = buf.toString('utf8')
  const title = fetched.title ?? opts.title
  const body = htmlToMarkdown(html)
  const content = title !== undefined && title !== '' ? `# ${title}\n\n${body}` : body
  return {
    content: Buffer.from(content, 'utf8'),
    mime: 'text/markdown',
    ...(title !== undefined ? { title } : {}),
    meta: { url, sourceMime: contentType },
  }
}
