import {
  type LoadUrlOptions as BaseLoadUrlOptions,
  htmlToMarkdown as baseHtmlToMarkdown,
  loadUrl as loadUrlText,
} from '../url.js'
import type { LoadedSource, SourceFetchLike, SourceLoadOptions } from './types.js'

export type LoadUrlOptions = SourceLoadOptions & {
  readonly fetch?: SourceFetchLike
  readonly signal?: AbortSignal
  readonly maxBytes?: number
}

export type FetchedUrlSource = {
  readonly content: string
  readonly mime: string
  readonly title?: string
  readonly filename?: string
  readonly url: string
}

export const htmlToMarkdown = (html: string): string => baseHtmlToMarkdown(html)

export const fetchUrlSource = async (
  url: string,
  opts: LoadUrlOptions = {},
): Promise<FetchedUrlSource> => {
  const loaded = await loadUrlText(url, opts as BaseLoadUrlOptions)
  return {
    content: loaded.content,
    mime: loaded.mime,
    ...(loaded.title !== undefined ? { title: loaded.title } : {}),
    ...(loaded.filename !== undefined ? { filename: loaded.filename } : {}),
    url,
  }
}

export const loadUrl = async (url: string, opts: LoadUrlOptions = {}): Promise<LoadedSource> => {
  const fetched = await fetchUrlSource(url, opts)
  return {
    content: fetched.content,
    mime: fetched.mime,
    ...(fetched.title !== undefined ? { title: fetched.title } : {}),
    meta: {
      url,
      ...(fetched.mime === 'text/markdown' ? { sourceMime: 'text/html' } : {}),
    },
  }
}
