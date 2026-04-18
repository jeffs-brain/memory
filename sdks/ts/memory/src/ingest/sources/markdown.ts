import type { LoadedSource, SourceLoadOptions } from './types.js'

export const loadMarkdown = async (
  bytes: Buffer,
  opts: SourceLoadOptions = {},
): Promise<LoadedSource> => ({
  content: bytes,
  mime: 'text/markdown',
  ...(opts.title !== undefined ? { title: opts.title } : {}),
  meta: {},
})
