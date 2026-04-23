import { toText } from '../../knowledge/hash.js'
import type { LoadedSource, SourceLoadOptions } from './types.js'

export const loadMarkdown = async (
  bytes: Uint8Array | ArrayBuffer,
  opts: SourceLoadOptions = {},
): Promise<LoadedSource> => ({
  content: toText(bytes),
  mime: 'text/markdown',
  ...(opts.title !== undefined ? { title: opts.title } : {}),
  meta: {},
})
