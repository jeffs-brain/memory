import { toText } from '../../knowledge/hash.js'
import type { LoadedSource, SourceLoadOptions } from './types.js'

export const loadText = async (
  bytes: Uint8Array | ArrayBuffer,
  opts: SourceLoadOptions = {},
): Promise<LoadedSource> => ({
  content: toText(bytes),
  mime: 'text/plain',
  ...(opts.title !== undefined ? { title: opts.title } : {}),
  meta: {},
})
