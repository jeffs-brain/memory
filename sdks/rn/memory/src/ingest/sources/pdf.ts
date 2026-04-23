import type { LoadedSource, SourceLoadOptions } from './types.js'

export const loadPdf = async (
  _bytes: Uint8Array | ArrayBuffer,
  _opts: SourceLoadOptions = {},
): Promise<LoadedSource> => {
  throw new Error(
    'loadPdf: React Native does not ship a built-in PDF parser; use markitdown or supply extracted text',
  )
}
