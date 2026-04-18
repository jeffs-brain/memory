// SPDX-License-Identifier: Apache-2.0

/**
 * PDF source adapter. Delegates text extraction to `pdf-parse`, which is
 * a node-only dependency. When the module cannot be loaded we throw a
 * descriptive error instead of crashing the caller.
 */

import type { LoadedSource, SourceLoadOptions } from './types.js'

type PdfParseFn = (
  data: Buffer,
  opts?: Record<string, unknown>,
) => Promise<{ text: string; numpages?: number; info?: unknown }>

let pdfParseCache: PdfParseFn | null | undefined

const tryLoadPdfParse = async (): Promise<PdfParseFn | null> => {
  if (pdfParseCache !== undefined) return pdfParseCache
  try {
    // @ts-expect-error — pdf-parse is an optional runtime dependency
    const mod = (await import('pdf-parse')) as unknown as
      | { default?: PdfParseFn }
      | PdfParseFn
    if (typeof mod === 'function') {
      pdfParseCache = mod
    } else if (typeof (mod as { default?: PdfParseFn }).default === 'function') {
      pdfParseCache = (mod as { default: PdfParseFn }).default
    } else {
      pdfParseCache = null
    }
  } catch {
    pdfParseCache = null
  }
  return pdfParseCache
}

export const loadPdf = async (
  bytes: Buffer,
  opts: SourceLoadOptions = {},
): Promise<LoadedSource> => {
  const pdfParse = await tryLoadPdfParse()
  if (pdfParse === null) {
    throw new Error(
      'loadPdf: pdf-parse is not installed; add it or supply pre-extracted text',
    )
  }
  const result = await pdfParse(bytes)
  const body = (result.text ?? '').trim()
  const content = opts.title !== undefined && opts.title !== ''
    ? `# ${opts.title}\n\n${body}`
    : body
  return {
    content: Buffer.from(content, 'utf8'),
    mime: 'text/markdown',
    ...(opts.title !== undefined ? { title: opts.title } : {}),
    meta: { pages: result.numpages ?? 0 },
  }
}
