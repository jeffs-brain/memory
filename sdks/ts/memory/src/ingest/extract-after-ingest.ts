// SPDX-License-Identifier: Apache-2.0

/**
 * Post-ingest extraction: after a document is ingested successfully,
 * optionally run the memory extractor to derive structured facts from
 * the raw content. The extraction is non-fatal: if it fails, the ingest
 * result is still returned with the extraction error logged.
 */

import type { Logger } from '../llm/index.js'
import { noopLogger } from '../llm/index.js'
import type { Memory, ExtractedMemory } from '../memory/types.js'

/** Default maximum content length (in characters) passed to the extractor. */
const DEFAULT_MAX_CONTENT_CHARS = 128_000

export type ExtractAfterIngestOptions = {
  readonly brainId: string
  readonly documentPath: string
  readonly documentContent: string
  readonly memory: Pick<Memory, 'extract'>
  readonly actorId?: string | undefined
  readonly sessionId?: string | undefined
  readonly maxContentChars?: number | undefined
  readonly logger?: Logger | undefined
}

export type ExtractAfterIngestResult = {
  readonly factsExtracted: number
  readonly memories: readonly ExtractedMemory[]
}

/**
 * Run the memory extractor on document content after a successful ingest.
 *
 * Builds a synthetic user message from the document content (truncated to
 * maxContentChars) and passes it to `memory.extract()`. Returns the
 * extracted memories or an empty result if extraction fails.
 */
export const extractAfterIngest = async (
  opts: ExtractAfterIngestOptions,
): Promise<ExtractAfterIngestResult> => {
  const logger = opts.logger ?? noopLogger
  const maxChars = opts.maxContentChars ?? DEFAULT_MAX_CONTENT_CHARS

  if (opts.documentContent.trim() === '') {
    logger.debug('extract-after-ingest: empty content, skipping extraction')
    return { factsExtracted: 0, memories: [] }
  }

  const truncated =
    opts.documentContent.length > maxChars
      ? opts.documentContent.slice(0, maxChars)
      : opts.documentContent

  const syntheticMessages = [
    {
      role: 'user' as const,
      content: `The following document was ingested from "${opts.documentPath}". Extract any important facts, knowledge, or structured information from it:\n\n${truncated}`,
    },
  ]

  try {
    const extracted = await opts.memory.extract({
      messages: syntheticMessages,
      ...(opts.actorId !== undefined ? { actorId: opts.actorId } : {}),
      ...(opts.sessionId !== undefined ? { sessionId: opts.sessionId } : {}),
      scope: 'global',
    })

    return {
      factsExtracted: extracted.length,
      memories: extracted,
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    logger.warn(`extract-after-ingest: extraction failed: ${message}`)
    return { factsExtracted: 0, memories: [] }
  }
}
