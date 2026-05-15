// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const schema = z.object({
  url: z.string().url(),
  brain: z.string().optional(),
  extract: z.boolean().optional().describe('Extract structured facts after ingestion.'),
})

export const ingestUrlTool: Tool<typeof schema> = {
  name: 'memory_ingest_url',
  description:
    'Fetch a URL and ingest its contents into the brain. Uses the server-side /ingest/url endpoint when available; otherwise fetches locally and creates a document.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const ingestResult = await client.ingestUrl(
      { url: args.url, brain: args.brain },
      ctx?.progress,
    ) as Record<string, unknown>

    if (args.extract !== true) {
      // Strip internal field before returning to caller.
      const { _document_content: _, ...cleanResult } = ingestResult
      return jsonContent(cleanResult)
    }

    // Read document content from the ingest result (populated by the
    // local client from the fetched buffer). No URL re-fetch needed.
    const content = typeof ingestResult._document_content === 'string'
      ? ingestResult._document_content
      : ''
    const { _document_content: _, ...cleanResult } = ingestResult

    let extraction = { factsExtracted: 0, memories: [] as readonly { filename: string; content: string }[] }
    if (content.length > 0) {
      extraction = await client.extractAfterIngest({
        content,
        documentSource: args.url,
        brain: args.brain,
      })
    }

    return jsonContent({
      ingest: cleanResult,
      extraction,
    })
  },
}
