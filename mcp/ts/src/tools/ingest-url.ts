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
    )

    if (args.extract !== true) {
      return jsonContent(ingestResult)
    }

    // Run extraction after successful ingest
    const extraction = await client.extractAfterIngest({
      url: args.url,
      brain: args.brain,
    })

    return jsonContent({
      ingest: ingestResult,
      extraction,
    })
  },
}
