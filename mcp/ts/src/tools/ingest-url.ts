// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  url: z.string().url(),
  brain: z.string().optional(),
})

export const ingestUrlTool: Tool<typeof schema> = {
  name: 'memory_ingest_url',
  description:
    'Fetch a URL and ingest its contents into the brain. Uses the server-side /ingest/url endpoint when available; otherwise fetches locally and creates a document.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const result = await client.ingestUrl(args, ctx?.progress)
    return jsonContent(result)
  },
}
