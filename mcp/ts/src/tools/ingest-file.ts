// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const schema = z.object({
  path: z.string().min(1).describe('Absolute or relative local path.'),
  brain: z.string().optional(),
  as: z.enum(['markdown', 'text', 'pdf', 'json']).optional(),
})

export const ingestFileTool: Tool<typeof schema> = {
  name: 'memory_ingest_file',
  description: 'Ingest a local file (<= 25 MB) into the brain. Returns the ingest result.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const result = await client.ingestFile(args, ctx?.progress)
    return jsonContent(result)
  },
}
