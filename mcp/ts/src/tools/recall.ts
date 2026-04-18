// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  query: z.string().min(1).max(4096),
  brain: z.string().optional(),
  scope: z.enum(['global', 'project', 'agent']).optional(),
  session_id: z.string().optional(),
  top_k: z.number().int().min(1).max(50).optional(),
})

export const recallTool: Tool<typeof schema> = {
  name: 'memory_recall',
  description:
    'Recall memories for a query. Pass session_id to weight recent session context; otherwise uses the dedicated memory-search surface. `scope` selects the memory namespace rather than a generic metadata filter.',
  inputSchema: schema,
  async handler(args, client) {
    const result = await client.recall({
      query: args.query,
      brain: args.brain,
      scope: args.scope,
      sessionId: args.session_id,
      topK: args.top_k,
    })
    return jsonContent(result)
  },
}
