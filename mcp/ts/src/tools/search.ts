// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  query: z.string().min(1).max(4096),
  brain: z.string().optional(),
  top_k: z.number().int().min(1).max(100).optional(),
  scope: z.enum(['all', 'global', 'project', 'agent']).optional(),
  sort: z.enum(['relevance', 'recency', 'relevance_then_recency']).optional(),
})

export const searchTool: Tool<typeof schema> = {
  name: 'memory_search',
  description:
    'Search memory notes in a brain and return matching note content with citations. `scope` selects the memory namespace, and `sort` controls whether relevance or recency wins.',
  inputSchema: schema,
  async handler(args, client) {
    const result = await client.search({
      query: args.query,
      brain: args.brain,
      topK: args.top_k,
      scope: args.scope,
      sort: args.sort,
    })
    return jsonContent(result)
  },
}
