// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  brain: z.string().optional(),
})

export const consolidateTool: Tool<typeof schema> = {
  name: 'memory_consolidate',
  description:
    'Trigger a consolidation pass on the brain (compile summaries, promote stable notes, prune stale episodic memory).',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const result = await client.consolidate(args, ctx?.progress)
    return jsonContent(result)
  },
}
