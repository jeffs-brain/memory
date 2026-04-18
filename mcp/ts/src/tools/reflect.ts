// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  session_id: z.string().min(1),
  brain: z.string().optional(),
})

export const reflectTool: Tool<typeof schema> = {
  name: 'memory_reflect',
  description: 'Close a session and trigger server-side reflection over its messages.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const result = await client.reflect(
      {
        sessionId: args.session_id,
        brain: args.brain,
      },
      ctx?.progress,
    )
    return jsonContent(result)
  },
}
