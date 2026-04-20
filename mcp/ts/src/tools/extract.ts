// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const messageSchema = z.object({
  role: z.enum(['system', 'user', 'assistant', 'tool']),
  content: z.string().min(1),
})

const schema = z.object({
  messages: z.array(messageSchema).min(1).max(500),
  brain: z.string().optional(),
  actor_id: z.string().optional(),
  session_id: z.string().optional(),
})

export const extractTool: Tool<typeof schema> = {
  name: 'memory_extract',
  description:
    'Submit a conversation transcript so the server can asynchronously extract memorable facts. If session_id is provided the messages are appended to that session; otherwise a transcript document is created.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const result = await client.extract(
      {
        messages: args.messages,
        brain: args.brain,
        actorId: args.actor_id,
        sessionId: args.session_id,
      },
      ctx?.progress,
    )
    return jsonContent(result)
  },
}
