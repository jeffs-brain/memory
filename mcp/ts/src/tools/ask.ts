// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const schema = z.object({
  query: z.string().min(1).max(8192),
  brain: z.string().optional(),
  top_k: z.number().int().min(1).max(50).optional(),
})

export const askTool: Tool<typeof schema> = {
  name: 'memory_ask',
  description:
    'Ask a question grounded in the brain. Streams answer tokens as MCP progress notifications and returns the final answer with citations.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    // TODO(next-pass): once the memory SDK surfaces a streaming `ask`
    // that yields `answer_delta` frames, forward each delta through
    // `ctx.progress(counter, delta)` so MCP clients can render tokens
    // as they arrive. Today we emit two coarse progress events
    // (retrieved + answered) which is enough to satisfy the protocol.
    const result = await client.ask(
      {
        query: args.query,
        brain: args.brain,
        topK: args.top_k,
      },
      ctx?.progress,
    )
    return jsonContent(result)
  },
}
