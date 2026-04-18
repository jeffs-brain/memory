// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  content: z.string().min(1).max(5_000_000).describe('Markdown body of the new memory.'),
  title: z
    .string()
    .min(1)
    .max(512)
    .optional()
    .describe('Title. Derived from the first heading if omitted.'),
  brain: z
    .string()
    .optional()
    .describe('Brain id or slug. Defaults to JB_BRAIN or the singleton brain.'),
  tags: z.array(z.string().min(1).max(64)).max(64).optional(),
  path: z.string().min(1).max(1024).optional().describe('Destination path within the brain.'),
})

export const rememberTool: Tool<typeof schema> = {
  name: 'memory_remember',
  description:
    'Store a new memory (markdown document) in the brain. Returns the created document id and path.',
  inputSchema: schema,
  async handler(args, client) {
    const result = await client.remember(args)
    return jsonContent(result)
  },
}
