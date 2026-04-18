// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({
  name: z.string().min(1).max(128),
  slug: z
    .string()
    .min(1)
    .max(64)
    .regex(/^[a-z0-9][a-z0-9-]*$/)
    .optional(),
  visibility: z.enum(['private', 'tenant', 'public']).optional(),
})

export const createBrainTool: Tool<typeof schema> = {
  name: 'memory_create_brain',
  description: 'Create a new brain. Generates a slug from the name if one is not provided.',
  inputSchema: schema,
  async handler(args, client) {
    const result = await client.createBrain(args)
    return jsonContent(result)
  },
}
