// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { jsonContent, type Tool } from './types.js'

const schema = z.object({}).strict()

export const listBrainsTool: Tool<typeof schema> = {
  name: 'memory_list_brains',
  description: 'List all brains the caller has access to.',
  inputSchema: schema,
  async handler(_args, client) {
    const result = await client.listBrains()
    return jsonContent(result)
  },
}
