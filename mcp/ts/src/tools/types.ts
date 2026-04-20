// SPDX-License-Identifier: Apache-2.0

import type { z } from 'zod'
import type { MemoryClient, ProgressEmitter } from '../memory-client.js'

export type ToolContent = {
  readonly type: 'text'
  readonly text: string
}

export type ToolResult = {
  readonly content: readonly ToolContent[]
  readonly structuredContent?: unknown
}

/**
 * Context passed to each tool handler. `progress` is present only when
 * the MCP client supplied a `_meta.progressToken`. Handlers that emit
 * MCP `notifications/progress` events go through this hook.
 */
export type ToolContext = {
  readonly progress?: ProgressEmitter
}

export type Tool<TSchema extends z.ZodTypeAny = z.ZodTypeAny> = {
  readonly name: string
  readonly description: string
  readonly inputSchema: TSchema
  handler(args: z.infer<TSchema>, client: MemoryClient, ctx?: ToolContext): Promise<ToolResult>
}

export const jsonContent = (value: unknown): ToolResult => ({
  content: [{ type: 'text', text: JSON.stringify(value, null, 2) }],
  structuredContent: value,
})
