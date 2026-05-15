// SPDX-License-Identifier: Apache-2.0

import type { z } from 'zod'
import type { TriggerBus } from '@jeffs-brain/memory/ingest'
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
 *
 * `triggerBus` is optionally provided when the server is wired with an
 * event bus. Tools that support async dispatch (e.g. directory ingest)
 * will publish events to the bus and return immediately.
 */
export type ToolContext = {
  readonly progress?: ProgressEmitter
  readonly triggerBus?: TriggerBus
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
