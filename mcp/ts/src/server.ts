// SPDX-License-Identifier: Apache-2.0

/**
 * Stdio MCP server bootstrap. Wires the Model Context Protocol SDK to
 * the unified `MemoryClient` and registers the thirteen `memory_*` tools
 * from `./tools/`.
 */

import { readFileSync } from 'node:fs'
import { Server } from '@modelcontextprotocol/sdk/server/index.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import { z } from 'zod/v4'
import { resolveConfig } from './config.js'
import { type MemoryClient, type ProgressEmitter, createMemoryClient } from './memory-client.js'
import { type Tool, type ToolResult, tools } from './tools/index.js'
import type { ToolContext } from './tools/types.js'

export const SERVER_NAME = '@jeffs-brain/memory-mcp'

/** Read `version` from package.json at `relative` to this module. */
const readPackageVersion = (relative: string): string => {
  const raw: unknown = JSON.parse(readFileSync(new URL(relative, import.meta.url), 'utf8'))
  if (
    typeof raw === 'object' &&
    raw !== null &&
    'version' in raw &&
    typeof raw.version === 'string'
  ) {
    return raw.version
  }
  throw new Error(`package.json at ${relative} has no version`)
}

/** The published package version, read so it can never drift. */
export const SERVER_VERSION = readPackageVersion('../package.json')

type ToolRegistry = ReadonlyMap<string, Tool>

const buildRegistry = (): ToolRegistry => {
  const map = new Map<string, Tool>()
  for (const tool of tools) {
    if (map.has(tool.name)) {
      throw new Error(`duplicate tool registration: ${tool.name}`)
    }
    map.set(tool.name, tool)
  }
  return map
}

/**
 * The JSON Schema advertised in `tools/list`, generated from the same zod
 * schema that validates the call. `io: 'input'` keeps defaulted fields
 * optional, and the `$schema` marker is dropped to match the Go and
 * Python servers.
 */
export const toJsonSchema = (schema: z.ZodType): Record<string, unknown> => {
  const { $schema: _dialect, ...rest } = z.toJSONSchema(schema, { io: 'input' })
  return rest
}

const runTool = async (
  registry: ToolRegistry,
  client: MemoryClient,
  name: string,
  rawArgs: unknown,
  ctx: ToolContext,
): Promise<ToolResult> => {
  const tool = registry.get(name)
  if (!tool) {
    throw new Error(`unknown tool: ${name}`)
  }
  const parsed = tool.inputSchema.parse(rawArgs ?? {}) as z.infer<typeof tool.inputSchema>
  return tool.handler(parsed, client, ctx)
}

export type CreateServerResult = {
  readonly server: Server
  readonly client: MemoryClient
}

/**
 * Build a fully-wired MCP `Server` plus its backing `MemoryClient`. The
 * caller is responsible for connecting a transport (stdio in production,
 * `InMemoryTransport` in tests) and for calling `shutdown()` when done.
 */
export const createServer = (client: MemoryClient): Server => {
  const registry = buildRegistry()
  const server = new Server(
    { name: SERVER_NAME, version: SERVER_VERSION },
    { capabilities: { tools: {} } },
  )

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      inputSchema: toJsonSchema(tool.inputSchema),
    })),
  }))

  server.setRequestHandler(CallToolRequestSchema, async (req, extra) => {
    const { name, arguments: args } = req.params
    const progressToken = req.params._meta?.progressToken
    const sendNotification = extra.sendNotification.bind(extra)
    let counter = 0
    const progress: ProgressEmitter | undefined =
      progressToken === undefined
        ? undefined
        : (_progress, message) => {
            counter += 1
            void sendNotification({
              method: 'notifications/progress',
              params: {
                progressToken,
                progress: counter,
                ...(message !== undefined ? { message } : {}),
              },
            })
          }
    const ctx: ToolContext = progress !== undefined ? { progress } : {}
    try {
      const result = await runTool(registry, client, name, args, ctx)
      return {
        content: result.content.map((c) => ({ type: c.type, text: c.text })),
        ...(result.structuredContent !== undefined
          ? { structuredContent: result.structuredContent }
          : {}),
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      return {
        isError: true,
        content: [{ type: 'text' as const, text: message }],
      }
    }
  })

  return server
}

export const bootstrap = async (): Promise<void> => {
  const cfg = resolveConfig(process.env)
  const client = createMemoryClient(cfg)
  const server = createServer(client)

  const transport = new StdioServerTransport()
  await server.connect(transport)

  const shutdown = async (): Promise<void> => {
    await client.close().catch(() => {
      /* swallow shutdown errors */
    })
    await server.close().catch(() => {
      /* swallow shutdown errors */
    })
  }

  process.on('SIGINT', () => {
    void shutdown().finally(() => process.exit(0))
  })
  process.on('SIGTERM', () => {
    void shutdown().finally(() => process.exit(0))
  })
}
