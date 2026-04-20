#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0

/**
 * Entry point for `@jeffs-brain/memory-mcp`. Boots a stdio MCP server
 * wired to either a local brain (FsStore + sqlite search + optional
 * Ollama) or a hosted brain (HttpStore) depending on env.
 */

import { bootstrap } from './server.js'

bootstrap().catch((err: unknown) => {
  const message = err instanceof Error ? (err.stack ?? err.message) : String(err)
  process.stderr.write(`[memory-mcp] fatal: ${message}\n`)
  process.exit(1)
})
