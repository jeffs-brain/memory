// SPDX-License-Identifier: Apache-2.0

import { askTool } from './ask.js'
import { consolidateTool } from './consolidate.js'
import { createBrainTool } from './create-brain.js'
import { extractTool } from './extract.js'
import { ingestBatchTool } from './ingest-batch.js'
import { ingestDirectoryTool } from './ingest-directory.js'
import { ingestFileTool } from './ingest-file.js'
import { ingestUrlTool } from './ingest-url.js'
import { listBrainsTool } from './list-brains.js'
import { recallTool } from './recall.js'
import { reflectTool } from './reflect.js'
import { rememberTool } from './remember.js'
import { searchTool } from './search.js'
import type { Tool } from './types.js'

export const tools: readonly Tool[] = [
  rememberTool,
  recallTool,
  searchTool,
  askTool,
  ingestBatchTool,
  ingestDirectoryTool,
  ingestFileTool,
  ingestUrlTool,
  extractTool,
  reflectTool,
  consolidateTool,
  createBrainTool,
  listBrainsTool,
]

export type { Tool, ToolResult } from './types.js'
