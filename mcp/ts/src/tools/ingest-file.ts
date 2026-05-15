// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises'
import { isAbsolute, resolve } from 'node:path'
import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const schema = z.object({
  path: z.string().min(1).describe('Absolute or relative local path.'),
  brain: z.string().optional(),
  as: z.enum(['markdown', 'text', 'pdf', 'json']).optional(),
  extract: z.boolean().optional().describe('Extract structured facts after ingestion.'),
})

export const ingestFileTool: Tool<typeof schema> = {
  name: 'memory_ingest_file',
  description: 'Ingest a local file (<= 25 MB) into the brain. Returns the ingest result.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const ingestResult = await client.ingestFile(
      { path: args.path, brain: args.brain, as: args.as },
      ctx?.progress,
    )

    if (args.extract !== true) {
      return jsonContent(ingestResult)
    }

    // Read the file content directly (already on disk, no re-fetch needed).
    const absPath = isAbsolute(args.path) ? args.path : resolve(args.path)
    let extraction = { factsExtracted: 0, memories: [] as readonly { filename: string; content: string }[] }
    try {
      const raw = await readFile(absPath, 'utf8')
      if (raw.length > 0) {
        extraction = await client.extractAfterIngest({
          content: raw,
          documentSource: args.path,
          brain: args.brain,
        })
      }
    } catch {
      // Extraction failure is non-fatal; return ingest result with empty extraction
    }

    return jsonContent({
      ingest: ingestResult,
      extraction,
    })
  },
}
