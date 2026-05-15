// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { enumerateFiles } from '@jeffs-brain/memory/ingest'
import { type Tool, jsonContent } from './types.js'

const MAX_FILES = 500

const schema = z.object({
  directory: z.string().min(1).describe('Absolute path to directory'),
  glob: z.string().optional().describe('Glob pattern filter (e.g. "**/*.md")'),
  brain: z.string().optional(),
  recursive: z.boolean().optional().default(true),
  maxFiles: z.number().int().positive().max(MAX_FILES).optional().default(100),
})

export type DirectoryIngestFileResult = {
  readonly path: string
  readonly status: 'success' | 'error' | 'skipped'
  readonly documentId?: string | undefined
  readonly hash?: string | undefined
  readonly bytes?: number | undefined
  readonly error?: string | undefined
}

export type DirectoryIngestResult = {
  readonly total: number
  readonly succeeded: number
  readonly failed: number
  readonly skipped: number
  readonly skippedReasons: readonly string[]
  readonly results: readonly DirectoryIngestFileResult[]
}

export const ingestDirectoryTool: Tool<typeof schema> = {
  name: 'memory_ingest_directory',
  description:
    'Ingest files from a directory. Walks recursively, respects .gitignore, and supports glob filtering.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const enumerated = await enumerateFiles({
      directory: args.directory,
      glob: args.glob,
      recursive: args.recursive,
      maxFiles: args.maxFiles,
    })

    const total = enumerated.files.length
    const results: DirectoryIngestFileResult[] = []
    let succeeded = 0
    let failed = 0

    for (let i = 0; i < total; i++) {
      const file = enumerated.files[i]
      if (file === undefined) continue

      try {
        const ingestResult = (await client.ingestFile(
          {
            path: file.path,
            brain: args.brain,
          },
          ctx?.progress,
        )) as Record<string, unknown>

        results.push({
          path: file.path,
          status: 'success',
          documentId: ingestResult.document_id as string | undefined,
          hash: ingestResult.hash as string | undefined,
          bytes: ingestResult.byte_size as number | undefined,
        })
        succeeded++
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        results.push({
          path: file.path,
          status: 'error',
          error: message,
        })
        failed++
      }

      ctx?.progress?.(i + 1, `${i + 1}/${total} ${file.path}`)
    }

    const directoryResult: DirectoryIngestResult = {
      total,
      succeeded,
      failed,
      skipped: enumerated.skipped.length,
      skippedReasons: enumerated.skipped,
      results,
    }

    return jsonContent(directoryResult)
  },
}
