// SPDX-License-Identifier: Apache-2.0

import { isAbsolute, normalize, resolve } from 'node:path'
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

/** Typed subset of the ingest response relevant to result reporting. */
type IngestResult = {
  readonly document_id?: string
  readonly hash?: string
  readonly byte_size?: number
}

const isIngestResult = (value: unknown): value is IngestResult =>
  typeof value === 'object' && value !== null

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
    const cleanedDir = resolve(normalize(args.directory))
    if (!isAbsolute(cleanedDir)) {
      throw new Error('memory_ingest_directory: directory must be an absolute path')
    }
    if (cleanedDir.includes('..')) {
      throw new Error("memory_ingest_directory: directory path must not contain '..'")
    }

    const enumerated = await enumerateFiles({
      directory: cleanedDir,
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
        const raw = await client.ingestFile(
          {
            path: file.path,
            brain: args.brain,
          },
          ctx?.progress,
        )

        const ingestResult: IngestResult = isIngestResult(raw) ? raw : {}

        results.push({
          path: file.path,
          status: 'success',
          documentId: ingestResult.document_id,
          hash: ingestResult.hash,
          bytes: ingestResult.byte_size,
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
