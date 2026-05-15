// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const MAX_BATCH_SIZE = 50

const fileEntrySchema = z.object({
  path: z.string().min(1),
  as: z.enum(['markdown', 'text', 'pdf', 'json']).optional(),
  title: z.string().optional(),
})

const schema = z.object({
  files: z.array(fileEntrySchema).min(1).max(MAX_BATCH_SIZE),
  brain: z.string().optional(),
})

export type BatchIngestFileResult = {
  readonly path: string
  readonly status: 'success' | 'error'
  readonly documentId?: string | undefined
  readonly hash?: string | undefined
  readonly bytes?: number | undefined
  readonly error?: string | undefined
}

export type BatchIngestResult = {
  readonly total: number
  readonly succeeded: number
  readonly failed: number
  readonly results: readonly BatchIngestFileResult[]
}

export const ingestBatchTool: Tool<typeof schema> = {
  name: 'memory_ingest_batch',
  description: 'Ingest up to 50 local files in a single call. Returns per-file results.',
  inputSchema: schema,
  async handler(args, client, ctx) {
    const total = args.files.length
    const results: BatchIngestFileResult[] = []
    let succeeded = 0
    let failed = 0

    for (let i = 0; i < total; i++) {
      const file = args.files[i]
      if (file === undefined) continue

      try {
        const ingestResult = (await client.ingestFile(
          {
            path: file.path,
            brain: args.brain,
            as: file.as,
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

    const batchResult: BatchIngestResult = {
      total,
      succeeded,
      failed,
      results,
    }

    return jsonContent(batchResult)
  },
}
