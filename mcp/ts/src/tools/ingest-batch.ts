// SPDX-License-Identifier: Apache-2.0

import { z } from 'zod'
import { type Tool, jsonContent } from './types.js'

const MAX_BATCH_SIZE = 50
const MAX_CONCURRENCY = 5

const fileEntrySchema = z.object({
  path: z.string().min(1),
  as: z.enum(['markdown', 'text', 'pdf', 'json']).optional(),
})

const schema = z.object({
  files: z.array(fileEntrySchema).min(1).max(MAX_BATCH_SIZE),
  brain: z.string().optional(),
})

/** Typed subset of the ingest response relevant to batch result reporting. */
type IngestResult = {
  readonly document_id?: string
  readonly hash?: string
  readonly byte_size?: number
}

const isIngestResult = (value: unknown): value is IngestResult =>
  typeof value === 'object' && value !== null

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
    const results: BatchIngestFileResult[] = new Array(total)
    let succeeded = 0
    let failed = 0
    let completed = 0

    const processFile = async (index: number): Promise<void> => {
      const file = args.files[index]
      if (file === undefined) return

      try {
        const raw = await client.ingestFile(
          {
            path: file.path,
            brain: args.brain,
            as: file.as,
          },
          ctx?.progress,
        )

        const ingestResult: IngestResult = isIngestResult(raw) ? raw : {}

        results[index] = {
          path: file.path,
          status: 'success',
          documentId: ingestResult.document_id,
          hash: ingestResult.hash,
          bytes: ingestResult.byte_size,
        }
        succeeded++
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        results[index] = {
          path: file.path,
          status: 'error',
          error: message,
        }
        failed++
      }

      completed++
      ctx?.progress?.(completed, `${completed}/${total} ${file.path}`)
    }

    // Bounded concurrency pool: process up to MAX_CONCURRENCY files at a time.
    const executing: Set<Promise<void>> = new Set()
    for (let i = 0; i < total; i++) {
      const task = processFile(i).then(() => {
        executing.delete(task)
      })
      executing.add(task)
      if (executing.size >= MAX_CONCURRENCY) {
        await Promise.race(executing)
      }
    }
    await Promise.all(executing)

    const batchResult: BatchIngestResult = {
      total,
      succeeded,
      failed,
      results: results.filter((r): r is BatchIngestFileResult => r !== undefined),
    }

    return jsonContent(batchResult)
  },
}
