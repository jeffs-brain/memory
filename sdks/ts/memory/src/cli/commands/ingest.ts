import { readFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import { defineCommand } from 'citty'
import { createIngest } from '../../knowledge/ingest.js'
import { noopLogger } from '../../llm/index.js'
import { openBrain } from '../brain.js'
import { CliUsageError, resolveBrainDir } from '../config.js'

export const ingestCommand = defineCommand({
  meta: {
    name: 'ingest',
    description: 'Ingest a file into the brain',
  },
  args: {
    file: {
      type: 'positional',
      description: 'Path to the file to ingest',
      required: true,
    },
    brain: {
      type: 'string',
      description: 'Brain directory (overrides JBMEM_BRAIN)',
    },
  },
  run: async ({ args }) => {
    const fileArg = args.file
    if (typeof fileArg !== 'string' || fileArg === '') {
      throw new CliUsageError('ingest: <file> is required')
    }
    const brainDir = resolveBrainDir(
      typeof args.brain === 'string' ? args.brain : undefined,
    )
    const filePath = resolve(fileArg)
    const content = await readFile(filePath)
    const store = await openBrain(brainDir)
    try {
      const ingest = createIngest({ store, logger: noopLogger })
      const result = await ingest(content, { name: basename(filePath) })
      process.stdout.write(
        `${JSON.stringify({
          path: result.path,
          hash: result.hash,
          bytes: result.bytes,
          skipped: result.skipped ?? null,
        })}\n`,
      )
    } finally {
      await store.close()
    }
  },
})
