// SPDX-License-Identifier: Apache-2.0

import { defineCommand } from 'citty'
import { initBrain } from '../brain.js'
import { CliUsageError } from '../config.js'

export const initCommand = defineCommand({
  meta: {
    name: 'init',
    description: 'Create a new brain directory with git-backed storage',
  },
  args: {
    path: {
      type: 'positional',
      description: 'Directory to create the brain in',
      required: true,
    },
  },
  run: async ({ args }) => {
    const path = args.path
    if (typeof path !== 'string' || path === '') {
      throw new CliUsageError('init: <path> is required')
    }
    const store = await initBrain(path)
    try {
      process.stdout.write(`${JSON.stringify({ brain: path, ok: true })}\n`)
    } finally {
      await store.close()
    }
  },
})
