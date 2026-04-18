// SPDX-License-Identifier: Apache-2.0

import { defineCommand } from 'citty'
import { createStoreBackedCursorStore } from '../../memory/cursor.js'
import { createMemory } from '../../memory/index.js'
import { openBrain, readBrainConfig } from '../brain.js'
import {
  buildProvider,
  providerFromEnv,
  resolveBrainDir,
} from '../config.js'

export const consolidateCommand = defineCommand({
  meta: {
    name: 'consolidate',
    description: 'Run a consolidation pass across the brain',
  },
  args: {
    brain: {
      type: 'string',
      description: 'Brain directory (overrides JB_BRAIN)',
    },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(
      typeof args.brain === 'string' ? args.brain : undefined,
    )
    const store = await openBrain(brainDir)
    try {
      const cfg = await readBrainConfig(brainDir)
      const provider = buildProvider(providerFromEnv())
      const memory = createMemory({
        store,
        provider,
        scope: 'global',
        actorId: cfg.actorId,
        cursorStore: createStoreBackedCursorStore(store),
      })
      const report = await memory.consolidate()
      process.stdout.write(`${JSON.stringify(report)}\n`)
    } finally {
      await store.close()
    }
  },
})
