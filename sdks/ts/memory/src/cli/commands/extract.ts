// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises'
import { defineCommand } from 'citty'
import type { Message } from '../../llm/index.js'
import { createStoreBackedCursorStore } from '../../memory/cursor.js'
import { createMemory } from '../../memory/index.js'
import { openBrain } from '../brain.js'
import { readBrainConfig } from '../brain.js'
import {
  CliUsageError,
  buildEmbedder,
  buildProvider,
  embedderFromEnv,
  providerFromEnv,
  resolveBrainDir,
} from '../config.js'

export const extractCommand = defineCommand({
  meta: {
    name: 'extract',
    description: 'Run the memory extraction stage against a transcript',
  },
  args: {
    brain: {
      type: 'string',
      description: 'Brain directory (overrides brain id)',
    },
    from: {
      type: 'string',
      description: 'Path to messages.json with a chronological transcript',
    },
  },
  run: async ({ args }) => {
    const fromArg = typeof args.from === 'string' ? args.from : ''
    if (fromArg === '') {
      throw new CliUsageError('extract: --from <messages.json> is required')
    }
    const messages = parseMessages(await readFile(fromArg, 'utf8'))
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    const store = await openBrain(brainDir)
    try {
      const config = await readBrainConfig(brainDir)
      const provider = buildProvider(providerFromEnv())
      const embedderSettings = embedderFromEnv()
      const embedder = embedderSettings !== undefined ? buildEmbedder(embedderSettings) : undefined
      const memory = createMemory({
        store,
        provider,
        ...(embedder !== undefined ? { embedder } : {}),
        scope: 'global',
        actorId: config.actorId,
        cursorStore: createStoreBackedCursorStore(store),
      })
      const extracted = await memory.extract({ messages })
      process.stdout.write(`${JSON.stringify({ count: extracted.length, extracted })}\n`)
    } finally {
      await store.close()
    }
  },
})

const parseMessages = (raw: string): Message[] => {
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    throw new CliUsageError(
      `extract: invalid JSON in --from: ${err instanceof Error ? err.message : String(err)}`,
    )
  }
  if (!Array.isArray(parsed)) {
    throw new CliUsageError('extract: --from must be a JSON array of messages')
  }
  const out: Message[] = []
  for (const m of parsed) {
    if (
      m !== null &&
      typeof m === 'object' &&
      'role' in m &&
      'content' in m &&
      typeof (m as { role: unknown }).role === 'string' &&
      typeof (m as { content: unknown }).content === 'string'
    ) {
      const typed = m as { role: string; content: string }
      if (typed.role === 'system' || typed.role === 'user' || typed.role === 'assistant') {
        out.push({ role: typed.role, content: typed.content })
      }
    }
  }
  return out
}
