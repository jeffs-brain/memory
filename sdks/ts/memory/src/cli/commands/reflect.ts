// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises'
import { defineCommand } from 'citty'
import type { Message } from '../../llm/index.js'
import { createStoreBackedCursorStore } from '../../memory/cursor.js'
import { createMemory } from '../../memory/index.js'
import { openBrain, readBrainConfig } from '../brain.js'
import {
  buildProvider,
  CliUsageError,
  providerFromEnv,
  resolveBrainDir,
} from '../config.js'

export const reflectCommand = defineCommand({
  meta: {
    name: 'reflect',
    description: 'Run the reflection stage over a session transcript',
  },
  args: {
    brain: {
      type: 'string',
      description: 'Brain directory (overrides JB_BRAIN)',
    },
    session: {
      type: 'string',
      description: 'Session identifier (used as the reflection filename)',
    },
    from: {
      type: 'string',
      description: 'Path to messages.json for this session',
    },
  },
  run: async ({ args }) => {
    const sessionId = typeof args.session === 'string' ? args.session : ''
    if (sessionId === '') {
      throw new CliUsageError('reflect: --session <id> is required')
    }
    const fromArg = typeof args.from === 'string' ? args.from : ''
    if (fromArg === '') {
      throw new CliUsageError('reflect: --from <messages.json> is required')
    }
    const messages = parseMessages(await readFile(fromArg, 'utf8'))
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
      const result = await memory.reflect({ messages, sessionId })
      process.stdout.write(`${JSON.stringify({ sessionId, result: result ?? null })}\n`)
    } finally {
      await store.close()
    }
  },
})

const parseMessages = (raw: string): Message[] => {
  const parsed: unknown = JSON.parse(raw)
  if (!Array.isArray(parsed)) {
    throw new CliUsageError('reflect: --from must be a JSON array of messages')
  }
  const out: Message[] = []
  for (const m of parsed) {
    if (
      m !== null &&
      typeof m === 'object' &&
      'role' in m &&
      'content' in m
    ) {
      const role = (m as { role: unknown }).role
      const content = (m as { content: unknown }).content
      if (
        typeof content === 'string' &&
        (role === 'system' || role === 'user' || role === 'assistant')
      ) {
        out.push({ role, content })
      }
    }
  }
  return out
}
