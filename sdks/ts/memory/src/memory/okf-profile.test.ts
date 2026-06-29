// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Message,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { parseFrontmatter } from './frontmatter.js'
import { createMemory } from './index.js'
import { scopeTopic } from './paths.js'

const stubProvider = (content: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  complete: async (_req: CompletionRequest): Promise<CompletionResponse> => ({
    content,
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async (_req: StructuredRequest) => content,
})

const conversation = (n: number): Message[] => {
  const out: Message[] = []
  for (let i = 0; i < n; i++)
    out.push({ role: i % 2 === 0 ? 'user' : 'assistant', content: `msg ${i}` })
  return out
}

const extraction = JSON.stringify({
  memories: [
    {
      action: 'create',
      filename: 'feedback-testing.md',
      name: 'Feedback on testing',
      description: 'User prefers integration tests',
      type: 'feedback',
      scope: 'global',
      content: 'Prefer integration tests over snapshots.',
      index_entry: '- feedback-testing.md: testing preference',
    },
  ],
})

describe('frontmatterProfile threads through createMemory → extract', () => {
  it('writes OKF-shaped notes when the okf profile is selected', async () => {
    const store = createMemStore()
    const mem = createMemory({
      store,
      provider: stubProvider(extraction),
      cursorStore: createStoreBackedCursorStore(store),
      scope: 'project',
      actorId: 'tenant-a',
      frontmatterProfile: 'okf',
    })

    await mem.extract({ messages: conversation(8) })

    const note = (
      await store.read(scopeTopic('global', 'tenant-a', 'feedback-testing.md'))
    ).toString('utf8')
    // OKF keys present; native name/modified keys replaced by title/timestamp.
    expect(note).toContain('title: Feedback on testing')
    expect(note).toMatch(/^timestamp:/m)
    expect(note).not.toMatch(/^name:/m)
    expect(note).not.toMatch(/^modified:/m)
    // Body still follows the frontmatter.
    expect(note).toContain('Prefer integration tests over snapshots.')

    // And it reads back to the right logical fields.
    const { frontmatter } = parseFrontmatter(note)
    expect(frontmatter.name).toBe('Feedback on testing')
    expect(frontmatter.type).toBe('feedback')
    expect(frontmatter.modified).toBeDefined()
  })

  it('writes native-format notes by default (no profile set)', async () => {
    const store = createMemStore()
    const mem = createMemory({
      store,
      provider: stubProvider(extraction),
      cursorStore: createStoreBackedCursorStore(store),
      scope: 'project',
      actorId: 'tenant-a',
    })

    await mem.extract({ messages: conversation(8) })

    const note = (
      await store.read(scopeTopic('global', 'tenant-a', 'feedback-testing.md'))
    ).toString('utf8')
    expect(note).toMatch(/^name: Feedback on testing$/m)
    expect(note).not.toContain('title:')
    expect(note).not.toMatch(/^timestamp:/m)
  })
})
