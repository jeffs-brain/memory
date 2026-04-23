import { describe, expect, it } from 'vitest'

import type { Message } from '../llm/types.js'
import { createMobileStore } from '../store/index.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { createEpisodeRecorder, defaultEpisodeRecorderConfig, episodePath } from './episodes.js'
import { parseFrontmatter } from './frontmatter.js'

const freshStore = async () =>
  await createMobileStore({
    root: '/brains/episodes',
    adapter: createMemoryFileAdapter(),
  })

const BASE_MESSAGES: readonly Message[] = [
  { role: 'user', content: 'Please implement durable episode persistence.' },
  {
    role: 'assistant',
    content: 'I am updating the recorder and writing the markdown file now.',
    toolCalls: [{ id: 'write-1', name: 'write', arguments: '{"path":"episodes.ts"}' }],
  },
  { role: 'tool', name: 'write', toolCallId: 'write-1', content: 'Wrote episodes.ts' },
  { role: 'user', content: 'Make sure the gate stays strict and deterministic.' },
  {
    role: 'assistant',
    content: 'I edited the tests and saved the updated assertions.',
    toolCalls: [{ id: 'edit-1', name: 'edit', arguments: '{"path":"episodes.test.ts"}' }],
  },
  { role: 'tool', name: 'edit', toolCallId: 'edit-1', content: 'Updated episodes.test.ts' },
  { role: 'user', content: 'Confirm the reflection data is preserved too.' },
  { role: 'assistant', content: 'The reflection fields are now included in the stored note.' },
]

const reflection = () => ({
  outcome: 'success' as const,
  summary: 'Persisted a clean episode note for the session.',
  retryFeedback: 'Keep the gate strict and batch writes through the store.',
  shouldRecordEpisode: true,
  openQuestions: ['Should this wire into createMemory next?'],
  heuristics: [
    {
      rule: 'Keep persistence batched and deterministic.',
      context: 'memory episode recording',
      confidence: 'high' as const,
      category: 'architecture',
      scope: 'project' as const,
      antiPattern: false,
    },
  ],
})

describe('episodes', () => {
  it('records a durable markdown episode and reads it back', async () => {
    const store = await freshStore()
    const episodes = createEpisodeRecorder({
      store,
      defaultScope: 'project',
      defaultActorId: 'tenant-a',
    })

    const result = await episodes.record({
      sessionId: 'sess-123',
      messages: BASE_MESSAGES,
      reflection: reflection(),
      tags: ['release-review', 'Architecture'],
      startedAt: '2026-04-18T08:00:00.000Z',
      endedAt: '2026-04-18T08:30:00.000Z',
    })

    expect(result.recorded).toBe(true)
    expect(result.path).toBe(episodePath('sess-123'))
    const raw = await store.read(episodePath('sess-123'))
    const parsed = parseFrontmatter(raw)
    expect(parsed.frontmatter.type).toBe('episode')
    expect(parsed.frontmatter.description).toBe('Persisted a clean episode note for the session.')
    expect(parsed.frontmatter.extra.actor_id).toBe('tenant-a')
    expect(raw).toContain('"should_record_episode": true')

    await expect(episodes.get('sess-123')).resolves.toMatchObject({
      sessionId: 'sess-123',
      actorId: 'tenant-a',
      outcome: 'success',
    })
    await expect(episodes.list()).resolves.toHaveLength(1)
  })

  it('enforces the threshold and action-signal gates', async () => {
    const store = await freshStore()
    const episodes = createEpisodeRecorder({
      store,
      defaultScope: 'project',
      defaultActorId: 'tenant-a',
      config: defaultEpisodeRecorderConfig({
        minMessages: 4,
        minSubstantiveMessages: 2,
        requireActionSignal: true,
      }),
    })

    await expect(
      episodes.record({
        sessionId: 'too-short',
        messages: [{ role: 'user', content: 'hello' }],
        reflection: reflection(),
      }),
    ).resolves.toMatchObject({ recorded: false, reason: 'below_threshold' })

    await expect(
      episodes.record({
        sessionId: 'no-action',
        messages: [
          { role: 'user', content: 'Can you reflect on the session?' },
          { role: 'assistant', content: 'I reviewed the transcript and summarised it.' },
          { role: 'user', content: 'What stands out?' },
          { role: 'assistant', content: 'The outcome looks significant.' },
        ],
        reflection: reflection(),
      }),
    ).resolves.toMatchObject({ recorded: false, reason: 'no_action_signal' })
  })
})
