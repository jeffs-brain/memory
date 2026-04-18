import { describe, expect, it } from 'vitest'
import type { Message } from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { parseFrontmatter } from './frontmatter.js'
import {
  createEpisodeRecorder,
  defaultEpisodeRecorderConfig,
  episodePath,
} from './episodes.js'

const BASE_MESSAGES: readonly Message[] = [
  { role: 'user', content: 'Please implement durable episode persistence.' },
  {
    role: 'assistant',
    content: 'I am updating the recorder and writing the markdown file now.',
    toolCalls: [{ id: 'write-1', name: 'write', arguments: '{"path":"episodes.ts"}' }],
  },
  { role: 'tool', name: 'write', toolCallId: 'write-1', content: 'Wrote packages/memory/src/memory/episodes.ts' },
  { role: 'user', content: 'Make sure the gate stays strict and deterministic.' },
  {
    role: 'assistant',
    content: 'I edited the tests and saved the updated assertions.',
    toolCalls: [{ id: 'edit-1', name: 'edit', arguments: '{"path":"episodes.test.ts"}' }],
  },
  { role: 'tool', name: 'edit', toolCallId: 'edit-1', content: 'Updated packages/memory/src/memory/episodes.test.ts' },
  { role: 'user', content: 'Confirm the reflection data is preserved too.' },
  { role: 'assistant', content: 'The reflection fields are now included in the stored episode note.' },
]

const reflection = (overrides: Partial<Parameters<ReturnType<typeof createEpisodeRecorder>['record']>[0]['reflection']> = {}) => ({
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
  ...overrides,
})

describe('episodes', () => {
  it('records a durable markdown episode and reads it back through get and list', async () => {
    const store = createMemStore()
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
    expect(result.reason).toBe('passed')
    expect(result.disposition).toBe('created')
    expect(result.path).toBe(episodePath('sess-123'))
    expect(result.episode).toMatchObject({
      sessionId: 'sess-123',
      actorId: 'tenant-a',
      scope: 'project',
      outcome: 'success',
      shouldRecordEpisode: true,
      tags: expect.arrayContaining([
        'episode',
        'release-review',
        'architecture',
        'outcome-success',
        'signal-write',
        'signal-edit',
        'signal-tool',
        'heuristic-architecture',
      ]),
    })

    const raw = (await store.read(episodePath('sess-123'))).toString('utf8')
    const parsed = parseFrontmatter(raw)
    expect(parsed.frontmatter.type).toBe('episode')
    expect(parsed.frontmatter.scope).toBe('project')
    expect(parsed.frontmatter.session_id).toBe('sess-123')
    expect(parsed.frontmatter.description).toBe('Persisted a clean episode note for the session.')
    expect(parsed.frontmatter.extra.actor_id).toBe('tenant-a')
    expect(parsed.frontmatter.extra.outcome).toBe('success')
    expect(parsed.frontmatter.extra.write_signal).toBe('true')
    expect(raw).toContain('## Summary')
    expect(raw).toContain('## Heuristics')
    expect(raw).toContain('## Episode data')
    expect(raw).toContain('"session_id": "sess-123"')
    expect(raw).toContain('"should_record_episode": true')

    const stored = await episodes.get('sess-123')
    expect(stored).toMatchObject({
      sessionId: 'sess-123',
      actorId: 'tenant-a',
      summary: 'Persisted a clean episode note for the session.',
      outcome: 'success',
      retryFeedback: 'Keep the gate strict and batch writes through the store.',
      openQuestions: ['Should this wire into createMemory next?'],
    })
    expect(stored?.heuristics).toEqual([
      expect.objectContaining({
        rule: 'Keep persistence batched and deterministic.',
        antiPattern: false,
      }),
    ])

    const listed = await episodes.list()
    expect(listed).toHaveLength(1)
    expect(listed[0]?.sessionId).toBe('sess-123')
  })

  it('enforces threshold, action-signal, and model significance gates', async () => {
    const store = createMemStore()
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
    ).resolves.toMatchObject({
      recorded: false,
      allowed: false,
      reason: 'below_threshold',
    })

    await expect(
      episodes.record({
        sessionId: 'model-declined',
        messages: BASE_MESSAGES,
        reflection: reflection({ shouldRecordEpisode: false }),
      }),
    ).resolves.toMatchObject({
      recorded: false,
      allowed: false,
      reason: 'model_declined',
    })

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
    ).resolves.toMatchObject({
      recorded: false,
      allowed: false,
      reason: 'no_action_signal',
    })

    expect(await episodes.list()).toHaveLength(0)
  })

  it('updates an existing episode in place and preserves the original created timestamp', async () => {
    const store = createMemStore()
    const episodes = createEpisodeRecorder({
      store,
      defaultScope: 'project',
      defaultActorId: 'tenant-a',
    })

    const first = await episodes.record({
      sessionId: 'sess-update',
      messages: BASE_MESSAGES,
      reflection: reflection({
        summary: 'Initial summary.',
        heuristics: [],
      }),
      endedAt: '2026-04-18T08:30:00.000Z',
    })
    const firstCreated = first.episode?.created
    expect(first.disposition).toBe('created')
    expect(firstCreated).toBeDefined()

    const second = await episodes.record({
      sessionId: 'sess-update',
      messages: BASE_MESSAGES,
      reflection: reflection({
        outcome: 'partial',
        summary: 'Updated summary after a second reflection pass.',
        heuristics: [
          {
            rule: 'Prefer idempotent episode writes.',
            context: 'rewriting the same session note',
            confidence: 'medium',
            category: 'testing',
            scope: 'project',
            antiPattern: false,
          },
        ],
      }),
      endedAt: '2026-04-18T08:45:00.000Z',
    })

    expect(second.recorded).toBe(true)
    expect(second.disposition).toBe('updated')
    expect(second.episode?.created).toBe(firstCreated)
    expect(second.episode?.summary).toBe('Updated summary after a second reflection pass.')
    expect(second.episode?.outcome).toBe('partial')
    expect(second.episode?.heuristics).toEqual([
      expect.objectContaining({ rule: 'Prefer idempotent episode writes.' }),
    ])

    const listed = await episodes.list()
    expect(listed).toHaveLength(1)
    expect(listed[0]?.sessionId).toBe('sess-update')
  })

  it('queries and filters recorded episodes by text, actor, scope, outcome, and tags', async () => {
    const store = createMemStore()
    const episodes = createEpisodeRecorder({
      store,
      defaultScope: 'project',
      defaultActorId: 'tenant-a',
    })

    await episodes.record({
      sessionId: 'sess-routes',
      messages: BASE_MESSAGES,
      reflection: reflection({
        summary: 'Tightened the route contract tests and persisted the recorder.',
        heuristics: [
          {
            rule: 'Keep route tests narrow and explicit.',
            context: 'backend routes',
            confidence: 'high',
            category: 'testing',
            scope: 'project',
            antiPattern: false,
          },
        ],
      }),
      tags: ['backend'],
      endedAt: '2026-04-18T09:00:00.000Z',
    })

    await episodes.record({
      sessionId: 'sess-global',
      messages: BASE_MESSAGES,
      reflection: reflection({
        summary: 'Captured a cross-project workflow reminder.',
        heuristics: [
          {
            rule: 'Start from the ticket context before changing shared code.',
            context: 'cross-project planning',
            confidence: 'medium',
            category: 'communication',
            scope: 'global',
            antiPattern: false,
          },
        ],
      }),
      actorId: 'tenant-b',
      scope: 'global',
      tags: ['planning'],
      endedAt: '2026-04-18T10:00:00.000Z',
    })

    const hits = await episodes.query({
      query: 'route tests',
      actorId: 'tenant-a',
      tags: ['backend'],
      outcome: 'success',
      scope: 'project',
    })

    expect(hits).toHaveLength(1)
    expect(hits[0]?.sessionId).toBe('sess-routes')
    expect(hits[0]?.score).toBeGreaterThan(0)

    const filtered = await episodes.list({
      actorId: 'tenant-b',
      scope: 'global',
      tags: ['planning'],
    })
    expect(filtered).toHaveLength(1)
    expect(filtered[0]?.sessionId).toBe('sess-global')
  })
})
