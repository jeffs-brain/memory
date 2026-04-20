// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { Message } from '../llm/index.js'
import {
  L0_BUFFER_SNAPSHOT_VERSION,
  appendL0Observation,
  compactL0Buffer,
  defaultL0BufferConfig,
  estimateL0BufferTokens,
  exportL0BufferSnapshot,
  formatL0Observation,
  needsL0BufferCompaction,
  observeMessages,
  renderL0Reminder,
  restoreL0BufferSnapshot,
} from './buffer.js'

describe('buffer', () => {
  it('builds deterministic observations from the last turn', () => {
    const messages: Message[] = [
      {
        role: 'user',
        content:
          'Please update the sync worker. <system-reminder>ignore this stale note</system-reminder> Touch the queue handler too.',
      },
      {
        role: 'assistant',
        toolCalls: [
          {
            id: 'read-a',
            name: 'read',
            arguments: JSON.stringify({ path: 'src/sync/worker.ts' }),
          },
          {
            id: 'write-b',
            name: 'write',
            arguments: JSON.stringify({ file_path: 'src/queue/handler.ts' }),
          },
        ],
      },
      {
        role: 'tool',
        toolCallId: 'read-a',
        name: 'read',
        content: 'Loaded file successfully.',
      },
      {
        role: 'tool',
        toolCallId: 'write-b',
        name: 'write',
        content: 'Error: permission denied',
      },
    ]

    expect(
      observeMessages(messages, {
        observedAt: '2026-04-18T09:10:11Z',
      }),
    ).toEqual({
      at: '2026-04-18T09:10:11.000Z',
      intent: 'edit',
      outcome: 'error',
      summary: 'Please update the sync worker. Touch the queue handler too.',
      entities: ['src/sync/worker.ts', 'src/queue/handler.ts'],
    })
  })

  it('renders reminders and compacts to the newest observations when the budget is exceeded', () => {
    const observations = [
      {
        at: '2026-04-18T09:00:00Z',
        intent: 'read',
        outcome: 'ok',
        summary: 'Read the sync worker.',
        entities: ['src/sync/worker.ts'],
      },
      {
        at: '2026-04-18T09:05:00Z',
        intent: 'edit',
        outcome: 'partial',
        summary: 'Updated queue handling and left one follow-up.',
        entities: ['src/queue/handler.ts'],
      },
      {
        at: '2026-04-18T09:10:00Z',
        intent: 'chat',
        outcome: 'ok',
        summary: 'Explained the next repair step.',
        entities: [],
      },
      {
        at: '2026-04-18T09:15:00Z',
        intent: 'plan',
        outcome: 'ok',
        summary: 'Planned the final verification pass.',
        entities: [],
      },
    ] as const

    const config = defaultL0BufferConfig({
      tokenBudget: 40,
      keepRecentPercent: 50,
    })

    expect(renderL0Reminder(observations.slice(0, 2), config)).toContain('<system-reminder>')
    expect(formatL0Observation(observations[1])).toContain('[partial]')
    expect(estimateL0BufferTokens(observations, config)).toBeGreaterThan(config.tokenBudget)
    expect(needsL0BufferCompaction(observations, config)).toBe(true)

    const compacted = compactL0Buffer(observations, config)
    expect(compacted.removed).toBe(3)
    expect(compacted.observations).toEqual([observations[3]])
    expect(estimateL0BufferTokens(compacted.observations, config)).toBeLessThanOrEqual(
      config.tokenBudget,
    )
  })

  it('appends observations, dedupes entities, and marks incomplete tool runs as partial', () => {
    const messages: Message[] = [
      { role: 'user', content: 'Can you inspect the docs and follow up?' },
      {
        role: 'assistant',
        toolCalls: [
          {
            id: 'read-docs',
            name: 'read',
            arguments: JSON.stringify({
              path: 'docs/plan.md',
              files: ['docs/plan.md', 'docs/notes.md'],
            }),
          },
          {
            id: 'grep-docs',
            name: 'grep',
            arguments: JSON.stringify({ paths: ['docs/plan.md'] }),
          },
        ],
      },
      {
        role: 'tool',
        toolCallId: 'read-docs',
        name: 'read',
        content: 'Loaded docs.',
      },
    ]

    const observation = observeMessages(messages, {
      observedAt: '2026-04-18T09:20:00Z',
      maxEntities: 3,
    })
    expect(observation).toMatchObject({
      intent: 'read',
      outcome: 'partial',
      entities: ['docs/plan.md', 'docs/notes.md'],
    })
    if (observation === undefined) {
      throw new Error('Expected an observation to be produced')
    }

    const appended = appendL0Observation(
      [
        {
          at: '2026-04-18T09:00:00Z',
          intent: 'chat',
          outcome: 'ok',
          summary: 'Started the session.',
          entities: [],
        },
      ],
      {
        ...observation,
        entities: ['docs/plan.md', 'docs/plan.md', 'docs/notes.md'],
        summary:
          'Can you inspect the docs and follow up? This should stay concise once the observation is sanitised for the rolling buffer.',
      },
      {
        maxObservationLength: 70,
      },
    )

    expect(appended.compacted).toBe(false)
    expect(appended.removed).toBe(0)
    expect(appended.observations[1]?.summary).toMatch(
      /^Can you inspect the docs and follow up\? This should stay concise /,
    )
    expect(appended.observations[1]?.summary).toHaveLength(70)
    expect(appended.observations[1]?.entities).toEqual(['docs/plan.md', 'docs/notes.md'])
  })

  it('exports a versioned snapshot and restores it without mutating the source buffer', () => {
    const observations = [
      {
        at: '2026-04-18T09:00:00.000Z',
        intent: 'chat',
        outcome: 'ok',
        summary: 'Started the session.',
        entities: ['docs/plan.md'],
      },
    ] as const

    const snapshot = exportL0BufferSnapshot(observations, {
      createdAt: '2026-04-19T10:11:12Z',
    })

    expect(snapshot.metadata).toEqual({
      format: 'l0-buffer-snapshot',
      version: L0_BUFFER_SNAPSHOT_VERSION,
      createdAt: '2026-04-19T10:11:12.000Z',
      observationCount: 1,
    })
    expect(snapshot.observations).toEqual(observations)
    expect(snapshot.observations).not.toBe(observations)
    expect(snapshot.observations[0]).not.toBe(observations[0])

    const restored = restoreL0BufferSnapshot(snapshot)
    expect(restored).toEqual(observations)
    expect(restored).not.toBe(snapshot.observations)
    expect(restored[0]).not.toBe(snapshot.observations[0])
  })

  it('rejects incompatible snapshot versions and mismatched observation counts', () => {
    const snapshot = exportL0BufferSnapshot([
      {
        at: '2026-04-18T09:00:00.000Z',
        intent: 'chat',
        outcome: 'ok',
        summary: 'Started the session.',
        entities: [],
      },
    ])

    expect(() =>
      restoreL0BufferSnapshot({
        ...snapshot,
        metadata: {
          ...snapshot.metadata,
          version: '2.0.0',
        },
      }),
    ).toThrow('Invalid L0 buffer snapshot version: 2.0.0')

    expect(() =>
      restoreL0BufferSnapshot({
        ...snapshot,
        metadata: {
          ...snapshot.metadata,
          observationCount: 2,
        },
      }),
    ).toThrow('expected 2 observations, got 1')
  })
})
