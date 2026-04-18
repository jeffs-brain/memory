// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { Message } from '../llm/index.js'
import { detectProceduralRecords, formatProceduralRecord } from './procedural.js'

describe('procedural', () => {
  it('detects skill invocations with user context and successful outcome', () => {
    const messages: Message[] = [
      { role: 'user', content: 'Deploy the chart and make sure the ingress values are correct.' },
      {
        role: 'assistant',
        toolCalls: [
          {
            id: 'call-skill',
            name: 'skill',
            arguments: JSON.stringify({
              skill: 'kubernetes-deploy',
              args: 'chart=api',
            }),
          },
        ],
      },
      {
        role: 'tool',
        toolCallId: 'call-skill',
        name: 'skill',
        content: 'Completed successfully.',
      },
    ]

    expect(
      detectProceduralRecords(messages, {
        observedAt: '2026-04-18T10:00:00Z',
      }),
    ).toEqual([
      {
        tier: 'skill',
        name: 'kubernetes-deploy',
        taskContext: 'Deploy the chart and make sure the ingress values are correct.',
        outcome: 'ok',
        observedAt: '2026-04-18T10:00:00.000Z',
        toolCalls: ['skill'],
        tags: ['procedural', 'skill', 'kubernetes-deploy'],
      },
    ])
  })

  it('detects agent invocations, prefers prompt context, and marks missing results as partial', () => {
    const messages: Message[] = [
      { role: 'user', content: 'Work out why the sync is slow.' },
      {
        role: 'assistant',
        toolCalls: [
          {
            id: 'call-agent',
            name: 'agent',
            arguments: JSON.stringify({
              type: 'research',
              prompt:
                'Investigate the git sync path, inspect fetch behaviour, and return a concise repair plan for the slowdown.',
            }),
          },
        ],
      },
    ]

    const [record] = detectProceduralRecords(messages, {
      observedAt: new Date('2026-04-18T11:30:00Z'),
      maxContextLength: 80,
    })

    expect(record).toMatchObject({
      tier: 'agent',
      name: 'research',
      outcome: 'partial',
      observedAt: '2026-04-18T11:30:00.000Z',
      toolCalls: ['agent'],
      tags: ['procedural', 'agent', 'research'],
    })
    expect(record?.taskContext).toMatch(
      /^Investigate the git sync path, inspect fetch behaviour, and return a concise /,
    )
    expect(record?.taskContext).toHaveLength(80)
  })

  it('ignores malformed procedural calls and formats records as markdown safely', () => {
    const messages: Message[] = [
      { role: 'user', content: 'Please help.' },
      {
        role: 'assistant',
        toolCalls: [
          { id: 'bad-skill', name: 'skill', arguments: '{' },
          { id: 'other', name: 'read', arguments: JSON.stringify({ path: 'src/index.ts' }) },
        ],
      },
      {
        role: 'assistant',
        toolCalls: [
          {
            id: 'good-agent',
            name: 'agent',
            arguments: JSON.stringify({
              type: 'planner:alpha',
              prompt: '',
            }),
          },
        ],
      },
      {
        role: 'tool',
        toolCallId: 'good-agent',
        name: 'agent',
        content: 'Error: provider timeout',
      },
    ]

    const [record] = detectProceduralRecords(messages, {
      observedAt: '2026-04-18T12:00:00Z',
    })
    expect(record).toMatchObject({
      tier: 'agent',
      name: 'planner:alpha',
      taskContext: 'Please help.',
      outcome: 'error',
    })

    expect(formatProceduralRecord(record!)).toContain('name: "planner:alpha"')
    expect(formatProceduralRecord(record!)).toContain('type: procedural')
    expect(formatProceduralRecord(record!)).toContain('## Tool sequence')
  })
})
