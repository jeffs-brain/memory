// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type { Message } from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { toPath } from '../store/path.js'
import { parseFrontmatter } from './frontmatter.js'
import { createStoreBackedProceduralStore, proceduralSessionPrefix } from './procedural-store.js'
import type { ProceduralRecord } from './types.js'

const makeRecord = (
  overrides: Partial<ProceduralRecord> & Pick<ProceduralRecord, 'name'>,
): ProceduralRecord => ({
  tier: overrides.tier ?? 'skill',
  name: overrides.name,
  taskContext: overrides.taskContext ?? '',
  outcome: overrides.outcome ?? 'ok',
  observedAt: overrides.observedAt ?? '2026-04-18T10:00:00Z',
  toolCalls: overrides.toolCalls ?? ['skill'],
  tags: overrides.tags ?? ['procedural', overrides.tier ?? 'skill', overrides.name],
})

describe('StoreBackedProceduralStore', () => {
  it('detects, persists, and reloads procedural notes durably', async () => {
    const store = createMemStore()
    const proceduralStore = createStoreBackedProceduralStore(store)
    const messages: Message[] = [
      { role: 'user', content: 'Deploy the chart and verify the ingress values.' },
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

    const [stored] = await proceduralStore.detectAndPersist({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      messages,
      observedAt: '2026-04-18T10:00:00Z',
    })

    expect(stored).toMatchObject({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      tier: 'skill',
      name: 'kubernetes-deploy',
      outcome: 'ok',
      toolCalls: ['skill'],
    })
    expect(stored).toBeDefined()
    if (stored === undefined) {
      throw new Error('expected stored procedural record')
    }

    expect(stored.path.startsWith(proceduralSessionPrefix('tenant-a', 'sess-1'))).toBe(true)

    const raw = (await store.read(stored.path)).toString('utf8')
    const parsed = parseFrontmatter(raw)
    expect(parsed.frontmatter.type).toBe('procedural')
    expect(parsed.frontmatter.source).toBe('procedural')
    expect(parsed.frontmatter.session_id).toBe('sess-1')
    expect(parsed.frontmatter.observed_on).toBe('2026-04-18T10:00:00.000Z')
    expect(parsed.frontmatter.extra.actor_id).toBe('tenant-a')
    expect(parsed.body).toContain('## Context')
    expect(parsed.body).toContain('## Tool sequence')
    expect(parsed.body).toContain('Deploy the chart and verify the ingress values.')

    const reloaded = createStoreBackedProceduralStore(store)
    await expect(reloaded.list({ actorId: 'tenant-a', sessionId: 'sess-1' })).resolves.toEqual([
      expect.objectContaining({
        path: stored.path,
        actorId: 'tenant-a',
        sessionId: 'sess-1',
        name: 'kubernetes-deploy',
        tags: ['procedural', 'skill', 'kubernetes-deploy'],
        taskContext: 'Deploy the chart and verify the ingress values.',
        toolCalls: ['skill'],
      }),
    ])
  })

  it('keeps duplicate records as separate files within one persist call', async () => {
    const store = createMemStore()
    const proceduralStore = createStoreBackedProceduralStore(store)
    const duplicate = makeRecord({
      name: 'planner',
      tier: 'agent',
      toolCalls: ['agent'],
      observedAt: '2026-04-18T11:00:00Z',
      tags: ['procedural', 'agent', 'planner'],
    })

    const stored = await proceduralStore.persist({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      records: [duplicate, duplicate],
    })

    expect(stored).toHaveLength(2)
    expect(new Set(stored.map((record) => record.path)).size).toBe(2)

    const entries = await store.list(proceduralSessionPrefix('tenant-a', 'sess-1'), {
      recursive: true,
    })
    expect(entries.filter((entry) => !entry.isDir && entry.path.endsWith('.md'))).toHaveLength(2)
  })

  it('lists persisted records with actor, session, tag, tier, outcome, and date filters', async () => {
    const store = createMemStore()
    const proceduralStore = createStoreBackedProceduralStore(store)

    await proceduralStore.persist({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      records: [
        makeRecord({
          name: 'kubernetes-deploy',
          tags: ['procedural', 'skill', 'kubernetes-deploy', 'deploy'],
          observedAt: '2026-04-18T10:00:00Z',
        }),
        makeRecord({
          name: 'research',
          tier: 'agent',
          outcome: 'error',
          taskContext: 'Investigate the timeout path.',
          toolCalls: ['agent'],
          tags: ['procedural', 'agent', 'research', 'timeout'],
          observedAt: '2026-04-18T11:00:00Z',
        }),
      ],
    })
    await proceduralStore.persist({
      actorId: 'tenant-a',
      sessionId: 'sess-2',
      records: [
        makeRecord({
          name: 'docker-build',
          tags: ['procedural', 'skill', 'docker-build', 'build'],
          observedAt: '2026-04-19T09:00:00Z',
        }),
      ],
    })
    await proceduralStore.persist({
      actorId: 'tenant-b',
      sessionId: 'sess-1',
      records: [
        makeRecord({
          name: 'external-run',
          tags: ['procedural', 'skill', 'external-run', 'deploy'],
          observedAt: '2026-04-20T09:00:00Z',
        }),
      ],
    })

    await expect(proceduralStore.list({ actorId: 'tenant-a' })).resolves.toMatchObject([
      { name: 'docker-build' },
      { name: 'research' },
      { name: 'kubernetes-deploy' },
    ])

    await expect(
      proceduralStore.list({
        actorId: 'tenant-a',
        sessionId: 'sess-1',
        tier: 'agent',
        outcome: 'error',
        tags: ['timeout'],
        since: '2026-04-18T10:30:00Z',
      }),
    ).resolves.toMatchObject([{ name: 'research' }])

    await expect(
      proceduralStore.list({
        actorId: 'tenant-a',
        tier: 'skill',
        tags: ['deploy'],
        until: '2026-04-18T23:59:59Z',
      }),
    ).resolves.toMatchObject([{ name: 'kubernetes-deploy' }])
  })

  it('queries persisted records by name, tags, tool calls, and task context', async () => {
    const store = createMemStore()
    const proceduralStore = createStoreBackedProceduralStore(store)

    await proceduralStore.persist({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      records: [
        makeRecord({
          name: 'kubernetes-deploy',
          taskContext: 'Deploy the chart and inspect the ingress controller settings.',
          tags: ['procedural', 'skill', 'kubernetes-deploy', 'deploy', 'ingress'],
          toolCalls: ['skill'],
          observedAt: '2026-04-18T10:00:00Z',
        }),
        makeRecord({
          name: 'research',
          tier: 'agent',
          taskContext: 'Investigate the timeout in the session sync path.',
          outcome: 'error',
          toolCalls: ['agent'],
          tags: ['procedural', 'agent', 'research', 'timeout'],
          observedAt: '2026-04-18T11:00:00Z',
        }),
      ],
    })

    const deployHits = await proceduralStore.query({
      actorId: 'tenant-a',
      text: 'kubernetes deploy ingress',
    })
    expect(deployHits[0]).toMatchObject({
      name: 'kubernetes-deploy',
      toolCalls: ['skill'],
    })
    expect(deployHits[0]?.score).toBeGreaterThan(0)

    const researchHits = await proceduralStore.query({
      actorId: 'tenant-a',
      text: 'agent timeout research',
      limit: 1,
    })
    expect(researchHits).toMatchObject([{ name: 'research', outcome: 'error' }])
    expect(researchHits[0]?.score).toBeGreaterThan(0)
  })

  it('ignores malformed or non-procedural notes under the procedural prefix', async () => {
    const store = createMemStore()
    const proceduralStore = createStoreBackedProceduralStore(store)

    await store.write(
      toPath('memory/_procedural/actor_tenant-a/session_sess-1/2026/04/18/bad.md'),
      Buffer.from('not frontmatter at all', 'utf8'),
    )
    await store.write(
      toPath('memory/_procedural/actor_tenant-a/session_sess-1/2026/04/18/other.md'),
      Buffer.from('---\ntype: reflection\n---\nhello\n', 'utf8'),
    )

    await expect(proceduralStore.list({ actorId: 'tenant-a' })).resolves.toEqual([])
    await expect(proceduralStore.query({ actorId: 'tenant-a', text: 'anything' })).resolves.toEqual(
      [],
    )
  })
})
