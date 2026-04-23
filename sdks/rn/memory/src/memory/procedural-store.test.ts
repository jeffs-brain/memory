import { describe, expect, it } from 'vitest'

import { createMobileStore, toPath } from '../store/index.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { parseFrontmatter } from './frontmatter.js'
import { createStoreBackedProceduralStore, proceduralSessionPrefix } from './procedural-store.js'
import type { ProceduralRecord } from './types.js'

const freshStore = async () =>
  await createMobileStore({
    root: '/brains/procedural',
    adapter: createMemoryFileAdapter(),
  })

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
    const store = await freshStore()
    const proceduralStore = createStoreBackedProceduralStore(store)
    const [stored] = await proceduralStore.detectAndPersist({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      observedAt: '2026-04-18T10:00:00Z',
      messages: [
        { role: 'user', content: 'Deploy the chart and verify the ingress values.' },
        {
          role: 'assistant',
          toolCalls: [
            {
              id: 'call-skill',
              name: 'skill',
              arguments: JSON.stringify({ skill: 'kubernetes-deploy', args: 'chart=api' }),
            },
          ],
        },
        {
          role: 'tool',
          toolCallId: 'call-skill',
          name: 'skill',
          content: 'Completed successfully.',
        },
      ],
    })

    expect(stored).toBeDefined()
    if (stored === undefined) throw new Error('expected stored procedural record')
    expect(stored.path.startsWith(proceduralSessionPrefix('tenant-a', 'sess-1'))).toBe(true)

    const raw = await store.read(stored.path)
    const parsed = parseFrontmatter(raw)
    expect(parsed.frontmatter.type).toBe('procedural')
    expect(parsed.frontmatter.source).toBe('procedural')
    expect(parsed.frontmatter.session_id).toBe('sess-1')
    expect(parsed.frontmatter.observed_on).toBe('2026-04-18T10:00:00.000Z')
    expect(parsed.frontmatter.extra.actor_id).toBe('tenant-a')

    await expect(
      proceduralStore.list({ actorId: 'tenant-a', sessionId: 'sess-1' }),
    ).resolves.toEqual([
      expect.objectContaining({
        path: stored.path,
        actorId: 'tenant-a',
        sessionId: 'sess-1',
        name: 'kubernetes-deploy',
      }),
    ])
  })

  it('filters and queries persisted records', async () => {
    const store = await freshStore()
    const proceduralStore = createStoreBackedProceduralStore(store)

    await proceduralStore.persist({
      actorId: 'tenant-a',
      sessionId: 'sess-1',
      records: [
        makeRecord({
          name: 'kubernetes-deploy',
          taskContext: 'Deploy the chart and inspect the ingress settings.',
          tags: ['procedural', 'skill', 'kubernetes-deploy', 'deploy'],
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

    await expect(
      proceduralStore.list({
        actorId: 'tenant-a',
        sessionId: 'sess-1',
        tier: 'agent',
        outcome: 'error',
        tags: ['timeout'],
      }),
    ).resolves.toMatchObject([{ name: 'research' }])

    await expect(
      proceduralStore.query({
        actorId: 'tenant-a',
        text: 'kubernetes deploy ingress',
      }),
    ).resolves.toMatchObject([{ name: 'kubernetes-deploy' }])

    await store.write(
      toPath('memory/_procedural/actor_tenant-a/session_sess-1/2026/04/18/bad.md'),
      'not frontmatter at all',
    )
    await expect(
      proceduralStore.query({ actorId: 'tenant-a', text: 'anything' }),
    ).resolves.toBeDefined()
  })
})
