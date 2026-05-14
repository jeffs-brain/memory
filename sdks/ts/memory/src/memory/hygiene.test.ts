// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StructuredRequest,
} from '../llm/index.js'
import { createMemStore } from '../store/memstore.js'
import { toPath } from '../store/path.js'
import { createStoreBackedCursorStore } from './cursor.js'
import { type Frontmatter, buildFrontmatter, parseFrontmatter } from './frontmatter.js'
import { runMemoryHygiene } from './hygiene.js'
import { createMemory } from './index.js'

const dummyProvider = (): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  complete: async (_req: CompletionRequest): Promise<CompletionResponse> => ({
    content: '',
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async (_req: StructuredRequest) => '',
})

const writeMemoryFile = async (
  store: ReturnType<typeof createMemStore>,
  path: string,
  frontmatter: Partial<Frontmatter>,
  body: string,
): Promise<void> => {
  const fm: Frontmatter = {
    name: frontmatter.name ?? path.split('/').pop() ?? path,
    type: frontmatter.type ?? 'project',
    scope: frontmatter.scope ?? 'project',
    modified: frontmatter.modified ?? '2026-05-15T10:00:00.000Z',
    extra: frontmatter.extra ?? {},
    ...(frontmatter.description !== undefined ? { description: frontmatter.description } : {}),
    ...(frontmatter.created !== undefined ? { created: frontmatter.created } : {}),
    ...(frontmatter.confidence !== undefined ? { confidence: frontmatter.confidence } : {}),
    ...(frontmatter.source !== undefined ? { source: frontmatter.source } : {}),
    ...(frontmatter.supersedes !== undefined ? { supersedes: frontmatter.supersedes } : {}),
    ...(frontmatter.superseded_by !== undefined
      ? { superseded_by: frontmatter.superseded_by }
      : {}),
    ...(frontmatter.claim_key !== undefined ? { claim_key: frontmatter.claim_key } : {}),
    ...(frontmatter.state_key !== undefined ? { state_key: frontmatter.state_key } : {}),
    ...(frontmatter.state_subject !== undefined
      ? { state_subject: frontmatter.state_subject }
      : {}),
    ...(frontmatter.retired !== undefined ? { retired: frontmatter.retired } : {}),
    ...(frontmatter.retired_on !== undefined ? { retired_on: frontmatter.retired_on } : {}),
    ...(frontmatter.retired_reason !== undefined
      ? { retired_reason: frontmatter.retired_reason }
      : {}),
    ...(frontmatter.tags !== undefined ? { tags: frontmatter.tags } : {}),
  }
  await store.write(toPath(path), Buffer.from(`${buildFrontmatter(fm)}\n${body}\n`, 'utf8'))
}

describe('memory hygiene', () => {
  it('detects contradiction groups by name in dry-run mode', async () => {
    const store = createMemStore()
    await writeMemoryFile(
      store,
      'memory/global/gym-time-a.md',
      { name: 'Gym time', scope: 'global', confidence: 'medium' },
      'Trains at 8am.',
    )
    await writeMemoryFile(
      store,
      'memory/global/gym-time-b.md',
      { name: 'Gym time', scope: 'global', confidence: 'high' },
      'Trains at 7am.',
    )

    const report = await runMemoryHygiene({ store, scope: 'global', actorId: 'tenant-a' })
    expect(report.contradictions).toHaveLength(1)
    expect(report.contradictions[0]?.keyReason).toBe('name')
    expect(report.contradictions[0]?.members).toHaveLength(2)
    expect(report.contradictions[0]?.canonical).toBeUndefined()
  })

  it('applies superseded_by to non-canonical contradictions', async () => {
    const store = createMemStore()
    await writeMemoryFile(
      store,
      'memory/global/gym-time-a.md',
      {
        name: 'Gym time',
        scope: 'global',
        confidence: 'medium',
        modified: '2026-05-15T09:00:00.000Z',
      },
      'Trains at 8am.',
    )
    await writeMemoryFile(
      store,
      'memory/global/gym-time-b.md',
      {
        name: 'Gym time',
        scope: 'global',
        confidence: 'high',
        modified: '2026-05-15T10:00:00.000Z',
      },
      'Trains at 7am.',
    )

    const report = await runMemoryHygiene({
      store,
      scope: 'global',
      actorId: 'tenant-a',
      apply: true,
      now: '2026-05-15T10:00:00.000Z',
    })

    expect(report.contradictions[0]?.canonical).toBe(toPath('memory/global/gym-time-b.md'))
    const old = parseFrontmatter(
      (await store.read(toPath('memory/global/gym-time-a.md'))).toString('utf8'),
    )
    expect(old.frontmatter.superseded_by).toBe('gym-time-b.md')
  })

  it('soft-retires old superseded files in place', async () => {
    const store = createMemStore()
    await writeMemoryFile(
      store,
      'memory/global/old-fact.md',
      {
        name: 'Old',
        scope: 'global',
        modified: '2026-03-31T10:00:00.000Z',
        superseded_by: 'new-fact.md',
      },
      'old body',
    )

    const report = await runMemoryHygiene({
      store,
      scope: 'global',
      actorId: 'tenant-a',
      apply: true,
      retiredAgeDays: 30,
      now: '2026-05-15T10:00:00.000Z',
    })

    expect(report.agingRetired).toHaveLength(1)
    const retired = parseFrontmatter(
      (await store.read(toPath('memory/global/old-fact.md'))).toString('utf8'),
    )
    expect(retired.frontmatter.retired).toBe(true)
    expect(retired.frontmatter.retired_on).toBe('2026-05-15')
  })

  it('does not treat distinct subjects sharing a state schema as contradictions', async () => {
    const store = createMemStore()
    await writeMemoryFile(
      store,
      'memory/global/alex.md',
      {
        name: 'Alex context',
        scope: 'global',
        state_key: 'state.owned.item.set.context',
        state_subject: 'alex',
      },
      'alex body',
    )
    await writeMemoryFile(
      store,
      'memory/global/boudewijn.md',
      {
        name: 'Boudewijn context',
        scope: 'global',
        state_key: 'state.owned.item.set.context',
        state_subject: 'boudewijn',
      },
      'boudewijn body',
    )

    const report = await runMemoryHygiene({ store, scope: 'global', actorId: 'tenant-a' })
    expect(report.contradictions).toHaveLength(0)
  })

  it('is exposed on createMemory with default actor and scope wiring', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    await writeMemoryFile(
      store,
      'memory/project/tenant-a/a.md',
      { name: 'Project setting', confidence: 'low' },
      'old',
    )
    await writeMemoryFile(
      store,
      'memory/project/tenant-a/b.md',
      { name: 'Project setting', confidence: 'high' },
      'new',
    )

    const memory = createMemory({
      store,
      provider: dummyProvider(),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    const report = await memory.hygiene({ apply: true })
    expect(report.contradictions[0]?.canonical).toBe(toPath('memory/project/tenant-a/b.md'))
  })
})
