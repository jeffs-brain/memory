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
import { createMemory } from './index.js'
import type { Plugin } from './types.js'

const provider = (verdict: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  async *stream() {
    yield { type: 'done', stopReason: 'end_turn' as const }
  },
  complete: async (_req: CompletionRequest): Promise<CompletionResponse> => ({
    content: JSON.stringify({ verdict, reason: 'overlap' }),
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  supportsStructuredDecoding: () => false,
  structured: async (_req: StructuredRequest) => JSON.stringify({ verdict, reason: 'overlap' }),
})

const writeNote = async (
  store: ReturnType<typeof createMemStore>,
  path: string,
  body: string,
  frontmatter: Partial<Frontmatter> = {},
) => {
  const builtFrontmatter: Frontmatter = {
    name: frontmatter.name ?? path.split('/').pop() ?? path,
    type: frontmatter.type ?? 'project',
    scope: frontmatter.scope ?? 'project',
    modified: frontmatter.modified ?? isoDaysAgo(1),
    extra: frontmatter.extra ?? {},
    ...(frontmatter.description !== undefined ? { description: frontmatter.description } : {}),
    ...(frontmatter.created !== undefined ? { created: frontmatter.created } : {}),
    ...(frontmatter.confidence !== undefined ? { confidence: frontmatter.confidence } : {}),
    ...(frontmatter.source !== undefined ? { source: frontmatter.source } : {}),
    ...(frontmatter.supersedes !== undefined ? { supersedes: frontmatter.supersedes } : {}),
    ...(frontmatter.superseded_by !== undefined
      ? { superseded_by: frontmatter.superseded_by }
      : {}),
    ...(frontmatter.session_id !== undefined ? { session_id: frontmatter.session_id } : {}),
    ...(frontmatter.session_date !== undefined ? { session_date: frontmatter.session_date } : {}),
    ...(frontmatter.observed_on !== undefined ? { observed_on: frontmatter.observed_on } : {}),
    ...(frontmatter.tags !== undefined ? { tags: frontmatter.tags } : {}),
  }
  const content = `${buildFrontmatter(builtFrontmatter)}\n${body}\n`
  await store.write(toPath(path), Buffer.from(content, 'utf8'))
}

const isoDaysAgo = (days: number): string =>
  new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString()

describe('consolidate', () => {
  it('merges duplicate notes, rebuilds the scope index, and logs the operation', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    // Shared significant word: "auth".
    await writeNote(store, 'memory/project/tenant-a/auth-notes.md', 'Uses OIDC via Lleverage.', {
      description: 'Primary auth implementation note',
      modified: isoDaysAgo(2),
    })
    await writeNote(
      store,
      'memory/project/tenant-a/auth-extra.md',
      'Refresh token rotation enabled.',
      {
        description: 'Additional auth operational detail',
        modified: isoDaysAgo(1),
      },
    )
    await store.write(
      toPath('memory/project/tenant-a/MEMORY.md'),
      Buffer.from('- stale-entry.md: out of date\n', 'utf8'),
    )

    const hooks: string[] = []
    const plugin: Plugin = {
      name: 'probe',
      onConsolidationStart: (ctx) => {
        hooks.push(`start:${ctx.scope}`)
      },
      onConsolidationEnd: (ctx) => {
        hooks.push(`end:${ctx.scope}:${ctx.report?.merged ?? 0}`)
      },
    }

    const mem = createMemory({
      store,
      provider: provider('merge'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
      plugins: [plugin],
    })

    const report = await mem.consolidate()
    expect(report.merged).toBe(1)
    const mergeOp = report.ops.find((op) => op.kind === 'merge')
    expect(mergeOp).toBeDefined()
    if (mergeOp?.kind !== 'merge') throw new Error('expected merge op')

    const keeper = (await store.read(mergeOp.keeper)).toString('utf8')
    expect(keeper).toContain('Uses OIDC via Lleverage.')
    expect(keeper).toContain('Refresh token rotation enabled.')

    await expect(store.read(mergeOp.donor)).rejects.toThrow()
    const index = (await store.read(toPath('memory/project/tenant-a/MEMORY.md'))).toString('utf8')
    expect(index).not.toContain('stale-entry.md')
    expect(index).not.toContain('auth-notes.md')
    expect(index).toContain('auth-extra.md: Additional auth operational detail')

    expect(hooks).toEqual(['start:project', 'end:project:1'])
  })

  it('flags stale notes and decays stale heuristic confidence without clobbering modified history', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    const createdAt = isoDaysAgo(220)
    const modifiedAt = isoDaysAgo(120)
    await writeNote(
      store,
      'memory/project/tenant-a/heuristic-build-pipeline.md',
      'Rule: Keep build steps isolated.\n\nContext: CI pipelines.\n\nWhy: This worked well before.',
      {
        description: 'Build pipeline heuristic',
        created: createdAt,
        modified: modifiedAt,
        confidence: 'high',
        tags: ['heuristic', 'delivery', 'high', 'pattern'],
      },
    )

    const mem = createMemory({
      store,
      provider: provider('distinct'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    const report = await mem.consolidate()
    expect(report.ops).toEqual(
      expect.arrayContaining([
        { kind: 'rewrite', path: toPath('memory/project/tenant-a/heuristic-build-pipeline.md') },
        { kind: 'rewrite', path: toPath('memory/project/tenant-a/MEMORY.md') },
      ]),
    )

    const heuristic = parseFrontmatter(
      (await store.read(toPath('memory/project/tenant-a/heuristic-build-pipeline.md'))).toString(
        'utf8',
      ),
    )
    expect(heuristic.frontmatter.confidence).toBe('medium')
    expect(heuristic.frontmatter.modified).toBe(modifiedAt)
    expect(heuristic.frontmatter.tags).toEqual(
      expect.arrayContaining(['heuristic', 'delivery', 'pattern', 'stale', 'medium']),
    )
    expect(heuristic.frontmatter.tags).not.toContain('high')
    expect(heuristic.frontmatter.extra.stale_since).toBeTruthy()

    const index = (await store.read(toPath('memory/project/tenant-a/MEMORY.md'))).toString('utf8')
    expect(index).toContain('heuristic-build-pipeline.md: Build pipeline heuristic [medium, stale]')
  })

  it('reinforces long-lived heuristics and clears stale markers once the note is recent again', async () => {
    const store = createMemStore()
    const cursorStore = createStoreBackedCursorStore(store)
    await writeNote(
      store,
      'memory/project/tenant-a/heuristic-route-contracts.md',
      'Rule: Check the route contract before patching handlers.\n\nContext: backend routes.\n\nWhy: The same fix kept being useful.',
      {
        description: 'Route contract heuristic',
        created: isoDaysAgo(70),
        modified: isoDaysAgo(10),
        confidence: 'low',
        tags: ['heuristic', 'backend', 'stale', 'low', 'pattern'],
        extra: { stale_since: isoDaysAgo(5) },
      },
    )

    const mem = createMemory({
      store,
      provider: provider('distinct'),
      cursorStore,
      scope: 'project',
      actorId: 'tenant-a',
    })

    await mem.consolidate()

    const heuristic = parseFrontmatter(
      (await store.read(toPath('memory/project/tenant-a/heuristic-route-contracts.md'))).toString(
        'utf8',
      ),
    )
    expect(heuristic.frontmatter.confidence).toBe('high')
    expect(heuristic.frontmatter.tags).toEqual(
      expect.arrayContaining(['heuristic', 'backend', 'pattern', 'high']),
    )
    expect(heuristic.frontmatter.tags).not.toContain('stale')
    expect(heuristic.frontmatter.tags).not.toContain('low')
    expect(heuristic.frontmatter.extra.stale_since).toBeUndefined()

    const index = (await store.read(toPath('memory/project/tenant-a/MEMORY.md'))).toString('utf8')
    expect(index).toContain('heuristic-route-contracts.md: Route contract heuristic [high]')
    expect(index).not.toContain('stale]')
  })
})
