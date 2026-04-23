import { describe, expect, it } from 'vitest'

import type {
  CompletionRequest,
  CompletionResponse,
  Provider,
  StructuredRequest,
} from '../llm/types.js'
import { createMobileStore, toPath } from '../store/index.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { createConsolidate } from './consolidate.js'
import { type Frontmatter, buildFrontmatter } from './frontmatter.js'

const freshStore = async () =>
  await createMobileStore({
    root: '/brains/consolidate',
    adapter: createMemoryFileAdapter(),
  })

const provider = (verdict: string): Provider => ({
  name: () => 'stub',
  modelName: () => 'stub-model',
  supportsStructuredDecoding: () => false,
  complete: async (_request: CompletionRequest): Promise<CompletionResponse> => ({
    content: JSON.stringify({ verdict, reason: 'overlap' }),
    toolCalls: [],
    usage: { inputTokens: 0, outputTokens: 0 },
    stopReason: 'end_turn',
  }),
  structured: async (_request: StructuredRequest) => JSON.stringify({ verdict, reason: 'overlap' }),
})

const writeNote = async (
  store: Awaited<ReturnType<typeof freshStore>>,
  path: string,
  body: string,
  frontmatter: Partial<Frontmatter> = {},
) => {
  const builtFrontmatter: Frontmatter = {
    name: frontmatter.name ?? path.split('/').pop() ?? path,
    type: frontmatter.type ?? 'project',
    scope: frontmatter.scope ?? 'project',
    modified: frontmatter.modified ?? new Date().toISOString(),
    extra: frontmatter.extra ?? {},
    ...(frontmatter.description === undefined ? {} : { description: frontmatter.description }),
    ...(frontmatter.created === undefined ? {} : { created: frontmatter.created }),
    ...(frontmatter.tags === undefined ? {} : { tags: frontmatter.tags }),
  }
  await store.write(toPath(path), `${buildFrontmatter(builtFrontmatter)}\n${body}\n`)
}

describe('createConsolidate', () => {
  it('merges duplicate notes and rebuilds the scope index', async () => {
    const store = await freshStore()
    await writeNote(store, 'memory/project/tenant-a/auth-notes.md', 'Uses OIDC via Lleverage.', {
      description: 'Primary auth implementation note',
      modified: '2026-04-18T09:00:00.000Z',
    })
    await writeNote(
      store,
      'memory/project/tenant-a/auth-extra.md',
      'Refresh token rotation enabled.',
      {
        description: 'Additional auth operational detail',
        modified: '2026-04-19T09:00:00.000Z',
      },
    )
    await store.write(
      toPath('memory/project/tenant-a/MEMORY.md'),
      '- stale-entry.md: out of date\n',
    )

    const consolidate = createConsolidate({
      store,
      provider: provider('merge'),
      logger: { debug: () => {}, info: () => {}, warn: () => {}, error: () => {} },
      defaultScope: 'project',
      defaultActorId: 'tenant-a',
    })

    const report = await consolidate()
    expect(report.merged).toBe(1)
    expect(report.ops.some((op) => op.kind === 'merge')).toBe(true)

    const donorStillExists = await store.exists(toPath('memory/project/tenant-a/auth-notes.md'))
    const keeper = await store.read(toPath('memory/project/tenant-a/auth-extra.md'))
    const index = await store.read(toPath('memory/project/tenant-a/MEMORY.md'))

    expect(donorStillExists).toBe(false)
    expect(keeper).toContain('Refresh token rotation enabled.')
    expect(keeper).toContain('Uses OIDC via Lleverage.')
    expect(index).toContain('auth-extra.md: Additional auth operational detail')
    expect(index).not.toContain('stale-entry.md')
  })
})
