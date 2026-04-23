import { afterEach, describe, expect, it } from 'vitest'

import type { Provider, StructuredRequest } from '../llm/types.js'
import { createRetrieval } from '../retrieval/index.js'
import { type SearchIndex, createSearchIndex } from '../search/index.js'
import { type Store, createMobileStore, toPath } from '../store/index.js'
import { createBetterSqliteOpenDb } from '../testing/better-sqlite-driver.js'
import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { createMemoryClient } from './client.js'
import type { MemoryClient } from './types.js'

const resources: Array<{
  readonly client: MemoryClient
  readonly store: Store
  readonly searchIndex: SearchIndex
}> = []

const fakeEmbedder = {
  name: () => 'test-embedder',
  model: () => 'minilm-mobile',
  dimension: () => 2,
  embed: async (texts: readonly string[]) =>
    texts.map((text) => {
      const lowered = text.toLocaleLowerCase('en')
      if (lowered.includes('cycling')) return [1, 0]
      if (lowered.includes('coffee')) return [0, 1]
      return [0.5, 0.5]
    }),
}

const createProvider = (): Provider => ({
  name: () => 'test-provider',
  modelName: () => 'test-provider-model',
  supportsStructuredDecoding: () => true,
  complete: async () => {
    throw new Error('not used in tests')
  },
  structured: async (request: StructuredRequest) => {
    switch (request.taskType) {
      case 'memory-extract':
        return JSON.stringify({
          notes: [
            {
              filename: 'coffee-preference',
              name: 'Coffee preference',
              description: 'Prefers flat whites',
              type: 'user',
              content: 'Usually orders a flat white.',
              tags: ['drink', 'coffee'],
            },
          ],
        })
      case 'memory-reflect':
        return JSON.stringify({
          outcome: 'success',
          summary: 'Talked about coffee preferences.',
          retryFeedback: 'Capture specific drink choices and follow-up buying decisions.',
          shouldRecordEpisode: true,
          openQuestions: ['Which beans should we order next?'],
          heuristics: [
            {
              rule: 'Preserve concrete user preferences in the stored reflection.',
              context: 'memory reflection',
              confidence: 'medium',
              category: 'communication',
              scope: 'global',
              antiPattern: false,
            },
          ],
        })
      default:
        throw new Error(`unexpected task type: ${request.taskType ?? 'unknown'}`)
    }
  },
})

const freshClient = async (
  provider?: Provider,
): Promise<{
  readonly client: MemoryClient
  readonly store: Store
  readonly searchIndex: SearchIndex
}> => {
  const store = await createMobileStore({
    root: '/brains/demo',
    adapter: createMemoryFileAdapter(),
  })
  const searchIndex = await createSearchIndex({
    dbPath: ':memory:',
    openDb: createBetterSqliteOpenDb(),
    vectorDim: 2,
  })
  const retrieval = createRetrieval({
    index: searchIndex,
    embedder: fakeEmbedder,
  })
  const client = createMemoryClient({
    brainId: 'demo',
    store,
    searchIndex,
    retrieval,
    embedder: fakeEmbedder,
    ...(provider === undefined ? {} : { provider }),
  })
  const resource = { client, store, searchIndex }
  resources.push(resource)
  return resource
}

afterEach(async () => {
  while (resources.length > 0) {
    const resource = resources.pop()
    if (resource !== undefined) await resource.client.close()
  }
})

describe('createMemoryClient', () => {
  it('writes notes, rebuilds the scope index, recalls them, and removes them cleanly', async () => {
    const { client, store, searchIndex } = await freshClient()

    const note = await client.remember({
      filename: 'cycling',
      name: 'Cycling',
      description: 'Likes cycling on Sundays',
      content: 'Prefers road cycling and long weekend rides.',
      tags: ['sport', 'cycling'],
    })

    expect(note.path).toBe('memory/global/cycling.md')
    expect(await store.read(toPath('memory/global/cycling.md'))).toContain('name: Cycling')
    expect(await store.read(toPath('memory/global/MEMORY.md'))).toContain(
      '- cycling.md: Likes cycling on Sundays',
    )

    const hits = await client.recall({ query: 'cycling', topK: 1 })
    expect(hits[0]?.path).toBe('memory/global/cycling.md')
    expect(searchIndex.chunkIdsWithVectorForModel('minilm-mobile')).toEqual([
      'memory/global/cycling.md',
    ])

    await client.forget(toPath('memory/global/cycling.md'))

    expect(await client.listNotes()).toEqual([])
    expect(await store.read(toPath('memory/global/MEMORY.md'))).toBe('\n')
  })

  it('runs extract and reflect with a configured provider', async () => {
    const { client, store } = await freshClient(createProvider())

    const extract = await client.extract({
      messages: [{ role: 'user', content: 'I usually order a flat white.' }],
      sessionId: 'session-1',
      sessionDate: '2026-04-23',
    })
    expect(extract.skipped).toBe(false)
    expect(extract.created.map((note) => note.path)).toEqual(['memory/global/coffee-preference.md'])

    const reflection = await client.reflect({
      messages: [{ role: 'user', content: 'Let us review the coffee discussion.' }],
      sessionId: 'session-1',
    })

    expect(reflection?.summary).toBe('Talked about coffee preferences.')
    expect(reflection?.outcome).toBe('success')
    expect(reflection?.shouldRecordEpisode).toBe(true)
    expect(await store.read(toPath('reflections/session-1.md'))).toContain(
      'Which beans should we order next?',
    )
  })

  it('skips extraction gracefully when no provider is configured', async () => {
    const { client } = await freshClient()

    const result = await client.extract({
      messages: [{ role: 'user', content: 'remember this' }],
    })

    expect(result).toEqual({
      created: [],
      skipped: true,
      reason: 'no provider configured',
    })
  })
})
