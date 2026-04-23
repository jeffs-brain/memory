import { afterEach, describe, expect, it } from 'vitest'

import type { ConnectivityMonitor } from '../connectivity/monitor.js'
import { ProviderRouter } from '../llm/provider-router.js'
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
    expect(request.model).toBeUndefined()
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

const createConnectivity = (online: boolean): ConnectivityMonitor => ({
  snapshot: () => ({
    online,
    reachable: online,
    changedAt: new Date('2026-04-23T00:00:00.000Z'),
  }),
  refresh: async () => ({
    online,
    reachable: online,
    changedAt: new Date('2026-04-23T00:00:00.000Z'),
  }),
  subscribe: () => () => {},
  close: async () => {},
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

  it('routes extraction through the cloud provider without leaking a local model override', async () => {
    const localProvider: Provider = {
      name: () => 'local',
      modelName: () => 'local-model',
      supportsStructuredDecoding: () => true,
      complete: async () => {
        throw new Error('not used in tests')
      },
      structured: async () => {
        throw new Error('should not route to the local provider')
      },
    }
    const cloudProvider: Provider = {
      name: () => 'cloud',
      modelName: () => 'cloud-model',
      supportsStructuredDecoding: () => true,
      complete: async () => {
        throw new Error('not used in tests')
      },
      structured: async (request: StructuredRequest) => {
        expect(request.model).toBeUndefined()
        return JSON.stringify({
          notes: [
            {
              filename: 'cloud-note',
              name: 'Cloud note',
              description: 'Created via the routed cloud provider',
              type: 'reference',
              content: 'This extract path used the cloud provider.',
            },
          ],
        })
      },
    }

    const provider = new ProviderRouter({
      localProvider,
      cloudProvider,
      strategy: 'auto',
      connectivity: createConnectivity(true),
      autoConfig: {
        preferLocal: true,
        cloudTriggers: {
          taskTypes: ['memory-extract'],
        },
      },
    })

    const { client } = await freshClient(provider)
    const result = await client.extract({
      messages: [{ role: 'user', content: 'Please remember this routed extract.' }],
    })

    expect(result.created.map((note) => note.path)).toEqual(['memory/global/cloud-note.md'])
    expect(provider.lastRoute()).toMatchObject({
      kind: 'provider',
      route: 'cloud',
      reason: 'cloud-trigger',
    })
  })

  it('rebuilds the generated scope index once for multi-note extraction', async () => {
    const store = await createMobileStore({
      root: '/brains/batched',
      adapter: createMemoryFileAdapter(),
    })
    let generatedIndexWrites = 0
    const originalWrite = store.write.bind(store)
    store.write = async (path, content) => {
      if (path === toPath('memory/global/MEMORY.md')) {
        generatedIndexWrites += 1
      }
      await originalWrite(path, content)
    }

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
      brainId: 'batched',
      store,
      searchIndex,
      retrieval,
      embedder: fakeEmbedder,
      provider: {
        name: () => 'batch-provider',
        modelName: () => 'batch-provider-model',
        supportsStructuredDecoding: () => true,
        complete: async () => {
          throw new Error('not used in tests')
        },
        structured: async () =>
          JSON.stringify({
            notes: [
              {
                filename: 'first-note',
                name: 'First note',
                description: 'The first extracted note',
                type: 'user',
                content: 'One',
              },
              {
                filename: 'second-note',
                name: 'Second note',
                description: 'The second extracted note',
                type: 'reference',
                content: 'Two',
              },
            ],
          }),
      },
    })

    resources.push({ client, store, searchIndex })

    const result = await client.extract({
      messages: [{ role: 'user', content: 'Capture both notes.' }],
    })

    expect(result.created).toHaveLength(2)
    expect(generatedIndexWrites).toBe(1)
    expect(await store.read(toPath('memory/global/MEMORY.md'))).toContain(
      '- first-note.md: The first extracted note',
    )
    expect(await store.read(toPath('memory/global/MEMORY.md'))).toContain(
      '- second-note.md: The second extracted note',
    )
  })

  it('supports previewExtract without persisting notes or advancing the cursor', async () => {
    let extractCalls = 0
    const provider: Provider = {
      name: () => 'preview-provider',
      modelName: () => 'preview-provider-model',
      supportsStructuredDecoding: () => true,
      complete: async () => {
        throw new Error('not used in tests')
      },
      structured: async (request: StructuredRequest) => {
        if (request.taskType !== 'memory-extract') {
          throw new Error(`unexpected task type: ${request.taskType ?? 'unknown'}`)
        }
        extractCalls += 1
        return JSON.stringify({
          notes: [
            {
              filename: 'preview-note',
              name: 'Preview note',
              description: 'Previewed note',
              type: 'user',
              content: 'Preview content.',
            },
          ],
        })
      },
    }

    const { client, store } = await freshClient(provider)
    const messages = [{ role: 'user', content: 'Please remember the previewed note.' }] as const

    const preview = await client.previewExtract({
      messages,
      sessionId: 'preview-session',
    })

    expect(preview).toHaveLength(1)
    expect(preview[0]).toMatchObject({
      filename: 'preview-note.md',
      indexEntry: '- preview-note.md: Previewed note',
      scope: 'global',
    })
    expect(await store.exists(toPath('memory/global/preview-note.md'))).toBe(false)

    const extract = await client.extract({
      messages,
      sessionId: 'preview-session',
    })

    expect(extract.created.map((note) => note.path)).toEqual(['memory/global/preview-note.md'])
    expect(extractCalls).toBe(2)
  })

  it('tracks extraction progress with a session cursor', async () => {
    let extractCalls = 0
    const store = await createMobileStore({
      root: '/brains/cursor',
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
      brainId: 'cursor',
      store,
      searchIndex,
      retrieval,
      embedder: fakeEmbedder,
      extractMinMessages: 2,
      provider: {
        name: () => 'cursor-provider',
        modelName: () => 'cursor-provider-model',
        supportsStructuredDecoding: () => true,
        complete: async () => {
          throw new Error('not used in tests')
        },
        structured: async (request: StructuredRequest) => {
          if (request.taskType !== 'memory-extract') {
            throw new Error(`unexpected task type: ${request.taskType ?? 'unknown'}`)
          }
          extractCalls += 1
          return JSON.stringify({
            notes: [
              {
                filename: 'cursor-note',
                name: 'Cursor note',
                description: 'Created after enough unseen messages',
                type: 'user',
                content: 'Cursor-backed extraction persisted this note.',
              },
            ],
          })
        },
      },
    })

    resources.push({ client, store, searchIndex })

    const first = await client.extract({
      messages: [{ role: 'user', content: 'First message only.' }],
      sessionId: 'cursor-session',
    })
    expect(first).toEqual({
      created: [],
      skipped: true,
      reason: 'extract threshold not reached',
    })

    const secondMessages = [
      { role: 'user', content: 'First message only.' },
      { role: 'assistant', content: 'Second message arrives.' },
    ] as const
    const second = await client.extract({
      messages: secondMessages,
      sessionId: 'cursor-session',
    })
    expect(second.created.map((note) => note.path)).toEqual(['memory/global/cursor-note.md'])

    const third = await client.extract({
      messages: secondMessages,
      sessionId: 'cursor-session',
    })
    expect(third).toEqual({
      created: [],
      skipped: true,
      reason: 'no new messages to extract',
    })
    expect(extractCalls).toBe(1)
  })

  it('supports richer recall and contextualise args with scope fallbacks', async () => {
    const provider: Provider = {
      name: () => 'selector-provider',
      modelName: () => 'selector-provider-model',
      supportsStructuredDecoding: () => true,
      complete: async () => {
        throw new Error('not used in tests')
      },
      structured: async (request: StructuredRequest) => {
        if (request.taskType !== 'memory-recall-selector') {
          throw new Error(`unexpected task type: ${request.taskType ?? 'unknown'}`)
        }
        return JSON.stringify({
          selected: ['memory/global/working-style.md'],
        })
      },
    }

    const { client } = await freshClient(provider)
    await client.remember({
      filename: 'working-style',
      name: 'Working style',
      description: 'Plans work carefully before editing',
      content: 'Prefers to plan work carefully before editing.',
    })
    await client.remember({
      filename: 'project-plan',
      name: 'Project plan',
      description: 'Repo-specific planning note',
      content: 'Use the repo plan when changing auth behaviour.',
      scope: 'project',
      actorId: 'repo',
    })

    const hits = await client.recall({
      query: 'plan work',
      scope: 'project',
      actorId: 'repo',
      fallbackScopes: ['global'],
      selector: 'auto',
      k: 1,
    })
    expect(hits.map((hit) => hit.path)).toEqual(['memory/global/working-style.md'])

    const context = await client.contextualise({
      message: 'How should I plan work?',
      scope: 'project',
      actorId: 'repo',
      topK: 1,
      excludedPaths: [toPath('memory/project/repo/project-plan.md')],
    })
    expect(context.userMessage).toBe('How should I plan work?')
    expect(context.memories[0]?.path).toBe('memory/global/working-style.md')
    expect(context.systemReminder).toContain('<system-reminder>')
    expect(context.systemReminder).toContain('Global memory: working-style.md')
  })

  it('exposes store subscriptions on the memory client', async () => {
    const { client } = await freshClient()
    const events: string[] = []

    const handle = client.subscribe((event) => {
      events.push(`${event.kind}:${event.path}`)
    })

    await client.remember({
      filename: 'subscribed-note',
      name: 'Subscribed note',
      description: 'Triggers store events',
      content: 'This note should surface through the subscription.',
    })

    const eventCountBeforeUnsubscribe = events.length
    client.unsubscribe(handle)

    await client.remember({
      filename: 'after-unsubscribe',
      name: 'After unsubscribe',
      description: 'Should not be observed',
      content: 'No more store events should be captured.',
    })

    expect(events.some((event) => event.includes('memory/global/subscribed-note.md'))).toBe(true)
    expect(events).toHaveLength(eventCountBeforeUnsubscribe)
  })
})
