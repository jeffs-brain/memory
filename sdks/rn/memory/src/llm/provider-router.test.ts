import { describe, expect, it } from 'vitest'

import type { ConnectivityMonitor } from '../connectivity/monitor.js'
import { ProviderRouter } from './provider-router.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Provider,
  StructuredRequest,
} from './types.js'

const completion = (content: string): CompletionResponse => ({
  content,
  toolCalls: [],
  usage: {
    inputTokens: 1,
    outputTokens: 1,
  },
  stopReason: 'end_turn',
})

const createProvider = (name: string): Provider => ({
  name: () => name,
  modelName: () => `${name}-model`,
  supportsStructuredDecoding: () => true,
  complete: async (_request: CompletionRequest) => completion(name),
  structured: async (_request: StructuredRequest) => `{"provider":"${name}"}`,
})

const createEmbedder = (name: string): Embedder => ({
  name: () => name,
  model: () => `${name}-embedder`,
  dimension: () => 2,
  embed: async () => [[1, 0]],
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

describe('ProviderRouter', () => {
  it('routes to the local provider when offline in auto mode', async () => {
    const router = new ProviderRouter({
      localProvider: createProvider('local'),
      cloudProvider: createProvider('cloud'),
      strategy: 'auto',
      connectivity: createConnectivity(false),
    })

    const response = await router.complete({
      messages: [{ role: 'user', content: 'hello' }],
    })

    expect(response.content).toBe('local')
    expect(router.lastRoute()?.route).toBe('local')
    expect(router.lastRoute()?.reason).toBe('offline')
  })

  it('routes configured pipeline tasks to the cloud provider in auto mode', async () => {
    const router = new ProviderRouter({
      localProvider: createProvider('local'),
      cloudProvider: createProvider('cloud'),
      strategy: 'auto',
      connectivity: createConnectivity(true),
      autoConfig: {
        preferLocal: true,
        cloudTriggers: {
          taskTypes: ['memory-extract'],
        },
      },
    })

    const response = await router.structured({
      messages: [{ role: 'user', content: 'extract this' }],
      taskType: 'memory-extract',
      schema: '{"type":"object"}',
    })

    expect(response).toContain('"provider":"cloud"')
    expect(router.lastRoute()?.route).toBe('cloud')
    expect(router.lastRoute()?.reason).toBe('cloud-trigger')
  })

  it('falls back to the cloud embedder when no local embedder is configured', async () => {
    const router = new ProviderRouter({
      cloudEmbedder: createEmbedder('cloud'),
      strategy: 'auto',
      connectivity: createConnectivity(true),
    })

    const embeddings = await router.embed(['hello'])

    expect(embeddings).toEqual([[1, 0]])
    expect(router.lastRoute()?.kind).toBe('embedder')
    expect(router.lastRoute()?.route).toBe('cloud')
  })

  it('honours cloud-only embedder routing and keeps provider model identity separate', async () => {
    const router = new ProviderRouter({
      localProvider: createProvider('local-provider'),
      localEmbedder: createEmbedder('local'),
      cloudEmbedder: createEmbedder('cloud'),
      strategy: 'cloud-only',
      connectivity: createConnectivity(true),
    })

    expect(router.model()).toBe('cloud-embedder')
    expect(router.dimension()).toBe(2)
    expect(router.modelName()).toBe('local-provider-model')

    const embeddings = await router.embed(['hello'])

    expect(embeddings).toEqual([[1, 0]])
    expect(router.lastRoute()?.kind).toBe('embedder')
    expect(router.lastRoute()?.route).toBe('cloud')
    expect(router.lastRoute()?.reason).toBe('strategy-cloud-only')
    expect(router.model()).toBe('cloud-embedder')
    expect(router.modelName()).toBe('local-provider-model')
  })
})
