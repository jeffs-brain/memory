// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('../brain.js', () => ({
  openBrain: vi.fn(),
}))

vi.mock('../config.js', () => {
  class MockCliUsageError extends Error {}
  return {
    CliUsageError: MockCliUsageError,
    resolveBrainDir: vi.fn(),
    embedderFromEnv: vi.fn(),
    buildEmbedder: vi.fn(),
    providerFromEnvOptional: vi.fn(),
    buildProvider: vi.fn(),
    rerankerFromEnv: vi.fn(),
    buildReranker: vi.fn(),
  }
})

vi.mock('../search-runner.js', () => {
  return {
    isSearchMode: (v: string): boolean =>
      v === 'hybrid' || v === 'hybrid-rerank' || v === 'bm25' || v === 'semantic',
    runSearch: vi.fn(),
  }
})

import { openBrain } from '../brain.js'
import {
  buildEmbedder,
  buildProvider,
  buildReranker,
  embedderFromEnv,
  providerFromEnvOptional,
  rerankerFromEnv,
  resolveBrainDir,
} from '../config.js'
import { runSearch } from '../search-runner.js'
import { searchCommand } from './search.js'

const mocked = {
  openBrain: openBrain as ReturnType<typeof vi.fn>,
  resolveBrainDir: resolveBrainDir as ReturnType<typeof vi.fn>,
  embedderFromEnv: embedderFromEnv as ReturnType<typeof vi.fn>,
  buildEmbedder: buildEmbedder as ReturnType<typeof vi.fn>,
  providerFromEnvOptional: providerFromEnvOptional as ReturnType<typeof vi.fn>,
  buildProvider: buildProvider as ReturnType<typeof vi.fn>,
  rerankerFromEnv: rerankerFromEnv as ReturnType<typeof vi.fn>,
  buildReranker: buildReranker as ReturnType<typeof vi.fn>,
  runSearch: runSearch as ReturnType<typeof vi.fn>,
}

describe('search command', () => {
  beforeEach(() => {
    mocked.openBrain.mockReset()
    mocked.resolveBrainDir.mockReset()
    mocked.embedderFromEnv.mockReset()
    mocked.buildEmbedder.mockReset()
    mocked.providerFromEnvOptional.mockReset()
    mocked.buildProvider.mockReset()
    mocked.rerankerFromEnv.mockReset()
    mocked.buildReranker.mockReset()
    mocked.runSearch.mockReset()
    mocked.resolveBrainDir.mockReturnValue('/tmp/brain')
    mocked.embedderFromEnv.mockReturnValue(undefined)
    mocked.providerFromEnvOptional.mockReturnValue(undefined)
    mocked.rerankerFromEnv.mockReturnValue(undefined)
    mocked.runSearch.mockResolvedValue([])
    vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('forwards --rerank and the configured reranker to runSearch', async () => {
    const store = { close: vi.fn().mockResolvedValue(undefined) }
    const rerankerSettings = {
      kind: 'http',
      baseURL: 'http://rerank.local',
      label: 'http-rerank',
      batchSize: 8,
      parallelism: 2,
      concurrencyCap: 4,
      preferHttp: true,
    } as const
    const reranker = {
      name: () => 'http-rerank',
      rerank: vi.fn(async () => []),
    }

    mocked.openBrain.mockResolvedValue(store)
    mocked.rerankerFromEnv.mockReturnValue(rerankerSettings)
    mocked.buildReranker.mockReturnValue(reranker)

    await searchCommand.run?.({
      args: {
        query: 'hotel',
        mode: 'hybrid',
        rerank: true,
        json: true,
        limit: '5',
      },
    })

    expect(mocked.buildReranker).toHaveBeenCalledWith(rerankerSettings, {})
    expect(mocked.runSearch).toHaveBeenCalledWith(
      'hotel',
      expect.objectContaining({
        store,
        limit: 5,
        mode: 'hybrid',
        rerank: true,
        reranker,
      }),
    )
    expect(store.close).toHaveBeenCalledTimes(1)
  })

  it('treats hybrid-rerank mode as an explicit rerank request', async () => {
    const store = { close: vi.fn().mockResolvedValue(undefined) }
    mocked.openBrain.mockResolvedValue(store)

    await searchCommand.run?.({
      args: {
        query: 'hotel',
        mode: 'hybrid-rerank',
        rerank: false,
        json: true,
        limit: '5',
      },
    })

    expect(mocked.runSearch).toHaveBeenCalledWith(
      'hotel',
      expect.objectContaining({
        store,
        limit: 5,
        mode: 'hybrid-rerank',
        rerank: true,
      }),
    )
    expect(store.close).toHaveBeenCalledTimes(1)
  })
})
