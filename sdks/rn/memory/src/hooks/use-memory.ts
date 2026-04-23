import { startTransition, useEffect, useEffectEvent, useState } from 'react'

import type { Embedder, Provider } from '../llm/types.js'
import { createMemoryClient } from '../memory/client.js'
import type { Scope } from '../memory/paths.js'
import type { MemoryClient } from '../memory/types.js'
import { createRetrieval } from '../retrieval/index.js'
import { type OpenSqliteDb, createSearchIndex } from '../search/index.js'
import { createOpSqliteOpenDb } from '../search/op-sqlite-driver.js'
import { resolveExpoDocumentDirectory } from '../store/expo-file-adapter.js'
import { createExpoFileAdapter } from '../store/expo-file-adapter.js'
import type { FileAdapter } from '../store/index.js'
import { createMobileStore } from '../store/index.js'

export type UseMemoryConfig = {
  readonly brainId: string
  readonly brainRoot?: string
  readonly fileAdapter?: FileAdapter
  readonly openDb?: OpenSqliteDb
  readonly provider?: Provider
  readonly embedder?: Embedder
  readonly actorId?: string
  readonly defaultScope?: Scope
  readonly vectorDim?: number
  readonly rebuildIndexOnReady?: boolean
}

export const useMemory = (
  config: UseMemoryConfig,
): {
  readonly memory: MemoryClient | null
  readonly isReady: boolean
  readonly error: Error | null
} => {
  const [memory, setMemory] = useState<MemoryClient | null>(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const initialise = useEffectEvent(async (nextConfig: UseMemoryConfig): Promise<MemoryClient> => {
    const fileAdapter = nextConfig.fileAdapter ?? (await createExpoFileAdapter())
    const openDb = nextConfig.openDb ?? createOpSqliteOpenDb()
    const root =
      nextConfig.brainRoot ??
      fileAdapter.join(await resolveExpoDocumentDirectory(), `brains/${nextConfig.brainId}`)

    const store = await createMobileStore({ root, adapter: fileAdapter })
    const searchIndex = await createSearchIndex({
      dbPath: fileAdapter.join(root, '.search.db'),
      openDb,
      ...(nextConfig.vectorDim === undefined ? {} : { vectorDim: nextConfig.vectorDim }),
    })
    const retrieval = createRetrieval({
      index: searchIndex,
      ...(nextConfig.embedder === undefined ? {} : { embedder: nextConfig.embedder }),
    })
    const client = createMemoryClient({
      brainId: nextConfig.brainId,
      store,
      searchIndex,
      retrieval,
      ...(nextConfig.provider === undefined ? {} : { provider: nextConfig.provider }),
      ...(nextConfig.embedder === undefined ? {} : { embedder: nextConfig.embedder }),
      ...(nextConfig.defaultScope === undefined ? {} : { defaultScope: nextConfig.defaultScope }),
      ...(nextConfig.actorId === undefined ? {} : { defaultActorId: nextConfig.actorId }),
    })
    if (nextConfig.rebuildIndexOnReady !== false) {
      await client.rebuildIndex()
    }
    return client
  })

  useEffect(() => {
    const nextConfig: UseMemoryConfig = {
      brainId: config.brainId,
      ...(config.actorId === undefined ? {} : { actorId: config.actorId }),
      ...(config.brainRoot === undefined ? {} : { brainRoot: config.brainRoot }),
      ...(config.defaultScope === undefined ? {} : { defaultScope: config.defaultScope }),
      ...(config.embedder === undefined ? {} : { embedder: config.embedder }),
      ...(config.fileAdapter === undefined ? {} : { fileAdapter: config.fileAdapter }),
      ...(config.openDb === undefined ? {} : { openDb: config.openDb }),
      ...(config.provider === undefined ? {} : { provider: config.provider }),
      ...(config.rebuildIndexOnReady === undefined
        ? {}
        : { rebuildIndexOnReady: config.rebuildIndexOnReady }),
      ...(config.vectorDim === undefined ? {} : { vectorDim: config.vectorDim }),
    }
    let cancelled = false
    let current: MemoryClient | null = null

    void (async () => {
      setIsReady(false)
      setError(null)
      try {
        const client = await initialise(nextConfig)
        if (cancelled) {
          await client.close()
          return
        }
        current = client
        startTransition(() => {
          setMemory(client)
          setIsReady(true)
        })
      } catch (resolved) {
        const nextError = resolved instanceof Error ? resolved : new Error(String(resolved))
        if (cancelled) return
        startTransition(() => {
          setMemory(null)
          setIsReady(false)
          setError(nextError)
        })
      }
    })()

    return () => {
      cancelled = true
      setIsReady(false)
      startTransition(() => {
        setMemory(null)
      })
      if (current !== null) {
        void current.close()
      }
    }
  }, [
    config.actorId,
    config.brainId,
    config.brainRoot,
    config.defaultScope,
    config.embedder,
    config.fileAdapter,
    config.openDb,
    config.provider,
    config.rebuildIndexOnReady,
    config.vectorDim,
    initialise,
  ])

  return { memory, isReady, error }
}
