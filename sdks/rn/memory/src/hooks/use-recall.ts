import {
  startTransition,
  useDeferredValue,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from 'react'

import type { MemoryClient, RecallHit } from '../memory/types.js'
import type { RecallArgs } from '../memory/types.js'

export const useRecall = (
  memory: MemoryClient | null,
  query: string,
  opts: Omit<RecallArgs, 'query'> = {},
): {
  readonly results: readonly RecallHit[]
  readonly isLoading: boolean
  readonly error: Error | null
} => {
  const deferredQuery = useDeferredValue(query)
  const [results, setResults] = useState<readonly RecallHit[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const requestIdRef = useRef(0)

  const runRecall = useEffectEvent(
    async (
      currentMemory: MemoryClient | null,
      currentQuery: string,
      currentTopK: RecallArgs['topK'],
      currentScope: RecallArgs['scope'],
      currentActorId: RecallArgs['actorId'],
      requestId: number,
    ): Promise<void> => {
      if (currentMemory === null || currentQuery.trim() === '') {
        if (requestId !== requestIdRef.current) return
        startTransition(() => {
          setResults([])
          setIsLoading(false)
          setError(null)
        })
        return
      }

      startTransition(() => {
        setError(null)
        setIsLoading(true)
      })

      try {
        const next = await currentMemory.recall({
          query: currentQuery,
          ...(currentTopK === undefined ? {} : { topK: currentTopK }),
          ...(currentScope === undefined ? {} : { scope: currentScope }),
          ...(currentActorId === undefined ? {} : { actorId: currentActorId }),
        })
        if (requestId !== requestIdRef.current) return
        startTransition(() => {
          setResults(next)
          setIsLoading(false)
        })
      } catch (resolved) {
        const nextError = resolved instanceof Error ? resolved : new Error(String(resolved))
        if (requestId !== requestIdRef.current) return
        startTransition(() => {
          setError(nextError)
          setIsLoading(false)
        })
      }
    },
  )

  useEffect(() => {
    const requestId = requestIdRef.current + 1
    requestIdRef.current = requestId
    void runRecall(memory, deferredQuery, opts.topK, opts.scope, opts.actorId, requestId)
    return () => {
      if (requestIdRef.current === requestId) {
        requestIdRef.current += 1
      }
    }
  }, [memory, deferredQuery, opts.actorId, opts.scope, opts.topK, runRecall])

  return { results, isLoading, error }
}
