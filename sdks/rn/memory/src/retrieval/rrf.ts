import type { RetrievalResult } from './types.js'

export const RRF_DEFAULT_K = 60

export type RrfCandidate = {
  readonly id: string
  readonly path: string
  readonly title?: string
  readonly summary?: string
  readonly content?: string
  readonly metadata?: Record<string, unknown>
  readonly bm25Rank?: number
  readonly vectorSimilarity?: number
}

export const reciprocalRankFusion = (
  lists: ReadonlyArray<readonly RrfCandidate[]>,
  k: number = RRF_DEFAULT_K,
): RetrievalResult[] => {
  const safeK = k > 0 ? k : RRF_DEFAULT_K

  type Bucket = {
    id: string
    path: string
    title: string
    summary: string
    content: string
    metadata?: Record<string, unknown>
    bm25Rank?: number
    vectorSimilarity?: number
    score: number
  }

  const buckets = new Map<string, Bucket>()

  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank += 1) {
      const candidate = list[rank]
      if (candidate === undefined) continue
      const existing = buckets.get(candidate.id)
      if (existing === undefined) {
        buckets.set(candidate.id, {
          id: candidate.id,
          path: candidate.path,
          title: candidate.title ?? '',
          summary: candidate.summary ?? '',
          content: candidate.content ?? '',
          ...(candidate.metadata === undefined ? {} : { metadata: candidate.metadata }),
          ...(candidate.bm25Rank === undefined ? {} : { bm25Rank: candidate.bm25Rank }),
          ...(candidate.vectorSimilarity === undefined
            ? {}
            : { vectorSimilarity: candidate.vectorSimilarity }),
          score: 1 / (safeK + rank + 1),
        })
        continue
      }

      if (existing.title === '' && (candidate.title ?? '') !== '')
        existing.title = candidate.title ?? ''
      if (existing.summary === '' && (candidate.summary ?? '') !== '') {
        existing.summary = candidate.summary ?? ''
      }
      if (existing.content === '' && (candidate.content ?? '') !== '') {
        existing.content = candidate.content ?? ''
      }
      if (existing.metadata === undefined && candidate.metadata !== undefined) {
        existing.metadata = candidate.metadata
      }
      if (existing.bm25Rank === undefined && candidate.bm25Rank !== undefined) {
        existing.bm25Rank = candidate.bm25Rank
      }
      if (existing.vectorSimilarity === undefined && candidate.vectorSimilarity !== undefined) {
        existing.vectorSimilarity = candidate.vectorSimilarity
      }
      existing.score += 1 / (safeK + rank + 1)
    }
  }

  const fused = [...buckets.values()].map<RetrievalResult>((bucket) => ({
    id: bucket.id,
    path: bucket.path,
    title: bucket.title,
    summary: bucket.summary,
    content: bucket.content,
    ...(bucket.metadata === undefined ? {} : { metadata: bucket.metadata }),
    score: bucket.score,
    ...(bucket.bm25Rank === undefined ? {} : { bm25Rank: bucket.bm25Rank }),
    ...(bucket.vectorSimilarity === undefined ? {} : { vectorSimilarity: bucket.vectorSimilarity }),
  }))

  fused.sort((left, right) => {
    if (left.score !== right.score) return right.score - left.score
    return left.path.localeCompare(right.path)
  })

  return fused
}
