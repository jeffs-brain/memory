/**
 * Reciprocal Rank Fusion (RRF) — the fusion step that merges the BM25
 * and vector candidate lists into a single ranked slate.
 *
 *   rrf_score(doc) = sum over lists of 1 / (k + rank(doc))
 *
 * where `rank` is 1-indexed and `k` defaults to 60. Documents missing
 * from a list contribute 0 to that list's term. Cormack, Clarke,
 * Buettcher (SIGIR 2009) tested k in [10, 1000] and found k = 60 robust
 * across TREC tracks; it has been the industry default ever since.
 *
 * Title/summary preservation: the first list a document appears in
 * seeds the merged metadata. Subsequent lists fill in gaps when they
 * carry richer fields. In practice, BM25 candidates carry full title
 * and summary; vector-only hits can arrive with empty frontmatter.
 * Merging lets the richer record win without overwriting.
 */

import type { RetrievalResult } from './types.js'

export const RRF_DEFAULT_K = 60

/**
 * A single input candidate. Mirrors the shape emitted by BM25 and
 * vector search but leaves it to the caller to normalise upstream
 * fields (bm25Rank, vectorSimilarity). The `id` field is the fusion
 * key; use `chunk.id` from the search layer.
 */
export type RRFCandidate = {
  readonly id: string
  readonly path: string
  readonly title?: string
  readonly summary?: string
  readonly content?: string
  readonly bm25Rank?: number
  readonly vectorSimilarity?: number
}

/**
 * reciprocalRankFusion fuses an arbitrary number of ranked lists into a
 * single ranking. Order of lists is semantically significant only for
 * metadata preservation: the first list wins title/summary. The fused
 * score is deterministic and independent of list order.
 *
 * Ties are broken by path ascending so the output is stable across runs
 * with identical inputs.
 */
export function reciprocalRankFusion(
  lists: ReadonlyArray<readonly RRFCandidate[]>,
  k: number = RRF_DEFAULT_K,
): RetrievalResult[] {
  const safeK = k > 0 ? k : RRF_DEFAULT_K

  type Bucket = {
    id: string
    path: string
    title: string
    summary: string
    content: string
    bm25Rank?: number
    vectorSimilarity?: number
    score: number
  }

  const buckets = new Map<string, Bucket>()

  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const c = list[rank]
      if (c === undefined) continue
      const existing = buckets.get(c.id)
      if (existing === undefined) {
        const next: Bucket = {
          id: c.id,
          path: c.path,
          title: c.title ?? '',
          summary: c.summary ?? '',
          content: c.content ?? '',
          score: 1 / (safeK + rank + 1),
        }
        if (c.bm25Rank !== undefined) next.bm25Rank = c.bm25Rank
        if (c.vectorSimilarity !== undefined) next.vectorSimilarity = c.vectorSimilarity
        buckets.set(c.id, next)
        continue
      }

      // Preserve richer metadata from later lists only when the first
      // list left the field empty. BM25 carries full title+summary;
      // vector-only hits may arrive with empty strings, so this merge
      // is effectively a one-way fill-in.
      if (existing.title === '' && (c.title ?? '') !== '') existing.title = c.title ?? ''
      if (existing.summary === '' && (c.summary ?? '') !== '') existing.summary = c.summary ?? ''
      if (existing.content === '' && (c.content ?? '') !== '') existing.content = c.content ?? ''
      if (existing.bm25Rank === undefined && c.bm25Rank !== undefined) {
        existing.bm25Rank = c.bm25Rank
      }
      if (existing.vectorSimilarity === undefined && c.vectorSimilarity !== undefined) {
        existing.vectorSimilarity = c.vectorSimilarity
      }
      existing.score += 1 / (safeK + rank + 1)
    }
  }

  const fused: RetrievalResult[] = []
  for (const b of buckets.values()) {
    const entry: RetrievalResult = {
      id: b.id,
      path: b.path,
      title: b.title,
      summary: b.summary,
      content: b.content,
      score: b.score,
      ...(b.bm25Rank !== undefined ? { bm25Rank: b.bm25Rank } : {}),
      ...(b.vectorSimilarity !== undefined ? { vectorSimilarity: b.vectorSimilarity } : {}),
    }
    fused.push(entry)
  }

  fused.sort((a, b) => {
    if (a.score !== b.score) return b.score - a.score
    return a.path < b.path ? -1 : a.path > b.path ? 1 : 0
  })

  return fused
}
