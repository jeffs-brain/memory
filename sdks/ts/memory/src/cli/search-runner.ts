/**
 * Ad-hoc hybrid search over a brain.
 *
 * Each `jbmem search` invocation builds an in-memory SQLite index from
 * every markdown/text file the brain exposes through its Store, then
 * delegates to the shared retrieval pipeline (`createRetrieval`) so the
 * CLI and the backend share the same BM25 + vector + RRF + rerank +
 * retry-ladder code path.
 *
 * For large brains this is obviously slow; the per-brain index should be
 * persisted once the worker pipeline in `apps/backend` lands.
 */

import type { Embedder } from '../llm/index.js'
import { createRetrieval } from '../retrieval/index.js'
import type { HybridMode } from '../retrieval/types.js'
import { createSearchIndex, type Chunk } from '../search/index.js'
import type { Store } from '../store/index.js'

export type SearchMode = 'hybrid' | 'bm25' | 'semantic'

export const isSearchMode = (v: string): v is SearchMode =>
  v === 'hybrid' || v === 'bm25' || v === 'semantic'

export type SearchHit = {
  readonly id: string
  readonly path: string
  readonly score: number
  readonly snippet: string
}

const MAX_SNIPPET = 240
const INDEXABLE_EXT = /\.(md|txt|markdown)$/i

type SearchDeps = {
  readonly store: Store
  readonly embedder?: Embedder
  readonly limit: number
  readonly mode: SearchMode
}

const TO_HYBRID_MODE: Record<SearchMode, HybridMode> = {
  hybrid: 'hybrid',
  bm25: 'bm25',
  semantic: 'semantic',
}

export const runSearch = async (
  query: string,
  deps: SearchDeps,
): Promise<readonly SearchHit[]> => {
  const docs = await collectDocuments(deps.store)
  if (docs.length === 0) return []

  // 1024 is the default; for tests we keep the same so synthetic embeds
  // and the index agree on dimension.
  const vectorDim = 1024
  const index = await createSearchIndex({ dbPath: ':memory:', vectorDim })
  try {
    const chunks: Chunk[] = []
    for (const doc of docs) {
      const content = doc.content.slice(0, 32_000)
      const base: Chunk = {
        id: doc.path,
        path: doc.path,
        ordinal: 0,
        title: doc.path,
        content,
      }
      if (deps.mode !== 'bm25' && deps.embedder !== undefined) {
        const [vec] = await deps.embedder.embed([content])
        if (vec !== undefined) {
          chunks.push({ ...base, embedding: Float32Array.from(vec) })
          continue
        }
      }
      chunks.push(base)
    }
    index.upsertChunks(chunks)

    const retrieval = createRetrieval({
      index,
      ...(deps.embedder !== undefined ? { embedder: deps.embedder } : {}),
    })

    const results = await retrieval.search({
      query,
      topK: deps.limit,
      mode: TO_HYBRID_MODE[deps.mode],
      // Rerank is a no-op without a reranker configured; keep it off to
      // avoid the skip-reason bookkeeping.
      rerank: false,
    })

    return results.map((r) => ({
      id: r.id,
      path: r.path,
      score: r.score,
      snippet: (r.content ?? '').slice(0, MAX_SNIPPET),
    }))
  } finally {
    await index.close()
  }
}

type BrainDoc = {
  readonly path: string
  readonly content: string
}

const collectDocuments = async (store: Store): Promise<readonly BrainDoc[]> => {
  const out: BrainDoc[] = []
  const entries = await store.list('', { recursive: true, includeGenerated: true })
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!INDEXABLE_EXT.test(entry.path)) continue
    try {
      const buf = await store.read(entry.path)
      out.push({ path: entry.path, content: buf.toString('utf8') })
    } catch {
      // ignore unreadable files; still want partial search.
    }
  }
  return out
}
