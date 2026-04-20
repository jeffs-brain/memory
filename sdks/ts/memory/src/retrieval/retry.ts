// SPDX-License-Identifier: Apache-2.0

/**
 * Retry ladder for zero-hit queries. Ported from
 * apps/jeff/internal/knowledge/search.go:SearchWithOpts. Each rung
 * returns as soon as the index produces at least one hit; callers are
 * expected to stop walking the ladder when `hits > 0`.
 *
 * Rungs:
 *   1. Strongest-term only — drop stop-words, keep the longest token.
 *   2. Force-refresh FTS + rerun sanitised query. For our architecture
 *      this is a NO-OP because SQLite FTS5 under better-sqlite3 is a
 *      live view of the knowledge_chunks table: any uncommitted writes
 *      are already visible to readers inside the same connection, and
 *      WAL-mode readers pick up committed writes on the next statement.
 *      The rung is retained (as a pass-through) so the ladder shape
 *      and attempt-trace stay 1:1 with the Go reference for eval
 *      reports. See docs on `forceRefreshIndex` below.
 *   3. Refreshed sanitised — rerun the original query after the no-op
 *      refresh. This catches cases where a concurrent writer committed
 *      between rung 1 and rung 3 in the Go implementation; our port
 *      inherits the shape for symmetry.
 *   4. Refreshed strongest term — same as rung 1 but after the refresh.
 *   5. Trigram fuzzy fallback — Jaccard ≥ 0.3 over path slugs. Pure-JS
 *      trigram set implementation so we stay free of pg_trgm / SQLite
 *      extensions (which would differ between better-sqlite3 and a
 *      hypothetical Postgres adapter).
 */

const STOP_WORDS: ReadonlySet<string> = new Set([
  'the',
  'a',
  'an',
  'and',
  'or',
  'but',
  'is',
  'are',
  'was',
  'what',
  'who',
  'when',
  'where',
  'why',
  'how',
  'you',
  'for',
  'from',
  'about',
  'advice',
  'any',
  'been',
  'can',
  'choose',
  'current',
  'decide',
  'deciding',
  'feeling',
  'find',
  'getting',
  'help',
  'helpful',
  'idea',
  'ideas',
  'interesting',
  'ive',
  'look',
  'looking',
  'make',
  'making',
  'might',
  'need',
  'needs',
  'noticed',
  'planning',
  'recent',
  'recently',
  'recommend',
  'recommendation',
  'recommendations',
  'should',
  'some',
  'soon',
  'suggest',
  'suggestion',
  'suggestions',
  'sure',
  'thinking',
  'tips',
  'together',
  'trying',
  'upcoming',
  'useful',
  'want',
  'weekend',
  'with',
  'would',
  'again',
  'becoming',
  'bit',
  'combined',
  'having',
  'items',
  'keep',
  'keeping',
  'kind',
  'kinds',
  'lately',
  'many',
  'seen',
  'show',
  'tonight',
  'trouble',
  'type',
  'types',
  'watch',
  'have',
  'has',
  'had',
  'de',
  'het',
  'een',
  'en',
  'of',
])

const TRIGRAM_JACCARD_THRESHOLD = 0.3

/**
 * Returns the longest non-stop-word token of at least three characters
 * from the raw query. Undefined when nothing survives filtering.
 */
export function strongestTerm(query: string): string | undefined {
  const tokens = normaliseRetryTokens(query)
  let best: string | undefined
  for (const tok of tokens) {
    if (tok.length < 3) continue
    if (STOP_WORDS.has(tok)) continue
    if (best === undefined || tok.length > best.length) best = tok
  }
  return best
}

/**
 * Returns every non-stop-word token of at least three characters,
 * deduplicated and lowercased. Used by the trigram fallback so every
 * query token gets a chance to match a slug, not just the longest.
 */
export function queryTokens(query: string): string[] {
  const tokens = normaliseRetryTokens(query)
  const out: string[] = []
  const seen = new Set<string>()
  for (const tok of tokens) {
    if (tok.length < 3) continue
    if (STOP_WORDS.has(tok)) continue
    if (seen.has(tok)) continue
    seen.add(tok)
    out.push(tok)
  }
  return out
}

/**
 * Strips punctuation so the "refreshed sanitised" rung retries a
 * cleaner form of the original input. Mirrors the Go retry ladder's
 * third rung without requiring an actual index refresh.
 */
export function sanitiseQuery(query: string): string {
  return query
    .replace(/[\p{P}\p{S}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const normaliseRetryTokens = (query: string): string[] =>
  sanitiseQuery(query)
    .toLowerCase()
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token !== '')

/**
 * forceRefreshIndex is a no-op for our SQLite + WAL-backed architecture.
 * Documented explicitly so readers can reason about why the ladder
 * skips straight from rung 2 to rung 3 without touching the database.
 *
 * In the Go reference, `searchIndex.Update(ctx)` re-reads the FTS
 * metadata table to detect files written since the last index build.
 * Our TypeScript indexer is write-through: every `upsertChunk` commits
 * to FTS5 + the vector table in the same transaction, so there is
 * nothing to refresh on the reader side. Leaving the function here
 * (and calling it from the ladder) keeps the attempt shape identical
 * to Jeff's, which matters for eval reports and tooling.
 */
export function forceRefreshIndex(): void {
  /* no-op; see docstring above. */
}

/**
 * Boundary-padded 3-gram set for `text`, lowercased, with
 * non-alphanumerics squashed to spaces. Mirrors Jeff's
 * `computeTrigrams`.
 */
export function computeTrigrams(text: string): Set<string> {
  const out = new Set<string>()
  if (text === '') return out
  const cleaned = text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ')
  for (const word of cleaned.split(/\s+/).filter((w) => w !== '')) {
    const padded = `$${word}$`
    if (padded.length < 3) continue
    for (let i = 0; i + 3 <= padded.length; i++) {
      out.add(padded.slice(i, i + 3))
    }
  }
  return out
}

/**
 * Keeps only the filename stem so single-word queries match slugs
 * without being drowned out by parent-directory noise.
 */
export function slugTextFor(path: string): string {
  let s = path.toLowerCase()
  const slash = s.lastIndexOf('/')
  if (slash >= 0) s = s.slice(slash + 1)
  if (s.endsWith('.md')) s = s.slice(0, -3)
  return s.replace(/[^\p{L}\p{N}]+/gu, ' ').trim()
}

export type TrigramHit = {
  id: string
  path: string
  similarity: number
  title: string
  summary: string
  content: string
  tags?: readonly string[] | string
  metadata?: Readonly<Record<string, unknown>>
}

/**
 * TrigramIndex is the lazy slug-fuzzy index used by the retry ladder's
 * final rung. Build once per retrieval instance from chunk metadata;
 * queries are Jaccard-overlap lookups over the trigram sets.
 */
export type TrigramIndex = {
  search(tokens: readonly string[], limit: number): TrigramHit[]
}

export type TrigramSourceChunk = {
  id: string
  path: string
  title?: string
  summary?: string
  content?: string
  tags?: readonly string[] | string
  metadata?: Readonly<Record<string, unknown>>
}

export function buildTrigramIndex(chunks: readonly TrigramSourceChunk[]): TrigramIndex {
  type Entry = {
    id: string
    path: string
    grams: Set<string>
    title: string
    summary: string
    content: string
    tags?: readonly string[] | string
    metadata?: Readonly<Record<string, unknown>>
  }
  const entries: Entry[] = []
  const byGram = new Map<string, Entry[]>()
  const seen = new Set<string>()

  for (const c of chunks) {
    if (c.id === '' || seen.has(c.id)) continue
    seen.add(c.id)
    const grams = computeTrigrams(slugTextFor(c.path))
    const entry: Entry = {
      id: c.id,
      path: c.path,
      grams,
      title: c.title ?? '',
      summary: c.summary ?? '',
      content: c.content ?? '',
      ...(c.tags !== undefined ? { tags: c.tags } : {}),
      ...(c.metadata !== undefined ? { metadata: c.metadata } : {}),
    }
    entries.push(entry)
    for (const g of grams) {
      const list = byGram.get(g)
      if (list === undefined) byGram.set(g, [entry])
      else list.push(entry)
    }
  }

  return {
    search(tokens, limit) {
      if (tokens.length === 0 || entries.length === 0 || limit <= 0) return []
      const best = new Map<string, { entry: Entry; similarity: number }>()
      for (const tok of tokens) {
        const queryGrams = computeTrigrams(tok)
        if (queryGrams.size === 0) continue

        const candidates = new Set<Entry>()
        for (const g of queryGrams) {
          const bucket = byGram.get(g)
          if (bucket !== undefined) {
            for (const entry of bucket) candidates.add(entry)
          }
        }

        for (const entry of candidates) {
          if (entry.grams.size === 0) continue
          const sim = jaccard(queryGrams, entry.grams)
          if (sim < TRIGRAM_JACCARD_THRESHOLD) continue
          const prev = best.get(entry.id)
          if (prev === undefined || sim > prev.similarity) {
            best.set(entry.id, { entry, similarity: sim })
          }
        }
      }

      const hits: TrigramHit[] = []
      for (const { entry, similarity } of best.values()) {
        hits.push({
          id: entry.id,
          path: entry.path,
          similarity,
          title: entry.title,
          summary: entry.summary,
          content: entry.content,
          ...(entry.tags !== undefined ? { tags: entry.tags } : {}),
          ...(entry.metadata !== undefined ? { metadata: entry.metadata } : {}),
        })
      }
      hits.sort((a, b) => {
        if (a.similarity !== b.similarity) return b.similarity - a.similarity
        return a.path < b.path ? -1 : a.path > b.path ? 1 : 0
      })
      return hits.slice(0, limit)
    },
  }
}

function jaccard(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 || b.size === 0) return 0
  const [small, large] = a.size <= b.size ? [a, b] : [b, a]
  let intersection = 0
  for (const g of small) if (large.has(g)) intersection++
  const union = a.size + b.size - intersection
  return union === 0 ? 0 : intersection / union
}
