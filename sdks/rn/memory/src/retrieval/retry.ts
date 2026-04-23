import { isStopWord, normalise } from '../query/index.js'
import type { RetrievalFilters } from './types.js'

const TRIGRAM_JACCARD_THRESHOLD = 0.3

export type TrigramChunk = {
  readonly id: string
  readonly path: string
  readonly title?: string
  readonly summary?: string
  readonly content: string
  readonly tags?: readonly string[] | string
  readonly metadata?: Readonly<Record<string, unknown>>
}

export type TrigramHit = TrigramChunk & {
  readonly similarity: number
}

export type TrigramIndex = {
  search(
    tokens: readonly string[],
    limit: number,
    filters?: RetrievalFilters,
  ): readonly TrigramHit[]
}

export const sanitiseQuery = (query: string): string => {
  return query
    .replace(/[\p{P}\p{S}]+/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

const retryTokens = (query: string): string[] => {
  return sanitiseQuery(normalise(query))
    .toLocaleLowerCase('en')
    .split(/\s+/)
    .map((token) => token.trim())
    .filter((token) => token !== '')
}

export const strongestTerm = (query: string): string | undefined => {
  let best: string | undefined
  for (const token of retryTokens(query)) {
    if (token.length < 3 || isStopWord(token)) continue
    if (best === undefined || token.length > best.length) best = token
  }
  return best
}

export const queryTokens = (query: string): string[] => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const token of retryTokens(query)) {
    if (token.length < 3 || isStopWord(token) || seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

export const computeTrigrams = (text: string): Set<string> => {
  const out = new Set<string>()
  const cleaned = text
    .toLocaleLowerCase('en')
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
  if (cleaned === '') return out
  for (const word of cleaned.split(/\s+/)) {
    const padded = `$${word}$`
    if (padded.length < 3) continue
    for (let index = 0; index + 3 <= padded.length; index += 1) {
      out.add(padded.slice(index, index + 3))
    }
  }
  return out
}

export const slugTextFor = (path: string): string => {
  let output = path.toLocaleLowerCase('en')
  const slashIndex = output.lastIndexOf('/')
  if (slashIndex >= 0) output = output.slice(slashIndex + 1)
  if (output.endsWith('.md')) output = output.slice(0, -3)
  return output.replace(/[^\p{L}\p{N}]+/gu, ' ').trim()
}

export const buildTrigramIndex = (chunks: readonly TrigramChunk[]): TrigramIndex => {
  type IndexedChunk = TrigramChunk & {
    readonly grams: Set<string>
  }

  const byGram = new Map<string, IndexedChunk[]>()
  const entries: IndexedChunk[] = []
  const seen = new Set<string>()

  for (const chunk of chunks) {
    if (chunk.id === '' || seen.has(chunk.id)) continue
    seen.add(chunk.id)
    const grams = computeTrigrams(slugTextFor(chunk.path))
    const entry = { ...chunk, grams }
    entries.push(entry)
    for (const gram of grams) {
      const current = byGram.get(gram)
      if (current === undefined) {
        byGram.set(gram, [entry])
        continue
      }
      current.push(entry)
    }
  }

  return {
    search(tokens, limit, filters) {
      if (limit <= 0 || tokens.length === 0 || entries.length === 0) return []
      const best = new Map<string, TrigramHit>()
      for (const token of tokens) {
        const tokenGrams = computeTrigrams(token)
        if (tokenGrams.size === 0) continue
        const candidates = new Set<(typeof entries)[number]>()
        for (const gram of tokenGrams) {
          const bucket = byGram.get(gram)
          if (bucket === undefined) continue
          for (const entry of bucket) candidates.add(entry)
        }
        for (const entry of candidates) {
          if (!matchesFilters(entry, filters)) continue
          const similarity = jaccard(tokenGrams, entry.grams)
          if (similarity < TRIGRAM_JACCARD_THRESHOLD) continue
          const existing = best.get(entry.id)
          if (existing !== undefined && existing.similarity >= similarity) continue
          best.set(entry.id, {
            id: entry.id,
            path: entry.path,
            content: entry.content,
            ...(entry.title === undefined ? {} : { title: entry.title }),
            ...(entry.summary === undefined ? {} : { summary: entry.summary }),
            ...(entry.tags === undefined ? {} : { tags: entry.tags }),
            ...(entry.metadata === undefined ? {} : { metadata: entry.metadata }),
            similarity,
          })
        }
      }
      return [...best.values()]
        .sort((left, right) => {
          if (left.similarity !== right.similarity) return right.similarity - left.similarity
          return left.path.localeCompare(right.path)
        })
        .slice(0, limit)
    },
  }
}

const jaccard = (left: ReadonlySet<string>, right: ReadonlySet<string>): number => {
  if (left.size === 0 || right.size === 0) return 0
  let intersection = 0
  for (const value of left) {
    if (right.has(value)) intersection += 1
  }
  return intersection / (left.size + right.size - intersection)
}

const matchesFilters = (chunk: TrigramChunk, filters: RetrievalFilters | undefined): boolean => {
  if (filters === undefined) return true
  if (
    filters.paths !== undefined &&
    filters.paths.length > 0 &&
    !filters.paths.includes(chunk.path)
  ) {
    return false
  }
  if (filters.pathPrefix !== undefined && !chunk.path.startsWith(filters.pathPrefix)) {
    return false
  }
  const metadataTags = Array.isArray(chunk.metadata?.tags)
    ? chunk.metadata.tags.filter((value): value is string => typeof value === 'string')
    : typeof chunk.tags === 'string'
      ? chunk.tags.split(/\s+/)
      : (chunk.tags ?? [])
  if (filters.tags !== undefined && filters.tags.length > 0) {
    const tagSet = new Set(metadataTags)
    if (!filters.tags.every((tag) => tagSet.has(tag))) return false
  }
  if (filters.scope !== undefined) {
    const scope = typeof chunk.metadata?.scope === 'string' ? chunk.metadata.scope : undefined
    if (scope !== filters.scope) return false
  }
  return true
}
