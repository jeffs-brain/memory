import { type Path, type Store, pathUnder, toPath } from '../store/index.js'
import { parseFrontmatter } from './frontmatter.js'
import { hashContent } from './hash.js'
import type { DedupReport, DedupSuggestion } from './types.js'

const DRAFTS_PREFIX = 'drafts'
const WIKI_PREFIX = 'wiki'
const DEFAULT_TITLE_THRESHOLD = 0.85

type DedupDeps = {
  store: Store
}

export type DedupOptions = {
  titleThreshold?: number
  scopes?: readonly ('drafts' | 'wiki')[]
}

type IndexedArticle = {
  path: Path
  titleKey: string
  titleTrigrams: Set<string>
  bodyHash: string
}

export const createDedup = (deps: DedupDeps) => {
  const { store } = deps

  return async (opts: DedupOptions = {}): Promise<DedupReport> => {
    const threshold = opts.titleThreshold ?? DEFAULT_TITLE_THRESHOLD
    const scopes = opts.scopes ?? ['drafts', 'wiki']
    const prefixes = scopes.map((scope) => (scope === 'drafts' ? DRAFTS_PREFIX : WIKI_PREFIX))

    const articles: IndexedArticle[] = []
    for (const prefix of prefixes) {
      for (const article of await collect(store, prefix)) {
        articles.push(article)
      }
    }

    const suggestions: DedupSuggestion[] = []
    const consumed = new Set<Path>()

    for (let leftIndex = 0; leftIndex < articles.length; leftIndex += 1) {
      const left = articles[leftIndex]
      if (left === undefined || consumed.has(left.path)) continue
      const merges: Path[] = []
      let reason: 'title' | 'content_hash' | 'both' = 'title'
      let score = 0

      for (let rightIndex = leftIndex + 1; rightIndex < articles.length; rightIndex += 1) {
        const right = articles[rightIndex]
        if (right === undefined || consumed.has(right.path)) continue

        const hashMatch = left.bodyHash === right.bodyHash
        const titleExact = left.titleKey !== '' && left.titleKey === right.titleKey
        const titleScore = jaccard(left.titleTrigrams, right.titleTrigrams)
        const titleNear = titleScore >= threshold

        if (!hashMatch && !titleExact && !titleNear) continue

        merges.push(right.path)
        consumed.add(right.path)
        if (hashMatch && (titleExact || titleNear)) reason = 'both'
        else if (hashMatch) reason = reason === 'title' ? 'content_hash' : reason
        score = Math.max(score, hashMatch ? 1 : titleExact ? 1 : titleScore)
      }

      if (merges.length === 0) continue
      consumed.add(left.path)
      suggestions.push({ keep: left.path, merge: merges, score, reason })
    }

    return { suggestions }
  }
}

const collect = async (store: Store, prefix: string): Promise<readonly IndexedArticle[]> => {
  const root = toPath(prefix)
  const exists = await store.exists(root).catch(() => false)
  if (!exists) return []

  const entries = await store.list(root, { recursive: true })
  const output: IndexedArticle[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!pathUnder(entry.path, prefix, true)) continue
    if (!entry.path.endsWith('.md')) continue
    if (entry.path.slice(prefix.length + 1).startsWith('_')) continue

    const content = await store.read(entry.path)
    const { frontmatter, body } = parseFrontmatter(content)
    const titleKey = normaliseTitle(frontmatter.title)
    output.push({
      path: entry.path,
      titleKey,
      titleTrigrams: trigrams(titleKey),
      bodyHash: hashContent(body.trim()),
    })
  }

  return output
}

const normaliseTitle = (title: string): string => {
  const source = title.trim().toLowerCase()
  if (source === '') return ''
  let output = ''
  let previousSpace = false
  for (const rune of source) {
    const isAlphanumeric = /[\p{L}\p{N}]/u.test(rune)
    if (isAlphanumeric) {
      output += rune
      previousSpace = false
      continue
    }
    if (!previousSpace && output.length > 0) {
      output += ' '
      previousSpace = true
    }
  }
  return output.trim()
}

const trigrams = (value: string): Set<string> => {
  const output = new Set<string>()
  if (value.length < 3) {
    if (value !== '') output.add(value)
    return output
  }
  for (let index = 0; index <= value.length - 3; index += 1) {
    output.add(value.slice(index, index + 3))
  }
  return output
}

const jaccard = (left: Set<string>, right: Set<string>): number => {
  if (left.size === 0 && right.size === 0) return 0
  let intersection = 0
  for (const token of left) {
    if (right.has(token)) intersection += 1
  }
  const union = left.size + right.size - intersection
  return union === 0 ? 0 : intersection / union
}
