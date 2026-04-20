// SPDX-License-Identifier: Apache-2.0

/**
 * Article dedup. Suggests merges where two articles share either a
 * normalised title or a content-body hash, or whose titles are highly
 * similar (trigram Jaccard). Pure — returns suggestions, never mutates.
 * Ported from apps/jeff/internal/knowledge/dedup.go (structural half).
 */

import { createHash } from 'node:crypto'
import { type Path, type Store, pathUnder, toPath } from '../store/index.js'
import { parseFrontmatter } from './frontmatter.js'
import type { DedupReport, DedupSuggestion } from './types.js'

const DRAFTS_PREFIX = 'drafts'
const WIKI_PREFIX = 'wiki'

const DEFAULT_TITLE_THRESHOLD = 0.85

type DedupDeps = {
  store: Store
}

export type DedupOptions = {
  /** Jaccard similarity above which titles are considered near-duplicate. */
  titleThreshold?: number
  /** Restrict to a single prefix. Defaults to both drafts + wiki. */
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
    const prefixes = scopes.map((s) => (s === 'drafts' ? DRAFTS_PREFIX : WIKI_PREFIX))

    const articles: IndexedArticle[] = []
    for (const prefix of prefixes) {
      for (const a of await collect(store, prefix)) {
        articles.push(a)
      }
    }

    const suggestions: DedupSuggestion[] = []
    const consumed = new Set<Path>()

    for (let i = 0; i < articles.length; i++) {
      const a = articles[i]
      if (!a || consumed.has(a.path)) continue
      const merges: Path[] = []
      let reason: 'title' | 'content_hash' | 'both' = 'title'
      let score = 0
      for (let j = i + 1; j < articles.length; j++) {
        const b = articles[j]
        if (!b || consumed.has(b.path)) continue
        const hashMatch = a.bodyHash === b.bodyHash
        const titleExact = a.titleKey !== '' && a.titleKey === b.titleKey
        const titleScore = jaccard(a.titleTrigrams, b.titleTrigrams)
        const titleNear = titleScore >= threshold

        if (hashMatch || titleExact || titleNear) {
          merges.push(b.path)
          consumed.add(b.path)
          if (hashMatch && (titleExact || titleNear)) reason = 'both'
          else if (hashMatch) reason = reason === 'title' ? 'content_hash' : reason
          score = Math.max(score, hashMatch ? 1 : titleExact ? 1 : titleScore)
        }
      }
      if (merges.length > 0) {
        consumed.add(a.path)
        suggestions.push({ keep: a.path, merge: merges, score, reason })
      }
    }

    return { suggestions }
  }
}

const collect = async (store: Store, prefix: string): Promise<readonly IndexedArticle[]> => {
  const root = toPath(prefix)
  const exists = await store.exists(root).catch(() => false)
  if (!exists) return []
  const entries = await store.list(root, { recursive: true })
  const out: IndexedArticle[] = []
  for (const e of entries) {
    if (e.isDir) continue
    if (!pathUnder(e.path, prefix, true)) continue
    if (!e.path.endsWith('.md')) continue
    if (e.path.slice(prefix.length + 1).startsWith('_')) continue
    const content = (await store.read(e.path)).toString('utf8')
    const { frontmatter, body } = parseFrontmatter(content)
    const titleKey = normaliseTitle(frontmatter.title)
    out.push({
      path: e.path,
      titleKey,
      titleTrigrams: trigrams(titleKey),
      bodyHash: createHash('sha256').update(body.trim()).digest('hex'),
    })
  }
  return out
}

const normaliseTitle = (title: string): string => {
  const s = title.trim().toLowerCase()
  if (s === '') return ''
  let out = ''
  let prevSpace = false
  for (const r of s) {
    const isAlnum = /[\p{L}\p{N}]/u.test(r)
    if (isAlnum) {
      out += r
      prevSpace = false
    } else {
      if (!prevSpace && out.length > 0) {
        out += ' '
        prevSpace = true
      }
    }
  }
  return out.trim()
}

const trigrams = (s: string): Set<string> => {
  const set = new Set<string>()
  if (s.length < 3) {
    if (s !== '') set.add(s)
    return set
  }
  for (let i = 0; i <= s.length - 3; i++) set.add(s.slice(i, i + 3))
  return set
}

const jaccard = (a: Set<string>, b: Set<string>): number => {
  if (a.size === 0 && b.size === 0) return 0
  let intersection = 0
  for (const t of a) if (b.has(t)) intersection++
  const union = a.size + b.size - intersection
  return union === 0 ? 0 : intersection / union
}
