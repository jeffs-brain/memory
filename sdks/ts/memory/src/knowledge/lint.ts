/**
 * Structural lint. Pure — no LLM calls, no Store mutations. Inspects
 * draft + wiki articles in the Store and reports issues. Ported (and
 * simplified) from apps/jeff/internal/knowledge/lint.go.
 */

import { lastSegment, pathUnder, toPath, type Path, type Store } from '../store/index.js'
import { parseFrontmatter } from './frontmatter.js'
import type { LintIssue, LintReport } from './types.js'

const REQUIRED_FIELDS = ['title', 'summary'] as const
const WIKILINK_RE = /\[\[([^\]]+)\]\]/g
const STUB_WORD_THRESHOLD = 200

const DRAFTS_PREFIX = 'drafts'
const WIKI_PREFIX = 'wiki'

type LintDeps = {
  store: Store
}

export const createLint = (deps: LintDeps) => {
  const { store } = deps

  return async (): Promise<LintReport> => {
    const issues: LintIssue[] = []

    const wikiArticles = await collectArticles(store, WIKI_PREFIX)
    const draftArticles = await collectArticles(store, DRAFTS_PREFIX)
    const wikiSlugs = new Set<string>()
    for (const a of wikiArticles) {
      const base = a.path.slice(WIKI_PREFIX.length + 1).replace(/\.md$/, '')
      wikiSlugs.add(base)
    }

    const titleGroups = new Map<string, Path[]>()

    for (const a of [...wikiArticles, ...draftArticles]) {
      const { frontmatter, body, present } = parseFrontmatter(a.content)

      if (!present) {
        issues.push({
          kind: 'missing_frontmatter_field',
          path: a.path,
          message: 'no frontmatter block present',
          details: { field: 'frontmatter' },
        })
        continue
      }

      for (const field of REQUIRED_FIELDS) {
        if (frontmatter[field].trim() === '') {
          issues.push({
            kind: 'missing_frontmatter_field',
            path: a.path,
            message: `missing required field: ${field}`,
            details: { field },
          })
        }
      }

      if (body.trim() === '') {
        issues.push({
          kind: 'empty_body',
          path: a.path,
          message: 'article body is empty',
        })
      }

      for (const match of body.matchAll(WIKILINK_RE)) {
        const target = (match[1] ?? '').trim()
        if (target === '' || wikiSlugs.has(target)) continue
        issues.push({
          kind: 'broken_link',
          path: a.path,
          message: `wikilink target not found: ${target}`,
          details: { target },
        })
      }

      const titleKey = normaliseTitle(frontmatter.title)
      if (titleKey !== '') {
        const group = titleGroups.get(titleKey) ?? []
        group.push(a.path)
        titleGroups.set(titleKey, group)
      }

      if (a.path.startsWith(`${WIKI_PREFIX}/`)) {
        const wordCount = countBodyWords(body)
        if (wordCount < STUB_WORD_THRESHOLD) {
          issues.push({
            kind: 'stub_article',
            path: a.path,
            message: `article body is under ${STUB_WORD_THRESHOLD} words: ${wordCount}`,
            details: { wordCount },
          })
        }

        if (countBodyLinks(body) === 0) {
          issues.push({
            kind: 'zero_link_article',
            path: a.path,
            message: 'article body has no wikilinks',
          })
        }

        for (const source of frontmatter.sources) {
          const sourcePath = source.trim()
          if (sourcePath === '') continue
          const sourceInfo = await store.stat(sourcePath as Path).catch(() => undefined)
          if (sourceInfo === undefined || sourceInfo.isDir) continue
          if (sourceInfo.modTime > a.modTime) {
            issues.push({
              kind: 'stale_source',
              path: a.path,
              message: `source modified after article: ${sourcePath}`,
              details: { sourcePath },
            })
          }
        }
      }
    }

    for (const draft of draftArticles) {
      const slug = draft.path.slice(DRAFTS_PREFIX.length + 1).replace(/\.md$/, '')
      if (wikiSlugs.has(slug)) continue
      issues.push({
        kind: 'orphan_draft',
        path: draft.path,
        message: `draft has no corresponding wiki entry: ${slug}`,
      })
    }

    for (const [key, paths] of titleGroups) {
      if (paths.length < 2) continue
      for (const p of paths) {
        issues.push({
          kind: 'duplicate_title',
          path: p,
          message: `duplicate title group: ${key} (${paths.length} articles)`,
          details: {
            titleKey: key,
            relatedPaths: paths,
          },
        })
      }
    }

    return { ok: issues.length === 0, issues }
  }
}

const collectArticles = async (store: Store, prefix: string): Promise<readonly CollectedArticle[]> => {
  const root = toPath(prefix)
  const exists = await store.exists(root).catch(() => false)
  if (!exists) return []
  const entries = await store.list(root, { recursive: true })
  const out: CollectedArticle[] = []
  for (const e of entries) {
    if (e.isDir) continue
    if (!pathUnder(e.path, prefix, true)) continue
    if (!e.path.endsWith('.md')) continue
    if (lastSegment(e.path).startsWith('_')) continue
    const content = (await store.read(e.path)).toString('utf8')
    out.push({ path: e.path, content, modTime: e.modTime })
  }
  return out
}

type CollectedArticle = { path: Path; content: string; modTime: Date }

const countBodyWords = (body: string): number => {
  let count = 0
  for (const line of body.split('\n')) {
    const trimmed = line.trim()
    if (trimmed === '') continue
    if (/^#+\s/.test(trimmed)) continue
    count += trimmed.split(/\s+/).length
  }
  return count
}

const countBodyLinks = (body: string): number => {
  let count = 0
  for (const _match of body.matchAll(WIKILINK_RE)) {
    count += 1
  }
  return count
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
