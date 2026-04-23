import { type Path, type Store, lastSegment, pathUnder, toPath } from '../store/index.js'
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

type CollectedArticle = {
  path: Path
  content: string
  modTime: Date
}

export const createLint = (deps: LintDeps) => {
  const { store } = deps

  return async (): Promise<LintReport> => {
    const issues: LintIssue[] = []
    const wikiArticles = await collectArticles(store, WIKI_PREFIX)
    const draftArticles = await collectArticles(store, DRAFTS_PREFIX)
    const wikiSlugs = new Set<string>()
    for (const article of wikiArticles) {
      const base = article.path.slice(WIKI_PREFIX.length + 1).replace(/\.md$/, '')
      wikiSlugs.add(base)
    }

    const titleGroups = new Map<string, Path[]>()

    for (const article of [...wikiArticles, ...draftArticles]) {
      const { frontmatter, body, present } = parseFrontmatter(article.content)

      if (!present) {
        issues.push({
          kind: 'missing_frontmatter_field',
          path: article.path,
          message: 'no frontmatter block present',
          details: { field: 'frontmatter' },
        })
        continue
      }

      for (const field of REQUIRED_FIELDS) {
        if (frontmatter[field].trim() === '') {
          issues.push({
            kind: 'missing_frontmatter_field',
            path: article.path,
            message: `missing required field: ${field}`,
            details: { field },
          })
        }
      }

      if (body.trim() === '') {
        issues.push({
          kind: 'empty_body',
          path: article.path,
          message: 'article body is empty',
        })
      }

      for (const match of body.matchAll(WIKILINK_RE)) {
        const target = (match[1] ?? '').trim()
        if (target === '' || wikiSlugs.has(target)) continue
        issues.push({
          kind: 'broken_link',
          path: article.path,
          message: `wikilink target not found: ${target}`,
          details: { target },
        })
      }

      const titleKey = normaliseTitle(frontmatter.title)
      if (titleKey !== '') {
        const group = titleGroups.get(titleKey) ?? []
        group.push(article.path)
        titleGroups.set(titleKey, group)
      }

      if (article.path.startsWith(`${WIKI_PREFIX}/`)) {
        const wordCount = countBodyWords(body)
        if (wordCount < STUB_WORD_THRESHOLD) {
          issues.push({
            kind: 'stub_article',
            path: article.path,
            message: `article body is under ${STUB_WORD_THRESHOLD} words: ${wordCount}`,
            details: { wordCount },
          })
        }

        if (countBodyLinks(body) === 0) {
          issues.push({
            kind: 'zero_link_article',
            path: article.path,
            message: 'article body has no wikilinks',
          })
        }

        for (const source of frontmatter.sources) {
          const sourcePath = source.trim()
          if (sourcePath === '') continue
          const sourceInfo = await store.stat(sourcePath as Path).catch(() => undefined)
          if (sourceInfo === undefined || sourceInfo.isDir) continue
          if (sourceInfo.modTime > article.modTime) {
            issues.push({
              kind: 'stale_source',
              path: article.path,
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

    for (const [titleKey, paths] of titleGroups) {
      if (paths.length < 2) continue
      for (const path of paths) {
        issues.push({
          kind: 'duplicate_title',
          path,
          message: `duplicate title group: ${titleKey} (${paths.length} articles)`,
          details: {
            titleKey,
            relatedPaths: paths,
          },
        })
      }
    }

    return { ok: issues.length === 0, issues }
  }
}

const collectArticles = async (
  store: Store,
  prefix: string,
): Promise<readonly CollectedArticle[]> => {
  const root = toPath(prefix)
  const exists = await store.exists(root).catch(() => false)
  if (!exists) return []

  const entries = await store.list(root, { recursive: true })
  const output: CollectedArticle[] = []
  for (const entry of entries) {
    if (entry.isDir) continue
    if (!pathUnder(entry.path, prefix, true)) continue
    if (!entry.path.endsWith('.md')) continue
    if (lastSegment(entry.path).startsWith('_')) continue
    output.push({
      path: entry.path,
      content: await store.read(entry.path),
      modTime: entry.modTime,
    })
  }
  return output
}

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
