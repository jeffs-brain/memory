// SPDX-License-Identifier: Apache-2.0

/**
 * Frontmatter parser / serialiser. YAML-flavour only — matches the Go
 * implementation in apps/jeff/internal/knowledge/frontmatter.go which
 * does its own line-by-line YAML subset because real YAML libraries
 * drag in too much surface area for a few known fields.
 */

import type { Frontmatter } from './types.js'

const FENCE = '---'

const DEFAULT_FRONTMATTER: Frontmatter = {
  title: '',
  summary: '',
  tags: [],
  sources: [],
}

export type ParsedFrontmatter = {
  frontmatter: Frontmatter
  body: string
  /** True when a fenced frontmatter block was actually present. */
  present: boolean
}

export const parseFrontmatter = (content: string): ParsedFrontmatter => {
  const lines = content.split('\n')
  if (lines.length < 2 || lines[0]?.trim() !== FENCE) {
    return { frontmatter: { ...DEFAULT_FRONTMATTER }, body: content, present: false }
  }

  let closeIdx = -1
  for (let i = 1; i < lines.length; i++) {
    if (lines[i]?.trim() === FENCE) {
      closeIdx = i
      break
    }
  }
  if (closeIdx < 0) {
    return { frontmatter: { ...DEFAULT_FRONTMATTER }, body: content, present: false }
  }

  let title = ''
  let summary = ''
  let created: string | undefined
  let modified: string | undefined
  let archived: boolean | undefined
  let supersededBy: string | undefined
  const tags: string[] = []
  const sources: string[] = []

  let currentListKey = ''

  for (let i = 1; i < closeIdx; i++) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()

    if (currentListKey !== '' && trimmed.startsWith('- ')) {
      const val = trimmed.slice(2).trim()
      if (val !== '') {
        if (currentListKey === 'tags') tags.push(stripQuotes(val))
        else if (currentListKey === 'sources') sources.push(stripQuotes(val))
      }
      continue
    }

    const kv = splitKV(line)
    if (!kv) {
      currentListKey = ''
      continue
    }
    const [key, rawVal] = kv

    if (rawVal === '') {
      // Bare key → list follows.
      currentListKey = key
      continue
    }
    currentListKey = ''
    const val = stripQuotes(rawVal)

    switch (key) {
      case 'title':
        title = val
        break
      case 'summary':
        summary = val
        break
      case 'created':
        created = val
        break
      case 'modified':
        modified = val
        break
      case 'archived':
        archived = /^(true|yes|1)$/i.test(val)
        break
      case 'superseded_by':
      case 'supersededBy':
        supersededBy = val
        break
      case 'tags':
        for (const t of parseInlineList(val)) tags.push(t)
        break
      case 'sources':
        for (const s of parseInlineList(val)) sources.push(s)
        break
      default:
        break
    }
  }

  const body = lines.slice(closeIdx + 1).join('\n').replace(/^\n+/, '').replace(/\n+$/, '')
  const fm: Frontmatter = {
    title,
    summary,
    tags,
    sources,
    ...(created !== undefined ? { created } : {}),
    ...(modified !== undefined ? { modified } : {}),
    ...(archived !== undefined ? { archived } : {}),
    ...(supersededBy !== undefined ? { supersededBy } : {}),
  }

  return { frontmatter: fm, body, present: true }
}

export const serialiseFrontmatter = (fm: Frontmatter, body: string): string => {
  const out: string[] = [FENCE]
  out.push(`title: ${yamlScalar(fm.title)}`)
  out.push(`summary: ${yamlScalar(fm.summary)}`)
  if (fm.tags.length === 0) {
    out.push('tags: []')
  } else {
    out.push('tags:')
    for (const t of fm.tags) out.push(`  - ${yamlScalar(t)}`)
  }
  if (fm.sources.length === 0) {
    out.push('sources: []')
  } else {
    out.push('sources:')
    for (const s of fm.sources) out.push(`  - ${yamlScalar(s)}`)
  }
  if (fm.created !== undefined) out.push(`created: ${yamlScalar(fm.created)}`)
  if (fm.modified !== undefined) out.push(`modified: ${yamlScalar(fm.modified)}`)
  if (fm.archived !== undefined) out.push(`archived: ${fm.archived ? 'true' : 'false'}`)
  if (fm.supersededBy !== undefined) out.push(`superseded_by: ${yamlScalar(fm.supersededBy)}`)
  out.push(FENCE)
  const trimmedBody = body.replace(/^\n+/, '').replace(/\n+$/, '')
  return `${out.join('\n')}\n\n${trimmedBody}\n`
}

const splitKV = (line: string): readonly [string, string] | undefined => {
  const idx = line.indexOf(':')
  if (idx < 0) return undefined
  const key = line.slice(0, idx).trim()
  const val = line.slice(idx + 1).trim()
  if (key === '') return undefined
  return [key, val]
}

const stripQuotes = (v: string): string => {
  if (v.length < 2) return v
  const first = v[0]
  const last = v[v.length - 1]
  if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
    return v.slice(1, -1)
  }
  return v
}

const parseInlineList = (v: string): readonly string[] => {
  const trimmed = v.trim()
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    const inner = trimmed.slice(1, -1)
    if (inner.trim() === '') return []
    return inner
      .split(',')
      .map((s) => stripQuotes(s.trim()))
      .filter((s) => s !== '')
  }
  // Plain "a, b, c" comma list.
  return trimmed
    .split(',')
    .map((s) => stripQuotes(s.trim()))
    .filter((s) => s !== '')
}

const NEEDS_QUOTE_RE = /[:#\[\]{}&*!|>'"%@`,]/

const yamlScalar = (v: string): string => {
  if (v === '') return '""'
  if (NEEDS_QUOTE_RE.test(v) || /^\s|\s$/.test(v)) {
    return JSON.stringify(v)
  }
  return v
}
