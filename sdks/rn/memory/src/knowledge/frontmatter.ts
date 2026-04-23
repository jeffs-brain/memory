// SPDX-License-Identifier: Apache-2.0

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
  present: boolean
}

export const parseFrontmatter = (content: string): ParsedFrontmatter => {
  const lines = content.split('\n')
  if (lines.length < 2 || lines[0]?.trim() !== FENCE) {
    return { frontmatter: { ...DEFAULT_FRONTMATTER }, body: content, present: false }
  }

  let closeIdx = -1
  for (let index = 1; index < lines.length; index += 1) {
    if (lines[index]?.trim() === FENCE) {
      closeIdx = index
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

  for (let index = 1; index < closeIdx; index += 1) {
    const line = lines[index] ?? ''
    const trimmed = line.trim()

    if (currentListKey !== '' && trimmed.startsWith('- ')) {
      const value = trimmed.slice(2).trim()
      if (value !== '') {
        if (currentListKey === 'tags') tags.push(stripQuotes(value))
        else if (currentListKey === 'sources') sources.push(stripQuotes(value))
      }
      continue
    }

    const kv = splitKv(line)
    if (kv === undefined) {
      currentListKey = ''
      continue
    }
    const [key, rawValue] = kv
    if (rawValue === '') {
      currentListKey = key
      continue
    }

    currentListKey = ''
    const value = stripQuotes(rawValue)
    switch (key) {
      case 'title':
        title = value
        break
      case 'summary':
        summary = value
        break
      case 'created':
        created = value
        break
      case 'modified':
        modified = value
        break
      case 'archived':
        archived = /^(true|yes|1)$/i.test(value)
        break
      case 'superseded_by':
      case 'supersededBy':
        supersededBy = value
        break
      case 'tags':
        for (const tag of parseInlineList(value)) tags.push(tag)
        break
      case 'sources':
        for (const source of parseInlineList(value)) sources.push(source)
        break
      default:
        break
    }
  }

  const body = lines
    .slice(closeIdx + 1)
    .join('\n')
    .replace(/^\n+/, '')
    .replace(/\n+$/, '')

  return {
    frontmatter: {
      title,
      summary,
      tags,
      sources,
      ...(created !== undefined ? { created } : {}),
      ...(modified !== undefined ? { modified } : {}),
      ...(archived !== undefined ? { archived } : {}),
      ...(supersededBy !== undefined ? { supersededBy } : {}),
    },
    body,
    present: true,
  }
}

export const serialiseFrontmatter = (frontmatter: Frontmatter, body: string): string => {
  const out: string[] = [FENCE]
  out.push(`title: ${yamlScalar(frontmatter.title)}`)
  out.push(`summary: ${yamlScalar(frontmatter.summary)}`)
  if (frontmatter.tags.length === 0) {
    out.push('tags: []')
  } else {
    out.push('tags:')
    for (const tag of frontmatter.tags) out.push(`  - ${yamlScalar(tag)}`)
  }
  if (frontmatter.sources.length === 0) {
    out.push('sources: []')
  } else {
    out.push('sources:')
    for (const source of frontmatter.sources) out.push(`  - ${yamlScalar(source)}`)
  }
  if (frontmatter.created !== undefined) out.push(`created: ${yamlScalar(frontmatter.created)}`)
  if (frontmatter.modified !== undefined) out.push(`modified: ${yamlScalar(frontmatter.modified)}`)
  if (frontmatter.archived !== undefined)
    out.push(`archived: ${frontmatter.archived ? 'true' : 'false'}`)
  if (frontmatter.supersededBy !== undefined) {
    out.push(`superseded_by: ${yamlScalar(frontmatter.supersededBy)}`)
  }
  out.push(FENCE)
  const trimmedBody = body.replace(/^\n+/, '').replace(/\n+$/, '')
  return `${out.join('\n')}\n\n${trimmedBody}\n`
}

const splitKv = (line: string): readonly [string, string] | undefined => {
  const index = line.indexOf(':')
  if (index < 0) return undefined
  const key = line.slice(0, index).trim()
  const value = line.slice(index + 1).trim()
  if (key === '') return undefined
  return [key, value]
}

const stripQuotes = (value: string): string => {
  if (value.length < 2) return value
  const first = value[0]
  const last = value[value.length - 1]
  if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
    return value.slice(1, -1)
  }
  return value
}

const parseInlineList = (value: string): readonly string[] => {
  const trimmed = value.trim()
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    const inner = trimmed.slice(1, -1)
    if (inner.trim() === '') return []
    return inner
      .split(',')
      .map((entry) => stripQuotes(entry.trim()))
      .filter((entry) => entry !== '')
  }
  return trimmed
    .split(',')
    .map((entry) => stripQuotes(entry.trim()))
    .filter((entry) => entry !== '')
}

const NEEDS_QUOTE_RE = /[:#\[\]{}&*!|>'"%@`,]/

const yamlScalar = (value: string): string => {
  if (value === '') return '""'
  if (NEEDS_QUOTE_RE.test(value) || /^\s|\s$/.test(value)) {
    return JSON.stringify(value)
  }
  return value
}
