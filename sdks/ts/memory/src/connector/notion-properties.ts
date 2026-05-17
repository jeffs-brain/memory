// SPDX-License-Identifier: Apache-2.0

/**
 * Property extraction helpers for the Notion connector. Converts
 * Notion property objects to their string representations and
 * parses database entries into structured data.
 */

import { renderPlainRichText, type NotionRichText } from './notion-blocks.js'

// ---------- Types ----------

/**
 * Parsed database entry.
 */
export type ParsedDatabaseEntry = {
  readonly pageId: string
  readonly title: string
  readonly url: string
  readonly lastEditedTime: Date
  readonly propertiesMarkdown: string
}

/**
 * Parsed page metadata.
 */
export type ParsedPage = {
  readonly id: string
  readonly title: string
  readonly url: string
  readonly lastEditedTime: Date
}

// ---------- Title extraction ----------

/**
 * Extracts the title from a page's properties map.
 */
export const extractTitleFromProperties = (
  properties: Record<string, Record<string, unknown>>,
): string => {
  for (const prop of Object.values(properties)) {
    if (prop.type === 'title') {
      const titleArr = (prop.title ?? []) as readonly NotionRichText[]
      return renderPlainRichText(titleArr)
    }
  }
  return ''
}

// ---------- Property value extraction ----------

/**
 * Converts a Notion property to its string representation.
 */
export const extractPropertyValue = (
  prop: Record<string, unknown>,
): string => {
  const propType = prop.type as string

  const renderers: Readonly<Record<string, () => string>> = {
    title: () =>
      renderPlainRichText((prop.title ?? []) as readonly NotionRichText[]),
    rich_text: () =>
      renderPlainRichText(
        (prop.rich_text ?? []) as readonly NotionRichText[],
      ),
    number: () => {
      const num = prop.number as number | null
      return num !== null && num !== undefined ? String(num) : ''
    },
    select: () => {
      const sel = prop.select as { name: string } | null
      return sel?.name ?? ''
    },
    multi_select: () => {
      const items = (prop.multi_select ?? []) as readonly { name: string }[]
      return items.map((i) => i.name).join(', ')
    },
    date: () => {
      const d = prop.date as { start: string; end?: string } | null
      if (d === null || d === undefined) return ''
      return d.end !== undefined && d.end !== ''
        ? `${d.start} to ${d.end}`
        : d.start
    },
    checkbox: () => (prop.checkbox === true ? 'true' : 'false'),
    url: () => (prop.url as string) ?? '',
    email: () => (prop.email as string) ?? '',
    phone_number: () => (prop.phone_number as string) ?? '',
    status: () => {
      const st = prop.status as { name: string } | null
      return st?.name ?? ''
    },
    people: () => {
      const people = (prop.people ?? []) as readonly { name: string }[]
      return people.map((p) => p.name).join(', ')
    },
    relation: () => {
      const rels = (prop.relation ?? []) as readonly { id: string }[]
      return rels.map((r) => r.id).join(', ')
    },
  }

  const renderer = renderers[propType]
  return renderer !== undefined ? renderer() : ''
}

// ---------- Database entry parsing ----------

/**
 * Parses a database entry from the query response.
 */
export const parseDatabaseEntry = (
  raw: Record<string, unknown>,
): ParsedDatabaseEntry => {
  const properties = (raw.properties ?? {}) as Record<
    string,
    Record<string, unknown>
  >
  const lastEditedStr = raw.last_edited_time as string
  const lastEditedTime = new Date(lastEditedStr)

  let title = ''
  const propLines: string[] = []

  // Sort keys for deterministic output.
  const sortedKeys = Object.keys(properties).sort()

  for (const key of sortedKeys) {
    const prop = properties[key]
    if (prop === undefined) continue
    const val = extractPropertyValue(prop)

    if (prop.type === 'title') {
      title = val
      continue
    }

    if (val !== '') {
      propLines.push(`- **${key}**: ${val}`)
    }
  }

  let markdown = ''
  if (title !== '') {
    markdown += `## ${title}\n\n`
  }
  if (propLines.length > 0) {
    markdown += propLines.join('\n') + '\n'
  }

  return {
    pageId: raw.id as string,
    title,
    url: raw.url as string,
    lastEditedTime,
    propertiesMarkdown: markdown,
  }
}

// ---------- Page response parsing ----------

/**
 * Parses a Notion API page response into structured metadata.
 */
export const parsePageResponse = (
  data: Record<string, unknown>,
): ParsedPage => {
  const properties = (data.properties ?? {}) as Record<
    string,
    Record<string, unknown>
  >
  const title = extractTitleFromProperties(properties)
  const lastEditedStr = data.last_edited_time as string
  const lastEditedTime = new Date(lastEditedStr)

  return {
    id: data.id as string,
    title,
    url: data.url as string,
    lastEditedTime,
  }
}
