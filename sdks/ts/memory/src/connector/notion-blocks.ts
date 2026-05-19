// SPDX-License-Identifier: Apache-2.0

/**
 * Block rendering helpers for the Notion connector. Converts Notion
 * block objects to markdown representation and handles rich text
 * formatting.
 */

// ---------- Types ----------

/**
 * Notion rich text object.
 */
export type NotionRichText = {
  readonly type: string
  readonly plain_text: string
  readonly href?: string
  readonly annotations?: NotionAnnotations
}

/**
 * Notion text formatting annotations.
 */
type NotionAnnotations = {
  readonly bold?: boolean
  readonly italic?: boolean
  readonly strikethrough?: boolean
  readonly underline?: boolean
  readonly code?: boolean
  readonly color?: string
}

/**
 * Notion block object from the blocks API.
 */
export type NotionBlock = {
  readonly id: string
  readonly type: string
  readonly has_children: boolean
  readonly [key: string]: unknown
}

// ---------- Module-level constants ----------

/**
 * Set of block types considered list items for indentation.
 * Module-level constant avoids per-call allocation.
 */
const LIST_BLOCK_TYPES = new Set([
  'bulleted_list_item',
  'numbered_list_item',
  'to_do',
  'toggle',
])

// ---------- Rich text helpers ----------

/**
 * Renders a Notion rich text array to markdown with formatting.
 */
export const renderRichText = (texts: readonly NotionRichText[]): string => {
  const parts: string[] = []

  for (const t of texts) {
    let text = t.plain_text

    if (t.annotations?.code === true) {
      text = '`' + text + '`'
    }
    if (t.annotations?.bold === true) {
      text = '**' + text + '**'
    }
    if (t.annotations?.italic === true) {
      text = '*' + text + '*'
    }
    if (t.annotations?.strikethrough === true) {
      text = '~~' + text + '~~'
    }

    if (t.href !== undefined && t.href !== '') {
      text = `[${text}](${t.href})`
    }

    parts.push(text)
  }

  return parts.join('')
}

/**
 * Extracts plain text from rich text without formatting.
 */
export const renderPlainRichText = (
  texts: readonly NotionRichText[],
): string => texts.map((t) => t.plain_text).join('')

// ---------- Block rendering ----------

/**
 * Converts a single Notion block to markdown. Blocks whose type is
 * not in the allowedTypes filter (when provided) are skipped.
 */
export const blockToMarkdown = (
  block: NotionBlock,
  allowedTypes?: ReadonlySet<string>,
): string => {
  // Apply block type filter when configured.
  if (allowedTypes !== undefined && !allowedTypes.has(block.type)) {
    return ''
  }

  const blockData = block[block.type] as Record<string, unknown> | undefined
  if (blockData === undefined) return ''

  const richText = (blockData.rich_text ?? []) as readonly NotionRichText[]
  const text = renderRichText(richText)
  const caption = (blockData.caption ?? []) as readonly NotionRichText[]

  const renderers: Readonly<Record<string, () => string>> = {
    paragraph: () => `${text}\n`,
    heading_1: () => `# ${text}\n`,
    heading_2: () => `## ${text}\n`,
    heading_3: () => `### ${text}\n`,
    bulleted_list_item: () => `- ${text}`,
    numbered_list_item: () => `1. ${text}`,
    to_do: () => {
      const checked = blockData.checked === true
      const marker = checked ? '[x]' : '[ ]'
      return `- ${marker} ${text}`
    },
    toggle: () => `- ${text}`,
    quote: () => `> ${text}\n`,
    callout: () => `> ${text}\n`,
    divider: () => '---\n',
    code: () => {
      const lang = (blockData.language as string) ?? ''
      return '```' + lang + '\n' + text + '\n```\n'
    },
    equation: () => {
      const expression = (blockData.expression as string) ?? ''
      return `$$\n${expression}\n$$\n`
    },
    image: () => {
      const capText = renderRichText(caption)
      const url = extractFileUrl(blockData)
      return capText !== ''
        ? `![${capText}](${url})\n`
        : `![](${url})\n`
    },
    file: () => {
      const url = extractFileUrl(blockData)
      const capText = renderRichText(caption)
      const label = capText !== '' ? capText : 'file'
      return `[${label}](${url})\n`
    },
    bookmark: () => {
      const capText = renderRichText(caption)
      const url = blockData.url as string | undefined
      if (url !== undefined) {
        return capText !== '' ? `[${capText}](${url})\n` : `${url}\n`
      }
      return ''
    },
    embed: () => {
      const url = blockData.url as string | undefined
      return url !== undefined ? `${url}\n` : ''
    },
    table_of_contents: () => '',
    breadcrumb: () => '',
    column_list: () => '',
    column: () => '',
    child_page: () => '',
    child_database: () => '',
    synced_block: () => '',
    link_preview: () => {
      const url = blockData.url as string | undefined
      return url !== undefined ? `${url}\n` : ''
    },
    table: () => '',
    table_row: () => {
      const cells = (blockData.cells ?? []) as readonly (readonly NotionRichText[])[]
      const rendered = cells.map((cell) => renderRichText(cell))
      return `| ${rendered.join(' | ')} |`
    },
  }

  const renderer = renderers[block.type]
  return renderer !== undefined ? renderer() : text
}

/**
 * Checks whether a block type is a list item.
 */
export const isListBlock = (blockType: string): boolean =>
  LIST_BLOCK_TYPES.has(blockType)

/**
 * Extracts a file URL from a file-type block's data.
 */
const extractFileUrl = (data: Record<string, unknown>): string => {
  const file = data.file as { url?: string } | undefined
  const external = data.external as { url?: string } | undefined
  return file?.url ?? external?.url ?? ''
}
