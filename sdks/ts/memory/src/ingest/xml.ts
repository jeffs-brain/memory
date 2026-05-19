// SPDX-License-Identifier: Apache-2.0

/**
 * XML structured data extractor with element path context, CDATA
 * handling, namespace stripping, and processing instruction removal.
 */

import { detectEncoding } from './encoding.js'
import type { ExtractResult } from './structured.js'

/** Upper bound on raw input in bytes (50 MiB). */
const DEFAULT_MAX_INPUT_SIZE = 50 * 1024 * 1024

export type XmlExtractorConfig = {
  readonly elementsPerChunk?: number
  readonly maxDepth?: number
  readonly maxInputSize?: number
}

const defaultXmlConfig = {
  elementsPerChunk: 50,
  maxDepth: 10,
  maxInputSize: DEFAULT_MAX_INPUT_SIZE,
} as const

type XmlNode = {
  readonly localName: string
  readonly attributes: readonly XmlAttr[]
  readonly children: XmlNode[]
  text: string
}

type XmlAttr = {
  readonly name: string
  readonly value: string
}

/**
 * Extract structured content from XML bytes. Elements are rendered
 * with their full path context. CDATA sections are treated as text,
 * namespace prefixes are stripped to local names, and processing
 * instructions are ignored.
 */
export const extractXML = (raw: Buffer, config: XmlExtractorConfig = {}): ExtractResult => {
  if (raw.length === 0) {
    throw new Error('structured: empty xml input')
  }

  const maxSize = config.maxInputSize ?? defaultXmlConfig.maxInputSize
  if (raw.length > maxSize) {
    throw new Error(`structured: xml input exceeds ${maxSize} byte limit`)
  }

  const { text, encoding } = detectEncoding(raw)

  const maxDepth = config.maxDepth ?? defaultXmlConfig.maxDepth
  const root = parseXMLTree(text, maxDepth)

  if (root === null) {
    throw new Error('structured: empty xml document')
  }

  const epc = config.elementsPerChunk ?? defaultXmlConfig.elementsPerChunk
  const { content, elementCount } = renderXMLNodes(root, epc, maxDepth)

  return {
    text: content,
    contentType: 'application/xml',
    encoding,
    metadata: {
      encoding,
      root_element: root.localName,
      element_count: String(elementCount),
    },
    pages: 0,
    language: '',
    confidence: 0,
    skipped: false,
  }
}

/**
 * Lightweight SAX-style XML parser that builds a tree of XmlNode
 * objects. Uses character-by-character scanning rather than a full
 * DOM parser to keep the implementation dependency-free.
 *
 * Handles: elements, attributes, text, CDATA, processing instructions.
 * Strips namespace prefixes from element and attribute names.
 */
const parseXMLTree = (text: string, maxDepth: number): XmlNode | null => {
  let root: XmlNode | null = null
  const stack: XmlNode[] = []
  let pos = 0

  while (pos < text.length) {
    const nextTag = text.indexOf('<', pos)
    if (nextTag === -1) {
      appendText(stack, text.substring(pos))
      break
    }

    if (nextTag > pos) {
      appendText(stack, text.substring(pos, nextTag))
    }

    // CDATA section.
    if (text.startsWith('<![CDATA[', nextTag)) {
      const cdataEnd = text.indexOf(']]>', nextTag + 9)
      if (cdataEnd === -1) {
        pos = text.length
        break
      }
      appendText(stack, text.substring(nextTag + 9, cdataEnd))
      pos = cdataEnd + 3
      continue
    }

    // Comment.
    if (text.startsWith('<!--', nextTag)) {
      const commentEnd = text.indexOf('-->', nextTag + 4)
      pos = commentEnd === -1 ? text.length : commentEnd + 3
      continue
    }

    // Processing instruction.
    if (text.startsWith('<?', nextTag)) {
      const piEnd = text.indexOf('?>', nextTag + 2)
      pos = piEnd === -1 ? text.length : piEnd + 2
      continue
    }

    // DOCTYPE.
    if (text.startsWith('<!', nextTag)) {
      const dtEnd = text.indexOf('>', nextTag + 2)
      pos = dtEnd === -1 ? text.length : dtEnd + 1
      continue
    }

    // Closing tag.
    if (text[nextTag + 1] === '/') {
      const closeEnd = text.indexOf('>', nextTag + 2)
      if (closeEnd === -1) {
        pos = text.length
        break
      }
      if (stack.length > 0) {
        stack.pop()
      }
      pos = closeEnd + 1
      continue
    }

    // Opening or self-closing tag.
    const tagEnd = text.indexOf('>', nextTag + 1)
    if (tagEnd === -1) {
      pos = text.length
      break
    }

    const tagContent = text.substring(nextTag + 1, tagEnd)
    const selfClosing = tagContent.endsWith('/')
    const cleanTag = selfClosing ? tagContent.slice(0, -1).trim() : tagContent.trim()

    const { localName, attributes } = parseTag(cleanTag)
    const node: XmlNode = {
      localName,
      attributes,
      children: [],
      text: '',
    }

    if (stack.length > 0) {
      const parent = stack[stack.length - 1]
      if (parent !== undefined) {
        parent.children.push(node)
      }
    }

    if (root === null) {
      root = node
    }

    if (!selfClosing && stack.length < maxDepth) {
      stack.push(node)
    }

    pos = tagEnd + 1
  }

  return root
}

/**
 * Parse a tag string into its local name and attributes.
 * Strips namespace prefixes (e.g. "ns:element" becomes "element").
 */
const parseTag = (tag: string): { localName: string; attributes: XmlAttr[] } => {
  const spaceIdx = tag.search(/\s/)
  const rawName = spaceIdx === -1 ? tag : tag.substring(0, spaceIdx)
  const localName = stripNamespace(rawName)

  const attributes: XmlAttr[] = []
  if (spaceIdx === -1) return { localName, attributes }

  const attrStr = tag.substring(spaceIdx)
  const attrRegex = /(\S+?)=(?:"([^"]*)"|'([^']*)')/g
  let match: RegExpExecArray | null

  while ((match = attrRegex.exec(attrStr)) !== null) {
    const name = stripNamespace(match[1] ?? '')
    const value = match[2] ?? match[3] ?? ''
    // Skip xmlns declarations.
    if (name === 'xmlns' || (match[1] ?? '').startsWith('xmlns:')) continue
    attributes.push({ name, value })
  }

  return { localName, attributes }
}

/** Strip namespace prefix from an element or attribute name. */
const stripNamespace = (name: string): string => {
  const colonIdx = name.indexOf(':')
  return colonIdx === -1 ? name : name.substring(colonIdx + 1)
}

/** Append trimmed text content to the current element on the stack. */
const appendText = (stack: readonly XmlNode[], raw: string): void => {
  const trimmed = raw.trim()
  if (trimmed === '' || stack.length === 0) return
  const current = stack[stack.length - 1]
  if (current === undefined) return
  if (current.text.length > 0) {
    current.text += ' '
  }
  current.text += trimmed
}

/** Render the XML tree as structured text with path context and chunks. */
const renderXMLNodes = (
  root: XmlNode,
  elementsPerChunk: number,
  maxDepth: number,
): { content: string; elementCount: number } => {
  if (root.children.length === 0) {
    const txt = root.text
    if (txt === '') {
      return { content: `<${root.localName}/>`, elementCount: 1 }
    }
    return { content: `${root.localName}: ${txt}`, elementCount: 1 }
  }

  const parts: string[] = []
  let totalElements = 0

  for (let i = 0; i < root.children.length; i += elementsPerChunk) {
    const end = Math.min(i + elementsPerChunk, root.children.length)
    const batch = root.children.slice(i, end)
    const lines: string[] = []

    for (let j = 0; j < batch.length; j++) {
      const child = batch[j]
      if (child === undefined) continue
      if (j > 0) lines.push('')
      const count = renderXMLNode(lines, child, root.localName, 0, maxDepth)
      totalElements += count
    }

    parts.push(lines.join('\n'))
  }

  return { content: parts.join('\n\n---\n\n'), elementCount: totalElements }
}

/** Recursively render a single XML node with path context. */
const renderXMLNode = (
  lines: string[],
  node: XmlNode,
  parentPath: string,
  depth: number,
  maxDepth: number,
): number => {
  if (depth >= maxDepth) {
    lines.push(`${parentPath}/${node.localName}: [max depth exceeded]`)
    return 1
  }

  const currentPath = `${parentPath}/${node.localName}`
  let count = 1

  for (const attr of node.attributes) {
    lines.push(`${currentPath}@${attr.name}: ${attr.value}`)
  }

  if (node.text !== '') {
    lines.push(`${currentPath}: ${node.text}`)
  }

  for (const child of node.children) {
    count += renderXMLNode(lines, child, currentPath, depth + 1, maxDepth)
  }

  return count
}
