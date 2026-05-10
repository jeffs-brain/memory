// SPDX-License-Identifier: Apache-2.0

/**
 * Markdown chunker: heading-aware splitting that preserves the heading
 * hierarchy in chunk metadata. Sections that exceed maxTokens are split
 * recursively using the separator hierarchy from the recursive chunker.
 */

import type { ChunkConfig } from '../chunk-config.js'
import type { Chunk, Chunker } from '../chunker-registry.js'
import { estimateTokens } from './recursive.js'

/** ATX heading regex: # through ###### followed by text. */
const ATX_HEADING_RE = /^(#{1,6})\s+(.+?)\s*#*\s*$/

/** Separator hierarchy for splitting within oversized sections. */
const SEPARATORS: readonly string[] = ['\n\n', '\n', '. ', ' ', '']

/**
 * Markdown chunker function. Parses heading structure, splits at heading
 * boundaries, and preserves the full heading path in chunk metadata.
 */
export const markdownChunker: Chunker = async (
  content: string,
  cfg: ChunkConfig,
  signal?: AbortSignal,
): Promise<readonly Chunk[]> => {
  signal?.throwIfAborted()
  if (content.trim() === '') return []

  const sections = splitMarkdownSections(content)
  const chunks: Chunk[] = []

  for (const section of sections) {
    signal?.throwIfAborted()
    const trimmed = section.content.trim()
    if (trimmed === '') continue

    const headingPath = section.headingPath.join(' > ')
    const tokens = estimateTokens(trimmed)

    if (tokens <= cfg.maxTokens) {
      chunks.push({
        id: '',
        content: trimmed,
        metadata: { chunker: 'markdown', headingPath },
      })
      continue
    }

    const subChunks = splitSectionWithOverlap(trimmed, cfg)
    for (const sub of subChunks) {
      chunks.push({
        id: '',
        content: sub,
        metadata: { chunker: 'markdown', headingPath },
      })
    }
  }
  return chunks
}

type MdSection = {
  readonly headingPath: readonly string[]
  readonly content: string
}

type HeadingEntry = {
  readonly level: number
  readonly title: string
}

/** Parses markdown into heading-bounded sections with hierarchy tracking. */
const splitMarkdownSections = (content: string): readonly MdSection[] => {
  const lines = content.split('\n')
  const sections: MdSection[] = []
  const stack: HeadingEntry[] = []
  let currentContent = ''
  let currentPath: readonly string[] = []

  const flush = (): void => {
    if (currentContent.trim() !== '') {
      sections.push({ headingPath: [...currentPath], content: currentContent })
    }
    currentContent = ''
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    const atxMatch = ATX_HEADING_RE.exec(line)

    if (atxMatch !== null) {
      flush()
      const hashes = atxMatch[1] ?? ''
      const title = (atxMatch[2] ?? '').trim()
      const level = hashes.length
      const heading = hashes + ' ' + title

      // Pop stack to appropriate depth.
      while (stack.length > 0 && (stack[stack.length - 1]?.level ?? 0) >= level) {
        stack.pop()
      }
      stack.push({ level, title: heading })
      currentPath = stack.map((h) => h.title)
      currentContent += line + '\n'
      continue
    }

    // Check for setext headings.
    if (i + 1 < lines.length && line.trim() !== '') {
      const nextLine = lines[i + 1] ?? ''
      if (isSetextUnderline(nextLine)) {
        flush()
        const level = nextLine.trim().startsWith('=') ? 1 : 2
        const title = line.trim()
        const heading = '#'.repeat(level) + ' ' + title

        while (stack.length > 0 && (stack[stack.length - 1]?.level ?? 0) >= level) {
          stack.pop()
        }
        stack.push({ level, title: heading })
        currentPath = stack.map((h) => h.title)
        currentContent += line + '\n' + nextLine + '\n'
        i++
        continue
      }
    }

    currentContent += line + '\n'
  }
  flush()
  return sections
}

/** Returns true when line is a setext heading underline (== or --). */
const isSetextUnderline = (line: string): boolean => {
  const trimmed = line.trim()
  if (trimmed.length < 2) return false
  return /^[=]+$/.test(trimmed) || /^[-]+$/.test(trimmed)
}

/** Splits an oversized section with overlap applied between pieces. */
const splitSectionWithOverlap = (text: string, cfg: ChunkConfig): readonly string[] => {
  const pieces = recursiveSplitLocal(text, cfg.maxTokens, 0)
  if (pieces.length <= 1) return pieces

  const result: string[] = []
  for (let i = 0; i < pieces.length; i++) {
    const piece = pieces[i]
    if (piece === undefined) continue
    const trimmed = piece.trim()
    if (trimmed === '') continue

    if (i > 0 && cfg.overlapTokens > 0) {
      const prevTrimmed = (pieces[i - 1] ?? '').trim()
      const tail = extractTail(prevTrimmed, cfg.overlapTokens)
      result.push(tail + '\n' + trimmed)
    } else {
      result.push(trimmed)
    }
  }
  return result
}

/** Local recursive split using the separator hierarchy. */
const recursiveSplitLocal = (
  text: string,
  maxTokens: number,
  sepIdx: number,
): readonly string[] => {
  if (estimateTokens(text) <= maxTokens) {
    return [text]
  }
  if (sepIdx >= SEPARATORS.length) {
    return hardSplit(text, maxTokens)
  }
  const sep = SEPARATORS[sepIdx]
  if (sep === undefined || sep === '') {
    return hardSplit(text, maxTokens)
  }
  const parts = text.split(sep)
  const merged: string[] = []
  let current = ''

  for (const part of parts) {
    const candidate = current === '' ? part : current + sep + part
    if (estimateTokens(candidate) > maxTokens) {
      if (current !== '') {
        merged.push(current)
        current = ''
      }
      if (estimateTokens(part) > maxTokens) {
        const sub = recursiveSplitLocal(part, maxTokens, sepIdx + 1)
        merged.push(...sub)
      } else {
        current = part
      }
    } else {
      current = candidate
    }
  }
  if (current !== '') {
    merged.push(current)
  }
  return merged
}

/** Character-level hard split as a last resort. */
const hardSplit = (text: string, maxTokens: number): readonly string[] => {
  const step = maxTokens * 4
  const out: string[] = []
  for (let i = 0; i < text.length; i += step) {
    out.push(text.slice(i, i + step))
  }
  return out
}

/** Extracts approximately overlapTokens worth of chars from the end. */
const extractTail = (text: string, overlapTokens: number): string => {
  const chars = overlapTokens * 4
  if (chars >= text.length) return text
  return text.slice(text.length - chars)
}
