// SPDX-License-Identifier: Apache-2.0

/**
 * Code-aware chunker: splits source code at function/class/type
 * boundaries using line-level heuristics. This is the Phase-1
 * implementation -- full AST-based splitting via tree-sitter is
 * deferred to a later phase.
 *
 * Algorithm:
 * 1. Extract leading import/package lines into a header block.
 * 2. Walk remaining lines and split at detected declaration boundaries.
 * 3. Prepend the header to each chunk for self-contained context.
 */

import type { ChunkConfig } from '../chunk-config.js'
import type { Chunk, Chunker } from '../chunker-registry.js'
import { estimateTokens } from './recursive.js'

const GO_FUNC_RE = /^func\s/
const GO_TYPE_RE = /^type\s/
const PY_DEF_RE = /^(def|class|async\s+def)\s/
const TS_FUNC_RE = /^(export\s+)?(function|class|const|interface|type|enum)\s/
const C_FUNC_RE = /^[a-zA-Z_].*\)\s*\{?\s*$/
const IMPORT_LINE_RE = /^(import|from|require|use|#include|package)\s/

const SEPARATORS: readonly string[] = ['\n\n', '\n', '. ', ' ', '']

export const codeChunker: Chunker = async (
  content: string,
  cfg: ChunkConfig,
  signal?: AbortSignal,
): Promise<readonly Chunk[]> => {
  signal?.throwIfAborted()
  if (content.trim() === '') return []

  const lines = content.split('\n')
  const { header, bodyStart } = extractImportHeader(lines)
  const sections = splitAtDeclarations(lines.slice(bodyStart))

  const chunks: Chunk[] = []
  for (const section of sections) {
    const text = section.trim()
    if (text === '') continue

    const full = header !== '' && !text.startsWith(header)
      ? header + '\n\n' + text
      : text

    if (estimateTokens(full) <= cfg.maxTokens) {
      chunks.push({
        id: '',
        content: full,
        metadata: { chunker: 'code' },
      })
      continue
    }
    const subPieces = recursiveSplitLocal(full, cfg.maxTokens, 0)
    for (const piece of subPieces) {
      const t = piece.trim()
      if (t === '') continue
      chunks.push({
        id: '',
        content: t,
        metadata: { chunker: 'code' },
      })
    }
  }

  return mergeUndersized(chunks, cfg)
}

const extractImportHeader = (lines: readonly string[]): { header: string; bodyStart: number } => {
  const headerLines: string[] = []
  let bodyStart = 0
  let inHeader = true

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()
    if (inHeader) {
      if (
        trimmed === '' ||
        IMPORT_LINE_RE.test(trimmed) ||
        trimmed.startsWith('//') ||
        trimmed.startsWith('/*') ||
        trimmed.startsWith('*') ||
        trimmed === ')'
      ) {
        headerLines.push(line)
        bodyStart = i + 1
        continue
      }
      inHeader = false
    }
  }

  return { header: headerLines.join('\n').trim(), bodyStart }
}

const splitAtDeclarations = (lines: readonly string[]): readonly string[] => {
  if (lines.length === 0) return []

  const sections: string[] = []
  let current = ''

  for (const line of lines) {
    const trimmed = line.trim()
    if (isDeclarationStart(trimmed) && current.length > 0) {
      sections.push(current)
      current = ''
    }
    current += line + '\n'
  }
  if (current.length > 0) {
    sections.push(current)
  }
  return sections
}

const isDeclarationStart = (line: string): boolean => {
  if (line === '') return false
  return (
    GO_FUNC_RE.test(line) ||
    GO_TYPE_RE.test(line) ||
    PY_DEF_RE.test(line) ||
    TS_FUNC_RE.test(line) ||
    C_FUNC_RE.test(line)
  )
}

const mergeUndersized = (chunks: Chunk[], cfg: ChunkConfig): readonly Chunk[] => {
  if (chunks.length === 0) return []
  const merged: Chunk[] = []
  for (const c of chunks) {
    if (estimateTokens(c.content) < cfg.minTokens && merged.length > 0) {
      const prev = merged[merged.length - 1]
      if (prev !== undefined) {
        merged[merged.length - 1] = {
          ...prev,
          content: prev.content + '\n' + c.content,
        }
        continue
      }
    }
    merged.push(c)
  }
  return merged
}

const recursiveSplitLocal = (text: string, maxTokens: number, sepIdx: number): readonly string[] => {
  if (estimateTokens(text) <= maxTokens) return [text]
  if (sepIdx >= SEPARATORS.length) return hardSplit(text, maxTokens)
  const sep = SEPARATORS[sepIdx]
  if (sep === undefined || sep === '') return hardSplit(text, maxTokens)
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
        merged.push(...recursiveSplitLocal(part, maxTokens, sepIdx + 1))
      } else {
        current = part
      }
    } else {
      current = candidate
    }
  }
  if (current !== '') merged.push(current)
  return merged
}

const hardSplit = (text: string, maxTokens: number): readonly string[] => {
  const step = maxTokens * 4
  const out: string[] = []
  for (let i = 0; i < text.length; i += step) {
    out.push(text.slice(i, i + step))
  }
  return out
}
