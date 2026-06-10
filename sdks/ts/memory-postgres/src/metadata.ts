// SPDX-License-Identifier: Apache-2.0

/**
 * Document metadata extraction for the Postgres store write path.
 *
 * The `memory.documents.metadata` jsonb column drives several of the
 * `document_edges` computations (see `./edges.ts`):
 *   - `computeSharedTagEdges`        reads `metadata->'tags'`
 *   - `computeSameSessionEdges`      reads `metadata->>'session_id'` / `sessionId`
 *   - `computeSupersedesEdges`       reads `metadata->>'supersedes'`
 *   - `computeDocumentOntologyEdges` reads `metadata->>'ontology_type'` / `ontologyType`
 *
 * The base `Store.write(path, content)` contract carries the metadata inside
 * the document's YAML frontmatter — there is no separate metadata channel on
 * the interface. The typed `parseFrontmatter` exported by `@jeffs-brain/memory`
 * deliberately recognises only a fixed field set (title/summary/tags/sources/
 * created/modified/archived/supersededBy) and DROPS every other key, so it
 * cannot surface `ontology_type`, `session_id`, or `supersedes`.
 *
 * This module parses the same YAML-subset fence (matching
 * `knowledge/frontmatter.ts` and the Go `internal/knowledge/frontmatter.go`)
 * but preserves EVERY scalar and list key as a generic record so the edge
 * computations can consume the persisted column. A document with no fenced
 * frontmatter yields an empty object, so existing callers keep writing `{}`
 * exactly as before.
 */

const FENCE = '---'

/**
 * A single frontmatter value. Scalars are kept as parsed strings/booleans;
 * lists are kept as string arrays. Mirrors the jsonb shapes the edge queries
 * read (`->>` for scalars, `jsonb_typeof(... ) = 'array'` for `tags`).
 */
export type DocumentMetadataValue = string | boolean | readonly string[]

/** Generic frontmatter map persisted into `memory.documents.metadata`. */
export type DocumentMetadata = Readonly<Record<string, DocumentMetadataValue>>

const BOOLEAN_RE = /^(true|false|yes|no)$/i

const stripQuotes = (v: string): string => {
  if (v.length < 2) return v
  const first = v[0]
  const last = v[v.length - 1]
  if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
    return v.slice(1, -1)
  }
  return v
}

const parseInlineList = (v: string): string[] => {
  const trimmed = v.trim()
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    const inner = trimmed.slice(1, -1)
    if (inner.trim() === '') return []
    return inner
      .split(',')
      .map((s) => stripQuotes(s.trim()))
      .filter((s) => s !== '')
  }
  return trimmed
    .split(',')
    .map((s) => stripQuotes(s.trim()))
    .filter((s) => s !== '')
}

const coerceScalar = (raw: string): string | boolean => {
  const value = stripQuotes(raw)
  if (BOOLEAN_RE.test(value)) return /^(true|yes)$/i.test(value)
  return value
}

const splitKV = (line: string): readonly [string, string] | undefined => {
  const idx = line.indexOf(':')
  if (idx < 0) return undefined
  const key = line.slice(0, idx).trim()
  const val = line.slice(idx + 1).trim()
  if (key === '') return undefined
  return [key, val]
}

/**
 * Extract a generic metadata map from a document's YAML-subset frontmatter.
 *
 * Returns an empty object when no fenced frontmatter block is present, so a
 * document written with no/empty metadata persists `{}` — byte-identical to
 * the prior behaviour where the column was never written and defaulted to
 * `'{}'::jsonb`.
 *
 * Inline lists (`tags: [a, b]` or `tags: a, b`) and block lists
 * (`tags:`\n`  - a`) both yield string arrays. `true`/`false`/`yes`/`no`
 * scalars coerce to booleans; everything else stays a string.
 *
 * @param content Raw UTF-8 document content.
 * @returns Parsed frontmatter map preserving every key.
 */
export const extractDocumentMetadata = (content: string): DocumentMetadata => {
  const lines = content.split('\n')
  if (lines.length < 2 || lines[0]?.trim() !== FENCE) return {}

  let closeIdx = -1
  for (let i = 1; i < lines.length; i++) {
    if (lines[i]?.trim() === FENCE) {
      closeIdx = i
      break
    }
  }
  if (closeIdx < 0) return {}

  const out: Record<string, DocumentMetadataValue> = {}
  let currentListKey = ''
  const listBuffers = new Map<string, string[]>()

  for (let i = 1; i < closeIdx; i++) {
    const line = lines[i] ?? ''
    const trimmed = line.trim()

    if (currentListKey !== '' && trimmed.startsWith('- ')) {
      const val = stripQuotes(trimmed.slice(2).trim())
      if (val !== '') {
        const buffer = listBuffers.get(currentListKey) ?? []
        buffer.push(val)
        listBuffers.set(currentListKey, buffer)
        out[currentListKey] = buffer
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
      // Bare key → block list follows. Seed an empty list so a list with no
      // items still records the key as an empty array.
      currentListKey = key
      if (!listBuffers.has(key)) {
        const buffer: string[] = []
        listBuffers.set(key, buffer)
        out[key] = buffer
      }
      continue
    }

    currentListKey = ''
    const inlineTrimmed = rawVal.trim()
    if (inlineTrimmed.startsWith('[') && inlineTrimmed.endsWith(']')) {
      out[key] = parseInlineList(inlineTrimmed)
      continue
    }
    out[key] = coerceScalar(rawVal)
  }

  return out
}
