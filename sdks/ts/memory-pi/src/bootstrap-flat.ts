// SPDX-License-Identifier: Apache-2.0

/**
 * One-shot indexer that walks an existing brain's content directories
 * (wiki/, memory/, raw/, ...) and populates the memory-pi SQLite FTS
 * index without writing to the Store.
 *
 * Intended for single-brain hosts where the on-disk markdown is the
 * canonical truth (typically a git working tree) and a separate runtime
 * already owns writes. memory-pi's BM25 + vector retrieval becomes a
 * read-only side index over those files. The Store is never touched
 * so source content is not duplicated or rewritten.
 *
 * Idempotent: paths already present in `knowledge_chunks` are skipped on
 * re-entry, so booting an already-bootstrapped brain is a no-op.
 */

import { readFile, readdir, stat } from 'node:fs/promises'
import { join, relative, sep } from 'node:path'
import { chunkMarkdown } from '@jeffs-brain/memory/ingest'
import type { Chunk as IndexChunk, SearchIndex } from '@jeffs-brain/memory/search'
import { type RuntimeLogger, noopRuntimeLogger } from './runtime-logger.js'

export const DEFAULT_BOOTSTRAP_SCAN_DIRS: readonly string[] = ['wiki', 'memory', 'raw']

export const DEFAULT_BOOTSTRAP_EXTENSIONS: readonly string[] = ['.md', '.markdown']

/** Upper bound on file size we are willing to chunk on first boot. */
const MAX_FILE_BYTES = 2 * 1024 * 1024

const BATCH_SIZE = 100

export type BootstrapFlatOptions = {
  readonly brainRoot: string
  readonly brainId: string
  readonly searchIndex: SearchIndex
  readonly scanDirs?: readonly string[]
  readonly extensions?: readonly string[]
  readonly logger?: RuntimeLogger
}

export type BootstrapFlatResult = {
  readonly scanned: number
  readonly indexed: number
  readonly skipped: number
  readonly skippedReasons: Readonly<Record<string, number>>
  readonly durationMs: number
}

const bumpReason = (reasons: Record<string, number>, key: string): void => {
  reasons[key] = (reasons[key] ?? 0) + 1
}

const toPosixRel = (rel: string): string => (sep === '/' ? rel : rel.split(sep).join('/'))

const isFileAccessible = async (path: string): Promise<boolean> => {
  try {
    const info = await stat(path)
    return info.isDirectory()
  } catch {
    return false
  }
}

const extractTitle = (path: string, content: string): string => {
  const trimmed = content.trim()
  // Skip a YAML frontmatter block before searching for headings or the
  // first content line. This keeps title extraction robust against the
  // mix of front-mattered and bare markdown that lives in real brains.
  let body = trimmed
  if (body.startsWith('---')) {
    const end = body.indexOf('\n---', 3)
    if (end >= 0) body = body.slice(end + 4)
  }
  const heading = body.match(/^#\s+(.+)$/m)?.[1]?.trim()
  if (heading !== undefined && heading !== '') return heading
  const firstLine = body
    .split('\n')
    .map((line) => line.trim())
    .find((line) => line !== '')
  if (firstLine !== undefined && firstLine !== '') return firstLine
  const base = path.split('/').pop() ?? path
  return base.replace(/\.(md|markdown)$/i, '')
}

async function* walk(
  root: string,
  extensions: readonly string[],
): AsyncGenerator<string, void, void> {
  const stack: string[] = [root]
  while (stack.length > 0) {
    const current = stack.pop()
    if (current === undefined) continue
    let entries: { name: string; isDir: boolean }[] = []
    try {
      const dirents = await readdir(current, { withFileTypes: true })
      entries = dirents.map((d) => ({ name: d.name, isDir: d.isDirectory() }))
    } catch {
      continue
    }
    for (const entry of entries) {
      if (entry.name.startsWith('.')) continue
      const next = join(current, entry.name)
      if (entry.isDir) {
        stack.push(next)
        continue
      }
      const lower = entry.name.toLowerCase()
      if (extensions.some((ext) => lower.endsWith(ext))) {
        yield next
      }
    }
  }
}

/**
 * Walk the brain content directories and upsert every markdown chunk
 * into the search index. Returns a summary record so the caller can log
 * how much was indexed.
 *
 * Indexing happens directly through `SearchIndex.upsertChunks` — we
 * deliberately bypass the ingest pipeline so we do NOT write copies of
 * the source files back into the brain Store.
 */
export const bootstrapFlatBrain = async (
  options: BootstrapFlatOptions,
): Promise<BootstrapFlatResult> => {
  const logger = options.logger ?? noopRuntimeLogger
  const scanDirs = options.scanDirs ?? DEFAULT_BOOTSTRAP_SCAN_DIRS
  const extensions = options.extensions ?? DEFAULT_BOOTSTRAP_EXTENSIONS
  const start = Date.now()
  const known = new Set(options.searchIndex.indexedPaths())

  let scanned = 0
  let indexed = 0
  let skipped = 0
  const skippedReasons: Record<string, number> = {}
  let batch: IndexChunk[] = []

  const flushBatch = (): void => {
    if (batch.length === 0) return
    options.searchIndex.upsertChunks(batch)
    batch = []
  }

  for (const sub of scanDirs) {
    const dir = join(options.brainRoot, sub)
    if (!(await isFileAccessible(dir))) {
      bumpReason(skippedReasons, `missing:${sub}`)
      continue
    }
    for await (const absPath of walk(dir, extensions)) {
      scanned++
      const rel = toPosixRel(relative(options.brainRoot, absPath))
      if (known.has(rel)) {
        skipped++
        bumpReason(skippedReasons, 'already-indexed')
        continue
      }
      let info
      try {
        info = await stat(absPath)
      } catch {
        skipped++
        bumpReason(skippedReasons, 'stat-failed')
        continue
      }
      if (info.size === 0) {
        skipped++
        bumpReason(skippedReasons, 'empty')
        continue
      }
      if (info.size > MAX_FILE_BYTES) {
        skipped++
        bumpReason(skippedReasons, 'too-large')
        logger.warn('memory-pi bootstrap: skipping oversize file', {
          path: rel,
          bytes: info.size,
        })
        continue
      }
      let content: string
      try {
        content = await readFile(absPath, 'utf8')
      } catch (err) {
        skipped++
        bumpReason(skippedReasons, 'read-failed')
        logger.warn('memory-pi bootstrap: read failed', {
          path: rel,
          err: err instanceof Error ? err.message : String(err),
        })
        continue
      }
      if (content.trim() === '') {
        skipped++
        bumpReason(skippedReasons, 'blank')
        continue
      }
      const sections = chunkMarkdown(content)
      if (sections.length === 0) {
        skipped++
        bumpReason(skippedReasons, 'no-chunks')
        continue
      }
      const title = extractTitle(rel, content)
      for (const section of sections) {
        batch.push({
          id: `${rel}#${section.ordinal}`,
          path: rel,
          ordinal: section.ordinal,
          title,
          content: section.content,
          metadata: {
            source: 'bootstrap-flat',
            brainId: options.brainId,
            headingPath: section.headingPath,
            startLine: section.startLine,
            endLine: section.endLine,
          },
        })
        if (batch.length >= BATCH_SIZE) flushBatch()
      }
      indexed++
    }
  }
  flushBatch()

  const result: BootstrapFlatResult = {
    scanned,
    indexed,
    skipped,
    skippedReasons,
    durationMs: Date.now() - start,
  }
  logger.info('memory-pi bootstrap: complete', {
    ...result,
    brainRoot: options.brainRoot,
    brainId: options.brainId,
    scanDirs,
  })
  return result
}
