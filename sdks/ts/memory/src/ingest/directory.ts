// SPDX-License-Identifier: Apache-2.0

/**
 * Directory enumeration for the ingest pipeline. Walks a directory
 * recursively, applies glob filtering, respects .gitignore patterns,
 * and excludes hidden files and symlinks.
 */

import { lstat, readFile, readdir } from 'node:fs/promises'
import { extname, join, normalize, relative, resolve } from 'node:path'

const FILE_SIZE_LIMIT = 25 * 1024 * 1024
const KNOWN_EXTENSIONS = new Set([
  '.md',
  '.markdown',
  '.txt',
  '.text',
  '.pdf',
  '.json',
  '.html',
  '.htm',
  '.csv',
  '.tsv',
  '.xml',
  '.yaml',
  '.yml',
  '.toml',
  '.rst',
  '.adoc',
  '.org',
  '.tex',
  '.rtf',
])

export type EnumerateOptions = {
  readonly directory: string
  readonly glob?: string | undefined
  readonly recursive?: boolean | undefined
  readonly maxFiles?: number | undefined
}

export type EnumeratedFile = {
  readonly path: string
  readonly size: number
}

export type EnumerateResult = {
  readonly files: readonly EnumeratedFile[]
  readonly skipped: readonly string[]
}

/**
 * Parse a simple .gitignore file into an array of patterns.
 * Supports basic patterns: exact matches, leading /, trailing /,
 * leading *, and negation is ignored for simplicity.
 */
const parseGitignore = (content: string): readonly string[] =>
  content
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line !== '' && !line.startsWith('#') && !line.startsWith('!'))

/**
 * Check whether a relative path matches any of the gitignore patterns.
 * This is a simplified matcher that handles the most common patterns.
 */
const matchesGitignore = (relativePath: string, patterns: readonly string[]): boolean => {
  const segments = relativePath.split('/')
  for (const pattern of patterns) {
    const cleaned = pattern.replace(/\/$/, '')
    // Direct segment match (e.g. "node_modules" matches any segment)
    if (segments.some((seg) => seg === cleaned)) return true
    // Pattern with wildcard prefix (e.g. "*.log")
    if (cleaned.startsWith('*') && relativePath.endsWith(cleaned.slice(1))) return true
    // Pattern starting with / means root-relative (with separator boundary)
    if (cleaned.startsWith('/')) {
      const trimmed = cleaned.slice(1)
      if (relativePath === trimmed || relativePath.startsWith(trimmed + '/')) return true
    }
    // Direct prefix match (with separator boundary)
    if (relativePath === cleaned || relativePath.startsWith(cleaned + '/')) return true
  }
  return false
}

/**
 * Check whether a filename matches a glob pattern.
 * Supports simple glob patterns: * (any chars), ? (single char), ** (recursive).
 */
const matchesGlob = (filename: string, pattern: string): boolean => {
  // Strip leading **/ for recursive matching
  const normalized = pattern.replace(/^\*\*\//, '')
  const regexStr = normalized
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*\*/g, '__DOUBLESTAR__')
    .replace(/\*/g, '[^/]*')
    .replace(/__DOUBLESTAR__/g, '.*')
    .replace(/\?/g, '[^/]')
  const regex = new RegExp(`^${regexStr}$`)
  return regex.test(filename) || regex.test(filename.split('/').pop() ?? '')
}

/**
 * Enumerate files in a directory suitable for ingestion.
 *
 * - Excludes hidden files (dot-prefix)
 * - Does not follow symlinks
 * - Respects .gitignore patterns when present
 * - Applies glob filter if provided
 * - Enforces maxFiles limit
 * - Skips files over 25 MiB
 */
export const enumerateFiles = async (opts: EnumerateOptions): Promise<EnumerateResult> => {
  const maxFiles = opts.maxFiles ?? 100
  const recursive = opts.recursive ?? true
  const cleanedDirectory = resolve(normalize(opts.directory))
  const files: EnumeratedFile[] = []
  const skipped: string[] = []

  // Try to load .gitignore
  let gitignorePatterns: readonly string[] = []
  try {
    const gitignoreContent = await readFile(join(cleanedDirectory, '.gitignore'), 'utf8')
    gitignorePatterns = parseGitignore(gitignoreContent)
  } catch {
    // No .gitignore or unreadable — proceed without patterns
  }

  const walk = async (dir: string): Promise<void> => {
    if (files.length >= maxFiles) return

    let entries: Awaited<ReturnType<typeof readdir>>
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      skipped.push(`${dir}: ${message}`)
      return
    }

    for (const entry of entries) {
      if (files.length >= maxFiles) {
        skipped.push(`max files limit (${maxFiles}) reached`)
        return
      }

      // Skip hidden files/directories
      if (entry.name.startsWith('.')) continue

      const fullPath = join(dir, entry.name)
      const normalizedFull = resolve(normalize(fullPath))
      const relativePath = relative(cleanedDirectory, normalizedFull)

      // Boundary check: ensure path stays within the directory root.
      if (relativePath.startsWith('..')) continue

      // Check .gitignore patterns
      if (gitignorePatterns.length > 0 && matchesGitignore(relativePath, gitignorePatterns)) {
        continue
      }

      // Skip symlinks
      try {
        const fileStat = await lstat(fullPath)
        if (fileStat.isSymbolicLink()) continue

        if (fileStat.isDirectory()) {
          if (recursive) {
            await walk(fullPath)
          }
          continue
        }

        if (!fileStat.isFile()) continue

        // Check file size
        if (fileStat.size > FILE_SIZE_LIMIT) {
          skipped.push(`${relativePath}: exceeds 25 MiB limit (${Math.round(fileStat.size / 1024 / 1024)} MiB)`)
          continue
        }

        // Check glob filter
        if (opts.glob !== undefined && !matchesGlob(relativePath, opts.glob)) {
          continue
        }

        // Check if file has a known text extension (skip binary files)
        const ext = extname(entry.name).toLowerCase()
        if (ext !== '' && !KNOWN_EXTENSIONS.has(ext)) {
          continue
        }

        files.push({ path: fullPath, size: fileStat.size })
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        skipped.push(`${relativePath}: ${message}`)
      }
    }
  }

  await walk(cleanedDirectory)
  return { files, skipped }
}
