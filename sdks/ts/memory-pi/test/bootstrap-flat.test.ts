// SPDX-License-Identifier: Apache-2.0

/**
 * Bootstrap walks an on-disk brain layout and populates the FTS index
 * without writing anything back to the Store. Tests pin behaviour for
 * the happy path, idempotency, edge cases (empty files, missing dirs),
 * and the title extraction heuristic.
 */

import { mkdtempSync, rmSync } from 'node:fs'
import { mkdir, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createSearchIndex } from '@jeffs-brain/memory/search'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { bootstrapFlatBrain } from '../src/bootstrap-flat.js'

let brainRoot: string

const writeAt = async (rel: string, body: string): Promise<void> => {
  const abs = join(brainRoot, rel)
  await mkdir(join(abs, '..'), { recursive: true })
  await writeFile(abs, body, 'utf8')
}

beforeEach(() => {
  brainRoot = mkdtempSync(join(tmpdir(), 'memory-pi-bootstrap-'))
})

afterEach(() => {
  rmSync(brainRoot, { recursive: true, force: true })
})

describe('bootstrapFlatBrain', () => {
  it('indexes markdown files across wiki, memory, and raw', async () => {
    await writeAt(
      'wiki/people/sample-user.md',
      '# Sample User\n\nProduct manager at example.com.\n',
    )
    await writeAt(
      'memory/heuristic-naming.md',
      '---\nname: naming\nscope: global\n---\n\n# Naming heuristic\n\nPrefer descriptive identifiers.\n',
    )
    await writeAt('raw/2026-05-15-meeting.md', 'Sprint review notes.\n')

    const idx = await createSearchIndex({ dbPath: ':memory:' })
    try {
      const report = await bootstrapFlatBrain({
        brainRoot,
        brainId: 'test-brain',
        searchIndex: idx,
      })

      expect(report.scanned).toBe(3)
      expect(report.indexed).toBe(3)
      expect(report.skipped).toBe(0)

      const indexed = idx.indexedPaths()
      expect(indexed.sort()).toEqual([
        'memory/heuristic-naming.md',
        'raw/2026-05-15-meeting.md',
        'wiki/people/sample-user.md',
      ])

      const hits = idx.searchBM25('Sample User', 10)
      expect(hits.length).toBeGreaterThan(0)
      expect(hits[0]?.chunk.path).toBe('wiki/people/sample-user.md')
    } finally {
      await idx.close()
    }
  })

  it('is idempotent across re-entries', async () => {
    await writeAt('memory/heuristic-foo.md', '# Heuristic foo\n\nbody\n')

    const idx = await createSearchIndex({ dbPath: ':memory:' })
    try {
      const first = await bootstrapFlatBrain({
        brainRoot,
        brainId: 'test-brain',
        searchIndex: idx,
      })
      expect(first.indexed).toBe(1)

      const second = await bootstrapFlatBrain({
        brainRoot,
        brainId: 'test-brain',
        searchIndex: idx,
      })
      expect(second.indexed).toBe(0)
      expect(second.skipped).toBe(1)
      expect(second.skippedReasons['already-indexed']).toBe(1)
    } finally {
      await idx.close()
    }
  })

  it('skips empty files, hidden directories, and non-markdown extensions', async () => {
    await writeAt('wiki/empty.md', '')
    await writeAt('wiki/.obsidian/cache.md', '# Hidden\n\nshould be skipped\n')
    await writeAt('wiki/keep.md', '# Keep\n\nthis should be indexed\n')
    await writeAt('wiki/binary.bin', 'unrelated')
    await writeAt('memory/notes.txt', 'plain text, no extension match')

    const idx = await createSearchIndex({ dbPath: ':memory:' })
    try {
      const report = await bootstrapFlatBrain({
        brainRoot,
        brainId: 'test-brain',
        searchIndex: idx,
      })

      const indexed = idx.indexedPaths()
      expect(indexed).toEqual(['wiki/keep.md'])
      // wiki/empty.md is zero bytes, so it trips the `empty` branch
      // before we even read it.
      expect(report.skippedReasons.empty).toBe(1)
    } finally {
      await idx.close()
    }
  })

  it('reports missing scan directories without crashing', async () => {
    const idx = await createSearchIndex({ dbPath: ':memory:' })
    try {
      const report = await bootstrapFlatBrain({
        brainRoot,
        brainId: 'test-brain',
        searchIndex: idx,
        scanDirs: ['wiki', 'memory', 'does-not-exist'],
      })
      expect(report.scanned).toBe(0)
      expect(report.indexed).toBe(0)
      expect(report.skippedReasons['missing:does-not-exist']).toBe(1)
      expect(report.skippedReasons['missing:wiki']).toBe(1)
      expect(report.skippedReasons['missing:memory']).toBe(1)
    } finally {
      await idx.close()
    }
  })

  it('extracts a sensible title even when frontmatter is present', async () => {
    await writeAt(
      'memory/user-fact-sample.md',
      '---\nname: sample-user\nscope: global\ntype: user-fact\n---\n\n# Sample user role\n\nThe sample user is a product manager.\n',
    )
    const idx = await createSearchIndex({ dbPath: ':memory:' })
    try {
      await bootstrapFlatBrain({
        brainRoot,
        brainId: 'test-brain',
        searchIndex: idx,
      })
      const hits = idx.searchBM25('sample', 5)
      expect(hits[0]?.chunk.title).toBe('Sample user role')
    } finally {
      await idx.close()
    }
  })
})
