// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm, utimes } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { createFsStore } from '../store/fsstore.js'
import { createMemStore } from '../store/memstore.js'
import {
  archivedSourcePath,
  createSourceArchive,
  INGESTED_ARCHIVE_PREFIX,
} from './archive.js'

const createdDirs: string[] = []

const makeTempDir = async (): Promise<string> => {
  const dir = await mkdtemp(join(tmpdir(), 'memory-archive-'))
  createdDirs.push(dir)
  return dir
}

afterEach(async () => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir !== undefined) await rm(dir, { recursive: true, force: true })
  }
})

describe('source archive', () => {
  it('reports file count, bytes, and oldest or newest for archived sources', async () => {
    const store = createMemStore()
    await store.write(archivedSourcePath('alpha'), Buffer.from('one', 'utf8'))
    await store.write(archivedSourcePath('beta'), Buffer.from('three', 'utf8'))

    const archive = createSourceArchive({ store })
    const stats = await archive.info()

    expect(stats.fileCount).toBe(2)
    expect(stats.totalBytes).toBe(8)
    expect(stats.oldestFile).toBeDefined()
    expect(stats.newestFile).toBeDefined()
  })

  it('prunes archived sources older than the retention window', async () => {
    const dir = await makeTempDir()
    const store = await createFsStore({ root: dir })
    const oldPath = archivedSourcePath('old')
    const freshPath = archivedSourcePath('fresh')
    await store.write(oldPath, Buffer.from('old', 'utf8'))
    await store.write(freshPath, Buffer.from('fresh', 'utf8'))

    const oldLocal = store.localPath(oldPath)
    const freshLocal = store.localPath(freshPath)
    if (oldLocal === undefined || freshLocal === undefined) {
      throw new Error('expected fs store to expose local paths')
    }

    await utimes(
      oldLocal,
      new Date('2026-01-01T00:00:00.000Z'),
      new Date('2026-01-01T00:00:00.000Z'),
    )
    await utimes(
      freshLocal,
      new Date('2026-04-17T00:00:00.000Z'),
      new Date('2026-04-17T00:00:00.000Z'),
    )

    const archive = createSourceArchive({ store })
    const dryRun = await archive.prune({
      retentionMs: 30 * 24 * 60 * 60 * 1000,
      dryRun: true,
      now: new Date('2026-04-18T00:00:00.000Z'),
    })
    expect(dryRun).toEqual({
      pruned: 1,
      paths: [oldPath],
    })
    expect(await store.exists(oldPath)).toBe(true)

    const applied = await archive.prune({
      retentionMs: 30 * 24 * 60 * 60 * 1000,
      now: new Date('2026-04-18T00:00:00.000Z'),
    })
    expect(applied).toEqual({
      pruned: 1,
      paths: [oldPath],
    })
    expect(await store.exists(oldPath)).toBe(false)
    expect(await store.exists(freshPath)).toBe(true)
  })

  it('returns empty stats when no archived sources exist', async () => {
    const store = createMemStore()
    const archive = createSourceArchive({ store })

    expect(await archive.info()).toEqual({
      fileCount: 0,
      totalBytes: 0,
    })
    expect(
      await archive.prune({
        retentionMs: 1000,
        dryRun: true,
      }),
    ).toEqual({
      pruned: 0,
      paths: [],
    })
    expect(await store.exists(INGESTED_ARCHIVE_PREFIX)).toBe(false)
  })
})
