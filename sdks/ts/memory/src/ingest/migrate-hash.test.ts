// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { createMemStore, toPath } from '../store/index.js'
import { hashSlug, hashSlugSHA256 } from './hash.js'
import { createHashMigrator, resolveHash } from './migrate-hash.js'

const RAW_DOCUMENTS_PREFIX = 'raw/documents'

const rawDocPath = (slug: string) => toPath(`${RAW_DOCUMENTS_PREFIX}/${slug}.md`)

describe('createHashMigrator', () => {
  it('migrates empty store with 0 migrated', async () => {
    const store = createMemStore()
    const migrator = createHashMigrator(store)

    const result = await migrator.migrate()
    expect(result.migrated).toBe(0)
    expect(result.total).toBe(0)
    expect(result.nextCursor).toBe('')
  })

  it('migrates documents from SHA-256 to BLAKE3 hashes', async () => {
    const store = createMemStore()

    const docs = [
      Buffer.from('document one content', 'utf8'),
      Buffer.from('document two content', 'utf8'),
      Buffer.from('document three content', 'utf8'),
    ]

    for (const content of docs) {
      const sha256Slug = hashSlugSHA256(content)
      await store.write(rawDocPath(sha256Slug), content)
    }

    const migrator = createHashMigrator(store)
    const result = await migrator.migrate({ batchSize: 10 })

    expect(result.migrated).toBe(3)
    expect(result.total).toBe(3)

    for (const content of docs) {
      const blake3SlugValue = hashSlug(content)
      const newExists = await store.exists(rawDocPath(blake3SlugValue))
      expect(newExists).toBe(true)

      const sha256SlugValue = hashSlugSHA256(content)
      const oldExists = await store.exists(rawDocPath(sha256SlugValue))
      expect(oldExists).toBe(false)
    }
  })

  it('resumes migration from saved cursor', async () => {
    const store = createMemStore()

    const docs = [
      Buffer.from('alpha content', 'utf8'),
      Buffer.from('beta content', 'utf8'),
      Buffer.from('gamma content', 'utf8'),
      Buffer.from('delta content', 'utf8'),
      Buffer.from('epsilon content', 'utf8'),
    ]

    for (const content of docs) {
      const sha256Slug = hashSlugSHA256(content)
      await store.write(rawDocPath(sha256Slug), content)
    }

    const migrator = createHashMigrator(store)

    const result1 = await migrator.migrate({ batchSize: 2 })
    expect(result1.migrated).toBe(2)
    expect(result1.total).toBe(5)
    expect(result1.nextCursor).not.toBe('')

    // Second pass resumes from saved state.
    const result2 = await migrator.migrate({ batchSize: 10 })
    const totalProcessed = result2.migrated + result2.skipped
    expect(totalProcessed).toBeGreaterThan(0)
  })

  it('dry-run reports but does not write', async () => {
    const store = createMemStore()

    const content = Buffer.from('dry run test content', 'utf8')
    const sha256Slug = hashSlugSHA256(content)
    const originalPath = rawDocPath(sha256Slug)
    await store.write(originalPath, content)

    const migrator = createHashMigrator(store)
    const result = await migrator.migrate({ dryRun: true })

    expect(result.migrated).toBe(1)

    // Original still exists.
    const originalExists = await store.exists(originalPath)
    expect(originalExists).toBe(true)

    // BLAKE3 path not created.
    const blake3SlugValue = hashSlug(content)
    const newExists = await store.exists(rawDocPath(blake3SlugValue))
    expect(newExists).toBe(false)

    // State file not written.
    const stateExists = await store.exists(toPath('raw/.pipeline-state/.blake3-migration.json'))
    expect(stateExists).toBe(false)
  })
})

describe('resolveHash', () => {
  it('finds document by BLAKE3 hash first', async () => {
    const store = createMemStore()

    const content = Buffer.from('dual-read test content', 'utf8')
    const blake3SlugValue = hashSlug(content)
    const sha256SlugValue = hashSlugSHA256(content)

    // Write at both paths.
    await store.write(rawDocPath(blake3SlugValue), content)
    await store.write(rawDocPath(sha256SlugValue), content)

    const resolved = await resolveHash(store, content)
    expect(resolved).toBe(blake3SlugValue)
  })

  it('falls back to SHA-256 when BLAKE3 not found', async () => {
    const store = createMemStore()

    const content = Buffer.from('legacy content sha256 only', 'utf8')
    const sha256SlugValue = hashSlugSHA256(content)
    await store.write(rawDocPath(sha256SlugValue), content)

    const resolved = await resolveHash(store, content)
    expect(resolved).toBe(sha256SlugValue)
  })

  it('returns BLAKE3 hash for new documents', async () => {
    const store = createMemStore()

    const content = Buffer.from('brand new document', 'utf8')
    const blake3SlugValue = hashSlug(content)

    const resolved = await resolveHash(store, content)
    expect(resolved).toBe(blake3SlugValue)
  })
})
