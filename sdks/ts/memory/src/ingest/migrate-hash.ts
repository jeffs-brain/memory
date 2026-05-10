// SPDX-License-Identifier: Apache-2.0

/**
 * SHA-256 to BLAKE3 hash migration for the ingest pipeline. Provides a
 * non-blocking, resumable migrator that re-hashes existing documents
 * and a dual-read resolver that tries BLAKE3 first with SHA-256 fallback.
 */

import type { Path, Store } from '../store/index.js'
import { isNotFound, joinPath, toPath } from '../store/index.js'
import { hashContent, hashContentSHA256, hashSlug, hashSlugSHA256 } from './hash.js'

const MIGRATION_STATE_PATH = 'raw/.pipeline-state/.blake3-migration.json'
const RAW_DOCUMENTS_PREFIX = 'raw/documents'
const DEFAULT_BATCH_SIZE = 100

export type MigrateOpts = {
  readonly batchSize?: number
  readonly dryRun?: boolean
  readonly cursor?: string
}

export type MigrateResult = {
  readonly migrated: number
  readonly skipped: number
  readonly total: number
  readonly nextCursor: string
}

type MigrationState = {
  readonly cursor: string
  readonly migrated: number
  readonly total: number
}

type HashMigrator = {
  migrate(opts?: MigrateOpts, signal?: AbortSignal): Promise<MigrateResult>
}

/**
 * Create a hash migrator bound to the given store. The migrator
 * processes documents in batches, renaming SHA-256-hashed files to
 * BLAKE3-hashed paths. Migration is non-blocking and resumable.
 */
export const createHashMigrator = (store: Store): HashMigrator => {
  const loadState = async (): Promise<MigrationState> => {
    try {
      const data = await store.read(toPath(MIGRATION_STATE_PATH))
      return JSON.parse(data.toString('utf8')) as MigrationState
    } catch (err: unknown) {
      if (isNotFound(err)) {
        return { cursor: '', migrated: 0, total: 0 }
      }
      throw err
    }
  }

  const saveState = async (state: MigrationState): Promise<void> => {
    const data = Buffer.from(JSON.stringify(state), 'utf8')
    await store.write(toPath(MIGRATION_STATE_PATH), data)
  }

  const rawDocumentPath = (slug: string): Path => joinPath(RAW_DOCUMENTS_PREFIX, `${slug}.md`)

  const extractSlug = (p: Path): string => {
    const s = String(p)
    const prefix = `${RAW_DOCUMENTS_PREFIX}/`
    if (!s.startsWith(prefix)) return ''
    const name = s.slice(prefix.length)
    if (!name.endsWith('.md')) return ''
    return name.slice(0, -3)
  }

  const migrateDocument = async (docPath: Path, dryRun: boolean): Promise<boolean> => {
    const slug = extractSlug(docPath)
    if (slug === '') return false

    const content = await store.read(docPath)
    const blake3SlugValue = hashSlug(content)

    if (slug === blake3SlugValue) return false

    if (dryRun) return true

    const newPath = rawDocumentPath(blake3SlugValue)
    if (String(docPath) === String(newPath)) return false

    await store.batch({ reason: 'blake3-migration' }, async (batch) => {
      await batch.write(newPath, content)
      await batch.delete(docPath)
    })
    return true
  }

  const migrate = async (opts?: MigrateOpts, signal?: AbortSignal): Promise<MigrateResult> => {
    const batchSize = opts?.batchSize ?? DEFAULT_BATCH_SIZE
    const dryRun = opts?.dryRun ?? false

    let cursor = opts?.cursor ?? ''
    if (cursor === '') {
      const state = await loadState()
      if (state.cursor !== '') {
        cursor = state.cursor
      }
    }

    const entries = await store.list(toPath(RAW_DOCUMENTS_PREFIX), {
      recursive: true,
      includeGenerated: true,
    })

    const docPaths = entries
      .filter((e) => !e.isDir && String(e.path).endsWith('.md'))
      .map((e) => e.path)
      .sort()

    const total = docPaths.length
    const startIdx = cursorIndex(docPaths, cursor)
    const endIdx = Math.min(startIdx + batchSize, total)

    let migrated = 0
    let skipped = 0

    for (let i = startIdx; i < endIdx; i++) {
      if (signal?.aborted) {
        throw new Error('ingest: migration aborted')
      }

      const docPath = docPaths[i]
      if (docPath === undefined) break

      const didMigrate = await migrateDocument(docPath, dryRun)
      if (didMigrate) {
        migrated++
      } else {
        skipped++
      }
    }

    const nextCursor = endIdx < total ? String(docPaths[endIdx] ?? '') : ''

    if (!dryRun) {
      await saveState({ cursor: nextCursor, migrated, total })
    }

    return { migrated, skipped, total, nextCursor }
  }

  return { migrate }
}

/**
 * Dual-read hash resolver. Computes the BLAKE3 hash of content and
 * checks whether a document exists under that hash. If not found, falls
 * back to the SHA-256 hash. Returns the resolved hash slug, or the
 * BLAKE3 hash for new documents that do not exist under either scheme.
 */
export const resolveHash = async (store: Store, content: Buffer): Promise<string> => {
  const blake3SlugValue = hashSlug(content)
  const blake3Path = joinPath(RAW_DOCUMENTS_PREFIX, `${blake3SlugValue}.md`)

  const blake3Exists = await store.exists(toPath(blake3Path))
  if (blake3Exists) return blake3SlugValue

  const sha256SlugValue = hashSlugSHA256(content)
  const sha256Path = joinPath(RAW_DOCUMENTS_PREFIX, `${sha256SlugValue}.md`)

  const sha256Exists = await store.exists(toPath(sha256Path))
  if (sha256Exists) return sha256SlugValue

  return blake3SlugValue
}

const cursorIndex = (paths: readonly Path[], cursor: string): number => {
  if (cursor === '') return 0
  for (let i = 0; i < paths.length; i++) {
    if (String(paths[i]) >= cursor) return i
  }
  return paths.length
}
