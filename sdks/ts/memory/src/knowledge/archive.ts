// SPDX-License-Identifier: Apache-2.0

import {
  joinPath,
  pathUnder,
  toPath,
  type FileInfo,
  type Path,
  type Store,
} from '../store/index.js'
import { INGESTED_PREFIX } from './ingest.js'

export const INGESTED_ARCHIVE_PREFIX = joinPath(INGESTED_PREFIX, '_archived')

export type SourceArchiveStats = {
  readonly fileCount: number
  readonly totalBytes: number
  readonly oldestFile?: Date
  readonly newestFile?: Date
}

export type PruneArchivedSourcesOptions = {
  readonly retentionMs: number
  readonly dryRun?: boolean
  readonly now?: Date
}

export type PruneArchivedSourcesResult = {
  readonly pruned: number
  readonly paths: readonly Path[]
}

type SourceArchiveDeps = {
  readonly store: Store
}

export const archivedSourcePath = (sourceId: string): Path =>
  joinPath(INGESTED_ARCHIVE_PREFIX, `${sourceId}.md`)

export const createSourceArchive = (deps: SourceArchiveDeps) => {
  const { store } = deps

  return {
    info: async (): Promise<SourceArchiveStats> => {
      const entries = await listArchivedSourceEntries(store)
      let totalBytes = 0
      let oldestFile: Date | undefined
      let newestFile: Date | undefined

      for (const entry of entries) {
        totalBytes += entry.size
        if (oldestFile === undefined || entry.modTime < oldestFile) {
          oldestFile = entry.modTime
        }
        if (newestFile === undefined || entry.modTime > newestFile) {
          newestFile = entry.modTime
        }
      }

      return {
        fileCount: entries.length,
        totalBytes,
        ...(oldestFile !== undefined ? { oldestFile } : {}),
        ...(newestFile !== undefined ? { newestFile } : {}),
      }
    },

    prune: async (
      opts: PruneArchivedSourcesOptions,
    ): Promise<PruneArchivedSourcesResult> => {
      if (!Number.isFinite(opts.retentionMs) || opts.retentionMs <= 0) {
        throw new Error(
          `archive: retentionMs must be a positive number, got ${String(opts.retentionMs)}`,
        )
      }

      const cutoff = (opts.now ?? new Date()).getTime() - opts.retentionMs
      const entries = await listArchivedSourceEntries(store)
      const paths = entries
        .filter((entry) => entry.modTime.getTime() < cutoff)
        .map((entry) => entry.path)
        .sort((left, right) => left.localeCompare(right))

      if (opts.dryRun === true || paths.length === 0) {
        return { pruned: paths.length, paths }
      }

      await store.batch(
        {
          reason: 'sources-prune',
          message: `pruned ${String(paths.length)} archived source file(s)`,
        },
        async (batch) => {
          for (const path of paths) {
            await batch.delete(path)
          }
        },
      )

      return { pruned: paths.length, paths }
    },
  }
}

const listArchivedSourceEntries = async (
  store: Store,
): Promise<readonly FileInfo[]> => {
  const root = toPath(INGESTED_ARCHIVE_PREFIX)
  const exists = await store.exists(root).catch(() => false)
  if (!exists) return []

  const entries = await store.list(root, { recursive: true })
  return entries.filter(
    (entry) =>
      !entry.isDir &&
      pathUnder(entry.path, INGESTED_ARCHIVE_PREFIX, true) &&
      entry.path.endsWith('.md'),
  )
}
