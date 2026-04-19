// SPDX-License-Identifier: Apache-2.0

/**
 * Raw-content ingest. Produces a raw/documents/<hash>.md entry and a matching
 * operations-log line in a single Batch with Reason: "ingest". Ported from
 * apps/jeff/internal/knowledge/ingest.go (the ingestInBatch helper).
 */

import { createHash } from 'node:crypto'
import { joinPath, type Path, type Store } from '../store/index.js'
import type { Logger } from '../llm/index.js'
import { appendLogInBatch } from './log.js'
import type { IngestOptions, IngestResult } from './types.js'

export const RAW_DOCUMENTS_PREFIX = 'raw/documents'

export const hashContent = (buf: Buffer): string =>
  createHash('sha256').update(buf).digest('hex')

export const rawDocumentPath = (hashOrName: string): Path =>
  joinPath(RAW_DOCUMENTS_PREFIX, `${hashOrName}.md`)

type IngestDeps = {
  store: Store
  logger: Logger
}

export const createIngest = (deps: IngestDeps) => {
  const { store, logger } = deps

  return async (
    input: string | Buffer,
    opts: IngestOptions = {},
  ): Promise<IngestResult> => {
    const content = typeof input === 'string' ? Buffer.from(input, 'utf8') : input
    const hash = hashContent(content)
    const name = opts.name && opts.name !== '' ? slugify(opts.name) : hash
    const path = rawDocumentPath(name)

    const existed = await store.exists(path)
    if (existed) {
      // Same content, same name -> deterministic skip. Same name, different
      // content is surfaced by the Store write later (overwrite semantics
      // vary by backend). By default we still overwrite because the Go
      // ingest dedup is checksum-based and a hit here means either the
      // same payload or a deliberate re-name, both safe to overwrite.
      const existing = await store.read(path)
      if (hashContent(existing) === hash) {
        logger.debug('ingest skipped: duplicate', { path, hash })
        return { path, hash, bytes: content.length, skipped: 'duplicate' }
      }
    }

    const logTitle = opts.logTitle ?? hash.slice(0, 12)
    const logDetail = opts.logDetail ?? `ingested ${content.length} bytes → ${path}`

    await store.batch({ reason: 'ingest' }, async (batch) => {
      await batch.write(path, content)
      await appendLogInBatch(batch, {
        kind: 'ingest',
        title: logTitle,
        detail: logDetail,
        when: new Date().toISOString(),
      })
    })

    logger.debug('ingest wrote', { path, hash, bytes: content.length })
    return { path, hash, bytes: content.length }
  }
}

const slugify = (s: string): string => {
  const lower = s.toLowerCase().trim()
  const replaced = lower.replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return replaced === '' ? 'untitled' : replaced
}
