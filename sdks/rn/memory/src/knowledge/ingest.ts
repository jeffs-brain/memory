// SPDX-License-Identifier: Apache-2.0

import type { Logger } from '../llm/index.js'
import { type Path, type Store, joinPath } from '../store/index.js'
import { type HashContentInput, contentByteLength, hashContent, toText } from './hash.js'
import { appendLogInBatch } from './log.js'
import type { IngestOptions, IngestResult } from './types.js'

export const RAW_DOCUMENTS_PREFIX = 'raw/documents'

export { hashContent } from './hash.js'

export const rawDocumentPath = (hashOrName: string): Path =>
  joinPath(RAW_DOCUMENTS_PREFIX, `${hashOrName}.md`)

type IngestDeps = {
  store: Store
  logger: Logger
}

export const createIngest = (deps: IngestDeps) => {
  const { store, logger } = deps

  return async (input: HashContentInput, opts: IngestOptions = {}): Promise<IngestResult> => {
    const content = input
    const text = toText(content)
    const hash = hashContent(content)
    const name = opts.name && opts.name !== '' ? slugify(opts.name) : hash
    const path = rawDocumentPath(name)

    const existed = await store.exists(path)
    if (existed) {
      const existing = await store.read(path)
      if (hashContent(existing) === hash) {
        logger.debug('ingest skipped: duplicate', { path, hash })
        return { path, hash, bytes: contentByteLength(content), skipped: 'duplicate' }
      }
    }

    const logTitle = opts.logTitle ?? hash.slice(0, 12)
    const logDetail = opts.logDetail ?? `ingested ${contentByteLength(content)} bytes → ${path}`

    await store.batch({ reason: 'ingest' }, async (batch) => {
      await batch.write(path, text)
      await appendLogInBatch(batch, {
        kind: 'ingest',
        title: logTitle,
        detail: logDetail,
        when: new Date().toISOString(),
      })
    })

    logger.debug('ingest wrote', { path, hash, bytes: contentByteLength(content) })
    return { path, hash, bytes: contentByteLength(content) }
  }
}

const slugify = (value: string): string => {
  const lower = value.toLowerCase().trim()
  const replaced = lower.replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return replaced === '' ? 'untitled' : replaced
}
