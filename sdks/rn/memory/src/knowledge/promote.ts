// SPDX-License-Identifier: Apache-2.0

import type { Logger } from '../llm/index.js'
import { type Path, type Store, joinPath, lastSegment } from '../store/index.js'
import { createKnowledgeEventEmitter } from './events.js'
import { appendLogInBatch } from './log.js'
import { normaliseKnowledgeArticleStem } from './paths.js'
import type { KnowledgeEventSink, PromoteOptions } from './types.js'

export const DRAFTS_PREFIX = 'drafts'
export const WIKI_PREFIX = 'wiki'

type PromoteDeps = {
  store: Store
  logger: Logger
  onEvent?: KnowledgeEventSink
}

export type PromoteResult = {
  from: Path
  to: Path
}

export const createPromote = (deps: PromoteDeps) => {
  const { store, logger } = deps
  const emitEvent = createKnowledgeEventEmitter({
    logger,
    ...(deps.onEvent !== undefined ? { onEvent: deps.onEvent } : {}),
  })

  return async (slug: string, opts: PromoteOptions = {}): Promise<PromoteResult> => {
    const articleStem = normaliseKnowledgeArticleStem(slug)
    const filename = `${articleStem}.md`
    const src = joinPath(DRAFTS_PREFIX, filename)
    const dst = joinPath(WIKI_PREFIX, filename)

    if (!(await store.exists(src))) {
      throw new Error(`promote: draft not found: ${src}`)
    }

    const content = await store.read(src)
    const detail = opts.detail ?? `promoted ${src} → ${dst}`

    await store.batch({ reason: 'promote' }, async (batch) => {
      await batch.write(dst, content)
      await batch.delete(src)
      await appendLogInBatch(batch, {
        kind: 'promote',
        title: lastSegment(dst),
        detail,
        when: new Date().toISOString(),
      })
    })

    await emitEvent({
      kind: 'promotion.landed',
      slug: articleStem,
      from: src,
      to: dst,
      detail,
      when: new Date().toISOString(),
    })

    logger.info('promote', { src, dst })
    return { from: src, to: dst }
  }
}
