// SPDX-License-Identifier: Apache-2.0

/**
 * Promote: move drafts/<slug>.md → wiki/<slug>.md inside a single Batch
 * with Reason: "promote". Ported from apps/jeff/internal/knowledge/
 * promote.go (the wiki-write half; we do not run the LLM curation call
 * because our compile step already produced the article).
 */

import type { Logger } from '../llm/index.js'
import { joinPath, lastSegment, type Path, type Store } from '../store/index.js'
import { appendLogInBatch } from './log.js'
import { normaliseKnowledgeArticleStem } from './paths.js'
import type { PromoteOptions } from './types.js'

export const DRAFTS_PREFIX = 'drafts'
export const WIKI_PREFIX = 'wiki'

type PromoteDeps = {
  store: Store
  logger: Logger
}

export type PromoteResult = {
  from: Path
  to: Path
}

export const createPromote = (deps: PromoteDeps) => {
  const { store, logger } = deps

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

    logger.info('promote', { src, dst })
    return { from: src, to: dst }
  }
}
