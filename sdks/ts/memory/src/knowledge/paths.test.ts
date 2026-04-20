// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { normaliseKnowledgeArticleStem, normaliseWikiRelativeArticlePath } from './paths.js'

describe('knowledge paths', () => {
  it('normalises a safe article stem', () => {
    expect(normaliseKnowledgeArticleStem('topic/sub-topic.md')).toBe('topic/sub-topic')
  })

  it('normalises a safe wiki-relative path', () => {
    expect(normaliseWikiRelativeArticlePath('wiki/topic/sub-topic')).toBe('topic/sub-topic.md')
  })

  it('rejects prefixed or non-canonical article stems', () => {
    expect(() => normaliseKnowledgeArticleStem('wiki/topic')).toThrow(/relative/)
    expect(() => normaliseKnowledgeArticleStem('../topic')).toThrow(/lowercase kebab-case/)
    expect(() => normaliseKnowledgeArticleStem('Topic')).toThrow(/lowercase kebab-case/)
  })
})
