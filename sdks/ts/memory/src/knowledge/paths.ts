// SPDX-License-Identifier: Apache-2.0

import { ErrInvalidPath } from '../store/index.js'

const ARTICLE_SEGMENT_RE = /^[a-z0-9][a-z0-9-]*$/

const trimMarkdownExtension = (value: string): string =>
  value.toLowerCase().endsWith('.md') ? value.slice(0, -'.md'.length) : value

const splitArticleSegments = (value: string): string[] => {
  const trimmed = value.trim()
  if (trimmed === '') throw new ErrInvalidPath('empty article path')
  if (trimmed.startsWith('/') || trimmed.endsWith('/')) {
    throw new ErrInvalidPath(`article path must not start or end with "/": ${value}`)
  }
  if (trimmed.includes('\\')) {
    throw new ErrInvalidPath(`article path must not contain backslash: ${value}`)
  }

  const stem = trimMarkdownExtension(trimmed)
  if (stem === '') throw new ErrInvalidPath('empty article path')
  if (stem.startsWith('wiki/') || stem.startsWith('drafts/')) {
    throw new ErrInvalidPath(`article path must be relative, not prefixed: ${value}`)
  }

  const segments = stem.split('/')
  for (const segment of segments) {
    if (!ARTICLE_SEGMENT_RE.test(segment)) {
      throw new ErrInvalidPath(
        `article path segment must be lowercase kebab-case: ${segment}`,
      )
    }
  }
  return segments
}

export const normaliseKnowledgeArticleStem = (value: string): string =>
  splitArticleSegments(value).join('/')

export const tryNormaliseKnowledgeArticleStem = (
  value: string,
): string | undefined => {
  try {
    return normaliseKnowledgeArticleStem(value)
  } catch (error) {
    if (error instanceof ErrInvalidPath) return undefined
    throw error
  }
}

export const normaliseWikiRelativeArticlePath = (value: string): string => {
  const trimmed = value.trim()
  if (trimmed === '') throw new ErrInvalidPath('empty article path')
  if (trimmed.startsWith('drafts/')) {
    throw new ErrInvalidPath(`article path must not target drafts/: ${value}`)
  }
  const relative = trimmed.startsWith('wiki/') ? trimmed.slice('wiki/'.length) : trimmed
  return `${normaliseKnowledgeArticleStem(relative)}.md`
}

export const tryNormaliseWikiRelativeArticlePath = (
  value: string,
): string | undefined => {
  try {
    return normaliseWikiRelativeArticlePath(value)
  } catch (error) {
    if (error instanceof ErrInvalidPath) return undefined
    throw error
  }
}
