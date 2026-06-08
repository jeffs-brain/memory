// SPDX-License-Identifier: Apache-2.0

/**
 * Codec priors adapter for the memory-note extraction path.
 *
 * Codec priors are the project's canonical vocabulary (the codec
 * `ontology:` block) expressed as plain string lists. This module turns
 * them into a bounded, deduplicated, truncated SOFT priors block that is
 * appended to the extraction system prompt so the extractor reuses the
 * project's canonical entity / relation / term names.
 *
 * Design invariants:
 * - Pure and side-effect free; no I/O, no provider calls.
 * - Backward compatible: an absent or wholly empty priors object yields
 *   an empty block, so the extraction prompt is byte-identical to the
 *   no-priors baseline.
 * - Bounded output: per-list item count, per-item length, and total
 *   character budget are all capped so priors can never grow the prompt
 *   without limit.
 * - Deduplication is case-insensitive and O(n) using a Set; first
 *   occurrence wins so caller ordering is preserved.
 * - Malformed input (non-array list, non-string item) raises a typed
 *   error rather than crashing or silently coercing.
 */

import type { CodecPriors } from './types.js'

/** Maximum number of items rendered per prior list. */
export const CODEC_PRIORS_MAX_ITEMS_PER_LIST = 64
/** Maximum rendered length of a single prior item, in characters. */
export const CODEC_PRIORS_MAX_ITEM_LENGTH = 80
/** Overall character budget for the rendered priors block body. */
export const CODEC_PRIORS_MAX_BLOCK_CHARS = 4000

/** Typed error raised when codec priors are structurally malformed. */
export class CodecPriorsError extends Error {
  constructor(message: string) {
    super(`memory: invalid codec priors: ${message}`)
    this.name = 'CodecPriorsError'
  }
}

type PriorList = {
  readonly heading: string
  readonly label: keyof CodecPriors
}

const PRIOR_LISTS: readonly PriorList[] = [
  { heading: 'Entities', label: 'entities' },
  { heading: 'Relations', label: 'relations' },
  { heading: 'Domain terms', label: 'domainTerms' },
] as const

/**
 * Build the soft known-entity priors block appended to the extraction
 * system prompt. Returns an empty string when there is nothing to render
 * (no priors, or all lists empty after sanitisation), guaranteeing the
 * baseline prompt is unchanged.
 *
 * @throws {CodecPriorsError} when a list is not an array of strings.
 */
export const buildCodecPriorsBlock = (priors?: CodecPriors): string => {
  if (priors === undefined) return ''

  const sections: string[] = []
  let remainingBudget = CODEC_PRIORS_MAX_BLOCK_CHARS

  for (const { heading, label } of PRIOR_LISTS) {
    const items = sanitisePriorList(label, priors[label])
    if (items.length === 0) continue

    const lines: string[] = []
    for (const item of items) {
      const line = `- ${item}`
      if (line.length > remainingBudget) break
      lines.push(line)
      remainingBudget -= line.length
    }
    if (lines.length === 0) continue
    sections.push(`### ${heading}\n${lines.join('\n')}`)
  }

  if (sections.length === 0) return ''

  return [
    '',
    '## Project codec priors',
    'The following are the project’s canonical entities, relations, and domain terms. Treat them as SOFT hints: when a saved fact references one of these, prefer the canonical name below. Do NOT invent facts to match them, and do NOT discard durable knowledge that falls outside this list.',
    '',
    ...sections,
  ].join('\n')
}

/**
 * Compose the effective extraction system prompt from the verbatim base
 * prompt and the optional codec priors block.
 */
export const applyCodecPriors = (basePrompt: string, priors?: CodecPriors): string => {
  const block = buildCodecPriorsBlock(priors)
  return block === '' ? basePrompt : `${basePrompt}\n${block}`
}

const sanitisePriorList = (
  label: keyof CodecPriors,
  value: CodecPriors[keyof CodecPriors],
): readonly string[] => {
  if (value === undefined) return []
  if (!Array.isArray(value)) {
    throw new CodecPriorsError(`${label} must be an array of strings`)
  }

  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of value) {
    if (typeof raw !== 'string') {
      throw new CodecPriorsError(`${label} must contain only strings`)
    }
    if (/[\n\r]/.test(raw)) {
      throw new CodecPriorsError(`${label} must not contain line breaks`)
    }
    const trimmed = raw.trim()
    if (trimmed === '') continue
    const item = truncateItem(trimmed)
    const key = item.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(item)
    if (out.length >= CODEC_PRIORS_MAX_ITEMS_PER_LIST) break
  }
  return out
}

const truncateItem = (value: string): string => {
  if (value.length <= CODEC_PRIORS_MAX_ITEM_LENGTH) return value
  return value.slice(0, CODEC_PRIORS_MAX_ITEM_LENGTH)
}
