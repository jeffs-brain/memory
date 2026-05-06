// SPDX-License-Identifier: Apache-2.0

/**
 * Regex-based classification of implicit user feedback on surfaced memories.
 *
 * Consumed by the memory reinforcement loop. The classifier is intentionally
 * conservative so a false positive never demotes a good memory.
 *
 * Ported from Go: go/memory/feedback/classifier.go
 */

import type { Path } from '../../store/index.js'

/** Classifies implicit user feedback on surfaced memories. */
export type Reaction = 'reinforced' | 'corrected' | 'neutral'

/** Records a detected reaction on a specific memory. */
export type FeedbackEvent = {
  readonly memoryPath: Path
  readonly reaction: Reaction
  readonly confidence: number
  readonly pattern: string
  readonly snippet: string
}

/** Holds the classification outcome for one turn. */
export type ClassifyResult = {
  readonly events: readonly FeedbackEvent[]
  readonly turnContent: string
}

/** Detects implicit user feedback from the next user turn. */
export type FeedbackClassifier = {
  classify(userInput: string, surfacedThisTurn: readonly Path[]): ClassifyResult
}

/** Patterns that detect reinforcement signals. */
const positivePatterns: readonly string[] = [
  '\\b(perfect|exactly|great|thanks|correct|right|yes)\\b',
  "\\bthat('s| is| was) (right|correct|helpful|useful|what i needed)\\b",
  '\\b(good|nice) (memory|recall|find)\\b',
  '\\byou remembered\\b',
  '\\bthat helps\\b',
  '\\bspot on\\b',
]

/** Patterns that detect correction signals. */
const negativePatterns: readonly string[] = [
  '\\b(wrong|incorrect|no|nope|not right)\\b',
  "\\bthat('s| is| was) (wrong|incorrect|outdated|old|stale)\\b",
  '\\b(forget|remove|delete) (that|this|it)\\b',
  '\\bnot what i (meant|asked|wanted)\\b',
  '\\btry again\\b',
  "\\bthat('s| is) (not|no longer) (true|accurate|relevant)\\b",
  '\\bactually[,.]?\\s',
]

const compilePatterns = (patterns: readonly string[]): readonly RegExp[] =>
  patterns.reduce<RegExp[]>((acc, p) => {
    try {
      acc.push(new RegExp(p, 'i'))
    } catch {
      // Skip invalid patterns — mirrors Go's silent skip on compile error.
    }
    return acc
  }, [])

const clamp = (v: number): number => (v > 1.0 ? 1.0 : v)

const truncateSnippet = (s: string, n: number): string =>
  s.length <= n ? s : `${s.slice(0, n)}...`

const detectReaction = (
  input: string,
  positive: readonly RegExp[],
  negative: readonly RegExp[],
): { reaction: Reaction; confidence: number; pattern: string } => {
  let posMatches = 0
  let posPattern = ''
  for (const r of positive) {
    const match = r.exec(input)
    if (match) {
      posMatches++
      if (posPattern === '') {
        posPattern = match[0]
      }
    }
  }

  let negMatches = 0
  let negPattern = ''
  for (const r of negative) {
    const match = r.exec(input)
    if (match) {
      negMatches++
      if (negPattern === '') {
        negPattern = match[0]
      }
    }
  }

  if (posMatches === 0 && negMatches === 0) {
    return { reaction: 'neutral', confidence: 0.0, pattern: '' }
  }

  if (posMatches > negMatches) {
    return { reaction: 'reinforced', confidence: clamp(posMatches * 0.3), pattern: posPattern }
  }

  if (negMatches > posMatches) {
    return { reaction: 'corrected', confidence: clamp(negMatches * 0.3), pattern: negPattern }
  }

  // Tie goes to neutral.
  return { reaction: 'neutral', confidence: 0.2, pattern: '' }
}

/** Creates a regex-based feedback classifier. */
export const createFeedbackClassifier = (): FeedbackClassifier => {
  const positive = compilePatterns(positivePatterns)
  const negative = compilePatterns(negativePatterns)

  return {
    classify(userInput: string, surfacedThisTurn: readonly Path[]): ClassifyResult {
      const turnContent = truncateSnippet(userInput, 500)

      if (surfacedThisTurn.length === 0 || userInput.trim() === '') {
        return { events: [], turnContent }
      }

      const { reaction, confidence, pattern } = detectReaction(userInput, positive, negative)
      const snippet = truncateSnippet(userInput, 200)

      const events: FeedbackEvent[] = surfacedThisTurn.map((memoryPath) => ({
        memoryPath,
        reaction,
        confidence,
        pattern,
        snippet,
      }))

      return { events, turnContent }
    },
  }
}
