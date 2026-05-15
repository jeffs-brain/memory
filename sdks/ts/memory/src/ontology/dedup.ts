// SPDX-License-Identifier: Apache-2.0

/**
 * Three-tier ontology type deduplication: exact ID match, fuzzy label
 * match (Jaro-Winkler >= 0.85 within same prefix), and semantic
 * embedding match (cosine >= 0.9 auto-merge, >= 0.75 review).
 *
 * Port of apps/intelligence-service/src/services/document-processing/
 * ontology-type-deduplicator.ts adapted for the standalone memory
 * package.
 */

import type { Embedder } from '../llm/types.js'
import type { TypeDefinition } from './types.js'

import { cosineSimilarity, jaroWinklerDistance } from './similarity.js'

export const FUZZY_LABEL_THRESHOLD = 0.85
export const EMBEDDING_AUTO_MERGE_THRESHOLD = 0.9
export const EMBEDDING_REVIEW_THRESHOLD = 0.75

export type DedupResultKind = 'exact' | 'fuzzy_match' | 'semantic_match' | 'semantic_review' | 'new'

export type DedupResult = {
  readonly kind: DedupResultKind
  readonly existingType?: TypeDefinition
  readonly similarity: number
}

export type DedupMethod = 'exact' | 'fuzzy_label' | 'embedding'

export type MergedPair = {
  readonly extracted: TypeDefinition
  readonly existingMatch: TypeDefinition
  readonly similarity: number
  readonly method: DedupMethod
}

export type DeduplicationResult = {
  readonly unique: TypeDefinition[]
  readonly autoMerged: MergedPair[]
  readonly reviewCandidates: MergedPair[]
}

/**
 * A pluggable string similarity function. Accepts two strings and
 * returns a similarity value in [0, 1].
 */
export type SimilarityFn = (a: string, b: string) => number

export type DeduplicatorConfig = {
  readonly embedder?: Embedder
  /** Minimum string similarity for fuzzy label matching. Default: 0.85. */
  readonly fuzzyThreshold?: number
  /** Minimum cosine similarity for automatic semantic merging. Default: 0.9. */
  readonly autoMergeThreshold?: number
  /** Minimum cosine similarity to flag for human review. Default: 0.75. */
  readonly reviewThreshold?: number
  /** Custom string similarity function. Default: jaroWinklerDistance. */
  readonly similarity?: SimilarityFn
}

/**
 * Deduplicator performs three-tier type deduplication against an
 * existing type registry. Thresholds and similarity algorithm are
 * configurable via DeduplicatorConfig.
 *
 * Time: O(E*X) for exact+fuzzy tiers; semantic tier adds embedding
 * cost proportional to batch sizes.
 */
export class Deduplicator {
  private readonly embedder: Embedder | undefined
  private readonly fuzzyThreshold: number
  private readonly autoMergeThreshold: number
  private readonly reviewThreshold: number
  private readonly similarityFn: SimilarityFn

  constructor(config: DeduplicatorConfig) {
    this.embedder = config.embedder
    this.fuzzyThreshold = config.fuzzyThreshold ?? FUZZY_LABEL_THRESHOLD
    this.autoMergeThreshold = config.autoMergeThreshold ?? EMBEDDING_AUTO_MERGE_THRESHOLD
    this.reviewThreshold = config.reviewThreshold ?? EMBEDDING_REVIEW_THRESHOLD
    this.similarityFn = config.similarity ?? jaroWinklerDistance
  }

  async deduplicate(
    extracted: readonly TypeDefinition[],
    existing: readonly TypeDefinition[],
    signal?: AbortSignal,
  ): Promise<DeduplicationResult> {
    const unique: TypeDefinition[] = []
    const autoMerged: MergedPair[] = []
    const reviewCandidates: MergedPair[] = []

    if (extracted.length === 0) {
      return { unique, autoMerged, reviewCandidates }
    }
    if (existing.length === 0) {
      unique.push(...extracted)
      return { unique, autoMerged, reviewCandidates }
    }

    const existingByID = buildTypeIDIndex(existing)
    const existingByPrefix = groupByPrefix(existing)

    const afterExact = deduplicateExact(extracted, existingByID, autoMerged)
    const afterFuzzy = deduplicateFuzzy(afterExact, existingByPrefix, autoMerged, this.fuzzyThreshold, this.similarityFn)

    if (this.embedder === undefined || afterFuzzy.length === 0) {
      unique.push(...afterFuzzy)
      return { unique, autoMerged, reviewCandidates }
    }

    await this.deduplicateSemantic(afterFuzzy, existing, unique, autoMerged, reviewCandidates, signal)
    return { unique, autoMerged, reviewCandidates }
  }

  private async deduplicateSemantic(
    candidates: readonly TypeDefinition[],
    existing: readonly TypeDefinition[],
    unique: TypeDefinition[],
    autoMerged: MergedPair[],
    reviewCandidates: MergedPair[],
    signal?: AbortSignal,
  ): Promise<void> {
    const candidateTexts = candidates.map((c) => `${c.label}: ${c.description}`)
    const existingTexts = existing.map((e) => `${e.label}: ${e.description}`)

    const [candidateEmbeddings, existingEmbeddings] = await Promise.all([
      this.embedder!.embed(candidateTexts, signal),
      this.embedder!.embed(existingTexts, signal),
    ])

    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i]!
      const candidateVec = candidateEmbeddings[i]
      if (candidateVec === undefined) {
        unique.push(candidate)
        continue
      }

      let bestSimilarity = 0
      let bestMatch: TypeDefinition | undefined

      for (let j = 0; j < existing.length; j++) {
        const existingVec = existingEmbeddings[j]
        if (existingVec === undefined) continue
        const similarity = cosineSimilarity(candidateVec, existingVec)
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity
          bestMatch = existing[j]
        }
      }

      if (bestMatch !== undefined && bestSimilarity >= this.autoMergeThreshold) {
        autoMerged.push({
          extracted: candidate,
          existingMatch: bestMatch,
          similarity: bestSimilarity,
          method: 'embedding',
        })
      } else if (bestMatch !== undefined && bestSimilarity >= this.reviewThreshold) {
        reviewCandidates.push({
          extracted: candidate,
          existingMatch: bestMatch,
          similarity: bestSimilarity,
          method: 'embedding',
        })
      } else {
        unique.push(candidate)
      }
    }
  }
}

function deduplicateExact(
  extracted: readonly TypeDefinition[],
  existingByID: Map<string, TypeDefinition>,
  autoMerged: MergedPair[],
): TypeDefinition[] {
  const remaining: TypeDefinition[] = []
  for (const entry of extracted) {
    const match = existingByID.get(entry.type)
    if (match !== undefined) {
      autoMerged.push({
        extracted: entry,
        existingMatch: match,
        similarity: 1.0,
        method: 'exact',
      })
      continue
    }
    remaining.push(entry)
  }
  return remaining
}

function deduplicateFuzzy(
  candidates: readonly TypeDefinition[],
  existingByPrefix: Map<string, TypeDefinition[]>,
  autoMerged: MergedPair[],
  threshold: number = FUZZY_LABEL_THRESHOLD,
  similarityFn: SimilarityFn = jaroWinklerDistance,
): TypeDefinition[] {
  const remaining: TypeDefinition[] = []
  for (const entry of candidates) {
    const prefix = typePrefix(entry.type)
    const samePrefix = existingByPrefix.get(prefix)
    if (samePrefix === undefined || samePrefix.length === 0) {
      remaining.push(entry)
      continue
    }

    let matched = false
    for (const existingEntry of samePrefix) {
      const similarity = similarityFn(entry.label, existingEntry.label)
      if (similarity >= threshold) {
        autoMerged.push({
          extracted: entry,
          existingMatch: existingEntry,
          similarity,
          method: 'fuzzy_label',
        })
        matched = true
        break
      }
    }

    if (!matched) {
      remaining.push(entry)
    }
  }
  return remaining
}

function buildTypeIDIndex(types: readonly TypeDefinition[]): Map<string, TypeDefinition> {
  const index = new Map<string, TypeDefinition>()
  for (const t of types) {
    index.set(t.type, t)
  }
  return index
}

function groupByPrefix(types: readonly TypeDefinition[]): Map<string, TypeDefinition[]> {
  const groups = new Map<string, TypeDefinition[]>()
  for (const t of types) {
    const prefix = typePrefix(t.type)
    const group = groups.get(prefix)
    if (group !== undefined) {
      group.push(t)
    } else {
      groups.set(prefix, [t])
    }
  }
  return groups
}

function typePrefix(typeID: string): string {
  const dotIndex = typeID.indexOf('.')
  if (dotIndex < 0) return typeID
  return typeID.slice(0, dotIndex)
}

/**
 * Evaluate a single candidate type against a list of existing types
 * using three tiers: exact ID match, fuzzy label match (Jaro-Winkler
 * >= 0.85 within same prefix), and semantic embedding match (cosine
 * >= 0.9 auto-merge, >= 0.75 review). Pass undefined for embedder to
 * skip the semantic tier.
 *
 * Time: O(X) for exact+fuzzy; semantic tier adds embedding cost.
 */
export async function deduplicateType(
  candidate: TypeDefinition,
  existing: readonly TypeDefinition[],
  embedder?: Embedder,
  signal?: AbortSignal,
): Promise<DedupResult> {
  if (existing.length === 0) {
    return { kind: 'new', similarity: 0 }
  }

  for (const entry of existing) {
    if (candidate.type === entry.type) {
      return { kind: 'exact', existingType: entry, similarity: 1.0 }
    }
  }

  const candidatePrefix = typePrefix(candidate.type)
  for (const entry of existing) {
    if (typePrefix(entry.type) !== candidatePrefix) continue
    const similarity = jaroWinklerDistance(candidate.label, entry.label)
    if (similarity >= FUZZY_LABEL_THRESHOLD) {
      return { kind: 'fuzzy_match', existingType: entry, similarity }
    }
  }

  if (embedder === undefined) {
    return { kind: 'new', similarity: 0 }
  }

  const candidateText = `${candidate.label}: ${candidate.description}`
  const existingTexts = existing.map((e) => `${e.label}: ${e.description}`)

  const [candidateEmbeddings, existingEmbeddings] = await Promise.all([
    embedder.embed([candidateText], signal),
    embedder.embed(existingTexts, signal),
  ])

  const candidateVec = candidateEmbeddings[0]
  if (candidateVec === undefined) {
    return { kind: 'new', similarity: 0 }
  }

  let bestSimilarity = 0
  let bestEntry: TypeDefinition | undefined

  for (let j = 0; j < existing.length; j++) {
    const existingVec = existingEmbeddings[j]
    if (existingVec === undefined) continue
    const sim = cosineSimilarity(candidateVec, existingVec)
    if (sim > bestSimilarity) {
      bestSimilarity = sim
      bestEntry = existing[j]
    }
  }

  if (bestEntry !== undefined && bestSimilarity >= EMBEDDING_AUTO_MERGE_THRESHOLD) {
    return { kind: 'semantic_match', existingType: bestEntry, similarity: bestSimilarity }
  }
  if (bestEntry !== undefined && bestSimilarity >= EMBEDDING_REVIEW_THRESHOLD) {
    return { kind: 'semantic_review', existingType: bestEntry, similarity: bestSimilarity }
  }
  return { kind: 'new', similarity: 0 }
}
