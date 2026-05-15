// SPDX-License-Identifier: Apache-2.0

/**
 * Template matching: score extracted ontology types against all 6
 * industry templates and suggest the best match. Supports exact set
 * intersection and semantic embedding similarity with greedy bipartite
 * matching.
 *
 * Port of apps/intelligence-service/src/services/document-processing/
 * ontology-extraction.service.ts (matchAgainstTemplatesExact and
 * matchAgainstTemplatesSemantic).
 */

import type { Embedder } from '../llm/types.js'
import type { TypeEntry, IndustryTemplate } from './templates.js'
import type { ExtractionResult } from './extract.js'

import { listTemplates, getTemplate } from './templates.js'
import { cosineSimilarity } from './similarity.js'

export type { ExtractionResult } from './extract.js'

/** Minimum cosine similarity for a semantic embedding pair to count as a match. */
export const TEMPLATE_MATCH_SEMANTIC_THRESHOLD = 0.8

/** Minimum combined score (overlap + coverage) / 2 for a suggestion. */
export const TEMPLATE_MATCH_COMBINED_MINIMUM = 0.3

export type TemplateSuggestion = {
  readonly templateKey: string
  readonly templateLabel: string
  readonly overlapScore: number
  readonly coverageScore: number
  readonly additionalTypes: TypeEntry[]
  readonly missingFromTemplate: TypeEntry[]
}

export type TemplateMatcherOptions = {
  readonly embedder?: Embedder
  /** Override the default semantic threshold (0.8). */
  readonly semanticThreshold?: number
  /** Override the default combined minimum (0.3). */
  readonly combinedMinimum?: number
}

/**
 * TemplateMatcher suggests industry templates based on extracted types.
 * Uses exact set intersection by default, and semantic embedding similarity
 * when an embedder is provided.
 */
export class TemplateMatcher {
  private readonly embedder: Embedder | undefined
  private readonly semanticThreshold: number
  private readonly combinedMinimum: number
  private readonly embeddingCache = new Map<string, readonly (readonly number[])[]>()

  constructor(options: TemplateMatcherOptions) {
    this.embedder = options.embedder
    this.semanticThreshold = options.semanticThreshold ?? TEMPLATE_MATCH_SEMANTIC_THRESHOLD
    this.combinedMinimum = options.combinedMinimum ?? TEMPLATE_MATCH_COMBINED_MINIMUM
  }

  /**
   * Find the best-matching industry template for the given extraction.
   * Uses semantic matching when an embedder is available, falling back to exact.
   * Returns undefined if no template scores above the combined threshold.
   */
  async match(
    extracted: ExtractionResult,
    signal?: AbortSignal,
  ): Promise<TemplateSuggestion | undefined> {
    if (extracted.nodeTypes.length === 0 && extracted.edgeTypes.length === 0) {
      return undefined
    }

    if (this.embedder === undefined) {
      return this.matchExact(extracted)
    }

    return this.matchSemantic(extracted, signal)
  }

  /**
   * Perform set-intersection-only matching (no embeddings).
   * Returns undefined if no template scores above the combined threshold.
   */
  matchExact(extracted: ExtractionResult): TemplateSuggestion | undefined {
    if (extracted.nodeTypes.length === 0 && extracted.edgeTypes.length === 0) {
      return undefined
    }

    const extractedNodeSet = new Map<string, TypeEntry>()
    for (const nt of extracted.nodeTypes) {
      extractedNodeSet.set(nt.type, nt)
    }
    const extractedEdgeSet = new Map<string, TypeEntry>()
    for (const et of extracted.edgeTypes) {
      extractedEdgeSet.set(et.type, et)
    }

    const extractedCount = extracted.nodeTypes.length + extracted.edgeTypes.length

    let best: TemplateSuggestion | undefined
    let bestCombined = 0

    const keys = listTemplates()
    for (const key of keys) {
      const tmpl = getTemplate(key)
      if (tmpl === undefined) continue

      const templateNodeSet = new Map<string, TypeEntry>()
      for (const nt of tmpl.nodeTypes) {
        templateNodeSet.set(nt.type, nt)
      }
      const templateEdgeSet = new Map<string, TypeEntry>()
      for (const et of tmpl.edgeTypes) {
        templateEdgeSet.set(et.type, et)
      }

      const templateCount = tmpl.nodeTypes.length + tmpl.edgeTypes.length
      if (templateCount === 0) continue

      let intersectionCount = 0
      for (const typeId of extractedNodeSet.keys()) {
        if (templateNodeSet.has(typeId)) intersectionCount++
      }
      for (const typeId of extractedEdgeSet.keys()) {
        if (templateEdgeSet.has(typeId)) intersectionCount++
      }

      const overlapScore = extractedCount > 0 ? intersectionCount / extractedCount : 0
      const coverageScore = intersectionCount / templateCount
      const combined = (overlapScore + coverageScore) / 2

      if (combined >= this.combinedMinimum && combined > bestCombined) {
        const additional = computeAdditionalTypes(extracted, templateNodeSet, templateEdgeSet)
        const missing = computeMissingTypes(tmpl, extractedNodeSet, extractedEdgeSet)

        bestCombined = combined
        best = {
          templateKey: key,
          templateLabel: tmpl.label,
          overlapScore,
          coverageScore,
          additionalTypes: additional,
          missingFromTemplate: missing,
        }
      }
    }

    return best
  }

  private async matchSemantic(
    extracted: ExtractionResult,
    signal?: AbortSignal,
  ): Promise<TemplateSuggestion | undefined> {
    const extractedTypes = combineTypes(extracted.nodeTypes, extracted.edgeTypes)
    if (extractedTypes.length === 0) return undefined

    const extractedTexts = extractedTypes.map((t) => `${t.label}: ${t.description}`)
    const extractedEmbeddings = await this.embedder!.embed(extractedTexts, signal)

    let best: TemplateSuggestion | undefined
    let bestCombined = 0

    const keys = listTemplates()
    for (const key of keys) {
      const tmpl = getTemplate(key)
      if (tmpl === undefined) continue

      const templateTypes = combineTypes(tmpl.nodeTypes, tmpl.edgeTypes)
      if (templateTypes.length === 0) continue

      let templateEmbeddings = this.embeddingCache.get(key)
      if (templateEmbeddings === undefined) {
        const templateTexts = templateTypes.map((t) => `${t.label}: ${t.description}`)
        templateEmbeddings = await this.embedder!.embed(templateTexts, signal)
        this.embeddingCache.set(key, templateEmbeddings)
      }

      const matchCount = greedyBipartiteMatch(
        extractedEmbeddings,
        templateEmbeddings,
        this.semanticThreshold,
      )

      const extractedCount = extractedTypes.length
      const templateCount = templateTypes.length

      const overlapScore = extractedCount > 0 ? matchCount / extractedCount : 0
      const coverageScore = templateCount > 0 ? matchCount / templateCount : 0
      const combined = (overlapScore + coverageScore) / 2

      if (combined >= this.combinedMinimum && combined > bestCombined) {
        const extractedNodeSet = new Map<string, TypeEntry>()
        for (const nt of extracted.nodeTypes) {
          extractedNodeSet.set(nt.type, nt)
        }
        const extractedEdgeSet = new Map<string, TypeEntry>()
        for (const et of extracted.edgeTypes) {
          extractedEdgeSet.set(et.type, et)
        }
        const templateNodeSet = new Map<string, TypeEntry>()
        for (const nt of tmpl.nodeTypes) {
          templateNodeSet.set(nt.type, nt)
        }
        const templateEdgeSet = new Map<string, TypeEntry>()
        for (const et of tmpl.edgeTypes) {
          templateEdgeSet.set(et.type, et)
        }

        const additional = computeAdditionalTypes(extracted, templateNodeSet, templateEdgeSet)
        const missing = computeMissingTypes(tmpl, extractedNodeSet, extractedEdgeSet)

        bestCombined = combined
        best = {
          templateKey: key,
          templateLabel: tmpl.label,
          overlapScore,
          coverageScore,
          additionalTypes: additional,
          missingFromTemplate: missing,
        }
      }
    }

    return best
  }
}

/**
 * Greedy bipartite matching: for each pair, compute cosine similarity.
 * Sort by similarity descending, greedily assign each pair if both sides
 * are unmatched. Only counts pairs with similarity >= threshold.
 */
function greedyBipartiteMatch(
  extractedVecs: readonly (readonly number[])[],
  templateVecs: readonly (readonly number[])[],
  threshold: number,
): number {
  if (extractedVecs.length === 0 || templateVecs.length === 0) return 0

  const pairs: Array<{ extractedIdx: number; templateIdx: number; similarity: number }> = []

  for (let i = 0; i < extractedVecs.length; i++) {
    const ev = extractedVecs[i]
    if (ev === undefined) continue
    for (let j = 0; j < templateVecs.length; j++) {
      const tv = templateVecs[j]
      if (tv === undefined) continue
      try {
        const sim = cosineSimilarity(ev, tv)
        if (sim >= threshold) {
          pairs.push({ extractedIdx: i, templateIdx: j, similarity: sim })
        }
      } catch {
        // Skip if vectors have different lengths
      }
    }
  }

  pairs.sort((a, b) => b.similarity - a.similarity)

  const usedExtracted = new Set<number>()
  const usedTemplate = new Set<number>()
  let matchCount = 0

  for (const p of pairs) {
    if (usedExtracted.has(p.extractedIdx) || usedTemplate.has(p.templateIdx)) continue
    usedExtracted.add(p.extractedIdx)
    usedTemplate.add(p.templateIdx)
    matchCount++
  }

  return matchCount
}

function computeAdditionalTypes(
  extracted: ExtractionResult,
  templateNodeSet: Map<string, TypeEntry>,
  templateEdgeSet: Map<string, TypeEntry>,
): TypeEntry[] {
  const additional: TypeEntry[] = []
  for (const nt of extracted.nodeTypes) {
    if (!templateNodeSet.has(nt.type)) additional.push(nt)
  }
  for (const et of extracted.edgeTypes) {
    if (!templateEdgeSet.has(et.type)) additional.push(et)
  }
  return additional
}

function computeMissingTypes(
  tmpl: IndustryTemplate,
  extractedNodeSet: Map<string, TypeEntry>,
  extractedEdgeSet: Map<string, TypeEntry>,
): TypeEntry[] {
  const missing: TypeEntry[] = []
  for (const nt of tmpl.nodeTypes) {
    if (!extractedNodeSet.has(nt.type)) missing.push(nt)
  }
  for (const et of tmpl.edgeTypes) {
    if (!extractedEdgeSet.has(et.type)) missing.push(et)
  }
  return missing
}

function combineTypes(
  nodeTypes: readonly TypeEntry[],
  edgeTypes: readonly TypeEntry[],
): TypeEntry[] {
  return [...nodeTypes, ...edgeTypes]
}
