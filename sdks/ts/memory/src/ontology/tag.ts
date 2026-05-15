// SPDX-License-Identifier: Apache-2.0

/**
 * Entity tagging for chunk metadata enrichment. Scans chunk content
 * for mentions of known ontology entity types and attaches type
 * metadata for retrieval enrichment.
 *
 * Port of go/ontology/tag.go.
 */

import type { ResolvedOntology } from './store.js'
import type { ClassificationResult } from './classify.js'
import { determineCategory } from './classify.js'

/**
 * ChunkTag is type metadata attached to a chunk for retrieval
 * enrichment.
 */
export type ChunkTag = {
  readonly entityTypes?: readonly string[]
  readonly businessCategory?: string
  readonly confidence: number
  readonly documentClass: string
}

/**
 * Creates a ChunkTag for a given chunk based on the classification
 * result and resolved ontology. Scans the chunk content for mentions
 * of known entity types from the ontology.
 */
export function tagChunk(
  content: string,
  classification: ClassificationResult,
  ontology: ResolvedOntology | undefined,
): ChunkTag {
  let businessCategory = classification.category
  let entityTypes: string[] | undefined

  if (ontology !== undefined) {
    const lower = content.toLowerCase()
    const matched = matchEntityTypes(lower, ontology)
    if (matched.length > 0) {
      entityTypes = matched
    }

    const ontologyCategory = determineCategory(content, ontology)
    if (ontologyCategory !== 'general') {
      businessCategory = ontologyCategory
    }
  }

  return {
    entityTypes,
    businessCategory,
    confidence: classification.confidence,
    documentClass: classification.class,
  }
}

/**
 * Creates ChunkTags for a batch of chunks. Each chunk is
 * independently tagged against the classification and ontology.
 */
export function tagChunks(
  contents: readonly string[],
  classification: ClassificationResult,
  ontology: ResolvedOntology | undefined,
): ChunkTag[] {
  return contents.map((content) => tagChunk(content, classification, ontology))
}

function matchEntityTypes(
  lowerContent: string,
  ontology: ResolvedOntology,
): string[] {
  const seen = new Set<string>()
  const matched: string[] = []

  for (const nt of ontology.nodeTypes) {
    const dotIdx = nt.type.indexOf('.')
    if (dotIdx < 0) continue
    const name = nt.type.slice(dotIdx + 1)
    const spaced = name.replace(/_/g, ' ')

    if (lowerContent.includes(name) || lowerContent.includes(spaced)) {
      if (!seen.has(nt.type)) {
        seen.add(nt.type)
        matched.push(nt.type)
      }
    }
  }

  return matched
}
