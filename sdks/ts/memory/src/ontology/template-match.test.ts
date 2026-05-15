// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'

import type { Embedder } from '../llm/types.js'

import { getTemplate } from './templates.js'
import {
  TemplateMatcher,
  TEMPLATE_MATCH_COMBINED_MINIMUM,
  type ExtractionResult,
  type TemplateSuggestion,
} from './template-match.js'

function makeExtraction(
  nodeTypes: Array<{ type: string; label: string; description: string }>,
  edgeTypes: Array<{ type: string; label: string; description: string }>,
  categories: string[] = [],
): ExtractionResult {
  return {
    nodeTypes,
    edgeTypes,
    businessCategories: categories,
    domain: 'test',
    confidence: 0.8,
  }
}

function createFakeEmbedder(dimensions: number): Embedder {
  return {
    name: () => 'fake-embedder',
    model: () => 'fake-model',
    dimension: () => dimensions,
    embed: async (texts: readonly string[]): Promise<number[][]> => {
      // Create deterministic vectors based on text hash
      return texts.map((text) => {
        const vec = new Array<number>(dimensions).fill(0)
        for (let i = 0; i < text.length && i < dimensions; i++) {
          vec[i] = (text.charCodeAt(i) % 100) / 100
        }
        // Normalise
        const mag = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0))
        if (mag > 0) {
          for (let i = 0; i < vec.length; i++) {
            vec[i] = vec[i]! / mag
          }
        }
        return vec
      })
    },
  }
}

describe('TemplateMatcher.matchExact', () => {
  it('returns suggestion with perfect overlap for exact template match', () => {
    const matcher = new TemplateMatcher({})
    const tmpl = getTemplate('server_hardware')!

    const extracted = makeExtraction([...tmpl.nodeTypes], [...tmpl.edgeTypes])
    const suggestion = matcher.matchExact(extracted)

    expect(suggestion).not.toBeUndefined()
    expect(suggestion!.templateKey).toBe('server_hardware')
    expect(suggestion!.overlapScore).toBe(1.0)
    expect(suggestion!.coverageScore).toBe(1.0)
  })

  it('calculates correct scores for partial overlap', () => {
    const matcher = new TemplateMatcher({})
    const tmpl = getTemplate('healthcare')!

    const extracted = makeExtraction(
      [...tmpl.nodeTypes.slice(0, 5)],
      [...tmpl.edgeTypes.slice(0, 3)],
    )

    const suggestion = matcher.matchExact(extracted)

    expect(suggestion).not.toBeUndefined()
    expect(suggestion!.templateKey).toBe('healthcare')
    // All 8 extracted types match (8/8 = 1.0 overlap)
    expect(suggestion!.overlapScore).toBeCloseTo(1.0, 2)
    // 8 out of 45 template types
    expect(suggestion!.coverageScore).toBeGreaterThan(0.1)
    expect(suggestion!.coverageScore).toBeLessThan(0.5)
  })

  it('returns undefined when no overlap exceeds threshold', () => {
    const matcher = new TemplateMatcher({})
    const extracted = makeExtraction(
      [
        { type: 'entity.alien_species', label: 'Alien Species', description: 'An extraterrestrial' },
        { type: 'entity.space_station', label: 'Space Station', description: 'An orbital station' },
      ],
      [{ type: 'orbits', label: 'Orbits', description: 'Orbital relationship' }],
    )

    const suggestion = matcher.matchExact(extracted)
    expect(suggestion).toBeUndefined()
  })

  it('selects the highest-scoring template from all 6', () => {
    const matcher = new TemplateMatcher({})

    const extracted = makeExtraction(
      [
        { type: 'entity.account', label: 'Account', description: 'A financial account' },
        { type: 'entity.payment', label: 'Payment', description: 'A transfer of funds' },
        { type: 'rule.kyc', label: 'KYC', description: 'Know your customer' },
        { type: 'rule.aml', label: 'AML', description: 'Anti-money laundering' },
        { type: 'entity.policy', label: 'Policy', description: 'An insurance policy' },
      ],
      [{ type: 'authorises', label: 'Authorises', description: 'Authorisation' }],
    )

    const suggestion = matcher.matchExact(extracted)

    expect(suggestion).not.toBeUndefined()
    // 5 of 6 match finance, 1 matches insurance
    expect(suggestion!.templateKey).toBe('finance')
  })

  it('correctly identifies additional types in extracted but not template', () => {
    const matcher = new TemplateMatcher({})
    const tmpl = getTemplate('order_processing')!

    const extracted = makeExtraction(
      [
        ...tmpl.nodeTypes,
        { type: 'entity.custom_thing', label: 'Custom Thing', description: 'Not in template' },
      ],
      [...tmpl.edgeTypes],
    )

    const suggestion = matcher.matchExact(extracted)

    expect(suggestion).not.toBeUndefined()
    const customType = suggestion!.additionalTypes.find((t) => t.type === 'entity.custom_thing')
    expect(customType).not.toBeUndefined()
  })

  it('correctly identifies missing types in template but not extracted', () => {
    const matcher = new TemplateMatcher({})
    const tmpl = getTemplate('order_processing')!

    // Use only first 3 node types
    const extracted = makeExtraction(
      [...tmpl.nodeTypes.slice(0, 3)],
      [...tmpl.edgeTypes],
    )

    const suggestion = matcher.matchExact(extracted)

    expect(suggestion).not.toBeUndefined()
    expect(suggestion!.missingFromTemplate.length).toBeGreaterThan(0)

    const missingTypes = new Set(suggestion!.missingFromTemplate.map((t) => t.type))
    for (const nt of tmpl.nodeTypes.slice(3)) {
      expect(missingTypes.has(nt.type)).toBe(true)
    }
  })

  it('returns undefined for empty extraction', () => {
    const matcher = new TemplateMatcher({})
    const suggestion = matcher.matchExact(makeExtraction([], []))
    expect(suggestion).toBeUndefined()
  })
})

describe('TemplateMatcher.match', () => {
  it('falls back to exact matching when no embedder is provided', async () => {
    const matcher = new TemplateMatcher({})
    const tmpl = getTemplate('server_hardware')!

    const extracted = makeExtraction([...tmpl.nodeTypes], [...tmpl.edgeTypes])
    const suggestion = await matcher.match(extracted)

    expect(suggestion).not.toBeUndefined()
    expect(suggestion!.templateKey).toBe('server_hardware')
  })

  it('returns undefined for empty extraction', async () => {
    const matcher = new TemplateMatcher({})
    const suggestion = await matcher.match(makeExtraction([], []))
    expect(suggestion).toBeUndefined()
  })

  it('semantic match above threshold returns a suggestion', async () => {
    const tmpl = getTemplate('healthcare')!
    const extractedTypes = tmpl.nodeTypes.slice(0, 3)

    // Build a set of texts that will be requested for both extracted and
    // healthcare template embeddings, so we can return matching vectors
    // only for those.
    const healthcareTexts = new Set<string>()
    for (const nt of tmpl.nodeTypes) {
      healthcareTexts.add(`${nt.label}: ${nt.description}`)
    }
    for (const et of tmpl.edgeTypes) {
      healthcareTexts.add(`${et.label}: ${et.description}`)
    }

    const fakeEmbedder: Embedder = {
      name: () => 'fake-selective',
      model: () => 'fake-model',
      dimension: () => 3,
      embed: async (texts: readonly string[]): Promise<number[][]> =>
        texts.map((text) =>
          healthcareTexts.has(text) ? [1, 0, 0] : [0, 0, 0],
        ),
    }

    const matcher = new TemplateMatcher({
      embedder: fakeEmbedder,
      semanticThreshold: 0.8,
      combinedMinimum: 0.01, // low so that even 3/45 matches qualify
    })

    const extracted = makeExtraction([...extractedTypes], [])

    const suggestion = await matcher.match(extracted)
    expect(suggestion).not.toBeUndefined()
    expect(suggestion!.templateKey).toBe('healthcare')
    expect(suggestion!.overlapScore).toBeCloseTo(1.0, 2)
  })

  it('semantic match below threshold returns undefined', async () => {
    // Zero-vector embedder: cosine similarity is 0 (or NaN) -- no matches.
    const fakeEmbedder: Embedder = {
      name: () => 'fake-zero',
      model: () => 'fake-model',
      dimension: () => 3,
      embed: async (texts: readonly string[]): Promise<number[][]> =>
        texts.map(() => [0, 0, 0]),
    }

    const matcher = new TemplateMatcher({
      embedder: fakeEmbedder,
      semanticThreshold: 0.8,
      combinedMinimum: 0.3,
    })

    const extracted = makeExtraction(
      [{ type: 'entity.alien_species', label: 'Alien Species', description: 'Extraterrestrial' }],
      [{ type: 'orbits', label: 'Orbits', description: 'Orbital relationship' }],
    )

    const suggestion = await matcher.match(extracted)
    expect(suggestion).toBeUndefined()
  })
})

describe('combined score calculation', () => {
  it('correctly computes (overlap + coverage) / 2', () => {
    // overlapScore = intersectionCount / extractedCount
    // coverageScore = intersectionCount / templateCount
    // combined = (overlapScore + coverageScore) / 2

    const overlapScore = 2 / 3
    const coverageScore = 2 / 10
    const combined = (overlapScore + coverageScore) / 2

    expect(combined).toBeCloseTo(0.4333, 3)
  })

  it('threshold of 0.3 is enforced', () => {
    expect(TEMPLATE_MATCH_COMBINED_MINIMUM).toBe(0.3)
  })
})

